"""
CCMT Trainer for ESL grading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import gc
import logging
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional, Tuple, List

from .ccmt_model import ESLCCMTModel
from .ccmt_dataset import ESLCCMTDataset, get_ccmt_collate_fn
from .training_utils import (
    ESLLossFunction, 
    ClassWeightCalculator, 
    ValidationMetrics,
    selective_freeze_embedding_layer,
    get_param_groups,
    maybe_empty_cache,
    compute_correlation
)
from .samplers import create_balanced_sampler
from .text_processor import TextProcessor


class CCMTTrainer:
    """
    Trainer for CCMT ESL grading model
    """
    
    def __init__(self,
                 model: ESLCCMTModel,
                 train_path: str,
                 val_path: str,
                 test_path: str,
                 text_processor: Optional[TextProcessor] = None,
                 # Training parameters
                 batch_size: int = 8,
                 epochs: int = 20,
                 learning_rates: Dict[str, float] = None,
                 optimizer_name: str = "adamw",
                 scheduler_name: str = "cosine",
                 warmup_steps: int = 500,
                 # Loss parameters
                 lambda_kl: float = 0.9,
                 lambda_mse: float = 0.1,
                 soft_target_std: float = 0.3,
                 # Sampling and weighting
                 sampling_strategy: str = "inverse_score",
                 sampling_alpha: float = 0.5,
                 class_weight_beta: float = 0.99,
                 # Data processing
                 remove_low_content: bool = True,
                 filter_scores: bool = True,
                 # Other parameters
                 device: str = "cuda",
                 save_dir: str = "./checkpoints",
                 logger: Optional[logging.Logger] = None,
                 use_wandb: bool = False):
        
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Paths and data
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.text_processor = text_processor
        
        # Training parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rates = learning_rates or {
            'base': 1e-5, 'encoder': 1e-6, 'scale': 1e-3
        }
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.warmup_steps = warmup_steps
        
        # Loss parameters
        self.lambda_kl = lambda_kl
        self.lambda_mse = lambda_mse
        self.soft_target_std = soft_target_std
        
        # Sampling parameters
        self.sampling_strategy = sampling_strategy
        self.sampling_alpha = sampling_alpha
        self.class_weight_beta = class_weight_beta
        
        # Data processing
        self.remove_low_content = remove_low_content
        self.filter_scores = filter_scores
        
        # Utilities
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.use_wandb = use_wandb
        
        # Initialize training components
        self._prepare_data()
        self._setup_training()
        
        # Best model tracking
        self.best_val_mae = float('inf')
        self.best_state_dict = None
        self.best_epoch = 0

    def _prepare_data(self):
        """Prepare training, validation, and test datasets"""
        self.logger.info("Preparing datasets...")
        
        # Load dataframes
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)
        
        self.logger.info(f"Data sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create datasets
        self.train_dataset = ESLCCMTDataset(
            dataframe=train_df,
            text_processor=self.text_processor,
            is_train=True,
            remove_low_content=self.remove_low_content,
            filter_scores=self.filter_scores,
            use_text_cache=True
        )
        
        self.val_dataset = ESLCCMTDataset(
            dataframe=val_df,
            text_processor=self.text_processor,
            is_train=False,
            remove_low_content=self.remove_low_content,
            filter_scores=self.filter_scores,
            use_text_cache=True
        )
        
        self.test_dataset = ESLCCMTDataset(
            dataframe=test_df,
            text_processor=self.text_processor,
            is_train=False,
            remove_low_content=self.remove_low_content,
            filter_scores=self.filter_scores,
            use_text_cache=True
        )
        
        self.logger.info(f"Dataset sizes after filtering - Train: {len(self.train_dataset)}, "
                        f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        
        # Create data loaders
        collate_fn = get_ccmt_collate_fn()
        
        # Create sampler for training
        train_sampler = create_balanced_sampler(
            self.train_dataset, 
            strategy=self.sampling_strategy,
            alpha=self.sampling_alpha
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0,  # Disable multiprocessing to avoid CUDA fork issues
            pin_memory=True,
            persistent_workers=False  # Must be False when num_workers=0
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,  # Disable multiprocessing to avoid CUDA fork issues
            pin_memory=True,
            persistent_workers=False  # Must be False when num_workers=0
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,  # Disable multiprocessing to avoid CUDA fork issues
            pin_memory=True,
            persistent_workers=False  # Must be False when num_workers=0
        )
        
        # Calculate class weights
        class_bins = [i * 0.5 for i in range(21)]  # 0.0 to 10.0 in steps of 0.5
        class_counts = ClassWeightCalculator.get_class_counts_from_scores(
            self.train_dataset.scores, class_bins
        )
        eff_class_counts = (class_counts + 1) ** (1 - self.sampling_alpha)
        self.loss_weights = ClassWeightCalculator.get_effective_number_weights(
            eff_class_counts, self.class_weight_beta
        ).to(self.device)
        
        self.logger.info(f"Class weights calculated: {self.loss_weights}")

    def _setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        # Parameter groups with different learning rates
        param_groups = get_param_groups(
            self.model,
            base_lr=self.learning_rates['base'],
            encoder_lr=self.learning_rates['encoder'],
            scale_lr=self.learning_rates['scale']
        )
        
        # Optimizer
        if self.optimizer_name.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        # Scheduler
        total_steps = len(self.train_loader) * self.epochs
        if self.scheduler_name.lower() == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
                num_cycles=(total_steps - self.warmup_steps) / (4 * len(self.train_loader))
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.criterion = ESLLossFunction(
            lambda_kl=self.lambda_kl,
            lambda_mse=self.lambda_mse,
            num_bins=21
        )
        
        # Mixed precision scaler
        self.scaler = amp.GradScaler('cuda')
        
        self.logger.info(f"Training setup complete. Total steps: {total_steps}")

    def apply_selective_freezing(self, stopwords: List[str]):
        """Apply selective freezing to embedding layers"""
        try:
            # Apply to English encoder
            if hasattr(self.model, 'english_text_encoder'):
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                selective_freeze_embedding_layer(
                    self.model.english_text_encoder, tokenizer, stopwords
                )
            
            # Apply to Vietnamese encoder  
            if hasattr(self.model, 'vietnamese_text_encoder'):
                tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
                selective_freeze_embedding_layer(
                    self.model.vietnamese_text_encoder, tokenizer, stopwords
                )
            
            self.logger.info("Selective freezing applied to embedding layers")
        except Exception as e:
            self.logger.warning(f"Could not apply selective freezing: {e}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_kl_loss = 0.0
        total_mse_loss = 0.0
        total_loss = 0.0
        total_mae = 0.0
        total_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch in pbar:
            # Move batch to device
            audio_chunks = batch['audio_chunks'].to(self.device)
            english_input_ids = batch['english_input_ids'].to(self.device)
            english_attention_mask = batch['english_attention_mask'].to(self.device)
            vietnamese_input_ids = batch['vietnamese_input_ids'].to(self.device)
            vietnamese_attention_mask = batch['vietnamese_attention_mask'].to(self.device)
            true_scores = batch['score'].to(self.device)
            
            # Get sample weights
            target_indices = (true_scores * 2).long().clamp(0, 20)
            weights = self.loss_weights[target_indices]
            
            # Forward pass
            with amp.autocast('cuda'):
                outputs = self.model(
                    audio_chunks=audio_chunks,
                    english_input_ids=english_input_ids,
                    english_attention_mask=english_attention_mask,
                    vietnamese_input_ids=vietnamese_input_ids,
                    vietnamese_attention_mask=vietnamese_attention_mask
                )
                
                # Calculate loss
                loss_outputs = self.criterion(
                    logits=outputs['logits'],
                    expected_scores=outputs['expected_score'],
                    true_scores=true_scores,
                    class_weights=weights
                )
                
                loss = loss_outputs['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            mae = torch.abs(outputs['expected_score'] - true_scores).mean()
            total_kl_loss += loss_outputs['kl_loss'].item()
            total_mse_loss += loss_outputs['mse_loss'].item()
            total_loss += loss.item()
            total_mae += mae.item()
            total_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'MAE': f"{mae.item():.4f}",
                'KL': f"{loss_outputs['kl_loss'].item():.4f}",
                'MSE': f"{loss_outputs['mse_loss'].item():.4f}"
            })
            
            # Memory management
            maybe_empty_cache()
        
        # Average metrics
        avg_kl_loss = total_kl_loss / total_batches
        avg_mse_loss = total_mse_loss / total_batches
        avg_loss = total_loss / total_batches
        avg_mae = total_mae / total_batches
        
        return {
            'train_loss': avg_loss,
            'train_kl_loss': avg_kl_loss,
            'train_mse_loss': avg_mse_loss,
            'train_mae': avg_mae
        }

    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        total_weighted_loss = 0.0
        total_weight = 0.0
        total_mae = 0.0
        total_count = 0
        
        all_predictions = []
        all_true_scores = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                audio_chunks = batch['audio_chunks'].to(self.device)
                english_input_ids = batch['english_input_ids'].to(self.device)
                english_attention_mask = batch['english_attention_mask'].to(self.device)
                vietnamese_input_ids = batch['vietnamese_input_ids'].to(self.device)
                vietnamese_attention_mask = batch['vietnamese_attention_mask'].to(self.device)
                true_scores = batch['score'].to(self.device)
                
                # Forward pass
                with amp.autocast('cuda'):
                    outputs = self.model(
                        audio_chunks=audio_chunks,
                        english_input_ids=english_input_ids,
                        english_attention_mask=english_attention_mask,
                        vietnamese_input_ids=vietnamese_input_ids,
                        vietnamese_attention_mask=vietnamese_attention_mask
                    )
                    pred_scores = outputs['expected_score']
                
                # Calculate weights
                unique_scores, counts = torch.unique(true_scores, return_counts=True)
                freq_map = {score.item(): count.item() for score, count in zip(unique_scores, counts)}
                weights = torch.tensor(
                    [((1.0 / freq_map[score.item()]) ** 0.5) for score in true_scores],
                    device=self.device
                )
                
                # Update metrics
                per_example_loss = (pred_scores - true_scores) ** 2
                weighted_loss = (weights * per_example_loss).sum().item()
                mae_batch = torch.abs(pred_scores - true_scores).sum().item()
                
                total_loss += per_example_loss.sum().item()
                total_weighted_loss += weighted_loss
                total_weight += weights.sum().item()
                total_mae += mae_batch
                total_count += true_scores.size(0)
                
                # Store predictions for correlation
                all_predictions.extend(pred_scores.cpu().numpy().tolist())
                all_true_scores.extend(true_scores.cpu().numpy().tolist())
        
        # Calculate final metrics
        avg_mse = total_loss / total_count
        weighted_mse = total_weighted_loss / total_weight if total_weight > 0 else 0.0
        avg_mae = total_mae / total_count
        correlation = compute_correlation(np.array(all_true_scores), np.array(all_predictions))
        
        maybe_empty_cache()
        
        return {
            'val_mse': avg_mse,
            'val_weighted_mse': weighted_mse,
            'val_mae': avg_mae,
            'val_correlation': correlation
        }

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Apply selective freezing if needed
        try:
            from .text_processing import ALL_STOPWORDS, most_common_words
            train_df = pd.read_csv(self.train_path)
            stopwords = ALL_STOPWORDS.union(most_common_words(train_df, 0.05))
            self.apply_selective_freezing(list(stopwords))
        except Exception as e:
            self.logger.warning(f"Could not apply selective freezing: {e}")
            self.logger.warning("Continuing without selective embedding freezing")
        
        for epoch in range(self.epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            log_message = (
                f"Epoch {epoch + 1}/{self.epochs}: "
                f"Train Loss={train_metrics['train_loss']:.4f}, "
                f"Train MAE={train_metrics['train_mae']:.4f}, "
                f"Val MSE={val_metrics['val_mse']:.4f}, "
                f"Val MAE={val_metrics['val_mae']:.4f}, "
                f"Val Corr={val_metrics['val_correlation']:.4f}"
            )
            print(log_message)
            self.logger.info(log_message)
            
            # Log to wandb if available
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({**train_metrics, **val_metrics, "epoch": epoch + 1})
                except:
                    pass
            
            # Save best model
            current_mae = val_metrics['val_mae']
            if current_mae < self.best_val_mae:
                self.best_val_mae = current_mae
                self.best_epoch = epoch + 1
                self.best_state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                
                # Save checkpoint
                checkpoint_path = os.path.join(self.save_dir, "best_model.pth")
                self.model.save(checkpoint_path)
                
                save_message = f"Best model updated at epoch {epoch + 1} with MAE: {current_mae:.4f}"
                print(save_message)
                self.logger.info(save_message)
                
                if self.use_wandb:
                    try:
                        import wandb
                        wandb.log({"best_val_mae": current_mae, "best_epoch": epoch + 1})
                    except:
                        pass
            
            # Early stopping if performance degrades too much
            elif current_mae > self.best_val_mae * 1.15:
                if self.best_state_dict is not None:
                    self.model.load_state_dict(self.best_state_dict)
                    reload_message = "Performance degraded; reloaded best model"
                    print(reload_message)
                    self.logger.info(reload_message)
        
        # Load best model at end
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            final_message = f"Training completed. Best model from epoch {self.best_epoch} with MAE: {self.best_val_mae:.4f}"
            print(final_message)
            self.logger.info(final_message)

    def test(self, output_csv_path: Optional[str] = None) -> Dict[str, float]:
        """Test the model"""
        if output_csv_path is None:
            output_csv_path = os.path.join(self.save_dir, "test_predictions.csv")
        
        self.model.eval()
        
        total_mse = 0.0
        total_mae = 0.0
        total_count = 0
        
        all_predictions = []
        all_true_scores = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Move batch to device
                audio_chunks = batch['audio_chunks'].to(self.device)
                english_input_ids = batch['english_input_ids'].to(self.device)
                english_attention_mask = batch['english_attention_mask'].to(self.device)
                vietnamese_input_ids = batch['vietnamese_input_ids'].to(self.device)
                vietnamese_attention_mask = batch['vietnamese_attention_mask'].to(self.device)
                true_scores = batch['score'].to(self.device)
                
                # Forward pass
                with amp.autocast('cuda'):
                    outputs = self.model(
                        audio_chunks=audio_chunks,
                        english_input_ids=english_input_ids,
                        english_attention_mask=english_attention_mask,
                        vietnamese_input_ids=vietnamese_input_ids,
                        vietnamese_attention_mask=vietnamese_attention_mask
                    )
                    pred_scores = outputs['expected_score']
                
                # Calculate metrics
                mse = F.mse_loss(pred_scores, true_scores, reduction='sum').item()
                mae = torch.abs(pred_scores - true_scores).sum().item()
                
                total_mse += mse
                total_mae += mae
                total_count += true_scores.size(0)
                
                # Store predictions
                all_predictions.extend(pred_scores.cpu().numpy().tolist())
                all_true_scores.extend(true_scores.cpu().numpy().tolist())
        
        # Calculate final metrics
        avg_mse = total_mse / total_count
        avg_mae = total_mae / total_count
        correlation = compute_correlation(np.array(all_true_scores), np.array(all_predictions))
        
        # Save predictions
        results_df = pd.DataFrame({
            'GroundTruth': all_true_scores,
            'PredictedScore': all_predictions,
            'AbsoluteError': [abs(t - p) for t, p in zip(all_true_scores, all_predictions)],
            'SquaredError': [(t - p) ** 2 for t, p in zip(all_true_scores, all_predictions)]
        })
        
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results_df.to_csv(output_csv_path, index=False)
        
        # Log results
        test_message = f"""
        === TEST RESULTS ===
        Test MSE: {avg_mse:.4f}
        Test MAE: {avg_mae:.4f}
        Test Correlation: {correlation:.4f}
        Total samples: {total_count}
        Results saved to: {output_csv_path}
        ==================
        """
        print(test_message)
        self.logger.info(test_message.replace('\n', ' '))
        
        # Log to wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.log({
                    "test_mse": avg_mse,
                    "test_mae": avg_mae,
                    "test_correlation": correlation,
                    "test_samples": total_count
                })
            except:
                pass
        
        maybe_empty_cache()
        
        return {
            'test_mse': avg_mse,
            'test_mae': avg_mae,
            'test_correlation': correlation,
            'results_df': results_df
        }
"""
Main trainer for CCMT English speaking scoring
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from pathlib import Path
from dataclasses import dataclass
import os
from .losses import create_loss_function, compute_class_weights
from .metrics import ScoringMetrics, evaluate_model
from .callbacks import Callback, create_default_callbacks

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Training state container"""
    epoch: int = 0
    batch_idx: int = 0
    total_epochs: int = 0
    total_batches: int = 0
    model: Optional[nn.Module] = None  # This will be handled specially
    optimizer: Optional[torch.optim.Optimizer] = None  # This will be handled specially
    scheduler: Optional[Any] = None  # This will be handled specially
    metrics: Dict[str, float] = None
    batch_loss: float = 0.0
    learning_rate: float = 0.0
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert training state to dictionary"""
        # Don't include model/optimizer/scheduler in the dict
        return {
            'epoch': self.epoch,
            'batch_idx': self.batch_idx,
            'total_epochs': self.total_epochs,
            'total_batches': self.total_batches,
            'metrics': self.metrics,
            'batch_loss': self.batch_loss,
            'learning_rate': self.learning_rate
        }


class CCMTTrainer:
    """
    Main trainer for CCMT model
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_function: Optional[nn.Module] = None,
        callbacks: Optional[List[Callback]] = None,
        device: str = "cpu",
        task_type: str = "classification",
        num_classes: int = 21,
        mixed_precision: bool = False,
        gradient_clip: float = 1.0
    ):
        """
        Initialize CCMT trainer
        
        Args:
            model: CCMT model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader (optional)
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            loss_function: Loss function
            callbacks: List of training callbacks
            device: Training device
            task_type: "classification" or "regression"
            num_classes: Number of classes
            mixed_precision: Use mixed precision training
            gradient_clip: Gradient clipping value
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.task_type = task_type
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Set up loss function
        if loss_function is None:
            self.loss_function = create_loss_function(task_type)
        else:
            self.loss_function = loss_function
        
        # Set up callbacks
        if callbacks is None:
            self.callbacks = create_default_callbacks()
        else:
            self.callbacks = callbacks
        
        # Set up mixed precision training
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.state = TrainingState()
        self.best_val_metric = float('inf')
        
        logger.info(f"Trainer initialized with device: {device}")
        logger.info(f"Task type: {task_type}, Num classes: {num_classes}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train(
        self,
        num_epochs: int,
        save_dir: Optional[str] = None,
        eval_every_n_epochs: int = 1,
        log_every_n_batches: int = 10
    ) -> Dict[str, Any]:
        """
        Main training loop
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save results
            eval_every_n_epochs: Evaluate every N epochs
            log_every_n_batches: Log every N batches
        
        Returns:
            Training history
        """
        # Initialize training state
        self.state.total_epochs = num_epochs
        self.state.total_batches = len(self.train_dataloader)
        self.state.model = self.model
        self.state.optimizer = self.optimizer
        self.state.scheduler = self.scheduler
        
        # Create save directory
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Start training
        self._call_callbacks('on_train_begin', self.state.to_dict())
        
        try:
            for epoch in range(num_epochs):
                self.state.epoch = epoch
                
                # Training phase
                train_loss = self._train_epoch(log_every_n_batches)
                history['train_loss'].append(train_loss)
                
                # Validation phase
                if self.val_dataloader and (epoch + 1) % eval_every_n_epochs == 0:
                    val_metrics = self._validate_epoch()
                    val_loss = val_metrics.get('loss', float('inf'))
                    
                    history['val_loss'].append(val_loss)
                    history['val_metrics'].append(val_metrics)
                    
                    # Update state with metrics
                    self.state.metrics = {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        **val_metrics
                    }
                else:
                    self.state.metrics = {'train_loss': train_loss}
                
                # Learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                history['learning_rates'].append(current_lr)
                self.state.learning_rate = current_lr
                
                # Check for early stopping
                early_stop = any(
                    getattr(callback, 'early_stop', False) 
                    for callback in self.callbacks
                )
                
                if early_stop:
                    logger.info("Early stopping triggered")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            self._call_callbacks('on_train_end', self.state.to_dict())
        
        # Save training history
        if save_dir:
            self._save_history(history, save_dir)
        
        return history
    
    def _train_epoch(self, log_every_n_batches: int = 10) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Start epoch
        self._call_callbacks('on_epoch_begin', self.state.epoch, self.state.to_dict())
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            self.state.batch_idx = batch_idx
            
            # Start batch
            self._call_callbacks('on_batch_begin', batch_idx, self.state.to_dict())
            
            # Forward pass
            loss = self._training_step(batch)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update state
            self.state.batch_loss = loss.item()
            self.state.learning_rate = self.optimizer.param_groups[0]['lr']
            
            # End batch
            self._call_callbacks('on_batch_end', batch_idx, self.state.to_dict())
            
            # Log progress
            if (batch_idx + 1) % log_every_n_batches == 0:
                avg_loss = total_loss / num_batches
                logger.info(
                    f"Epoch {self.state.epoch + 1}, Batch {batch_idx + 1}/{len(self.train_dataloader)}: "
                    f"Loss = {loss.item():.4f}, Avg Loss = {avg_loss:.4f}"
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # End epoch
        self._call_callbacks('on_epoch_end', self.state.epoch, self.state.to_dict())
        
        return avg_loss
    
    def _training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Single training step"""
        # Move data to device
        inputs = batch['ccmt_input'].to(self.device)
        targets = batch['targets'].to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if self.mixed_precision and self.scaler:
            with torch.cuda.amp.autocast():
                predictions = self.model(inputs)
                loss = self.loss_function(predictions, targets)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        else:
            # Standard training
            predictions = self.model(inputs)
            loss = self.loss_function(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Optimizer step
            self.optimizer.step()
        
        # Scheduler step (if not epoch-based)
        if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if hasattr(self.scheduler, 'step_every_batch') and self.scheduler.step_every_batch:
                self.scheduler.step()
        
        return loss
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        if not self.val_dataloader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        metrics_calculator = ScoringMetrics(task_type=self.task_type, num_classes=self.num_classes)
        
        # Start validation
        self._call_callbacks('on_validation_begin', self.state.to_dict())
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move data to device
                inputs = batch['ccmt_input'].to(self.device)
                targets = batch['targets'].to(self.device)
                scores = batch.get('scores', targets * 0.5).to(self.device)
                
                # Forward pass
                predictions = self.model(inputs)
                loss = self.loss_function(predictions, targets)
                
                total_loss += loss.item()
                
                # Update metrics
                metrics_calculator.update(predictions, targets, scores)
        
        # Compute metrics
        metrics = metrics_calculator.compute()
        metrics['loss'] = total_loss / len(self.val_dataloader)
        
        # End validation
        self._call_callbacks('on_validation_end', self.state.to_dict())
        
        return metrics
    
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model on a dataset
        
        Args:
            dataloader: Data loader to evaluate on (uses test_dataloader if None)
        
        Returns:
            Evaluation metrics
        """
        if dataloader is None:
            dataloader = self.test_dataloader
        
        if dataloader is None:
            logger.warning("No dataloader provided for evaluation")
            return {}
        
        return evaluate_model(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
            task_type=self.task_type
        )
    
    def _call_callbacks(self, method_name: str, *args, **kwargs):
        """Call method on all callbacks"""
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                try:
                    # Convert state to dict before passing to callbacks
                    if 'trainer_state' in kwargs:
                        kwargs['trainer_state'] = kwargs['trainer_state'].to_dict() if hasattr(kwargs['trainer_state'], 'to_dict') else kwargs['trainer_state']
                    getattr(callback, method_name)(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Callback {type(callback).__name__}.{method_name} failed: {e}")
    
    def _save_history(self, history: Dict[str, Any], save_dir: str):
        """Save training history"""
        try:
            import json
            history_file = Path(save_dir) / "training_history.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_history = {}
            for key, value in history.items():
                if isinstance(value, list):
                    json_history[key] = [
                        item.tolist() if isinstance(item, np.ndarray) else item
                        for item in value
                    ]
                else:
                    json_history[key] = value
            
            with open(history_file, 'w') as f:
                json.dump(json_history, f, indent=2)
            
            logger.info(f"Training history saved to {history_file}")
        
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")
    
    def save_model(self, save_path: str, include_optimizer: bool = False):
        """Save model checkpoint"""
        try:
            # Only save state dictionaries
            checkpoint = {
                'model_state_dict': self.model.state_dict(),  # ✅ Only state_dict
                'model_config': {
                    'task_type': self.task_type,
                    'num_classes': self.num_classes
                }
            }
            
            if include_optimizer:
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                if self.scheduler:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            # Save checkpoint
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
        
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, load_path: str, load_optimizer: bool = False):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if requested
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint and self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                self.state.epoch = checkpoint.get('epoch', 0)
                self.best_val_metric = checkpoint.get('best_val_metric', float('inf'))
            
            logger.info(f"Model loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise


def create_trainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    config: Dict[str, Any],
    val_dataloader: Optional[DataLoader] = None,
    test_dataloader: Optional[DataLoader] = None
) -> CCMTTrainer:
    """
    Factory function to create trainer from configuration
    
    Args:
        model: CCMT model
        train_dataloader: Training data loader
        config: Training configuration
        val_dataloader: Validation data loader
        test_dataloader: Test data loader
    
    Returns:
        Configured trainer
    """
    # Set up optimizer    @dataclass
    class TrainingState:
        """Training state container"""
        epoch: int = 0
        batch_idx: int = 0
        total_epochs: int = 0
        total_batches: int = 0
        model: Optional[nn.Module] = None  # This will be handled specially
        optimizer: Optional[torch.optim.Optimizer] = None  # This will be handled specially
        scheduler: Optional[Any] = None  # This will be handled specially
        metrics: Dict[str, float] = None
        batch_loss: float = 0.0
        learning_rate: float = 0.0
        
        def __post_init__(self):
            if self.metrics is None:
                self.metrics = {}
                
        def to_dict(self) -> Dict[str, Any]:
            """Convert training state to dictionary"""
            # Don't include model/optimizer/scheduler in the dict
            return {
                'epoch': self.epoch,
                'batch_idx': self.batch_idx,
                'total_epochs': self.total_epochs,
                'total_batches': self.total_batches,
                'metrics': self.metrics,
                'batch_loss': self.batch_loss,
                'learning_rate': self.learning_rate
            }            @dataclass
            class TrainingState:
                """Training state container"""
                epoch: int = 0
                batch_idx: int = 0
                total_epochs: int = 0
                total_batches: int = 0
                model: Optional[nn.Module] = None  # This will be handled specially
                optimizer: Optional[torch.optim.Optimizer] = None  # This will be handled specially
                scheduler: Optional[Any] = None  # This will be handled specially
                metrics: Dict[str, float] = None
                batch_loss: float = 0.0
                learning_rate: float = 0.0
                
                def __post_init__(self):
                    if self.metrics is None:
                        self.metrics = {}
                        
                def to_dict(self) -> Dict[str, Any]:
                    """Convert training state to dictionary"""
                    # Don't include model/optimizer/scheduler in the dict
                    return {
                        'epoch': self.epoch,
                        'batch_idx': self.batch_idx,
                        'total_epochs': self.total_epochs,
                        'total_batches': self.total_batches,
                        'metrics': self.metrics,
                        'batch_loss': self.batch_loss,
                        'learning_rate': self.learning_rate
                    }                    @dataclass
                    class TrainingState:
                        """Training state container"""
                        epoch: int = 0
                        batch_idx: int = 0
                        total_epochs: int = 0
                        total_batches: int = 0
                        model: Optional[nn.Module] = None  # This will be handled specially
                        optimizer: Optional[torch.optim.Optimizer] = None  # This will be handled specially
                        scheduler: Optional[Any] = None  # This will be handled specially
                        metrics: Dict[str, float] = None
                        batch_loss: float = 0.0
                        learning_rate: float = 0.0
                        
                        def __post_init__(self):
                            if self.metrics is None:
                                self.metrics = {}
                                
                        def to_dict(self) -> Dict[str, Any]:
                            """Convert training state to dictionary"""
                            # Don't include model/optimizer/scheduler in the dict
                            return {
                                'epoch': self.epoch,
                                'batch_idx': self.batch_idx,
                                'total_epochs': self.total_epochs,
                                'total_batches': self.total_batches,
                                'metrics': self.metrics,
                                'batch_loss': self.batch_loss,
                                'learning_rate': self.learning_rate
                            }                            @dataclass
                            class TrainingState:
                                """Training state container"""
                                epoch: int = 0
                                batch_idx: int = 0
                                total_epochs: int = 0
                                total_batches: int = 0
                                model: Optional[nn.Module] = None  # This will be handled specially
                                optimizer: Optional[torch.optim.Optimizer] = None  # This will be handled specially
                                scheduler: Optional[Any] = None  # This will be handled specially
                                metrics: Dict[str, float] = None
                                batch_loss: float = 0.0
                                learning_rate: float = 0.0
                                
                                def __post_init__(self):
                                    if self.metrics is None:
                                        self.metrics = {}
                                        
                                def to_dict(self) -> Dict[str, Any]:
                                    """Convert training state to dictionary"""
                                    # Don't include model/optimizer/scheduler in the dict
                                    return {
                                        'epoch': self.epoch,
                                        'batch_idx': self.batch_idx,
                                        'total_epochs': self.total_epochs,
                                        'total_batches': self.total_batches,
                                        'metrics': self.metrics,
                                        'batch_loss': self.batch_loss,
                                        'learning_rate': self.learning_rate
                                    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Set up scheduler
    scheduler = None
    if config.get('use_scheduler', True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('num_epochs', 50),
            eta_min=config.get('min_lr', 1e-6)
        )
    
    # Set up loss function
    loss_config = {
        'num_classes': config.get('num_classes', 21),
        'label_smoothing': config.get('label_smoothing', 0.0)
    }
    
    # Compute class weights if needed
    if config.get('use_class_weights', False) and hasattr(train_dataloader.dataset, 'get_class_weights'):
        class_weights = train_dataloader.dataset.get_class_weights()
        loss_config['class_weights'] = class_weights
    
    loss_function = create_loss_function(
        task_type=config.get('task_type', 'classification'),
        loss_config=loss_config
    )
    
    # Set up callbacks
    callbacks = create_default_callbacks(
        output_dir=config.get('output_dir', './outputs'),
        patience=config.get('patience', 10)
    )
    
    # Create trainer
    trainer = CCMTTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        callbacks=callbacks,
        device=config.get('device', 'cpu'),
        task_type=config.get('task_type', 'classification'),
        num_classes=config.get('num_classes', 21),
        mixed_precision=config.get('mixed_precision', False),
        gradient_clip=config.get('gradient_clip', 1.0)
    )
    
    return trainer


# Example usage
if __name__ == "__main__":
    print("CCMT Trainer - Example usage")
    
    # This would require actual model and data to run
    # Just showing the basic structure
    
    config = {
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'task_type': 'classification',
        'num_classes': 21,
        'device': 'cpu',
        'output_dir': './outputs'
    }
    
    print(f"Example config: {config}")
    print("Trainer ready to use with actual model and data!")
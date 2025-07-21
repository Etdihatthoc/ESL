"""
Training utilities for CCMT ESL grading model
Contains loss functions, soft target generation, class weighting, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from scipy.stats import truncnorm
from collections import Counter
from typing import Dict, List, Tuple, Optional


class SoftTargetGenerator:
    """
    Generate soft target distributions using truncated Gaussian
    """
    def __init__(self, num_bins: int = 21, score_range: Tuple[float, float] = (0.0, 10.0)):
        self.num_bins = num_bins
        self.score_range = score_range
        self.bin_centers = np.linspace(score_range[0], score_range[1], num_bins)
    
    def create_soft_targets(self, scores: torch.Tensor, std: float = 0.3) -> torch.Tensor:
        """
        Generate soft target distributions using a truncated Gaussian centered on each score.
        
        Args:
            scores: [batch_size], scalar scores in [0, 10]
            std: Gaussian standard deviation
            
        Returns:
            torch.Tensor: [batch_size, num_bins], soft label distributions
        """
        scores_np = scores.cpu().numpy()
        batch_size = scores_np.shape[0]
        
        soft_labels = np.zeros((batch_size, self.num_bins), dtype=np.float32)
        
        for i, score in enumerate(scores_np):
            # Add small random variation to std
            scaled_std = std + random.uniform(-0.05, 0.05)
            
            # Define truncated Gaussian bounds
            a = (self.score_range[0] - score) / scaled_std
            b = (self.score_range[1] - score) / scaled_std
            dist = truncnorm(a, b, loc=score, scale=scaled_std)
            
            # Get normalized probabilities
            probs = dist.pdf(self.bin_centers)
            probs /= probs.sum()  # Normalize
            
            soft_labels[i] = probs
        
        return torch.from_numpy(soft_labels).to(scores.device)


class ESLLossFunction(nn.Module):
    """
    Combined loss function for ESL grading with KL divergence and MSE
    """
    def __init__(self, 
                 lambda_kl: float = 0.9,
                 lambda_mse: float = 0.1,
                 num_bins: int = 21):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_mse = lambda_mse
        self.num_bins = num_bins
        self.soft_target_generator = SoftTargetGenerator(num_bins)
        
    def forward(self, 
                logits: torch.Tensor,
                expected_scores: torch.Tensor,
                true_scores: torch.Tensor,
                class_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            logits: [batch_size, num_bins] - model logits
            expected_scores: [batch_size] - predicted continuous scores  
            true_scores: [batch_size] - ground truth scores
            class_weights: [batch_size] - per-sample weights
            
        Returns:
            Dictionary with loss components
        """
        # Generate soft targets
        soft_targets = self.soft_target_generator.create_soft_targets(true_scores)
        
        # KL divergence loss
        log_probs = F.log_softmax(logits, dim=-1)
        kl_loss_per_sample = F.kl_div(log_probs, soft_targets, reduction='none').sum(dim=-1)
        weighted_kl_loss = (kl_loss_per_sample * class_weights).sum() / class_weights.sum()
        
        # MSE loss
        mse_loss_per_sample = F.mse_loss(expected_scores, true_scores, reduction='none')
        weighted_mse_loss = (mse_loss_per_sample * class_weights).sum() / class_weights.sum()
        
        # Combined loss
        total_loss = self.lambda_kl * weighted_kl_loss + self.lambda_mse * weighted_mse_loss
        
        return {
            'total_loss': total_loss,
            'kl_loss': weighted_kl_loss,
            'mse_loss': weighted_mse_loss,
            'kl_loss_per_sample': kl_loss_per_sample,
            'mse_loss_per_sample': mse_loss_per_sample
        }


class ClassWeightCalculator:
    """
    Calculate effective class weights for imbalanced datasets
    """
    @staticmethod
    def get_class_counts_from_scores(scores: List[float], class_bins: List[float]) -> np.ndarray:
        """Get counts for each class bin"""
        class_to_index = {v: i for i, v in enumerate(class_bins)}
        indices = [class_to_index.get(score, len(class_bins)-1) for score in scores]
        counts = np.zeros(len(class_bins), dtype=int)
        for idx in indices:
            if idx < len(class_bins):
                counts[idx] += 1
        return counts
    
    @staticmethod
    def get_effective_number_weights(class_counts: np.ndarray, beta: float = 0.9999) -> torch.Tensor:
        """
        Implements Cui et al. (2019) class-balanced loss weights
        """
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)  # Avoid division by zero
        weights = weights / np.mean(weights)  # Normalize to mean 1
        return torch.tensor(weights, dtype=torch.float32)
    
    @staticmethod
    def get_sample_weights(scores: List[float], 
                          class_bins: List[float],
                          sampling_alpha: float = 0.5,
                          beta: float = 0.99) -> torch.Tensor:
        """
        Get per-sample weights for loss computation
        """
        class_counts = ClassWeightCalculator.get_class_counts_from_scores(scores, class_bins)
        
        # Compensate for sampling bias
        eff_class_counts = (class_counts + 1) ** (1 - sampling_alpha)
        loss_weights = ClassWeightCalculator.get_effective_number_weights(eff_class_counts, beta)
        
        return loss_weights


def selective_freeze_embedding_layer(model, tokenizer, unfrozen_words: List[str]):
    """
    Freeze embedding layer except for specified words
    """
    if not hasattr(model, 'encoder'):
        return
        
    embedding_layer = model.encoder.embeddings.word_embeddings
    embedding_layer.weight.requires_grad = True
    
    # Get token IDs of unfrozen words and special tokens
    token_ids = set()
    for word in unfrozen_words:
        try:
            ids = tokenizer(word, add_special_tokens=False)['input_ids']
            token_ids.update(ids)
        except:
            continue
    
    # Add all special token IDs
    if hasattr(tokenizer, "all_special_ids"):
        token_ids.update(tokenizer.all_special_ids)
    else:
        for tok in tokenizer.all_special_tokens:
            try:
                ids = tokenizer(tok, add_special_tokens=False)['input_ids']
                token_ids.update(ids)
            except:
                continue
    
    vocab_size, hidden_size = embedding_layer.weight.shape
    grad_mask = torch.zeros(vocab_size, 1, device=embedding_layer.weight.device)
    for idx in token_ids:
        if idx < vocab_size:
            grad_mask[idx] = 1.0
    
    # Register gradient hook
    def hook_fn(grad):
        return grad * grad_mask
    
    embedding_layer.weight.register_hook(hook_fn)
    print(f"Selective freezing applied to {len(token_ids)} tokens out of {vocab_size}")


def get_param_groups(model, base_lr: float = 1e-5, encoder_lr: float = 1e-6, scale_lr: float = 1e-3):
    """
    Group parameters for different learning rates
    """
    special_params = []
    encoder_params = []
    base_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'scale' in name or 'alpha' in name:
            special_params.append(param)
        elif ('encoder' in name or 'audio_encoder' in name or 
              'english_text_encoder' in name or 'vietnamese_text_encoder' in name):
            encoder_params.append(param)
        else:
            base_params.append(param)
    
    param_groups = []
    if base_params:
        param_groups.append({'params': base_params, 'lr': base_lr})
    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': encoder_lr})
    if special_params:
        param_groups.append({'params': special_params, 'lr': scale_lr})
    
    print(f"Parameter groups: base={len(base_params)}, encoder={len(encoder_params)}, special={len(special_params)}")
    
    return param_groups


def maybe_empty_cache(threshold: float = 0.93):
    """
    Empty CUDA cache if memory usage is high
    """
    if torch.cuda.is_available():
        try:
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            if reserved / total > threshold:
                torch.cuda.empty_cache()
        except Exception:
            torch.cuda.empty_cache()


class ValidationMetrics:
    """
    Calculate validation metrics with inverse frequency weighting
    """
    @staticmethod
    def calculate_weighted_metrics(true_scores: torch.Tensor, 
                                 pred_scores: torch.Tensor,
                                 alpha: float = 0.5) -> Tuple[float, float]:
        """
        Calculate weighted MSE with inverse frequency weighting
        """
        # Calculate frequency weights
        unique_scores, counts = torch.unique(true_scores, return_counts=True)
        freq_map = {score.item(): count.item() for score, count in zip(unique_scores, counts)}
        
        weights = torch.tensor(
            [((1.0 / freq_map[score.item()]) ** alpha) for score in true_scores],
            device=true_scores.device
        )
        
        # Weighted MSE
        per_example_loss = (pred_scores - true_scores) ** 2
        weighted_loss = (weights * per_example_loss).sum().item()
        total_weight = weights.sum().item()
        weighted_avg = weighted_loss / total_weight if total_weight > 0 else 0.0
        
        # Regular MSE
        per_item_avg = per_example_loss.mean().item()
        
        return weighted_avg, per_item_avg


def compute_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient
    """
    try:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    except:
        return 0.0
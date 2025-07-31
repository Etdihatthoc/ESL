"""
Loss functions for CCMT English speaking scoring task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class ScoringLoss(nn.Module):
    """
    Main loss function for speaking scoring task
    Supports both classification and regression modes
    """
    
    def __init__(
        self,
        task_type: str = "classification",
        num_classes: int = 21,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        """
        Initialize scoring loss
        
        Args:
            task_type: "classification" or "regression"
            num_classes: Number of classes for classification
            class_weights: Class weights for imbalanced data
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        
        if task_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
        else:  # regression
            self.loss_fn = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss
        
        Args:
            predictions: Model predictions (batch, num_classes) or (batch, 1)
            targets: Ground truth targets (batch,) for classification or (batch,) for regression
        
        Returns:
            Loss tensor
        """
        if self.task_type == "classification":
            # predictions: (batch, num_classes), targets: (batch,) with class indices
            return self.loss_fn(predictions, targets.long())
        else:
            # predictions: (batch, 1), targets: (batch,) with continuous scores
            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)
            return self.loss_fn(predictions, targets.float())


class ClassificationLoss(nn.Module):
    """Classification loss for speaking scoring"""
    
    def __init__(
        self,
        num_classes: int = 21,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        use_focal: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if use_focal:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        
        if self.use_focal:
            # Focal loss for handling class imbalance
            ce_loss = self.ce_loss(predictions, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
            return focal_loss.mean()
        else:
            return self.ce_loss(predictions, targets)


class RegressionLoss(nn.Module):
    """Regression loss for speaking scoring"""
    
    def __init__(
        self,
        loss_type: str = "mse",  # "mse", "mae", "huber", "smooth_l1"
        huber_delta: float = 1.0
    ):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss(delta=huber_delta)
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        return self.loss_fn(predictions, targets.float())


class CombinedLoss(nn.Module):
    """
    Combined loss that can mix classification and regression losses
    Useful for multi-task learning or ordinal regression
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        regression_weight: float = 0.1,
        num_classes: int = 21,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        
        self.classification_loss = ClassificationLoss(
            num_classes=num_classes,
            class_weights=class_weights
        )
        self.regression_loss = RegressionLoss(loss_type="mse")
    
    def forward(
        self, 
        class_predictions: torch.Tensor,
        reg_predictions: torch.Tensor, 
        targets: torch.Tensor,
        scores: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            class_predictions: Classification predictions (batch, num_classes)
            reg_predictions: Regression predictions (batch, 1)
            targets: Class targets (batch,)
            scores: Continuous score targets (batch,)
        
        Returns:
            Dictionary with loss components
        """
        class_loss = self.classification_loss(class_predictions, targets)
        reg_loss = self.regression_loss(reg_predictions, scores)
        
        total_loss = (self.classification_weight * class_loss + 
                     self.regression_weight * reg_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': class_loss,
            'regression_loss': reg_loss
        }


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for scoring tasks
    Treats the problem as a series of binary classifications
    """
    
    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes
        self.binary_ce = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Raw logits (batch, num_classes-1)
            targets: Class targets (batch,)
        """
        batch_size = targets.size(0)
        
        # Create ordinal targets
        # For class k, first k elements should be 1, rest should be 0
        ordinal_targets = torch.zeros(batch_size, self.num_classes - 1, device=targets.device)
        
        for i in range(batch_size):
            target_class = targets[i].long()
            if target_class > 0:
                ordinal_targets[i, :target_class] = 1.0
        
        return self.binary_ce(predictions, ordinal_targets)


class DistributionLoss(nn.Module):
    """
    Distribution-based loss for scoring
    Models scores as a distribution over possible values
    """
    
    def __init__(self, num_classes: int = 21, temperature: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def _create_target_distribution(self, targets: torch.Tensor, std: float = 0.5) -> torch.Tensor:
        """Create Gaussian distribution centered at target score"""
        batch_size = targets.size(0)
        distributions = torch.zeros(batch_size, self.num_classes, device=targets.device)
        
        # Score values: 0, 0.5, 1.0, ..., 10.0
        score_values = torch.arange(self.num_classes, device=targets.device) * 0.5
        
        for i in range(batch_size):
            target_score = targets[i].float()
            # Create Gaussian distribution
            dist = torch.exp(-0.5 * ((score_values - target_score) / std) ** 2)
            dist = dist / dist.sum()  # Normalize
            distributions[i] = dist
        
        return distributions
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert targets (class indices) to actual scores
        if targets.dtype in [torch.int, torch.long]:
            scores = targets.float() * 0.5  # Convert class index to score
        else:
            scores = targets
        
        # Create target distributions
        target_dist = self._create_target_distribution(scores)
        
        # Apply temperature scaling and softmax
        pred_log_softmax = F.log_softmax(predictions / self.temperature, dim=-1)
        
        return self.kl_div(pred_log_softmax, target_dist)


def create_loss_function(
    task_type: str = "classification",
    loss_config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Factory function to create loss function
    
    Args:
        task_type: "classification", "regression", "combined", "ordinal"
        loss_config: Loss configuration dictionary
    
    Returns:
        Loss function
    """
    if loss_config is None:
        loss_config = {}
    
    if task_type == "classification":
        return ClassificationLoss(**loss_config)
    elif task_type == "regression":
        return RegressionLoss(**loss_config)
    elif task_type == "combined":
        return CombinedLoss(**loss_config)
    elif task_type == "ordinal":
        return OrdinalRegressionLoss(**loss_config)
    elif task_type == "distribution":
        return DistributionLoss(**loss_config)
    else:
        return ScoringLoss(task_type=task_type, **loss_config)


def compute_class_weights(targets: torch.Tensor, num_classes: int = 21) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset
    
    Args:
        targets: Target tensor with class indices
        num_classes: Number of classes
    
    Returns:
        Class weights tensor
    """
    class_counts = torch.bincount(targets.long(), minlength=num_classes)
    total_samples = len(targets)
    
    # Inverse frequency weighting
    class_weights = total_samples / (num_classes * class_counts.float() + 1e-8)
    
    return class_weights


# Example usage and testing
if __name__ == "__main__":
    # Test classification loss
    batch_size, num_classes = 8, 21
    
    predictions_class = torch.randn(batch_size, num_classes)
    targets_class = torch.randint(0, num_classes, (batch_size,))
    
    class_loss = ClassificationLoss(num_classes=num_classes)
    loss_value = class_loss(predictions_class, targets_class)
    print(f"Classification loss: {loss_value.item():.4f}")
    
    # Test regression loss
    predictions_reg = torch.randn(batch_size, 1)
    targets_reg = torch.rand(batch_size) * 10  # Scores 0-10
    
    reg_loss = RegressionLoss()
    loss_value = reg_loss(predictions_reg, targets_reg)
    print(f"Regression loss: {loss_value.item():.4f}")
    
    # Test combined loss
    combined_loss = CombinedLoss(num_classes=num_classes)
    losses = combined_loss(predictions_class, predictions_reg, targets_class, targets_reg)
    print(f"Combined loss: {losses['total_loss'].item():.4f}")
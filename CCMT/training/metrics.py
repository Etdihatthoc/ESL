"""
Evaluation metrics for CCMT English speaking scoring task
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate classification accuracy
    
    Args:
        predictions: Model predictions (batch, num_classes) or (batch,)
        targets: Ground truth targets (batch,)
    
    Returns:
        Accuracy as float
    """
    if predictions.dim() > 1:
        # Convert logits to class predictions
        pred_classes = torch.argmax(predictions, dim=-1)
    else:
        pred_classes = predictions
    
    correct = (pred_classes == targets).float()
    return correct.mean().item()


def calculate_correlation(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate Pearson and Spearman correlation
    
    Args:
        predictions: Predicted scores
        targets: True scores
    
    Returns:
        Dictionary with correlation coefficients
    """
    try:
        pearson_corr, pearson_p = pearsonr(predictions, targets)
        spearman_corr, spearman_p = spearmanr(predictions, targets)
        
        return {
            'pearson': float(pearson_corr),
            'pearson_pvalue': float(pearson_p),
            'spearman': float(spearman_corr),
            'spearman_pvalue': float(spearman_p)
        }
    except Exception as e:
        logger.warning(f"Failed to calculate correlation: {e}")
        return {
            'pearson': 0.0,
            'pearson_pvalue': 1.0,
            'spearman': 0.0,
            'spearman_pvalue': 1.0
        }


def calculate_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    return float(mean_absolute_error(targets, predictions))


def calculate_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Mean Squared Error"""
    return float(mean_squared_error(targets, predictions))


def calculate_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Root Mean Squared Error"""
    return float(np.sqrt(mean_squared_error(targets, predictions)))


def exact_match_accuracy(predictions: np.ndarray, targets: np.ndarray, tolerance: float = 0.0) -> float:
    """
    Calculate exact match accuracy with optional tolerance
    
    Args:
        predictions: Predicted scores
        targets: True scores
        tolerance: Tolerance for considering a match
    
    Returns:
        Exact match accuracy
    """
    matches = np.abs(predictions - targets) <= tolerance
    return float(matches.mean())


def within_range_accuracy(predictions: np.ndarray, targets: np.ndarray, range_size: float = 0.5) -> float:
    """
    Calculate accuracy within a score range
    
    Args:
        predictions: Predicted scores
        targets: True scores
        range_size: Size of acceptable range (e.g., 0.5 points)
    
    Returns:
        Within-range accuracy
    """
    return exact_match_accuracy(predictions, targets, tolerance=range_size)


def classification_to_scores(class_predictions: torch.Tensor) -> np.ndarray:
    """
    Convert classification predictions to continuous scores
    
    Args:
        class_predictions: Class probabilities (batch, num_classes)
    
    Returns:
        Continuous scores (batch,)
    """
    if class_predictions.dim() == 1:
        # Already class indices
        return (class_predictions.cpu().numpy() * 0.5)
    
    # Convert probabilities to expected scores
    num_classes = class_predictions.shape[1]
    score_values = torch.arange(num_classes, device=class_predictions.device).float() * 0.5
    expected_scores = torch.sum(class_predictions * score_values.unsqueeze(0), dim=1)
    
    return expected_scores.cpu().numpy()


def scores_to_classification(scores: np.ndarray) -> np.ndarray:
    """
    Convert continuous scores to class indices
    
    Args:
        scores: Continuous scores (0-10)
    
    Returns:
        Class indices (0-20)
    """
    # Clamp scores to valid range
    scores = np.clip(scores, 0.0, 10.0)
    # Round to nearest 0.5 and convert to class index
    class_indices = np.round(scores * 2).astype(int)
    return np.clip(class_indices, 0, 20)


class ScoringMetrics:
    """
    Comprehensive metrics calculator for speaking scoring task
    """
    
    def __init__(self, task_type: str = "classification", num_classes: int = 21):
        """
        Initialize metrics calculator
        
        Args:
            task_type: "classification" or "regression"
            num_classes: Number of classes for classification
        """
        self.task_type = task_type
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.all_predictions = []
        self.all_targets = []
        self.all_scores = []
    
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        scores: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with batch results
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            scores: Ground truth continuous scores (optional)
        """
        # Convert to numpy and store
        if predictions.dim() > 1 and self.task_type == "classification":
            # Convert logits to class predictions for storage
            pred_classes = torch.argmax(predictions, dim=-1)
            self.all_predictions.extend(pred_classes.cpu().numpy())
        else:
            self.all_predictions.extend(predictions.cpu().numpy().flatten())
        
        self.all_targets.extend(targets.cpu().numpy().flatten())
        
        if scores is not None:
            self.all_scores.extend(scores.cpu().numpy().flatten())
        else:
            # Convert targets to scores if not provided
            target_scores = targets.cpu().numpy().flatten() * 0.5
            self.all_scores.extend(target_scores)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary of computed metrics
        """
        if not self.all_predictions:
            return {}
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        scores = np.array(self.all_scores)
        
        metrics = {}
        
        if self.task_type == "classification":
            # Classification metrics
            metrics['accuracy'] = accuracy_score(targets, predictions)
            metrics['exact_match'] = exact_match_accuracy(predictions * 0.5, scores)
            metrics['within_0.5'] = within_range_accuracy(predictions * 0.5, scores, 0.5)
            metrics['within_1.0'] = within_range_accuracy(predictions * 0.5, scores, 1.0)
            
            # Convert predictions to continuous scores for correlation
            pred_scores = predictions * 0.5
            
        else:  # regression
            # Convert predictions to class indices for accuracy
            pred_classes = scores_to_classification(predictions)
            target_classes = scores_to_classification(scores)
            
            metrics['accuracy'] = accuracy_score(target_classes, pred_classes)
            metrics['exact_match'] = exact_match_accuracy(predictions, scores)
            metrics['within_0.5'] = within_range_accuracy(predictions, scores, 0.5)
            metrics['within_1.0'] = within_range_accuracy(predictions, scores, 1.0)
            
            pred_scores = predictions
        
        # Correlation metrics
        correlations = calculate_correlation(pred_scores, scores)
        metrics.update(correlations)
        
        # Error metrics
        metrics['mae'] = calculate_mae(pred_scores, scores)
        metrics['mse'] = calculate_mse(pred_scores, scores)
        metrics['rmse'] = calculate_rmse(pred_scores, scores)
        
        return metrics
    
    def compute_per_class_metrics(self) -> Dict[str, Any]:
        """Compute per-class metrics for classification tasks"""
        if self.task_type != "classification" or not self.all_predictions:
            return {}
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        per_class_metrics = {}
        
        for class_idx in range(self.num_classes):
            class_mask = targets == class_idx
            if class_mask.sum() == 0:
                continue
            
            class_predictions = predictions[class_mask]
            class_targets = targets[class_mask]
            
            per_class_metrics[f'class_{class_idx}_accuracy'] = accuracy_score(
                class_targets, class_predictions
            )
            per_class_metrics[f'class_{class_idx}_count'] = int(class_mask.sum())
        
        return per_class_metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    task_type: str = "classification"
) -> Dict[str, float]:
    """
    Evaluate model on a dataset
    
    Args:
        model: CCMT model
        dataloader: Data loader
        device: Device to run evaluation on
        task_type: Task type
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics_calculator = ScoringMetrics(task_type=task_type)
    
    with torch.no_grad():
        for batch in dataloader:
            # Get inputs and targets
            if 'ccmt_input' in batch:
                inputs = batch['ccmt_input'].to(device)
            else:
                # Fallback if ccmt_input not available
                continue
            
            targets = batch['targets'].to(device)
            scores = batch.get('scores', targets * 0.5).to(device)
            
            # Forward pass
            predictions = model(inputs)
            
            # Update metrics
            metrics_calculator.update(predictions, targets, scores)
    
    return metrics_calculator.compute()


def create_metrics_summary(metrics: Dict[str, float]) -> str:
    """
    Create a readable summary of metrics
    
    Args:
        metrics: Dictionary of metrics
    
    Returns:
        Formatted string summary
    """
    summary_lines = []
    summary_lines.append("Evaluation Metrics:")
    summary_lines.append("=" * 40)
    
    # Main metrics
    if 'accuracy' in metrics:
        summary_lines.append(f"Accuracy:          {metrics['accuracy']:.4f}")
    if 'pearson' in metrics:
        summary_lines.append(f"Pearson Corr:      {metrics['pearson']:.4f}")
    if 'spearman' in metrics:
        summary_lines.append(f"Spearman Corr:     {metrics['spearman']:.4f}")
    if 'mae' in metrics:
        summary_lines.append(f"MAE:               {metrics['mae']:.4f}")
    if 'rmse' in metrics:
        summary_lines.append(f"RMSE:              {metrics['rmse']:.4f}")
    
    # Range-based accuracy
    if 'within_0.5' in metrics:
        summary_lines.append(f"Within 0.5 points: {metrics['within_0.5']:.4f}")
    if 'within_1.0' in metrics:
        summary_lines.append(f"Within 1.0 points: {metrics['within_1.0']:.4f}")
    
    summary_lines.append("=" * 40)
    
    return "\n".join(summary_lines)


# Example usage and testing
if __name__ == "__main__":
    # Test metrics calculation
    batch_size, num_classes = 100, 21
    
    # Simulate classification predictions and targets
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    scores = targets.float() * 0.5 + torch.randn(batch_size) * 0.1  # Add some noise
    
    # Test individual metric functions
    accuracy = calculate_accuracy(predictions, targets)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Convert to numpy for correlation
    pred_scores = classification_to_scores(predictions)
    true_scores = scores.numpy()
    
    correlations = calculate_correlation(pred_scores, true_scores)
    print(f"Pearson correlation: {correlations['pearson']:.4f}")
    
    mae = calculate_mae(pred_scores, true_scores)
    print(f"MAE: {mae:.4f}")
    
    # Test metrics calculator
    metrics_calc = ScoringMetrics(task_type="classification", num_classes=num_classes)
    metrics_calc.update(predictions, targets, scores)
    all_metrics = metrics_calc.compute()
    
    print("\nAll metrics:")
    print(create_metrics_summary(all_metrics))
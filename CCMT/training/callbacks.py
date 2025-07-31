"""
Training callbacks for CCMT
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import json
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base callback class"""
    
    def on_train_begin(self, trainer_state: Dict[str, Any]):
        """Called at the beginning of training"""
        pass
    
    def on_train_end(self, trainer_state: Dict[str, Any]):
        """Called at the end of training"""
        pass
    
    def on_epoch_begin(self, epoch: int, trainer_state: Dict[str, Any]):
        """Called at the beginning of each epoch"""
        pass
    
    def on_epoch_end(self, epoch: int, trainer_state: Dict[str, Any]):
        """Called at the end of each epoch"""
        pass
    
    def on_batch_begin(self, batch_idx: int, trainer_state: Dict[str, Any]):
        """Called at the beginning of each batch"""
        pass
    
    def on_batch_end(self, batch_idx: int, trainer_state: Dict[str, Any]):
        """Called at the end of each batch"""
        pass
    
    def on_validation_begin(self, trainer_state: Dict[str, Any]):
        """Called at the beginning of validation"""
        pass
    
    def on_validation_end(self, trainer_state: Dict[str, Any]):
        """Called at the end of validation"""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to prevent overfitting
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max" for metric
            restore_best_weights: Whether to restore best weights
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def on_train_begin(self, trainer_state: Dict[str, Any]):
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
    
    def on_epoch_end(self, epoch: int, trainer_state: Dict[str, Any]):
        metrics = trainer_state.get('metrics', {})
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return
        
        # Check if we have improvement
        if self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
            
            # Save best weights
            if self.restore_best_weights:
                model = trainer_state.get('model')
                if model is not None:
                    self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        # Check for early stopping
        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            
            # Restore best weights
            if self.restore_best_weights and self.best_weights is not None:
                model = trainer_state.get('model')
                if model is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best model weights")
    
    def _is_improvement(self, current_score: float) -> bool:
        """Check if current score is an improvement"""
        if self.best_score is None:
            return True
        
        if self.mode == 'min':
            return current_score < (self.best_score - self.min_delta)
        else:
            return current_score > (self.best_score + self.min_delta)


class ModelCheckpoint(Callback):
    """
    Model checkpointing callback
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        filename: str = "checkpoint_epoch_{epoch:03d}.pt",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = True,
        save_every_n_epochs: int = 5
    ):
        """
        Initialize model checkpoint
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            filename: Checkpoint filename template
            monitor: Metric to monitor for best model
            mode: "min" or "max" for metric
            save_best_only: Only save when metric improves
            save_last: Always save last epoch
            save_every_n_epochs: Save every N epochs regardless
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        
        self.best_score = None
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def on_train_begin(self, trainer_state: Dict[str, Any]):
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
    
    def on_epoch_end(self, epoch: int, trainer_state: Dict[str, Any]):
        metrics = trainer_state.get('metrics', {})
        current_score = metrics.get(self.monitor)
        
        should_save = False
        checkpoint_type = ""
        
        # Check if we should save based on metric improvement
        if current_score is not None and self._is_improvement(current_score):
            self.best_score = current_score
            should_save = True
            checkpoint_type = "best"
        
        # Save every N epochs
        if (epoch + 1) % self.save_every_n_epochs == 0:
            should_save = True
            checkpoint_type = "periodic"
        
        # Save if not save_best_only or if it's the last epoch
        if not self.save_best_only or self.save_last:
            should_save = True
            if not checkpoint_type:
                checkpoint_type = "regular"
        
        if should_save:
            self._save_checkpoint(epoch, trainer_state, checkpoint_type)
    
    def on_train_end(self, trainer_state: Dict[str, Any]):
        if self.save_last:
            epoch = trainer_state.get('epoch', 0)
            self._save_checkpoint(epoch, trainer_state, "final")
    
    def _is_improvement(self, current_score: float) -> bool:
        """Check if current score is an improvement"""
        if self.best_score is None:
            return True
        
        if self.mode == 'min':
            return current_score < self.best_score
        else:
            return current_score > self.best_score
    
    def _save_checkpoint(self, epoch: int, trainer_state: Dict[str, Any], checkpoint_type: str):
        """Save checkpoint to disk"""
        try:
            # Access model through trainer_state but only save its state_dict
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer_state['model'].state_dict(),
                'checkpoint_type': checkpoint_type
            }
            
            # Add optimizer state if available
            if 'optimizer' in trainer_state and trainer_state['optimizer'] is not None:
                checkpoint['optimizer_state_dict'] = trainer_state['optimizer'].state_dict()
            
            # Add scheduler state if available
            if 'scheduler' in trainer_state and trainer_state['scheduler'] is not None:
                checkpoint['scheduler_state_dict'] = trainer_state['scheduler'].state_dict()
                
            # Add metrics
            checkpoint['metrics'] = trainer_state.get('metrics', {})
            checkpoint['best_score'] = self.best_score
            
            # Generate filename and save
            if checkpoint_type == "best":
                filename = "best_model.pt"
            elif checkpoint_type == "final":
                filename = "final_model.pt"
            else:
                filename = self.filename.format(epoch=epoch)
            
            checkpoint_path = self.checkpoint_dir / filename
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved {checkpoint_type} checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")


class LearningRateScheduler(Callback):
    """
    Learning rate scheduling callback
    """
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: Optional[str] = None
    ):
        """
        Initialize LR scheduler
        
        Args:
            scheduler: PyTorch learning rate scheduler
            monitor: Metric to monitor (for ReduceLROnPlateau)
        """
        self.scheduler = scheduler
        self.monitor = monitor
    
    def on_epoch_end(self, epoch: int, trainer_state: Dict[str, Any]):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau needs a metric
            if self.monitor:
                metrics = trainer_state.get('metrics', {})
                metric_value = metrics.get(self.monitor)
                if metric_value is not None:
                    self.scheduler.step(metric_value)
            else:
                logger.warning("ReduceLROnPlateau requires a monitor metric")
        else:
            # Other schedulers step automatically
            self.scheduler.step()
        
        # Log current learning rate
        current_lr = self.scheduler.get_last_lr()[0]
        logger.info(f"Learning rate: {current_lr:.2e}")


class MetricsLogger(Callback):
    """
    Metrics logging callback
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        log_every_n_batches: int = 10,
        save_metrics: bool = True
    ):
        """
        Initialize metrics logger
        
        Args:
            log_dir: Directory to save logs
            log_every_n_batches: Log every N batches
            save_metrics: Save metrics to file
        """
        self.log_dir = Path(log_dir)
        self.log_every_n_batches = log_every_n_batches
        self.save_metrics = save_metrics
        
        self.training_log = []
        self.batch_count = 0
        self.start_time = None
        
        if self.save_metrics:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def on_train_begin(self, trainer_state: Dict[str, Any]):
        self.start_time = time.time()
        self.batch_count = 0
        self.training_log = []
        logger.info("Training started")
    
    def on_train_end(self, trainer_state: Dict[str, Any]):
        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        if self.save_metrics:
            self._save_training_log()
    
    def on_epoch_begin(self, epoch: int, trainer_state: Dict[str, Any]):
        logger.info(f"Epoch {epoch + 1} started")
    
    def on_epoch_end(self, epoch: int, trainer_state: Dict[str, Any]):
        metrics = trainer_state.get('metrics', {})
        
        # Log epoch metrics
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch + 1} - {metric_str}")
        
        # Save to training log
        log_entry = {
            'epoch': epoch + 1,
            'timestamp': time.time(),
            **metrics
        }
        self.training_log.append(log_entry)
    
    def on_batch_end(self, batch_idx: int, trainer_state: Dict[str, Any]):
        self.batch_count += 1
        
        if self.batch_count % self.log_every_n_batches == 0:
            loss = trainer_state.get('batch_loss', 0.0)
            lr = trainer_state.get('learning_rate', 0.0)
            logger.info(f"Batch {self.batch_count}: loss={loss:.4f}, lr={lr:.2e}")
    
    def _save_training_log(self):
        """Save training log to file"""
        try:
            log_file = self.log_dir / "training_log.json"
            with open(log_file, 'w') as f:
                json.dump(self.training_log, f, indent=2)
            logger.info(f"Training log saved to {log_file}")
        except Exception as e:
            logger.error(f"Failed to save training log: {e}")


class ProgressCallback(Callback):
    """
    Progress reporting callback
    """
    
    def __init__(self, print_every_n_batches: int = 100):
        self.print_every_n_batches = print_every_n_batches
        self.epoch_start_time = None
        self.batch_count = 0
    
    def on_epoch_begin(self, epoch: int, trainer_state: Dict[str, Any]):
        self.epoch_start_time = time.time()
        self.batch_count = 0
        total_epochs = trainer_state.get('total_epochs', '?')
        print(f"\nEpoch {epoch + 1}/{total_epochs}")
        print("-" * 50)
    
    def on_epoch_end(self, epoch: int, trainer_state: Dict[str, Any]):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch completed in {epoch_time:.2f}s")
    
    def on_batch_end(self, batch_idx: int, trainer_state: Dict[str, Any]):
        self.batch_count += 1
        
        if self.batch_count % self.print_every_n_batches == 0:
            total_batches = trainer_state.get('total_batches', '?')
            loss = trainer_state.get('batch_loss', 0.0)
            print(f"Batch {self.batch_count}/{total_batches} - Loss: {loss:.4f}")


def create_default_callbacks(
    output_dir: str = "./outputs",
    patience: int = 10,
    save_every_n_epochs: int = 5
) -> List[Callback]:
    """
    Create default set of callbacks for training
    
    Args:
        output_dir: Output directory
        patience: Early stopping patience
        save_every_n_epochs: Checkpoint saving frequency
    
    Returns:
        List of callbacks
    """
    output_path = Path(output_dir)
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            checkpoint_dir=output_path / "checkpoints",
            save_every_n_epochs=save_every_n_epochs
        ),
        MetricsLogger(
            log_dir=output_path / "logs"
        ),
        ProgressCallback()
    ]
    
    return callbacks


# Example usage
if __name__ == "__main__":
    # Test callbacks
    print("Testing callbacks...")
    
    # Mock trainer state
    trainer_state = {
        'epoch': 0,
        'metrics': {'val_loss': 0.5, 'val_accuracy': 0.8},
        'model': None,
        'optimizer': None
    }
    
    # Test early stopping
    early_stop = EarlyStopping(monitor="val_loss", patience=3)
    early_stop.on_train_begin(trainer_state)
    
    for epoch in range(5):
        trainer_state['epoch'] = epoch
        trainer_state['metrics']['val_loss'] = 0.5 - epoch * 0.1  # Improving loss
        early_stop.on_epoch_end(epoch, trainer_state)
        
        if early_stop.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print("Callbacks test completed")
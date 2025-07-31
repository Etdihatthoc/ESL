"""
utils.py - Utility functions for ESL binary classification
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os
import random
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir="./logs", experiment_name="binary_classifier"):
    """
    Setup logging configuration
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{experiment_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def count_parameters(model):
    """
    Count total and trainable parameters in the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    return total_params, trainable_params


def get_optimizer(model, lr=2e-5, weight_decay=1e-4, optimizer_type='adamw'):
    """
    Get optimizer with different learning rates for different parts of the model
    """
    if optimizer_type.lower() == 'adamw':
        # Different learning rates for encoder and classifier
        encoder_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                classifier_params.append(param)
        
        param_groups = [
            {'params': encoder_params, 'lr': lr * 0.1},  # Lower LR for pretrained encoder
            {'params': classifier_params, 'lr': lr}      # Normal LR for classifier
        ]
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer


def get_scheduler(optimizer, num_training_steps, warmup_ratio=0.1, scheduler_type='cosine'):
    """
    Get learning rate scheduler
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    if scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        scheduler = None
    
    return scheduler


def selective_freeze_embedding_layer(model, tokenizer, unfrozen_words):
    """
    Freezes the embedding layer of a transformer model,
    but allows selected tokens (from unfrozen_words) to remain trainable.
    (Adapted from original code)
    """
    # Freeze the entire embedding layer
    embedding_layer = model.encoder.embeddings.word_embeddings
    embedding_layer.weight.requires_grad = True  # must stay True for masking
    for param in model.encoder.embeddings.parameters():
        param.requires_grad = True  # required for backward hook to work

    # Get token IDs of unfrozen words and all special tokens
    token_ids = set()
    for word in unfrozen_words:
        ids = tokenizer(word, add_special_tokens=False)['input_ids']
        token_ids.update(ids)

    # Add all special token IDs
    if hasattr(tokenizer, "all_special_ids"):
        token_ids.update(tokenizer.all_special_ids)
    else:
        # Fallback for tokenizers without all_special_ids
        for tok in tokenizer.all_special_tokens:
            ids = tokenizer(tok, add_special_tokens=False)['input_ids']
            token_ids.update(ids)

    vocab_size, hidden_size = embedding_layer.weight.shape
    grad_mask = torch.zeros(vocab_size, 1, device=embedding_layer.weight.device)
    for idx in token_ids:
        if idx < vocab_size:
            grad_mask[idx] = 1.0

    # Register gradient hook to zero out updates for frozen tokens
    def hook_fn(grad):
        # grad: [vocab_size, hidden_size]
        return grad * grad_mask

    embedding_layer.weight.register_hook(hook_fn)


def calculate_dataset_stats(train_loader, val_loader, test_loader):
    """
    Calculate and print dataset statistics
    """
    print("\n=== DATASET STATISTICS ===")
    
    # Count samples in each loader
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    test_samples = len(test_loader.dataset)
    total_samples = train_samples + val_samples + test_samples
    
    print(f"Train samples: {train_samples} ({train_samples/total_samples*100:.1f}%)")
    print(f"Validation samples: {val_samples} ({val_samples/total_samples*100:.1f}%)")
    print(f"Test samples: {test_samples} ({test_samples/total_samples*100:.1f}%)")
    print(f"Total samples: {total_samples}")
    
    # Group distribution in training set
    train_groups = train_loader.dataset.groups
    group_0_count = train_groups.count(0)
    group_1_count = train_groups.count(1)
    
    print(f"\nTraining set group distribution:")
    print(f"Group 0 (3.5-6.5): {group_0_count} ({group_0_count/train_samples*100:.1f}%)")
    print(f"Group 1 (7.0-10.0): {group_1_count} ({group_1_count/train_samples*100:.1f}%)")
    print(f"Class imbalance ratio: {max(group_0_count, group_1_count) / min(group_0_count, group_1_count):.2f}")
    
    # Score distribution in training set
    train_scores = train_loader.dataset.raw_scores
    print(f"\nTraining set score range: {min(train_scores):.1f} - {max(train_scores):.1f}")
    print(f"Mean score: {np.mean(train_scores):.2f} ± {np.std(train_scores):.2f}")


def create_model_config(args):
    """
    Create model configuration from arguments
    """
    config = {
        'model_name': getattr(args, 'model_name', 'Alibaba-NLP/gte-multilingual-base'),
        'pooling_dropout': getattr(args, 'pooling_dropout', 0.3),
        'classifier_dropout': getattr(args, 'classifier_dropout', 0.5),
        'avg_last_k': getattr(args, 'avg_last_k', 4),
        'd_fuse': getattr(args, 'd_fuse', 256)
    }
    return config


def create_training_config(args, num_training_steps):
    """
    Create training configuration from arguments
    """
    config = {
        'learning_rate': getattr(args, 'lr', 2e-5),
        'weight_decay': getattr(args, 'weight_decay', 1e-4),
        'epochs': getattr(args, 'epochs', 10),
        'batch_size': getattr(args, 'batch_size', 32),
        'warmup_ratio': getattr(args, 'warmup_ratio', 0.1),
        'scheduler_type': getattr(args, 'scheduler_type', 'cosine'),
        'optimizer_type': getattr(args, 'optimizer_type', 'adamw'),
        'use_focal_loss': getattr(args, 'use_focal_loss', False),
        'focal_alpha': getattr(args, 'focal_alpha', 1.0),
        'focal_gamma': getattr(args, 'focal_gamma', 2.0),
        'label_smoothing': getattr(args, 'label_smoothing', True),
        'num_training_steps': num_training_steps
    }
    return config


def save_experiment_config(config, save_path):
    """
    Save experiment configuration to file
    """
    import json
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool, list, dict)):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Experiment configuration saved to: {save_path}")


def check_device_memory():
    """
    Check and print GPU memory usage
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
        max_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        
        print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB, Total: {max_memory:.2f}GB")
    else:
        print("CUDA not available")


def maybe_empty_cache(threshold=0.9):
    """
    Empty CUDA cache if memory usage is above threshold
    """
    if torch.cuda.is_available():
        try:
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            if reserved / total > threshold:
                torch.cuda.empty_cache()
                print("Cleared CUDA cache")
        except Exception:
            torch.cuda.empty_cache()
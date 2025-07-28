"""
pipeline_utils.py - Helper utilities for combined pipeline testing
Location: /media/gpus/Data/AES/ESL-Grading/pipeline_utils.py
"""

import torch


def get_binary_collate_fn_with_audio(tokenizer, max_length=8192):
    """
    Collate function for binary classification with audio (simplified version)
    """
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        groups = torch.stack([item['group'] for item in batch])
        raw_scores = torch.stack([item['raw_score'] for item in batch]) if 'raw_score' in batch[0] else torch.stack([item['score'] for item in batch])
        question_types = torch.tensor([item['question_type'] for item in batch], dtype=torch.long)
        
        # Text encoding
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Audio processing (simplified)
        has_audio = [item.get('has_audio', False) for item in batch]
        if any(has_audio):
            audios = torch.stack([item['audio'] for item in batch])
        else:
            audios = None

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'group': groups,
            'score': raw_scores,
            'question_type': question_types,
            'audio': audios,
            'has_audio': has_audio
        }
    
    return collate_fn


def print_model_info(model, model_name):
    """
    Print basic model information
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{model_name}: {total_params:,} total params, {trainable_params:,} trainable")
    except:
        print(f"{model_name}: loaded successfully")


def check_gpu_memory():
    """
    Check GPU memory usage
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        print("CUDA not available")


def simple_metrics_summary(ground_truth, predictions, dataset_name):
    """
    Calculate and print simple metrics
    """
    import numpy as np
    from scipy.stats import pearsonr
    
    gt = np.array(ground_truth)
    pred = np.array(predictions)
    
    mse = np.mean((gt - pred) ** 2)
    mae = np.mean(np.abs(gt - pred))
    
    try:
        correlation, _ = pearsonr(gt, pred)
    except:
        correlation = 0.0
    
    print(f"{dataset_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, Corr: {correlation:.4f}")
    
    return {'mse': mse, 'mae': mae, 'correlation': correlation}
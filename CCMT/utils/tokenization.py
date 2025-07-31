"""
Token sampling and alignment utilities for CCMT
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import random
import logging

logger = logging.getLogger(__name__)


def sample_tokens_uniform(tokens: torch.Tensor, target_length: int, keep_first: bool = True) -> torch.Tensor:
    """
    Sample tokens uniformly to match target length
    
    Args:
        tokens: Input token tensor (seq_len, dim)
        target_length: Target sequence length
        keep_first: Whether to always keep the first token (e.g., CLS token)
    
    Returns:
        Sampled token tensor (target_length, dim)
    """
    seq_len, dim = tokens.shape
    
    if seq_len == target_length:
        return tokens
    
    if seq_len < target_length:
        # Pad with zeros
        padding = torch.zeros(target_length - seq_len, dim, device=tokens.device, dtype=tokens.dtype)
        return torch.cat([tokens, padding], dim=0)
    
    # Sample uniformly
    if keep_first and seq_len > 1:
        # Keep first token, sample from the rest
        first_token = tokens[:1]  # (1, dim)
        remaining_tokens = tokens[1:]  # (seq_len-1, dim)
        
        if target_length == 1:
            return first_token
        
        # Sample remaining tokens
        remaining_target = target_length - 1
        indices = torch.linspace(0, len(remaining_tokens) - 1, remaining_target).long()
        sampled_remaining = remaining_tokens[indices]
        
        return torch.cat([first_token, sampled_remaining], dim=0)
    
    else:
        # Sample all tokens uniformly
        indices = torch.linspace(0, seq_len - 1, target_length).long()
        return tokens[indices]


def sample_tokens_random(tokens: torch.Tensor, target_length: int, keep_first: bool = True) -> torch.Tensor:
    """
    Sample tokens randomly to match target length
    
    Args:
        tokens: Input token tensor (seq_len, dim)
        target_length: Target sequence length
        keep_first: Whether to always keep the first token
    
    Returns:
        Sampled token tensor (target_length, dim)
    """
    seq_len, dim = tokens.shape
    
    if seq_len == target_length:
        return tokens
    
    if seq_len < target_length:
        # Pad with zeros
        padding = torch.zeros(target_length - seq_len, dim, device=tokens.device, dtype=tokens.dtype)
        return torch.cat([tokens, padding], dim=0)
    
    # Sample randomly
    if keep_first and seq_len > 1:
        # Keep first token, sample from the rest
        first_token = tokens[:1]  # (1, dim)
        remaining_tokens = tokens[1:]  # (seq_len-1, dim)
        
        if target_length == 1:
            return first_token
        
        # Random sample remaining tokens
        remaining_target = target_length - 1
        indices = torch.randperm(len(remaining_tokens))[:remaining_target]
        indices = indices.sort()[0]  # Sort to maintain some order
        sampled_remaining = remaining_tokens[indices]
        
        return torch.cat([first_token, sampled_remaining], dim=0)
    
    else:
        # Sample all tokens randomly
        indices = torch.randperm(seq_len)[:target_length]
        indices = indices.sort()[0]  # Sort to maintain some order
        return tokens[indices]


def align_token_lengths(token_lists: List[torch.Tensor], 
                       target_length: int,
                       sampling_method: str = "uniform",
                       keep_first: bool = True) -> List[torch.Tensor]:
    """
    Align multiple token sequences to the same length
    
    Args:
        token_lists: List of token tensors with potentially different lengths
        target_length: Target sequence length for all sequences
        sampling_method: "uniform" or "random" sampling
        keep_first: Whether to keep first token
    
    Returns:
        List of aligned token tensors
    """
    aligned_tokens = []
    
    for tokens in token_lists:
        if sampling_method == "uniform":
            aligned = sample_tokens_uniform(tokens, target_length, keep_first)
        elif sampling_method == "random":
            aligned = sample_tokens_random(tokens, target_length, keep_first)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        aligned_tokens.append(aligned)
    
    return aligned_tokens


def pad_or_truncate_tokens(tokens: torch.Tensor, 
                          target_length: int,
                          pad_value: float = 0.0) -> torch.Tensor:
    """
    Pad or truncate tokens to target length
    
    Args:
        tokens: Input token tensor (seq_len, dim)
        target_length: Target sequence length
        pad_value: Value to use for padding
    
    Returns:
        Tensor of target length
    """
    seq_len = tokens.shape[0]
    
    if seq_len == target_length:
        return tokens
    
    elif seq_len < target_length:
        # Pad
        padding_shape = (target_length - seq_len,) + tokens.shape[1:]
        padding = torch.full(padding_shape, pad_value, device=tokens.device, dtype=tokens.dtype)
        return torch.cat([tokens, padding], dim=0)
    
    else:
        # Truncate
        return tokens[:target_length]


def create_attention_mask(tokens: torch.Tensor, pad_token_id: float = 0.0) -> torch.Tensor:
    """
    Create attention mask for padded tokens
    
    Args:
        tokens: Token tensor (seq_len, dim) or (batch, seq_len, dim)
        pad_token_id: Value representing padded tokens
    
    Returns:
        Attention mask tensor (seq_len,) or (batch, seq_len)
    """
    if tokens.dim() == 2:
        # Single sequence (seq_len, dim)
        # Check if any dimension has all pad values
        mask = ~torch.all(tokens == pad_token_id, dim=-1)
        return mask.float()
    
    elif tokens.dim() == 3:
        # Batch (batch, seq_len, dim)
        mask = ~torch.all(tokens == pad_token_id, dim=-1)
        return mask.float()
    
    else:
        raise ValueError(f"Unexpected token tensor dimensions: {tokens.dim()}")


def batch_sample_tokens(token_batch: torch.Tensor,
                       target_length: int,
                       sampling_method: str = "uniform",
                       keep_first: bool = True) -> torch.Tensor:
    """
    Sample tokens for a batch of sequences
    
    Args:
        token_batch: Batch of token tensors (batch, seq_len, dim)
        target_length: Target sequence length
        sampling_method: Sampling method
        keep_first: Whether to keep first token
    
    Returns:
        Batch of sampled tokens (batch, target_length, dim)
    """
    batch_size, seq_len, dim = token_batch.shape
    
    if seq_len == target_length:
        return token_batch
    
    sampled_batch = []
    
    for i in range(batch_size):
        tokens = token_batch[i]  # (seq_len, dim)
        
        if sampling_method == "uniform":
            sampled = sample_tokens_uniform(tokens, target_length, keep_first)
        elif sampling_method == "random":
            sampled = sample_tokens_random(tokens, target_length, keep_first)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        sampled_batch.append(sampled)
    
    return torch.stack(sampled_batch)


def create_ccmt_input(english_tokens: torch.Tensor,
                     vietnamese_tokens: torch.Tensor, 
                     audio_tokens: torch.Tensor,
                     target_length_per_modality: int = 100) -> torch.Tensor:
    """
    Create CCMT input by concatenating aligned tokens from all modalities
    
    Args:
        english_tokens: English token tensor (seq_len, dim)
        vietnamese_tokens: Vietnamese token tensor (seq_len, dim) 
        audio_tokens: Audio token tensor (seq_len, dim)
        target_length_per_modality: Target length for each modality
    
    Returns:
        CCMT input tensor (3 * target_length_per_modality, dim)
    """
    # Align all token sequences to the same length
    aligned_tokens = align_token_lengths(
        [english_tokens, vietnamese_tokens, audio_tokens],
        target_length_per_modality,
        sampling_method="uniform",
        keep_first=True
    )
    
    # Concatenate along sequence dimension
    ccmt_input = torch.cat(aligned_tokens, dim=0)
    
    return ccmt_input


def batch_create_ccmt_input(english_batch: torch.Tensor,
                           vietnamese_batch: torch.Tensor,
                           audio_batch: torch.Tensor,
                           target_length_per_modality: int = 100) -> torch.Tensor:
    """
    Create CCMT input for a batch
    
    Args:
        english_batch: English tokens (batch, seq_len, dim)
        vietnamese_batch: Vietnamese tokens (batch, seq_len, dim)
        audio_batch: Audio tokens (batch, seq_len, dim)
        target_length_per_modality: Target length per modality
    
    Returns:
        CCMT input batch (batch, 3 * target_length_per_modality, dim)
    """
    batch_size = english_batch.shape[0]
    ccmt_inputs = []
    
    for i in range(batch_size):
        ccmt_input = create_ccmt_input(
            english_batch[i],
            vietnamese_batch[i], 
            audio_batch[i],
            target_length_per_modality
        )
        ccmt_inputs.append(ccmt_input)
    
    return torch.stack(ccmt_inputs)


def validate_token_alignment(token_lists: List[torch.Tensor]) -> bool:
    """
    Validate that all token sequences have the same length and dimension
    
    Args:
        token_lists: List of token tensors
    
    Returns:
        True if all sequences are properly aligned
    """
    if not token_lists:
        return True
    
    first_shape = token_lists[0].shape
    
    for tokens in token_lists[1:]:
        if tokens.shape != first_shape:
            return False
    
    return True


def get_token_statistics(tokens: torch.Tensor) -> Dict[str, float]:
    """
    Get statistics about token tensor
    
    Args:
        tokens: Token tensor (seq_len, dim)
    
    Returns:
        Dictionary of statistics
    """
    if tokens.numel() == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'norm': 0.0
        }
    
    return {
        'mean': tokens.mean().item(),
        'std': tokens.std().item(),
        'min': tokens.min().item(),
        'max': tokens.max().item(),
        'norm': torch.linalg.norm(tokens).item()
    }


def apply_token_dropout(tokens: torch.Tensor, dropout_rate: float = 0.1) -> torch.Tensor:
    """
    Apply dropout to token embeddings (set random tokens to zero)
    
    Args:
        tokens: Token tensor (seq_len, dim)
        dropout_rate: Probability of dropping each token
    
    Returns:
        Token tensor with dropout applied
    """
    if dropout_rate <= 0.0:
        return tokens
    
    # Create dropout mask
    mask = torch.rand(tokens.shape[0], device=tokens.device) > dropout_rate
    mask = mask.unsqueeze(-1).expand_as(tokens)
    
    return tokens * mask


def interpolate_tokens(tokens: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Interpolate tokens to target length using linear interpolation
    
    Args:
        tokens: Input token tensor (seq_len, dim)
        target_length: Target sequence length
    
    Returns:
        Interpolated token tensor (target_length, dim)
    """
    if tokens.shape[0] == target_length:
        return tokens
    
    # Use 1D interpolation for each dimension
    seq_len, dim = tokens.shape
    
    # Create interpolation indices
    old_indices = torch.linspace(0, seq_len - 1, seq_len)
    new_indices = torch.linspace(0, seq_len - 1, target_length)
    
    interpolated_tokens = torch.zeros(target_length, dim, device=tokens.device, dtype=tokens.dtype)
    
    for d in range(dim):
        # Interpolate each dimension separately
        interpolated_tokens[:, d] = torch.nn.functional.interpolate(
            tokens[:, d].unsqueeze(0).unsqueeze(0),  # (1, 1, seq_len)
            size=target_length,
            mode='linear',
            align_corners=True
        ).squeeze(0).squeeze(0)
    
    return interpolated_tokens


def create_position_embeddings(seq_length: int, dim: int, device: str = "cpu") -> torch.Tensor:
    """
    Create sinusoidal position embeddings
    
    Args:
        seq_length: Sequence length
        dim: Embedding dimension
        device: Device to create embeddings on
    
    Returns:
        Position embeddings (seq_length, dim)
    """
    position = torch.arange(seq_length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * 
                        -(np.log(10000.0) / dim))
    
    pos_embedding = torch.zeros(seq_length, dim, device=device)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    
    return pos_embedding


def add_position_embeddings(tokens: torch.Tensor, pos_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Add position embeddings to tokens
    
    Args:
        tokens: Token tensor (seq_len, dim)
        pos_embeddings: Position embeddings (seq_len, dim). If None, creates sinusoidal embeddings
    
    Returns:
        Tokens with position embeddings added
    """
    seq_len, dim = tokens.shape
    
    if pos_embeddings is None:
        pos_embeddings = create_position_embeddings(seq_len, dim, device=tokens.device)
    
    return tokens + pos_embeddings


def token_attention_weights(query_tokens: torch.Tensor, key_tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute attention weights between query and key tokens
    
    Args:
        query_tokens: Query tokens (seq_len_q, dim)
        key_tokens: Key tokens (seq_len_k, dim)
    
    Returns:
        Attention weights (seq_len_q, seq_len_k)
    """
    # Compute scaled dot-product attention weights
    scale = 1.0 / np.sqrt(query_tokens.shape[-1])
    attention_scores = torch.matmul(query_tokens, key_tokens.transpose(-2, -1)) * scale
    attention_weights = torch.softmax(attention_scores, dim=-1)
    
    return attention_weights
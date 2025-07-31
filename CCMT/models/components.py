"""
Core transformer components for CCMT architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class PreNorm(nn.Module):
    """Pre-normalization wrapper for transformer layers"""
    
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs):
        if context is not None:
            return self.fn(self.norm(x), context, **kwargs)
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """Cross-attention mechanism for multimodal fusion"""
    
    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        dim_head: int = 64, 
        dropout: float = 0.1
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        # Separate projections for queries, keys, and values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False) 
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if heads > 1 or dim_head != dim else nn.Identity()

    def forward(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, q_len, dim) - tokens that attend 
            key_value: (batch, kv_len, dim) - tokens being attended to
            mask: Optional attention mask
        Returns:
            (batch, q_len, dim) - attended output
        """
        batch_size = query.shape[0]
        
        # Generate Q, K, V
        q = self.to_q(query)  # (batch, q_len, inner_dim)
        k = self.to_k(key_value)  # (batch, kv_len, inner_dim) 
        v = self.to_v(key_value)  # (batch, kv_len, inner_dim)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = rearrange(mask, 'b i j -> b 1 i j')
            dots.masked_fill_(mask == 0, -float('inf'))
        
        # Apply attention
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        # Apply to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class CrossModalTransformerBlock(nn.Module):
    """Single cross-modal transformer block with cross-attention and feed-forward"""
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.cross_attn = PreNorm(
            dim, 
            CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        )
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout))

    def forward(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Query tokens (batch, q_len, dim)
            key_value: Key/Value tokens (batch, kv_len, dim)  
            mask: Optional attention mask
        Returns:
            Updated query tokens (batch, q_len, dim)
        """
        # Cross-attention with residual connection
        query = query + self.cross_attn(query, context=key_value, mask=mask)
        
        # Feed-forward with residual connection  
        query = query + self.ff(query)
        
        return query


class ScoringHead(nn.Module):
    """Classification/Regression head for English speaking scoring"""
    
    def __init__(
        self, 
        dim: int, 
        num_classes: int = 21,  # 0-10 in 0.5 increments
        task_type: str = "classification",  # or "regression"
        dropout: float = 0.2
    ):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes if task_type == "classification" else 1)
        )
        
        # Apply appropriate final activation
        if task_type == "classification":
            self.final_activation = nn.Softmax(dim=-1)
        elif task_type == "regression":
            self.final_activation = nn.Sigmoid()  # Scale to [0,1] then multiply by 10
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Class token from transformer (batch, dim)
        Returns:
            Predictions (batch, num_classes) or (batch, 1)
        """
        logits = self.head(x)
        
        if self.task_type == "classification":
            return self.final_activation(logits)
        else:  # regression
            return self.final_activation(logits) * 10.0  # Scale to [0,10]
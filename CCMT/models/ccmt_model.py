"""
Main CCMT architecture adapted for English speaking scoring task
Based on the original CCMT implementation from ristea/ccmt
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .components import PreNorm, FeedForward, CrossAttention, ScoringHead


class Attention(nn.Module):
    """Multi-head self-attention mechanism from original CCMT"""
    
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Key/Value source (batch, seq_len, dim)
            q: Query source (batch, seq_len, dim)
        Returns:
            Attended output (batch, seq_len, dim)
        """
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2), kv)
        q = self.to_q(q).view(q.shape[0], q.shape[1], self.heads, -1).transpose(1, 2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(out.shape[0], out.shape[2], -1)
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer block from original CCMT with cross-attention"""
    
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Key/Value tokens (batch, seq_len, dim)
            q: Query tokens (batch, seq_len, dim)
        Returns:
            Transformed tokens (batch, seq_len, dim)
        """
        for attn, ff in self.layers:
            x = attn(x, q) + x
            x = ff(x) + x
        return x


class CascadedCrossModalTransformer(nn.Module):
    """
    Main CCMT model adapted for English speaking scoring
    Architecture: English Text -> Vietnamese Text -> Audio
    """
    
    def __init__(
        self,
        # Task parameters
        num_classes: int = 21,  # 0-10 in 0.5 increments  
        task_type: str = "classification",  # or "regression"
        
        # Architecture parameters
        num_patches: int = 300,  # Total tokens (100 per modality)
        dim: int = 768,  # Token dimension (BERT-base size)
        depth: int = 6,  # Transformer depth
        heads: int = 8,  # Attention heads
        mlp_dim: int = 2048,  # Feed-forward hidden dim
        dim_head: int = 64,  # Dimension per attention head
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Validate input
        assert num_patches % 3 == 0, "num_patches must be divisible by 3 for 3 modalities!"
        self.patches_per_modality = num_patches // 3
        self.dim = dim
        self.task_type = task_type
        
        # Positional embeddings for each modality
        self.pos_embedding_english = nn.Parameter(
            torch.randn(1, self.patches_per_modality, dim) * 0.02
        )
        self.pos_embedding_vietnamese = nn.Parameter(
            torch.randn(1, self.patches_per_modality, dim) * 0.02
        )
        self.pos_embedding_audio = nn.Parameter(
            torch.randn(1, self.patches_per_modality, dim) * 0.02
        )
        
        # Two cascaded cross-modal transformers
        # First: English (query) x Vietnamese (key/value) 
        self.cross_tr_language = Transformer(
            dim=dim, depth=depth, heads=heads, 
            dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout
        )
        
        # Second: Multilingual text (query) x Audio (key/value)
        self.cross_tr_speech = Transformer(
            dim=dim, depth=depth, heads=heads,
            dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout  
        )
        
        # Classification/Regression head
        self.scoring_head = ScoringHead(
            dim=dim, num_classes=num_classes, 
            task_type=task_type, dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CCMT
        
        Args:
            x: Concatenated tokens (batch, num_patches, dim)
               Format: [english_tokens, vietnamese_tokens, audio_tokens]
        Returns:
            Predictions (batch, num_classes) or (batch, 1)
        """
        batch_size = x.shape[0]
        
        # Split into modalities and add positional embeddings
        english_tokens = x[:, :self.patches_per_modality] + self.pos_embedding_english
        vietnamese_tokens = x[:, self.patches_per_modality:2*self.patches_per_modality] + self.pos_embedding_vietnamese  
        audio_tokens = x[:, 2*self.patches_per_modality:] + self.pos_embedding_audio
        
        # First cascade: Language cross-attention
        # English queries attend to Vietnamese keys/values
        multilingual_tokens = self.cross_tr_language(vietnamese_tokens, english_tokens)
        
        # Second cascade: Speech cross-attention  
        # Multilingual queries attend to Audio keys/values
        final_tokens = self.cross_tr_speech(audio_tokens, multilingual_tokens)
        
        # Use first token (class token) for prediction
        class_token = final_tokens[:, 0]  # (batch, dim)
        
        # Generate predictions
        predictions = self.scoring_head(class_token)
        
        return predictions
    
    def get_attention_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention weights for visualization
        
        Args:
            x: Input tokens (batch, num_patches, dim)
        Returns:
            language_attn: Attention weights from language cross-attention
            speech_attn: Attention weights from speech cross-attention
        """
        # This would require modifying the Transformer class to return attention weights
        # For now, return None as placeholder
        return None, None
    
    def freeze_encoders(self):
        """Freeze positional embeddings (encoders are external)"""
        self.pos_embedding_english.requires_grad = False
        self.pos_embedding_vietnamese.requires_grad = False  
        self.pos_embedding_audio.requires_grad = False
    
    def unfreeze_encoders(self):
        """Unfreeze positional embeddings"""
        self.pos_embedding_english.requires_grad = True
        self.pos_embedding_vietnamese.requires_grad = True
        self.pos_embedding_audio.requires_grad = True


# Factory function for easy model creation
def create_ccmt_model(
    task_type: str = "classification",
    num_classes: int = 21,
    model_size: str = "base"  # "base", "large"
) -> CascadedCrossModalTransformer:
    """
    Factory function to create CCMT models with predefined configurations
    
    Args:
        task_type: "classification" or "regression"
        num_classes: Number of classes for classification
        model_size: Model size configuration
    Returns:
        Configured CCMT model
    """
    configs = {
        "base": {
            "dim": 768,
            "depth": 6, 
            "heads": 8,
            "mlp_dim": 2048,
            "dim_head": 64
        },
        "large": {
            "dim": 1024,
            "depth": 8,
            "heads": 12, 
            "mlp_dim": 4096,
            "dim_head": 64
        }
    }
    
    config = configs.get(model_size, configs["base"])
    
    return CascadedCrossModalTransformer(
        num_classes=num_classes,
        task_type=task_type,
        **config
    )
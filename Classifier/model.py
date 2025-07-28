"""
model.py - Binary classification model for ESL score grouping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
import math


class AttentionPooling(nn.Module):
    """
    Attention pooling layer (reused from original code)
    """
    def __init__(self, hidden_dim, expected_seq_len=32, attn_proj=None, dropout=None):
        super().__init__()
        self.attn_proj = attn_proj or nn.Linear(hidden_dim, 1)
        init_scale = 1.0 / math.log(expected_seq_len)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        if dropout is not None and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, hidden_states, attention_mask=None, visualize=False):
        """
        hidden_states: [B, T, D]
        attention_mask: [B, T] (1 = keep, 0 = pad); optional
        """
        B, T, D = hidden_states.size()
        device = hidden_states.device

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.float32, device=device)

        raw_scores = self.attn_proj(hidden_states)  # [B, T, 1]

        scale_factor = self.scale * math.log(T)
        scaled_scores = raw_scores * scale_factor  # [B, T, 1]

        attn_mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
        scaled_scores = scaled_scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = F.softmax(scaled_scores, dim=1)  # [B, T, 1]

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        pooled = torch.sum(attn_weights * hidden_states, dim=1)  # [B, D]

        if visualize:
            return pooled, attn_weights
        else:
            return pooled


class ESLBinaryClassifier(nn.Module):
    """
    Binary classifier for ESL score grouping
    Group 0: scores 3.5-6.5 (lower proficiency)
    Group 1: scores 7-10 (higher proficiency)
    """
    def __init__(self, 
                 model_name='Alibaba-NLP/gte-multilingual-base',
                 pooling_dropout=0.3,
                 classifier_dropout=0.5,
                 avg_last_k=4,
                 hidden_dim=256):
        super().__init__()
        
        self.pooling_dropout = pooling_dropout
        self.classifier_dropout = classifier_dropout
        self.avg_last_k = avg_last_k
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        
        # Text encoder
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        text_hidden_size = self.encoder.config.hidden_size
        self.encoder.gradient_checkpointing_enable()
        
        # Projection layer to common hidden dimension
        self.text_proj = nn.Linear(text_hidden_size, hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        
        # Attention pooling
        self.attn_proj = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Dropout(pooling_dropout),
            nn.Linear(128, 1, bias=False)
        )
        self.attention_pool = AttentionPooling(
            hidden_dim, 
            attn_proj=self.attn_proj,
            expected_seq_len=512,
            dropout=pooling_dropout
        )
        
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
    def encode_text(self, input_ids, attention_mask):
        """Encode text using the transformer encoder"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states
        
        # Average last k layers
        k = min(self.avg_last_k, len(all_hidden_states))
        if k == 1:
            hidden_states = all_hidden_states[-1]
        else:
            hidden_states = torch.stack(all_hidden_states[-k:], dim=0).mean(dim=0)
        
        hidden_states = hidden_states.float()
        return hidden_states  # [batch, seq_len, hidden_size]
    
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        # Encode text
        text_hidden = self.encode_text(input_ids, attention_mask)  # [B, T, H]
        
        # Project to common space
        text_proj = self.text_proj(text_hidden)  # [B, T, hidden_dim]
        text_proj = self.text_norm(text_proj)
        
        # Apply attention pooling
        with torch.amp.autocast('cuda', enabled=False):
            pooled = self.attention_pool(text_proj, attention_mask)  # [B, hidden_dim]
        
        # Classification
        logits = self.classifier(pooled)  # [B, 2]
        
        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=-1),
            'predictions': torch.argmax(logits, dim=-1)
        }
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'model_name': self.model_name,
                'pooling_dropout': self.pooling_dropout,
                'classifier_dropout': self.classifier_dropout,
                'avg_last_k': self.avg_last_k,
                'hidden_dim': self.hidden_dim
            }
        }, path)
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(
            model_name=config['model_name'],
            pooling_dropout=config.get('pooling_dropout', 0.3),
            classifier_dropout=config.get('classifier_dropout', 0.5),
            avg_last_k=config.get('avg_last_k', 4),
            hidden_dim=config.get('hidden_dim', 256)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_class_weights(dataset):
    """
    Calculate class weights for handling imbalance
    """
    from collections import Counter
    
    group_counts = Counter(dataset.groups)
    total = len(dataset.groups)
    
    # Calculate inverse frequency weights
    weights = torch.zeros(2)
    for group, count in group_counts.items():
        weights[group] = total / (2 * count)
    
    print(f"Class weights - Group 0: {weights[0]:.3f}, Group 1: {weights[1]:.3f}")
    return weights
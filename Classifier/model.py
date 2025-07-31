"""
model.py - Binary classification model for ESL score grouping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import math
import gc

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
    def __init__(self, 
                 model_name='Alibaba-NLP/gte-multilingual-base', 
                 audio_encoder_id="jonatasgrosman/wav2vec2-large-xlsr-53-english",
                 pooling_dropout=0.3, 
                 classifier_dropout=0.5, 
                 avg_last_k=4,
                 d_fuse=256):
        super().__init__()
        self.num_types = 3
        self.pooling_dropout = pooling_dropout
        self.classifier_dropout = classifier_dropout
        self.avg_last_k = avg_last_k
        self.d_fuse = d_fuse

        # ========== TEXT ENCODER ==========
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        text_hidden_size = self.encoder.config.hidden_size
        self.encoder.gradient_checkpointing_enable()

        # ========== AUDIO ENCODER ==========
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_id)
        self.audio_hidden_dim = self.audio_encoder.config.output_hidden_size
        
        # ========== PROJECTION LAYERS ==========
        # Audio projection to common space
        self.audio_proj = nn.Linear(self.audio_hidden_dim, d_fuse)
        self.audio_norm = nn.LayerNorm(d_fuse)
        
        # Text projection to common space
        self.text_proj = nn.Linear(text_hidden_size, d_fuse)
        self.text_norm = nn.LayerNorm(d_fuse)
        
        # ========== 3 ATTENTION MECHANISMS ==========
        # 1. Text Self-Attention (q=text, k=text, v=text)
        self.text_self_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.text_self_norm = nn.LayerNorm(d_fuse)
        
        # 2. Text-to-Audio Cross-Attention (q=text, k=audio, v=audio)
        self.text_to_audio_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.t2a_norm = nn.LayerNorm(d_fuse)
        
        # 3. Audio-to-Text Cross-Attention (q=audio, k=text, v=text)
        self.audio_to_text_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.a2t_norm = nn.LayerNorm(d_fuse)
        
        # ========== 3 ATTENTION POOLING LAYERS ==========
        # Attention pooling for text self-attention output
        self.text_self_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, 256),
            nn.Tanh(), 
            nn.Dropout(pooling_dropout),
            nn.Linear(256, 1, bias=False)
        )
        self.text_self_pool = AttentionPooling(d_fuse, attn_proj=self.text_self_attn_proj, 
                                              expected_seq_len=512, dropout=pooling_dropout)
        
        # Attention pooling for text-to-audio output
        self.t2a_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, 256),
            nn.Tanh(), 
            nn.Dropout(pooling_dropout),
            nn.Linear(256, 1, bias=False)
        )
        self.t2a_pool = AttentionPooling(d_fuse, attn_proj=self.t2a_attn_proj, 
                                        expected_seq_len=512, dropout=pooling_dropout)
        
        # Attention pooling for audio-to-text output
        self.a2t_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, 256),
            nn.Tanh(), 
            nn.Dropout(pooling_dropout),
            nn.Linear(256, 1, bias=False)
        )
        self.a2t_pool = AttentionPooling(d_fuse, attn_proj=self.a2t_attn_proj, 
                                        expected_seq_len=10, dropout=pooling_dropout)  # num_chunks
        
        # ========== REGRESSION HEAD ==========
        # Takes concatenated 3 vectors: 3 * d_fuse
        self.classifier = nn.Sequential(
            nn.Linear(3 * d_fuse, 2 * d_fuse, bias=False),
            nn.LayerNorm(2 * d_fuse),
            nn.GELU(),
            nn.Dropout(self.classifier_dropout),
            nn.Linear(2 * d_fuse, d_fuse, bias=False),
            nn.LayerNorm(d_fuse),
            nn.GELU(),
            nn.Dropout(self.classifier_dropout),
            nn.Linear(d_fuse, 2, bias=False)
        )

    def encode_text(self, input_ids, attention_mask):
        """Encode text without pooling"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states
        k = min(self.avg_last_k, len(all_hidden_states))
        if k == 1:
            hidden_states = all_hidden_states[-1]
        else:
            hidden_states = torch.stack(all_hidden_states[-k:], dim=0).mean(dim=0)
        hidden_states = hidden_states.float()
        return hidden_states  # [batch, seq_len, text_hidden_dim]

    def encode_audio(self, audio):
        """Encode audio chunks using Wav2Vec2"""
        if audio is None:
            return None

        batch_size, num_chunks, waveform_len = audio.shape
        device = next(self.parameters()).device

        audio_encoder_out = []
        for i in range(num_chunks):
            inp = audio[:, i, :].to(device)
            with torch.no_grad():
                out = self.audio_encoder(input_values=inp).last_hidden_state
                audio_encoder_out.append(out.mean(dim=1).detach().cpu())

            del inp, out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        audio_features = torch.stack(audio_encoder_out, dim=1).to(device)  # (batch, num_chunks, audio_hidden_dim)
        audio_features = self.audio_proj(audio_features)  # (batch, num_chunks, d_fuse)
        audio_features = self.audio_norm(audio_features)
        return audio_features

    def apply_three_attention_mechanisms(self, text_features, audio_features, attention_mask):
        """
        Apply 3 attention mechanisms and return 3 pooled vectors
        Args:
            text_features: [batch, seq_len, text_hidden_dim]
            audio_features: [batch, num_chunks, d_fuse] or None
            attention_mask: [batch, seq_len]
        Returns:
            Tuple of 3 pooled vectors, each [batch, d_fuse]
        """
        batch_size = text_features.size(0)
        device = text_features.device
        
        # Project text to common space
        text_proj = self.text_proj(text_features)  # [batch, seq_len, d_fuse]
        text_proj = self.text_norm(text_proj)
        
        # 1. Text Self-Attention (q=text, k=text, v=text)
        text_self_output, _ = self.text_self_attention(
            query=text_proj, 
            key=text_proj, 
            value=text_proj
        )
        text_self_output = self.text_self_norm(text_self_output)  # [batch, seq_len, d_fuse]
        
        # Pool text self-attention output
        with torch.amp.autocast('cuda', enabled=False):
            text_self_pooled = self.text_self_pool(text_self_output, attention_mask)  # [batch, d_fuse]
        
        if audio_features is None:
            # If no audio, create zero vectors for audio-related attentions
            t2a_pooled = torch.zeros(batch_size, self.d_fuse, device=device)
            a2t_pooled = torch.zeros(batch_size, self.d_fuse, device=device)
        else:
            # 2. Text-to-Audio Cross-Attention (q=text, k=audio, v=audio)
            t2a_output, _ = self.text_to_audio_attention(
                query=text_proj,      # [batch, seq_len, d_fuse]
                key=audio_features,   # [batch, num_chunks, d_fuse]
                value=audio_features  # [batch, num_chunks, d_fuse]
            )
            t2a_output = self.t2a_norm(t2a_output)  # [batch, seq_len, d_fuse]
            
            # Pool text-to-audio output
            with torch.amp.autocast('cuda', enabled=False):
                t2a_pooled = self.t2a_pool(t2a_output, attention_mask)  # [batch, d_fuse]
            
            # 3. Audio-to-Text Cross-Attention (q=audio, k=text, v=text)
            a2t_output, _ = self.audio_to_text_attention(
                query=audio_features, # [batch, num_chunks, d_fuse]
                key=text_proj,        # [batch, seq_len, d_fuse]
                value=text_proj       # [batch, seq_len, d_fuse]
            )
            a2t_output = self.a2t_norm(a2t_output)  # [batch, num_chunks, d_fuse]
            
            # Pool audio-to-text output (no mask needed for audio)
            with torch.amp.autocast('cuda', enabled=False):
                a2t_pooled = self.a2t_pool(a2t_output)  # [batch, d_fuse]
        
        return text_self_pooled, t2a_pooled, a2t_pooled

    def forward(self, input_ids, attention_mask, audio=None):
        """Forward pass with 3 attention mechanisms"""
        # Text encoding
        text_hidden_states = self.encode_text(input_ids, attention_mask)  # [batch, seq_len, text_hidden_dim]
        
        # Audio encoding
        audio_features = self.encode_audio(audio)  # [batch, num_chunks, d_fuse] or None
        
        # Apply 3 attention mechanisms and get 3 pooled vectors
        text_self_pooled, t2a_pooled, a2t_pooled = self.apply_three_attention_mechanisms(
            text_hidden_states, audio_features, attention_mask
        )
        
        # Concatenate 3 vectors
        combined_features = torch.cat([text_self_pooled, t2a_pooled, a2t_pooled], dim=1)  # [batch, 3*d_fuse]

        # Classification
        logits = self.classifier(combined_features)  # [B, 2]
        
        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=-1),
            'predictions': torch.argmax(logits, dim=-1)
        }

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'pooling_dropout': self.pooling_dropout,
                'regression_dropout': self.regression_dropout,
                'model_name': self.encoder.config._name_or_path,
                'avg_last_k': self.avg_last_k,
                'd_fuse': self.d_fuse
            }
        }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        model = cls(
            model_name=config.get('model_name', 'Alibaba-NLP/gte-multilingual-base'),
            pooling_dropout=config.get('pooling_dropout', 0.3),
            regression_dropout=config.get('regression_dropout', 0.5),
            avg_last_k=config.get('avg_last_k', 1),
            d_fuse=config.get('d_fuse', 256)
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
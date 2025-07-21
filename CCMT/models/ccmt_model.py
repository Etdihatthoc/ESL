import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Optional
from .audio_encoder import AudioEncoder
from .text_encoder import EnglishTextEncoder, VietnameseTextEncoder
from .ccmt import CascadedCrossModalTransformer


class MultiModalProjection(nn.Module):
    """
    Projects different modalities to a common dimension space
    """
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.projection(x)


class TokenSampler(nn.Module):
    """
    Samples fixed number of tokens from variable length sequences
    """
    def __init__(self, num_tokens=100, strategy="random"):
        super().__init__()
        self.num_tokens = num_tokens
        self.strategy = strategy
    
    def forward(self, tokens, attention_mask=None, keep_cls=True):
        """
        Sample fixed number of tokens from input
        
        Args:
            tokens: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len], optional
            keep_cls: whether to always keep the first token (CLS token)
            
        Returns:
            sampled_tokens: [batch_size, num_tokens, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = tokens.shape
        
        if seq_len <= self.num_tokens:
            # If sequence is shorter, pad with zeros
            padding = torch.zeros(
                batch_size, 
                self.num_tokens - seq_len, 
                hidden_dim,
                device=tokens.device,
                dtype=tokens.dtype
            )
            return torch.cat([tokens, padding], dim=1)
        
        # Sample tokens
        if self.strategy == "random":
            if keep_cls and seq_len > 1:
                # Always keep the first token (CLS)
                cls_token = tokens[:, 0:1, :]  # [batch_size, 1, hidden_dim]
                remaining_tokens = tokens[:, 1:, :]  # [batch_size, seq_len-1, hidden_dim]
                
                # Sample remaining tokens
                num_remaining = self.num_tokens - 1
                if remaining_tokens.shape[1] <= num_remaining:
                    sampled_tokens = torch.cat([cls_token, remaining_tokens], dim=1)
                else:
                    # Random sampling
                    indices = torch.randperm(remaining_tokens.shape[1])[:num_remaining]
                    indices = indices.sort()[0]  # Sort to maintain some order
                    sampled_remaining = remaining_tokens[:, indices, :]
                    sampled_tokens = torch.cat([cls_token, sampled_remaining], dim=1)
            else:
                # Random sampling without preserving CLS
                indices = torch.randperm(seq_len)[:self.num_tokens]
                indices = indices.sort()[0]
                sampled_tokens = tokens[:, indices, :]
        else:
            # Simply truncate
            sampled_tokens = tokens[:, :self.num_tokens, :]
        
        return sampled_tokens


class ESLCCMTModel(nn.Module):
    """
    CCMT model adapted for ESL grading task
    """
    def __init__(self,
                 # Audio encoder configs
                 audio_model_id="facebook/wav2vec2-base-960h",
                 # Text encoder configs  
                 english_model_name="bert-base-uncased",
                 vietnamese_model_name="vinai/phobert-base-v2",
                 # CCMT configs
                 common_dim=256,
                 num_tokens_per_modality=100,
                 ccmt_depth=6,
                 ccmt_heads=8,
                 ccmt_mlp_dim=1024,
                 # Task-specific configs
                 num_score_bins=21,  # 0 to 10 in 0.5 increments
                 dropout=0.2):
        super().__init__()
        
        self.common_dim = common_dim
        self.num_tokens_per_modality = num_tokens_per_modality
        self.num_score_bins = num_score_bins
        
        # Initialize encoders
        self.audio_encoder = AudioEncoder(
            model_id=audio_model_id,
            freeze_feature_encoder=False
        )
        
        self.english_text_encoder = EnglishTextEncoder(
            model_name=english_model_name
        )
        
        self.vietnamese_text_encoder = VietnameseTextEncoder(
            model_name=vietnamese_model_name
        )
        
        # Projection layers to common dimension
        self.audio_projection = MultiModalProjection(
            self.audio_encoder.get_output_dim(),
            common_dim,
            dropout
        )
        
        self.english_text_projection = MultiModalProjection(
            self.english_text_encoder.get_output_dim(),
            common_dim,
            dropout
        )
        
        self.vietnamese_text_projection = MultiModalProjection(
            self.vietnamese_text_encoder.get_output_dim(),
            common_dim,
            dropout
        )
        
        # Token samplers for each modality
        self.token_sampler = TokenSampler(
            num_tokens=num_tokens_per_modality,
            strategy="random"
        )
        
        # CCMT architecture
        total_patches = num_tokens_per_modality * 3  # 3 modalities
        self.ccmt = CascadedCrossModalTransformer(
            num_classes=num_score_bins,
            num_patches=total_patches,
            dim=common_dim,
            depth=ccmt_depth,
            heads=ccmt_heads,
            mlp_dim=ccmt_mlp_dim,
            dropout=dropout
        )
        
        # Task-specific head for regression
        self.score_head = nn.Sequential(
            nn.LayerNorm(common_dim),
            nn.Linear(common_dim, common_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim // 2, num_score_bins)
        )
        
    def forward(self, 
                audio_chunks,
                english_input_ids,
                english_attention_mask,
                vietnamese_input_ids,
                vietnamese_attention_mask):
        """
        Forward pass of the CCMT model
        
        Args:
            audio_chunks: [batch_size, num_chunks, waveform_length]
            english_input_ids: [batch_size, seq_len]
            english_attention_mask: [batch_size, seq_len]
            vietnamese_input_ids: [batch_size, seq_len]
            vietnamese_attention_mask: [batch_size, seq_len]
            
        Returns:
            Dictionary with logits, probabilities, and expected score
        """
        batch_size = english_input_ids.shape[0]
        device = english_input_ids.device
        
        # Encode each modality
        # 1. Audio encoding
        if audio_chunks is not None:
            audio_features = self.audio_encoder(audio_chunks)  # [batch_size, num_chunks, audio_hidden_dim]
            audio_features = self.audio_projection(audio_features)  # [batch_size, num_chunks, common_dim]
            audio_tokens = self.token_sampler(audio_features, keep_cls=False)
        else:
            # Create dummy audio tokens if no audio
            audio_tokens = torch.zeros(
                batch_size, self.num_tokens_per_modality, self.common_dim,
                device=device, dtype=torch.float32
            )
        
        # 2. English text encoding
        english_hidden_states = self.english_text_encoder(
            english_input_ids, english_attention_mask
        )  # [batch_size, seq_len, english_hidden_dim]
        english_features = self.english_text_projection(english_hidden_states)
        english_tokens = self.token_sampler(
            english_features, english_attention_mask, keep_cls=True
        )
        
        # 3. Vietnamese text encoding
        vietnamese_hidden_states = self.vietnamese_text_encoder(
            vietnamese_input_ids, vietnamese_attention_mask
        )  # [batch_size, seq_len, vietnamese_hidden_dim]
        vietnamese_features = self.vietnamese_text_projection(vietnamese_hidden_states)
        vietnamese_tokens = self.token_sampler(
            vietnamese_features, vietnamese_attention_mask, keep_cls=True
        )
        
        # Concatenate all modalities for CCMT
        # Order: English text, Vietnamese text, Audio
        multimodal_tokens = torch.cat([
            english_tokens,      # [batch_size, num_tokens, common_dim]
            vietnamese_tokens,   # [batch_size, num_tokens, common_dim] 
            audio_tokens         # [batch_size, num_tokens, common_dim]
        ], dim=1)  # [batch_size, 3*num_tokens, common_dim]
        
        # Apply CCMT
        ccmt_output = self.ccmt(multimodal_tokens)  # [batch_size, num_classes]
        
        # Get logits and probabilities
        logits = ccmt_output  # Already from classification head in CCMT
        
        # For additional regression head, we can use the class token features
        # Note: This is optional and can be removed if not needed
        try:
            if hasattr(self.ccmt, 'get_class_token_features'):
                class_features = self.ccmt.get_class_token_features(multimodal_tokens)
                additional_logits = self.score_head(class_features)
                # Combine logits (you can experiment with different strategies)
                logits = (logits + additional_logits) / 2
        except Exception as e:
            # If there's any issue with the additional head, just use CCMT output
            pass
        
        # Convert to probabilities and expected score
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate expected score (0 to 10 in 0.5 increments)
        score_bins = torch.linspace(0, 10, steps=self.num_score_bins, device=device)
        expected_score = (probs * score_bins).sum(dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'expected_score': expected_score,
            'multimodal_features': multimodal_tokens
        }
    
    def save(self, path):
        """Save model state"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': {
                    'common_dim': self.common_dim,
                    'num_tokens_per_modality': self.num_tokens_per_modality,
                    'num_score_bins': self.num_score_bins
                }
            }, path)
            print(f"Model saved successfully to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    @classmethod
    def load(cls, path, **kwargs):
        """Load model from checkpoint"""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            config = checkpoint.get('config', {})
            
            # Merge config with kwargs
            config.update(kwargs)
            
            model = cls(**config)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully from {path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
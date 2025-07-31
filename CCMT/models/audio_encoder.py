"""
Audio encoder using Wav2Vec2.0 for audio feature extraction
"""

import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Optional, Tuple
import random


class AudioEncoder(nn.Module):
    """
    Wav2Vec2.0 wrapper for audio encoding in CCMT pipeline
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        target_dim: int = 768,  # Target dimension to match text encoders
        max_tokens: int = 100,  # Number of tokens to sample
        freeze_feature_extractor: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.target_dim = target_dim
        
        # Load pretrained Wav2Vec2.0 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze feature extractor if specified
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
        
        # Get model dimension
        self.wav2vec2_dim = self.wav2vec2.config.hidden_size
        
        # Projection layer to match target dimension
        if self.wav2vec2_dim != target_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.wav2vec2_dim, target_dim),
                nn.LayerNorm(target_dim),
                nn.Dropout(dropout)
            )
        else:
            self.projection = nn.Identity()
        
        # Add a learnable class token
        self.class_token = nn.Parameter(torch.randn(1, 1, target_dim) * 0.02)
        
    def forward(
        self, 
        audio_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through audio encoder
        
        Args:
            audio_input: Raw audio waveform (batch, sequence_length)
            attention_mask: Attention mask for variable length audio
        Returns:
            Audio tokens (batch, max_tokens, target_dim)
        """
        batch_size = audio_input.shape[0]
        
        # Extract features using Wav2Vec2.0
        with torch.cuda.amp.autocast(enabled=False):  # Wav2Vec2 doesn't support autocast
            outputs = self.wav2vec2(
                audio_input,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
        
        # Get last hidden states
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, wav2vec2_dim)
        
        # Project to target dimension
        projected_features = self.projection(hidden_states)  # (batch, seq_len, target_dim)
        
        # Add class token at the beginning
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        features_with_class = torch.cat([class_tokens, projected_features], dim=1)
        
        # Sample tokens to match required length
        sampled_tokens = self._sample_tokens(features_with_class)
        
        return sampled_tokens
    
    def _sample_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens to match max_tokens requirement
        Always keeps the class token (first token)
        
        Args:
            features: Feature tokens (batch, seq_len, dim)
        Returns:
            Sampled tokens (batch, max_tokens, dim)
        """
        batch_size, seq_len, dim = features.shape
        
        if seq_len <= self.max_tokens:
            # Pad if too short
            if seq_len < self.max_tokens:
                padding = torch.zeros(
                    batch_size, self.max_tokens - seq_len, dim,
                    device=features.device, dtype=features.dtype
                )
                features = torch.cat([features, padding], dim=1)
            return features
        else:
            # Sample if too long (always keep class token)
            class_token = features[:, :1]  # (batch, 1, dim)
            other_tokens = features[:, 1:]  # (batch, seq_len-1, dim)
            
            # Randomly sample remaining tokens
            if self.training:
                # Random sampling during training
                indices = torch.randperm(other_tokens.shape[1])[:self.max_tokens-1]
                indices = indices.sort()[0]  # Sort to maintain some order
            else:
                # Uniform sampling during inference for consistency
                step = other_tokens.shape[1] / (self.max_tokens - 1)
                indices = torch.arange(0, other_tokens.shape[1], step, dtype=torch.long)[:self.max_tokens-1]
            
            sampled_tokens = other_tokens[:, indices]
            return torch.cat([class_token, sampled_tokens], dim=1)
    
    def freeze(self):
        """Freeze the entire audio encoder"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze the entire audio encoder"""
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_wav2vec2(self):
        """Freeze only the Wav2Vec2.0 backbone"""
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
    
    def unfreeze_wav2vec2(self):
        """Unfreeze only the Wav2Vec2.0 backbone"""
        for param in self.wav2vec2.parameters():
            param.requires_grad = True


class AudioPreprocessor:
    """
    Audio preprocessing utilities for CCMT pipeline
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        max_length: float = 30.0,  # Maximum audio length in seconds
        normalize: bool = True
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(max_length * sample_rate)
        self.normalize = normalize
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
        Returns:
            Preprocessed audio tensor (sequence_length,)
        """
        # Load audio
        waveform, orig_sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if orig_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate, 
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Squeeze to 1D
        waveform = waveform.squeeze(0)
        
        # Truncate or pad to max_samples
        if len(waveform) > self.max_samples:
            # Random crop during training, center crop during inference
            if self.training:
                start = random.randint(0, len(waveform) - self.max_samples)
                waveform = waveform[start:start + self.max_samples]
            else:
                start = (len(waveform) - self.max_samples) // 2
                waveform = waveform[start:start + self.max_samples]
        elif len(waveform) < self.max_samples:
            # Pad with zeros
            padding = self.max_samples - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Normalize
        if self.normalize:
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
        
        return waveform
    
    def create_attention_mask(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask for variable length audio
        
        Args:
            waveforms: Batch of audio waveforms (batch, sequence_length)
        Returns:
            Attention mask (batch, sequence_length)
        """
        # Simple implementation - in practice you might want more sophisticated masking
        # based on actual audio length before padding
        return torch.ones_like(waveforms, dtype=torch.bool)


# Factory function for easy creation
def create_audio_encoder(
    model_size: str = "base",
    target_dim: int = 768,
    max_tokens: int = 100
) -> AudioEncoder:
    """
    Factory function to create audio encoders with predefined configurations
    
    Args:
        model_size: "base" or "large"
        target_dim: Target feature dimension
        max_tokens: Number of tokens to output
    Returns:
        Configured AudioEncoder
    """
    model_configs = {
        "base": "facebook/wav2vec2-base-960h", 
        "large": "facebook/wav2vec2-large-960h"
    }
    
    model_name = model_configs.get(model_size, model_configs["base"])
    
    return AudioEncoder(
        model_name=model_name,
        target_dim=target_dim,
        max_tokens=max_tokens
    )
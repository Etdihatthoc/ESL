"""
Text encoders for English and Vietnamese text processing
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer
)
from typing import Optional, List, Dict, Tuple
import random


class BaseTextEncoder(nn.Module):
    """Base class for text encoders"""
    
    def __init__(
        self,
        model_name: str,
        target_dim: int = 768,
        max_tokens: int = 100,
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name
        self.target_dim = target_dim
        self.max_tokens = max_tokens
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Get model dimension
        self.model_dim = self.model.config.hidden_size
        
        # Projection layer if dimensions don't match
        if self.model_dim != target_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.model_dim, target_dim),
                nn.LayerNorm(target_dim),
                nn.Dropout(dropout)
            )
        else:
            self.projection = nn.Identity()
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize batch of texts
        
        Args:
            texts: List of text strings
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through text encoder
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
        Returns:
            Text tokens (batch, max_tokens, target_dim)
        """
        # Forward through transformer
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, model_dim)
        
        # Project to target dimension
        projected_features = self.projection(hidden_states)  # (batch, seq_len, target_dim)
        
        # Sample tokens to match required length
        sampled_tokens = self._sample_tokens(projected_features, attention_mask)
        
        return sampled_tokens
    
    def _sample_tokens(
        self, 
        features: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample tokens to match max_tokens requirement
        Always keeps CLS token (first token)
        
        Args:
            features: Feature tokens (batch, seq_len, dim)
            attention_mask: Attention mask (batch, seq_len)
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
            # Sample if too long (always keep CLS token)
            cls_token = features[:, :1]  # (batch, 1, dim)
            other_tokens = features[:, 1:]  # (batch, seq_len-1, dim)
            
            if attention_mask is not None:
                # Use attention mask to identify valid tokens
                valid_mask = attention_mask[:, 1:]  # Exclude CLS position
                
                sampled_tokens_list = []
                for i in range(batch_size):
                    valid_indices = torch.where(valid_mask[i])[0]
                    if len(valid_indices) >= self.max_tokens - 1:
                        # Randomly sample from valid tokens
                        if self.training:
                            selected = torch.randperm(len(valid_indices))[:self.max_tokens-1]
                            selected = selected.sort()[0]
                        else:
                            # Uniform sampling for consistency
                            step = len(valid_indices) / (self.max_tokens - 1)
                            selected = torch.arange(0, len(valid_indices), step, dtype=torch.long)[:self.max_tokens-1]
                        indices = valid_indices[selected]
                    else:
                        # Use all valid tokens and pad
                        indices = valid_indices
                    
                    selected_tokens = other_tokens[i, indices]  # (num_selected, dim)
                    
                    # Pad if necessary
                    if len(selected_tokens) < self.max_tokens - 1:
                        padding_needed = self.max_tokens - 1 - len(selected_tokens)
                        padding = torch.zeros(
                            padding_needed, dim,
                            device=features.device, dtype=features.dtype
                        )
                        selected_tokens = torch.cat([selected_tokens, padding], dim=0)
                    
                    sampled_tokens_list.append(selected_tokens)
                
                sampled_tokens = torch.stack(sampled_tokens_list)  # (batch, max_tokens-1, dim)
            else:
                # Simple random/uniform sampling without attention mask
                if self.training:
                    indices = torch.randperm(other_tokens.shape[1])[:self.max_tokens-1]
                    indices = indices.sort()[0]
                else:
                    step = other_tokens.shape[1] / (self.max_tokens - 1)
                    indices = torch.arange(0, other_tokens.shape[1], step, dtype=torch.long)[:self.max_tokens-1]
                
                sampled_tokens = other_tokens[:, indices]
            
            return torch.cat([cls_token, sampled_tokens], dim=1)
    
    def freeze(self):
        """Freeze the entire text encoder"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze the entire text encoder"""
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self):
        """Freeze only the transformer backbone"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze only the transformer backbone"""
        for param in self.model.parameters():
            param.requires_grad = True


class EnglishTextEncoder(BaseTextEncoder):
    """
    English text encoder using BERT or RoBERTa
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        target_dim: int = 768,
        max_tokens: int = 100,
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__(
            model_name=model_name,
            target_dim=target_dim,
            max_tokens=max_tokens,
            max_length=max_length,
            dropout=dropout
        )
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        High-level interface to encode texts
        
        Args:
            texts: List of English text strings
        Returns:
            Encoded tokens (batch, max_tokens, target_dim)
        """
        # Tokenize
        tokenized = self.tokenize_texts(texts)
        
        # Move to same device as model
        device = next(self.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        # Encode
        with torch.no_grad():
            encoded = self.forward(**tokenized)
        
        return encoded


class VietnameseTextEncoder(BaseTextEncoder):
    """
    Vietnamese text encoder using PhoBERT or multilingual models
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",  # PhoBERT for Vietnamese
        target_dim: int = 768,
        max_tokens: int = 100,
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__(
            model_name=model_name,
            target_dim=target_dim,
            max_tokens=max_tokens,
            max_length=max_length,
            dropout=dropout
        )
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        High-level interface to encode Vietnamese texts
        
        Args:
            texts: List of Vietnamese text strings
        Returns:
            Encoded tokens (batch, max_tokens, target_dim)
        """
        # Tokenize
        tokenized = self.tokenize_texts(texts)
        
        # Move to same device as model
        device = next(self.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        # Encode
        with torch.no_grad():
            encoded = self.forward(**tokenized)
        
        return encoded


class TextPreprocessor:
    """
    Text preprocessing utilities for CCMT pipeline
    """
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text string
        Returns:
            Cleaned text string
        """
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common ASR artifacts
        text = text.replace('[REPEAT]', '')
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed annotations
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of raw text strings
        Returns:
            List of cleaned text strings
        """
        return [self.clean_text(text) for text in texts]


# Factory functions for easy creation
def create_english_encoder(
    model_type: str = "bert",  # "bert", "roberta"
    model_size: str = "base",  # "base", "large"
    target_dim: int = 768,
    max_tokens: int = 100
) -> EnglishTextEncoder:
    """
    Factory function to create English text encoders
    
    Args:
        model_type: Type of model ("bert", "roberta")
        model_size: Model size ("base", "large")
        target_dim: Target feature dimension
        max_tokens: Number of tokens to output
    Returns:
        Configured EnglishTextEncoder
    """
    model_configs = {
        ("bert", "base"): "bert-base-uncased",
        ("bert", "large"): "bert-large-uncased",
        ("roberta", "base"): "roberta-base",
        ("roberta", "large"): "roberta-large"
    }
    
    model_name = model_configs.get((model_type, model_size), "bert-base-uncased")
    
    return EnglishTextEncoder(
        model_name=model_name,
        target_dim=target_dim,
        max_tokens=max_tokens
    )


def create_vietnamese_encoder(
    model_type: str = "phobert",  # "phobert", "multilingual"
    target_dim: int = 768,
    max_tokens: int = 100
) -> VietnameseTextEncoder:
    """
    Factory function to create Vietnamese text encoders
    
    Args:
        model_type: Type of model ("phobert", "multilingual")
        target_dim: Target feature dimension
        max_tokens: Number of tokens to output
    Returns:
        Configured VietnameseTextEncoder
    """
    model_configs = {
        "phobert": "vinai/phobert-base",
        "multilingual": "bert-base-multilingual-cased"
    }
    
    model_name = model_configs.get(model_type, "vinai/phobert-base")
    
    return VietnameseTextEncoder(
        model_name=model_name,
        target_dim=target_dim,
        max_tokens=max_tokens
    )
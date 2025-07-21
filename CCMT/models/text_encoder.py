import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class TextEncoder(nn.Module):
    """
    Text encoder using BERT-like models for different languages
    """
    def __init__(self, 
                 model_name,
                 avg_last_k=4,
                 max_length=256):
        super().__init__()
        self.model_name = model_name
        self.avg_last_k = avg_last_k
        self.max_length = max_length
        
        # Load model configuration and enable hidden states output
        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        
        # Load the transformer model
        self.encoder = AutoModel.from_pretrained(
            model_name, 
            config=config,
            trust_remote_code=True
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.encoder.gradient_checkpointing_enable()
        
        self.hidden_dim = self.encoder.config.hidden_size
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for text encoding
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_dim]
        """
        # Get all hidden states from the model
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        all_hidden_states = outputs.hidden_states
        
        # Average the last k layers
        k = min(self.avg_last_k, len(all_hidden_states))
        if k == 1:
            hidden_states = all_hidden_states[-1]
        else:
            # Stack and average the last k layers
            stacked_states = torch.stack(all_hidden_states[-k:], dim=0)
            hidden_states = stacked_states.mean(dim=0)
        
        # Ensure float precision
        hidden_states = hidden_states.float()
        
        return hidden_states
    
    def get_output_dim(self):
        """Return the output dimension of the encoder"""
        return self.hidden_dim


class EnglishTextEncoder(TextEncoder):
    """English text encoder using BERT"""
    def __init__(self, 
                 model_name="bert-base-uncased",
                 avg_last_k=4,
                 max_length=256):
        super().__init__(model_name, avg_last_k, max_length)


class VietnameseTextEncoder(TextEncoder):
    """Vietnamese text encoder using PhoBERT"""
    def __init__(self, 
                 model_name="vinai/phobert-base-v2",
                 avg_last_k=4,
                 max_length=256):
        super().__init__(model_name, avg_last_k, max_length)


def create_text_encoder(language="english", **kwargs):
    """
    Factory function to create text encoders for different languages
    
    Args:
        language: "english" or "vietnamese"
        **kwargs: additional arguments for the encoder
        
    Returns:
        TextEncoder instance
    """
    if language.lower() == "english":
        return EnglishTextEncoder(**kwargs)
    elif language.lower() == "vietnamese":
        return VietnameseTextEncoder(**kwargs)
    else:
        raise ValueError(f"Unsupported language: {language}")
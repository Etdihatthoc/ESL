import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import gc


class AudioEncoder(nn.Module):
    """
    Audio encoder using Wav2Vec2 for extracting audio features
    """
    def __init__(self, 
                 model_id="facebook/wav2vec2-base-960h",
                 hidden_dim=768,
                 freeze_feature_encoder=False):
        super().__init__()
        self.model_id = model_id
        self.hidden_dim = hidden_dim
        
        # Load Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_id)
        
        # Optionally freeze feature encoder (CNN layers)
        if freeze_feature_encoder:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        
        # Enable gradient checkpointing to save memory
        self.wav2vec2.gradient_checkpointing_enable()
        
    def forward(self, audio_chunks):
        """
        Forward pass for audio encoding
        
        Args:
            audio_chunks: [batch_size, num_chunks, waveform_length]
            
        Returns:
            audio_features: [batch_size, num_chunks, hidden_dim]
        """
        if audio_chunks is None:
            return None
            
        batch_size, num_chunks, waveform_len = audio_chunks.shape
        device = audio_chunks.device
        
        # Process each chunk separately to avoid memory issues
        chunk_features = []
        
        for i in range(num_chunks):
            chunk_input = audio_chunks[:, i, :].to(device)  # [batch_size, waveform_length]
            
            with torch.cuda.amp.autocast():
                # Get last hidden state from Wav2Vec2
                outputs = self.wav2vec2(input_values=chunk_input)
                
                # Average pool over sequence length to get fixed-size representation
                chunk_feat = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_dim]
                chunk_features.append(chunk_feat)
            
            # Clean up intermediate tensors
            del chunk_input, outputs
            
        # Stack all chunks
        audio_features = torch.stack(chunk_features, dim=1)  # [batch_size, num_chunks, hidden_dim]
        
        # Clean up
        del chunk_features
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return audio_features
    
    def get_output_dim(self):
        """Return the output dimension of the encoder"""
        return self.hidden_dim
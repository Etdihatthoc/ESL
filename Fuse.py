
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. GATED FUSION - MOST RECOMMENDED
# ==========================================

class GatedFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim, dropout=0.1):
        """
        Gated fusion with learnable gates for each modality
        Args:
            input_dim1: Dimension of first modality (text)
            input_dim2: Dimension of second modality (audio)
            output_dim: Output dimension
        """
        super().__init__()
        
        # Project both modalities to same dimension
        self.proj1 = nn.Linear(input_dim1, output_dim)
        self.proj2 = nn.Linear(input_dim2, output_dim)
        
        # Gating mechanism
        self.gate1 = nn.Sequential(
            nn.Linear(input_dim1 + input_dim2, output_dim),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(input_dim1 + input_dim2, output_dim),
            nn.Sigmoid()
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.GELU()
        )
        
    def forward(self, x1, x2):
        """
        x1: text features (batch, dim1)
        x2: audio features (batch, dim2)
        """
        # Project to same dimension
        h1 = self.proj1(x1)  # (batch, output_dim)
        h2 = self.proj2(x2)  # (batch, output_dim)
        
        # Compute gates
        concat = torch.cat([x1, x2], dim=1)
        g1 = self.gate1(concat)  # Gate for text
        g2 = self.gate2(concat)  # Gate for audio
        
        # Gated fusion
        fused = g1 * h1 + g2 * h2
        
        # Final transformation
        output = self.fusion(fused)
        return output
"""
CCMT Models Package
Contains all model components for Cascaded Cross-Modal Transformer
"""

from .ccmt_model import CascadedCrossModalTransformer, create_ccmt_model
from .audio_encoder import AudioEncoder, AudioPreprocessor, create_audio_encoder
from .text_encoders import (
    EnglishTextEncoder, VietnameseTextEncoder, TextPreprocessor,
    create_english_encoder, create_vietnamese_encoder
)
from .translator import (
    EnglishToVietnameseTranslator, FallbackTranslator,
    create_en_vi_translator, create_fallback_translator, quick_translate
)
from .components import (
    PreNorm, FeedForward, CrossAttention, 
    CrossModalTransformerBlock, ScoringHead
)

__all__ = [
    # Main model
    'CascadedCrossModalTransformer',
    'create_ccmt_model',
    
    # Audio components
    'AudioEncoder',
    'AudioPreprocessor',
    'create_audio_encoder',
    
    # Text components
    'EnglishTextEncoder', 
    'VietnameseTextEncoder',
    'TextPreprocessor',
    'create_english_encoder',
    'create_vietnamese_encoder',
    
    # Translation components
    'EnglishToVietnameseTranslator',
    'FallbackTranslator',
    'create_en_vi_translator',
    'create_fallback_translator',
    'quick_translate',
    
    # Core components
    'PreNorm',
    'FeedForward',
    'CrossAttention',
    'CrossModalTransformerBlock',
    'ScoringHead'
]
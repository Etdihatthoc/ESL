"""
CCMT Utils Package
Utility functions and helpers for CCMT implementation
"""

from .text_utils import (
    clean_transcript,
    normalize_text,
    extract_content_words,
    calculate_text_similarity,
    batch_process_texts
)

from .audio_utils import (
    load_audio_safe,
    calculate_audio_features,
    normalize_audio,
    detect_speech_segments,
    batch_process_audio
)

from .translation_utils import (
    batch_translate,
    validate_translation_quality,
    cache_translations,
    translation_fallback
)

from .tokenization import (
    sample_tokens_uniform,
    sample_tokens_random,
    align_token_lengths,
    create_attention_mask,
    pad_or_truncate_tokens
)

from .config import (
    CCMTConfig,
    load_config,
    save_config,
    merge_configs,
    get_default_config, 
    setup_directories, get_device_config, print_config
)

__all__ = [
    # Text utilities
    'clean_transcript',
    'normalize_text', 
    'extract_content_words',
    'calculate_text_similarity',
    'batch_process_texts',
    
    # Audio utilities
    'load_audio_safe',
    'calculate_audio_features',
    'normalize_audio',
    'detect_speech_segments', 
    'batch_process_audio',
    
    # Translation utilities
    'batch_translate',
    'validate_translation_quality',
    'cache_translations',
    'translation_fallback',
    
    # Tokenization utilities
    'sample_tokens_uniform',
    'sample_tokens_random',
    'align_token_lengths',
    'create_attention_mask',
    'pad_or_truncate_tokens',
    
    # Configuration
    'CCMTConfig',
    'load_config',
    'save_config', 
    'merge_configs',
    'get_default_config',
    'setup_directories',
    'get_device_config',
    'print_config'
]
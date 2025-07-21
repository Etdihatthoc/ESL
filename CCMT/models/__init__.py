"""
CCMT Models Package for ESL Grading

This package contains the implementation of Cascaded Cross-Modal Transformer (CCMT)
adapted for English as Second Language (ESL) grading tasks.

Main Components:
- ESLCCMTModel: Main model combining audio, English text, and Vietnamese text
- AudioEncoder: Wav2Vec2-based audio feature extraction
- TextEncoder: BERT-like text encoders for different languages
- TextProcessor: ASR transcription and translation utilities
- ESLCCMTDataset: Dataset class for CCMT training
- CascadedCrossModalTransformer: Core CCMT architecture
- CCMTTrainer: Complete training pipeline
- Training utilities: Loss functions, samplers, configurations
"""

# Core model components (no dependencies)
from .audio_encoder import AudioEncoder
from .text_encoder import (
    TextEncoder,
    EnglishTextEncoder, 
    VietnameseTextEncoder,
    create_text_encoder
)
from .text_processor import TextProcessor, AsyncTextProcessor
from .ccmt import CascadedCrossModalTransformer, Transformer, Attention
from .ccmt_model import ESLCCMTModel, MultiModalProjection, TokenSampler
from .ccmt_dataset import ESLCCMTDataset, get_ccmt_collate_fn

# Training utilities (may have external dependencies)
try:
    from .training_utils import (
        SoftTargetGenerator,
        ESLLossFunction,
        ClassWeightCalculator,
        ValidationMetrics,
        selective_freeze_embedding_layer,
        get_param_groups,
        maybe_empty_cache,
        compute_correlation
    )
    _training_utils_available = True
except ImportError as e:
    print(f"Warning: Could not import training_utils: {e}")
    _training_utils_available = False

# Samplers
try:
    from .samplers import (
        InverseScoreSampler,
        StratifiedBatchSampler,
        WeightedRandomSampler,
        create_balanced_sampler,
        analyze_sampling_distribution
    )
    _samplers_available = True
except ImportError as e:
    print(f"Warning: Could not import samplers: {e}")
    _samplers_available = False

# Training configurations
try:
    from .training_config import (
        ModelConfig,
        DataConfig, 
        TrainingConfig,
        LoggingConfig,
        CCMTTrainingConfig,
        get_quick_test_config,
        get_full_training_config,
        get_large_model_config,
        save_config,
        load_config
    )
    _training_config_available = True
except ImportError as e:
    print(f"Warning: Could not import training_config: {e}")
    _training_config_available = False

# Main trainer (depends on training_utils and samplers)
try:
    if _training_utils_available and _samplers_available:
        from .trainer import CCMTTrainer
        _trainer_available = True
    else:
        print("Warning: CCMTTrainer not available due to missing dependencies")
        _trainer_available = False
except ImportError as e:
    print(f"Warning: Could not import CCMTTrainer: {e}")
    _trainer_available = False

__version__ = "1.0.0"
__author__ = "ESL Grading Team"

# Core components (always available)
__all__ = [
    # Main model and components
    "ESLCCMTModel",
    "AudioEncoder",
    "TextEncoder", 
    "EnglishTextEncoder",
    "VietnameseTextEncoder",
    "create_text_encoder",
    "TextProcessor",
    "AsyncTextProcessor",
    "CascadedCrossModalTransformer",
    "Transformer",
    "Attention",
    "ESLCCMTDataset",
    "get_ccmt_collate_fn",
    "MultiModalProjection",
    "TokenSampler",
]

# Add training components if available
if _trainer_available:
    __all__.append("CCMTTrainer")

if _training_utils_available:
    __all__.extend([
        "SoftTargetGenerator",
        "ESLLossFunction", 
        "ClassWeightCalculator",
        "ValidationMetrics",
        "selective_freeze_embedding_layer",
        "get_param_groups",
        "maybe_empty_cache",
        "compute_correlation",
    ])

if _samplers_available:
    __all__.extend([
        "InverseScoreSampler",
        "StratifiedBatchSampler",
        "WeightedRandomSampler",
        "create_balanced_sampler",
        "analyze_sampling_distribution",
    ])

if _training_config_available:
    __all__.extend([
        "ModelConfig",
        "DataConfig",
        "TrainingConfig", 
        "LoggingConfig",
        "CCMTTrainingConfig",
        "get_quick_test_config",
        "get_full_training_config",
        "get_large_model_config",
        "save_config",
        "load_config",
    ])

# Default configurations
DEFAULT_MODEL_CONFIG = {
    # Model architectures
    "audio_model_id": "facebook/wav2vec2-base-960h",
    "english_model_name": "bert-base-uncased", 
    "vietnamese_model_name": "vinai/phobert-base-v2",
    
    # CCMT parameters
    "common_dim": 512,
    "num_tokens_per_modality": 100,
    "ccmt_depth": 6,
    "ccmt_heads": 8,
    "ccmt_mlp_dim": 1024,
    
    # Task parameters
    "num_score_bins": 21,  # 0 to 10 in 0.5 increments
    "dropout": 0.2,
}

DEFAULT_DATASET_CONFIG = {
    # Text parameters
    "max_text_length": 512,
    
    # Audio parameters  
    "num_audio_chunks": 10,
    "chunk_length_sec": 30,
    "sample_rate": 16000,
    
    # Tokenizer names (should match model names)
    "english_tokenizer_name": "bert-base-uncased",
    "vietnamese_tokenizer_name": "vinai/phobert-base-v2",
    "audio_processor_name": "facebook/wav2vec2-base-960h",
}

# Combined config for backward compatibility
DEFAULT_CONFIG = {**DEFAULT_MODEL_CONFIG, **DEFAULT_DATASET_CONFIG}

def create_esl_ccmt_model(config=None, **kwargs):
    """
    Factory function to create ESLCCMTModel with default or custom config
    
    Args:
        config: dictionary with model configuration
        **kwargs: additional arguments to override config
        
    Returns:
        ESLCCMTModel instance
    """
    # Start with default model config
    model_config = DEFAULT_MODEL_CONFIG.copy()
    
    # Update with provided config (filter only model parameters)
    if config is not None:
        # Only use keys that are valid for the model
        valid_keys = set(DEFAULT_MODEL_CONFIG.keys())
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        model_config.update(filtered_config)
    
    # Override with kwargs (filter only model parameters)
    valid_keys = set(DEFAULT_MODEL_CONFIG.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    model_config.update(filtered_kwargs)
    
    return ESLCCMTModel(**model_config)

def create_esl_ccmt_dataset(dataframe, config=None, **kwargs):
    """
    Factory function to create ESLCCMTDataset with default or custom config
    
    Args:
        dataframe: pandas DataFrame with the data
        config: dictionary with dataset configuration  
        **kwargs: additional arguments to override config
        
    Returns:
        ESLCCMTDataset instance
    """
    # Start with default dataset config
    dataset_config = DEFAULT_DATASET_CONFIG.copy()
    
    # Update with provided config
    if config is not None:
        dataset_config.update(config)
    
    # Override with kwargs  
    dataset_config.update(kwargs)
    
    return ESLCCMTDataset(dataframe, **dataset_config)

# Add factory functions to __all__
__all__.extend([
    "create_esl_ccmt_model",
    "create_esl_ccmt_dataset", 
    "DEFAULT_CONFIG",
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_DATASET_CONFIG"
])

# Provide helpful error message if CCMTTrainer is not available
def _get_trainer_error_msg():
    return """
CCMTTrainer is not available. This might be due to missing dependencies.
Please ensure you have installed:
- scipy (for SoftTargetGenerator)
- transformers (for get_cosine_schedule_with_warmup)

Try: pip install scipy transformers

For manual installation of training components, see the training_utils.py and samplers.py modules.
"""

# Create a dummy CCMTTrainer that provides helpful error message
if not _trainer_available:
    class CCMTTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError(_get_trainer_error_msg())
    
    __all__.append("CCMTTrainer")
"""
Configuration management for CCMT
"""

import json
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CCMTConfig:
    """
    Main configuration class for CCMT
    """
    # Model configuration
    model_name: str = "ccmt-base"
    task_type: str = "classification"  # or "regression" 
    num_classes: int = 21
    model_dim: int = 768
    num_patches: int = 300
    depth: int = 6
    heads: int = 8
    mlp_dim: int = 2048
    dropout: float = 0.1
    
    # Data configuration
    csv_path: str = ""
    score_column: str = "vocabulary"
    max_audio_length: float = 30.0
    sample_rate: int = 16000
    max_text_length: int = 512
    
    # Training configuration
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 50
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Model paths
    audio_encoder_name: str = "facebook/wav2vec2-base-960h"
    english_encoder_name: str = "bert-base-uncased"
    vietnamese_encoder_name: str = "vinai/phobert-base"
    translator_name: str = "Helsinki-NLP/opus-mt-en-vi"
    
    # Device and optimization
    device: str = "cuda"  # "auto", "cpu", "cuda"
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging and checkpointing
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    
    # Data processing
    cache_translations: bool = True
    preload_audio: bool = False
    augmentation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like get access"""
        return getattr(self, key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-like bracket access"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in config")
    
    def __setitem__(self, key: str, value: Any):
        """Dictionary-like bracket assignment"""
        setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        """Dictionary-like 'in' operator"""
        return hasattr(self, key)
    
    def keys(self):
        """Dictionary-like keys() method"""
        return self.to_dict().keys()
    
    def values(self):
        """Dictionary-like values() method"""
        return self.to_dict().values()
    
    def items(self):
        """Dictionary-like items() method"""
        return self.to_dict().items()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CCMTConfig':
        """Create config from dictionary"""
        # Filter out keys that are not in the dataclass fields
        import inspect
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.task_type not in ["classification", "regression"]:
            logger.error(f"Invalid task_type: {self.task_type}")
            return False
        
        if self.num_classes <= 0:
            logger.error(f"Invalid num_classes: {self.num_classes}")
            return False
        
        if self.batch_size <= 0:
            logger.error(f"Invalid batch_size: {self.batch_size}")
            return False
        
        if self.learning_rate <= 0:
            logger.error(f"Invalid learning_rate: {self.learning_rate}")
            return False
        
        return True


def load_config(config_path: Union[str, Path]) -> CCMTConfig:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file (.json or .yaml)
    
    Returns:
        CCMTConfig object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found, using default config")
        return CCMTConfig()
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                logger.error(f"Unsupported config file format: {config_path.suffix}")
                return CCMTConfig()
        
        config = CCMTConfig.from_dict(config_dict)
        
        if not config.validate():
            logger.error("Configuration validation failed")
            return CCMTConfig()
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return CCMTConfig()


def save_config(config: CCMTConfig, config_path: Union[str, Path]):
    """
    Save configuration to file
    
    Args:
        config: CCMTConfig object
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        config_dict = config.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(config_dict, f, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                logger.error(f"Unsupported config file format: {config_path.suffix}")
                return
        
        logger.info(f"Saved configuration to {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")


def merge_configs(base_config: CCMTConfig, override_config: Dict[str, Any]) -> CCMTConfig:
    """
    Merge base configuration with override values
    
    Args:
        base_config: Base configuration
        override_config: Dictionary of values to override
    
    Returns:
        Merged configuration
    """
    base_dict = base_config.to_dict()
    base_dict.update(override_config)
    return CCMTConfig.from_dict(base_dict)


def get_default_config() -> CCMTConfig:
    """Get default configuration"""
    return CCMTConfig()


def create_training_config(
    csv_path: str,
    output_dir: str,
    batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 1e-4
) -> CCMTConfig:
    """
    Create configuration for training
    
    Args:
        csv_path: Path to training CSV
        output_dir: Output directory
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        Training configuration
    """
    config = CCMTConfig()
    config.csv_path = csv_path
    config.output_dir = output_dir
    config.batch_size = batch_size
    config.num_epochs = num_epochs
    config.learning_rate = learning_rate
    
    return config


def create_inference_config(
    model_path: str,
    csv_path: str = "",
    batch_size: int = 32
) -> CCMTConfig:
    """
    Create configuration for inference
    
    Args:
        model_path: Path to trained model
        csv_path: Path to data CSV (optional)
        batch_size: Inference batch size
    
    Returns:
        Inference configuration
    """
    config = CCMTConfig()
    config.csv_path = csv_path
    config.batch_size = batch_size
    config.augmentation = False  # No augmentation during inference
    
    return config


def update_config_from_args(config: CCMTConfig, args: Any) -> CCMTConfig:
    """
    Update configuration from command line arguments
    
    Args:
        config: Base configuration
        args: Command line arguments (argparse.Namespace)
    
    Returns:
        Updated configuration
    """
    config_dict = config.to_dict()
    
    # Update config with non-None arguments
    for key, value in vars(args).items():
        if value is not None and key in config_dict:
            config_dict[key] = value
    
    return CCMTConfig.from_dict(config_dict)


def setup_directories(config: CCMTConfig):
    """
    Create necessary directories for training
    
    Args:
        config: Configuration object
    """
    output_dir = Path(config.output_dir)
    
    # Create directories
    directories = [
        output_dir,
        output_dir / "checkpoints",
        output_dir / "logs", 
        output_dir / "configs",
        output_dir / "results"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directories under {output_dir}")


def save_training_config(config: CCMTConfig):
    """
    Save configuration for training run
    
    Args:
        config: Configuration object
    """
    output_dir = Path(config.output_dir)
    config_path = output_dir / "configs" / "training_config.yaml"
    save_config(config, config_path)


def create_config_from_template(template_name: str = "default") -> CCMTConfig:
    """
    Create configuration from predefined template
    
    Args:
        template_name: Template name ("default", "large", "small")
    
    Returns:
        Configuration object
    """
    templates = {
        "default": {
            "model_dim": 768,
            "depth": 6,
            "heads": 8,
            "batch_size": 16,
            "learning_rate": 1e-4
        },
        "large": {
            "model_dim": 1024,
            "depth": 8,
            "heads": 12,
            "batch_size": 8,
            "learning_rate": 5e-5
        },
        "small": {
            "model_dim": 512,
            "depth": 4,
            "heads": 6,
            "batch_size": 32,
            "learning_rate": 2e-4
        }
    }
    
    base_config = CCMTConfig()
    
    if template_name in templates:
        return merge_configs(base_config, templates[template_name])
    else:
        logger.warning(f"Unknown template: {template_name}, using default")
        return base_config


def validate_paths(config: CCMTConfig) -> bool:
    """
    Validate that required paths exist
    
    Args:
        config: Configuration object
    
    Returns:
        True if all paths are valid
    """
    if config.csv_path and not Path(config.csv_path).exists():
        logger.error(f"CSV path does not exist: {config.csv_path}")
        return False
    
    return True


def get_device_config(config: CCMTConfig) -> str:
    """
    Get device configuration
    
    Args:
        config: Configuration object
    
    Returns:
        Device string ("cpu", "cuda", etc.)
    """
    if config.device == "cuda":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return config.device


def print_config(config: CCMTConfig):
    """
    Print configuration in a readable format
    
    Args:
        config: Configuration object
    """
    print("CCMT Configuration:")
    print("=" * 50)
    
    config_dict = config.to_dict()
    
    for key, value in config_dict.items():
        print(f"{key:25}: {value}")
    
    print("=" * 50)


# Configuration validation functions
def validate_model_config(config: CCMTConfig) -> List[str]:
    """Validate model configuration parameters"""
    errors = []
    
    if config.model_dim <= 0 or config.model_dim % config.heads != 0:
        errors.append(f"model_dim ({config.model_dim}) must be positive and divisible by heads ({config.heads})")
    
    if config.depth <= 0:
        errors.append(f"depth must be positive, got {config.depth}")
    
    if config.heads <= 0:
        errors.append(f"heads must be positive, got {config.heads}")
    
    return errors


def validate_training_config(config: CCMTConfig) -> List[str]:
    """Validate training configuration parameters"""
    errors = []
    
    if config.batch_size <= 0:
        errors.append(f"batch_size must be positive, got {config.batch_size}")
    
    if config.learning_rate <= 0:
        errors.append(f"learning_rate must be positive, got {config.learning_rate}")
    
    if config.num_epochs <= 0:
        errors.append(f"num_epochs must be positive, got {config.num_epochs}")
    
    return errors
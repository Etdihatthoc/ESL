"""
Training configuration for CCMT ESL grading
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import os


@dataclass
class ModelConfig:
    """Configuration for CCMT model architecture"""
    # Model architectures
    audio_model_id: str = "facebook/wav2vec2-base-960h"
    english_model_name: str = "bert-base-uncased"
    vietnamese_model_name: str = "vinai/phobert-base-v2"
    
    # CCMT parameters
    common_dim: int = 512
    num_tokens_per_modality: int = 100
    ccmt_depth: int = 6
    ccmt_heads: int = 8
    ccmt_mlp_dim: int = 1024
    
    # Task parameters
    num_score_bins: int = 21  # 0 to 10 in 0.5 increments
    dropout: float = 0.2


@dataclass
class DataConfig:
    """Configuration for data processing"""
    # Text parameters
    max_text_length: int = 512
    
    # Audio parameters
    num_audio_chunks: int = 10
    chunk_length_sec: int = 30
    sample_rate: int = 16000
    
    # Data filtering
    remove_low_content: bool = True
    filter_scores: bool = True
    
    # Audio augmentation (only during training)
    use_audio_augmentation: bool = True
    noise_prob: float = 0.7
    speed_prob: float = 0.7
    pitch_prob: float = 0.7
    
    # Text processing
    asr_model_name: str = "openai/whisper-base"
    translation_model_name: str = "Helsinki-NLP/opus-mt-en-vi"


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # Basic training parameters
    batch_size: int = 1
    epochs: int = 20
    device: str = "cuda"
    
    # Learning rates for different parameter groups
    learning_rates: Dict[str, float] = None
    
    # Optimizer and scheduler
    optimizer_name: str = "adamw"
    weight_decay: float = 1e-4
    scheduler_name: str = "cosine"
    warmup_steps: int = 500
    
    # Loss function parameters
    lambda_kl: float = 0.9
    lambda_mse: float = 0.1
    soft_target_std: float = 0.3
    
    # Sampling and class balancing
    sampling_strategy: str = "inverse_score"  # "inverse_score", "stratified", "weighted_random", "random"
    sampling_alpha: float = 0.5
    class_weight_beta: float = 0.99
    
    # Regularization
    use_selective_freezing: bool = True
    freezing_stopword_ratio: float = 0.05
    
    # Validation and early stopping
    patience: int = 5
    min_delta: float = 1e-4
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = {
                'base': 1e-5,
                'encoder': 1e-6,
                'scale': 1e-3
            }


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring"""
    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    
    # Wandb integration
    use_wandb: bool = False
    wandb_project: str = "esl-ccmt-grading"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # Model checkpointing
    save_best_only: bool = True
    save_every_n_epochs: int = 5
    
    # Testing and evaluation
    test_output_dir: str = "./results"


@dataclass
class CCMTTrainingConfig:
    """Complete configuration for CCMT training"""
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    logging: LoggingConfig = None
    
    # Data paths
    train_path: str = ""
    val_path: str = ""
    test_path: str = ""
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
    
    def validate(self):
        """Validate configuration"""
        assert os.path.exists(self.train_path), f"Train path does not exist: {self.train_path}"
        assert os.path.exists(self.val_path), f"Val path does not exist: {self.val_path}"
        assert os.path.exists(self.test_path), f"Test path does not exist: {self.test_path}"
        
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.training.epochs > 0, "Epochs must be positive"
        assert 0 <= self.training.lambda_kl <= 1, "lambda_kl must be in [0, 1]"
        assert 0 <= self.training.lambda_mse <= 1, "lambda_mse must be in [0, 1]"
        assert abs(self.training.lambda_kl + self.training.lambda_mse - 1.0) < 1e-6, "lambda_kl + lambda_mse must equal 1.0"
        
        assert self.model.common_dim > 0, "common_dim must be positive"
        assert self.model.num_tokens_per_modality > 0, "num_tokens_per_modality must be positive"
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'logging': self.logging.__dict__,
            'paths': {
                'train_path': self.train_path,
                'val_path': self.val_path,
                'test_path': self.test_path
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary"""
        config = cls()
        
        if 'model' in config_dict:
            for k, v in config_dict['model'].items():
                setattr(config.model, k, v)
        
        if 'data' in config_dict:
            for k, v in config_dict['data'].items():
                setattr(config.data, k, v)
        
        if 'training' in config_dict:
            for k, v in config_dict['training'].items():
                setattr(config.training, k, v)
        
        if 'logging' in config_dict:
            for k, v in config_dict['logging'].items():
                setattr(config.logging, k, v)
        
        if 'paths' in config_dict:
            config.train_path = config_dict['paths'].get('train_path', '')
            config.val_path = config_dict['paths'].get('val_path', '')
            config.test_path = config_dict['paths'].get('test_path', '')
        
        return config


# Predefined configurations for different scenarios

def get_quick_test_config() -> CCMTTrainingConfig:
    """Configuration for quick testing"""
    config = CCMTTrainingConfig()
    
    # Smaller model for faster training
    config.model.common_dim = 256
    config.model.ccmt_depth = 4
    config.model.num_tokens_per_modality = 50
    
    # Fewer epochs and larger batch size
    config.training.batch_size = 1
    config.training.epochs = 5
    config.training.warmup_steps = 100
    
    # Less data processing
    config.data.num_audio_chunks = 5
    config.data.chunk_length_sec = 15
    config.data.use_audio_augmentation = False
    
    return config


def get_full_training_config() -> CCMTTrainingConfig:
    """Configuration for full training"""
    config = CCMTTrainingConfig()
    
    # Full model size
    config.model.common_dim = 512
    config.model.ccmt_depth = 6
    config.model.num_tokens_per_modality = 100
    
    # Full training
    config.training.batch_size = 1
    config.training.epochs = 30
    config.training.warmup_steps = 500
    
    # Full data processing
    config.data.num_audio_chunks = 10
    config.data.chunk_length_sec = 30
    config.data.use_audio_augmentation = True
    
    # Enable wandb logging
    config.logging.use_wandb = True
    
    return config


def get_large_model_config() -> CCMTTrainingConfig:
    """Configuration for large model training"""
    config = CCMTTrainingConfig()
    
    # Large model
    config.model.common_dim = 768
    config.model.ccmt_depth = 8
    config.model.num_tokens_per_modality = 128
    config.model.ccmt_heads = 12
    config.model.ccmt_mlp_dim = 2048
    
    # Adjusted training for larger model
    config.training.batch_size = 1
    config.training.epochs = 25
    config.training.learning_rates = {
        'base': 5e-6,
        'encoder': 5e-7,
        'scale': 5e-4
    }
    
    return config


def save_config(config: CCMTTrainingConfig, path: str):
    """Save configuration to file"""
    import json
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def load_config(path: str) -> CCMTTrainingConfig:
    """Load configuration from file"""
    import json
    
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    return CCMTTrainingConfig.from_dict(config_dict)
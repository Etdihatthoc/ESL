#!/usr/bin/env python3
"""
Main training script for CCMT ESL grading model

Usage:
    python train_ccmt.py --config config.json
    python train_ccmt.py --train_path data/train.csv --val_path data/val.csv --test_path data/test.csv
    python train_ccmt.py --quick_test  # For quick testing
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models import (
    ESLCCMTModel,
    TextProcessor,
    create_esl_ccmt_model
)
from models.trainer import CCMTTrainer
from models.training_config import (
    CCMTTrainingConfig,
    get_quick_test_config,
    get_full_training_config,
    get_large_model_config,
    save_config,
    load_config
)


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def setup_wandb(config: CCMTTrainingConfig, logger: logging.Logger):
    """Setup Weights & Biases logging"""
    if not config.logging.use_wandb:
        return
    
    try:
        import wandb
        
        # Initialize wandb
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=config.logging.wandb_run_name,
            config=config.to_dict(),
            reinit=True
        )
        
        logger.info("Weights & Biases initialized successfully")
        
    except ImportError:
        logger.warning("wandb not installed. Install with: pip install wandb")
        config.logging.use_wandb = False
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        config.logging.use_wandb = False


def create_model_from_config(config: CCMTTrainingConfig, logger: logging.Logger) -> ESLCCMTModel:
    """Create CCMT model from configuration"""
    logger.info("Creating CCMT model...")
    
    model = create_esl_ccmt_model(
        audio_model_id=config.model.audio_model_id,
        english_model_name=config.model.english_model_name,
        vietnamese_model_name=config.model.vietnamese_model_name,
        common_dim=config.model.common_dim,
        num_tokens_per_modality=config.model.num_tokens_per_modality,
        ccmt_depth=config.model.ccmt_depth,
        ccmt_heads=config.model.ccmt_heads,
        ccmt_mlp_dim=config.model.ccmt_mlp_dim,
        num_score_bins=config.model.num_score_bins,
        dropout=config.model.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_text_processor_from_config(config: CCMTTrainingConfig, logger: logging.Logger) -> TextProcessor:
    """Create text processor from configuration"""
    logger.info("Creating text processor...")
    
    try:
        text_processor = TextProcessor(
            asr_model_name=config.data.asr_model_name,
            translation_model_name=config.data.translation_model_name,
            device=config.training.device
        )
        logger.info("Text processor created successfully")
        return text_processor
    except Exception as e:
        logger.warning(f"Failed to create text processor: {e}")
        logger.warning("Training will proceed without text processing")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train CCMT ESL grading model")
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--quick_test", action="store_true", help="Use quick test configuration")
    parser.add_argument("--large_model", action="store_true", help="Use large model configuration")
    
    # Data paths (required if not using config file)
    parser.add_argument("--train_path", type=str, help="Path to training CSV file")
    parser.add_argument("--val_path", type=str, help="Path to validation CSV file")
    parser.add_argument("--test_path", type=str, help="Path to test CSV file")
    
    # Training parameters (override config)
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Base learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    # Logging parameters
    parser.add_argument("--log_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Model save directory")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="esl-ccmt-grading", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, help="WandB run name")
    
    # Other options
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--test_only", action="store_true", help="Only run testing")
    parser.add_argument("--save_config", type=str, help="Save configuration to this path and exit")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    elif args.quick_test:
        config = get_quick_test_config()
        print("Using quick test configuration")
    elif args.large_model:
        config = get_large_model_config()
        print("Using large model configuration")
    else:
        config = get_full_training_config()
        print("Using default full training configuration")
    
    # Override configuration with command line arguments
    if args.train_path:
        config.train_path = args.train_path
    if args.val_path:
        config.val_path = args.val_path
    if args.test_path:
        config.test_path = args.test_path
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rates['base'] = args.learning_rate
    if args.device:
        config.training.device = args.device
    if args.log_dir:
        config.logging.log_dir = args.log_dir
    if args.save_dir:
        config.logging.save_dir = args.save_dir
    if args.wandb:
        config.logging.use_wandb = True
    if args.wandb_project:
        config.logging.wandb_project = args.wandb_project
    if args.wandb_run_name:
        config.logging.wandb_run_name = args.wandb_run_name
    
    # Save configuration if requested
    if args.save_config:
        save_config(config, args.save_config)
        print(f"Configuration saved to {args.save_config}")
        return
    
    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return 1
    
    # Setup logging
    logger = setup_logging(config.logging.log_dir, config.logging.log_level)
    logger.info("Starting CCMT training")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Setup wandb
    #setup_wandb(config, logger)
    
    # Check CUDA availability
    if config.training.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config.training.device = "cpu"
    
    logger.info(f"Using device: {config.training.device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Create components
        model = create_model_from_config(config, logger)
        text_processor = create_text_processor_from_config(config, logger)
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = CCMTTrainer(
            model=model,
            train_path=config.train_path,
            val_path=config.val_path,
            test_path=config.test_path,
            text_processor=text_processor,
            # Training parameters
            batch_size=config.training.batch_size,
            epochs=config.training.epochs,
            learning_rates=config.training.learning_rates,
            optimizer_name=config.training.optimizer_name,
            scheduler_name=config.training.scheduler_name,
            warmup_steps=config.training.warmup_steps,
            # Loss parameters
            lambda_kl=config.training.lambda_kl,
            lambda_mse=config.training.lambda_mse,
            soft_target_std=config.training.soft_target_std,
            # Sampling and weighting
            sampling_strategy=config.training.sampling_strategy,
            sampling_alpha=config.training.sampling_alpha,
            class_weight_beta=config.training.class_weight_beta,
            # Data processing
            remove_low_content=config.data.remove_low_content,
            filter_scores=config.data.filter_scores,
            # Other parameters
            device=config.training.device,
            save_dir=config.logging.save_dir,
            logger=logger,
            use_wandb=config.logging.use_wandb
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=config.training.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Run training or testing
        if args.test_only:
            logger.info("Running testing only...")
            test_results = trainer.test()
            logger.info(f"Test results: {test_results}")
        else:
            # Train the model
            logger.info("Starting training...")
            trainer.train()
            
            # Test the model
            logger.info("Running final testing...")
            test_results = trainer.test()
            logger.info(f"Final test results: {test_results}")
        
        logger.info("Training completed successfully!")
        
        # Finish wandb run
        if config.logging.use_wandb:
            try:
                import wandb
                wandb.finish()
            except:
                pass
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Finish wandb run with failure
        if config.logging.use_wandb:
            try:
                import wandb
                wandb.finish(exit_code=1)
            except:
                pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
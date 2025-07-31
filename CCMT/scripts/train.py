#!/usr/bin/env python3
"""
Main training script for CCMT English Speaking Scoring
Usage: python scripts/train.py --config configs/base_config.yaml --csv_path data/scores.csv
"""
import multiprocessing
import logging
from pathlib import Path
import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path and prioritize local modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # Insert at beginning to prioritize local modules

import torch
from models import create_ccmt_model, create_audio_encoder, create_english_encoder, create_vietnamese_encoder, create_en_vi_translator
from data import create_dataset_splits, create_dataloaders
from training import create_trainer
from utils import load_config, merge_configs, setup_directories, get_device_config, print_config


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train CCMT for English Speaking Scoring")
    
    # Configuration
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to base configuration file")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--training_config", type=str, default="configs/training_config.yaml",
                       help="Path to training configuration file")
    
    # Data
    parser.add_argument("--csv_path", type=str, required=True,
                       help="Path to CSV file with audio paths and scores")
    parser.add_argument("--score_column", type=str, default="vocabulary",
                       help="Column to use as target score")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate")
    
    # Model
    parser.add_argument("--task_type", type=str, choices=["classification", "regression"], default=None,
                       help="Task type")
    parser.add_argument("--model_size", type=str, choices=["small", "base", "large"], default=None,
                       help="Model size")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="ccmt_training",
                       help="Experiment name for output folder")
    
    # Device
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cpu, cuda, auto)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    return parser.parse_args()


def load_configurations(args):
    """Load and merge configuration files"""
    from utils import CCMTConfig
    
    # Load individual configs (these return CCMTConfig objects or dicts)
    base_config_dict = {}
    if Path(args.config).exists():
        base_config_obj = load_config(args.config)
        base_config_dict = base_config_obj.to_dict() if hasattr(base_config_obj, 'to_dict') else base_config_obj
    
    model_config_dict = {}
    if Path(args.model_config).exists():
        model_config_obj = load_config(args.model_config)
        model_config_dict = model_config_obj.to_dict() if hasattr(model_config_obj, 'to_dict') else model_config_obj
    
    training_config_dict = {}
    if Path(args.training_config).exists():
        training_config_obj = load_config(args.training_config)
        training_config_dict = training_config_obj.to_dict() if hasattr(training_config_obj, 'to_dict') else training_config_obj
    
    # Merge all config dictionaries
    merged_config_dict = {**base_config_dict, **model_config_dict, **training_config_dict}
    
    # Override with command line arguments
    if args.csv_path:
        merged_config_dict['csv_path'] = args.csv_path
    if args.score_column:
        merged_config_dict['score_column'] = args.score_column
    if args.batch_size:
        merged_config_dict['batch_size'] = args.batch_size
    if args.num_epochs:
        merged_config_dict['num_epochs'] = args.num_epochs
    if args.learning_rate:
        merged_config_dict['learning_rate'] = args.learning_rate
    if args.task_type:
        merged_config_dict['task_type'] = args.task_type
    if args.model_size:
        merged_config_dict['model_size'] = args.model_size
    if args.output_dir:
        merged_config_dict['output_dir'] = args.output_dir
    if args.device:
        merged_config_dict['device'] = args.device
    
    # Set experiment-specific output directory
    if args.experiment_name:
        merged_config_dict['output_dir'] = os.path.join(
            merged_config_dict.get('output_dir', './outputs'), 
            args.experiment_name
        )
    
    # Create CCMTConfig object from merged dictionary
    config = CCMTConfig.from_dict(merged_config_dict)
    
    return config


def create_models_and_processors(config):
    """Create CCMT model and preprocessing components"""
    device = get_device_config(config)
    
    # Create encoders
    logging.info("Creating encoders...")
    audio_encoder = create_audio_encoder(
        model_size=config.get('model_size', 'base'),
        max_tokens=config.get('max_tokens_per_modality', 100)
    )
    
    english_encoder = create_english_encoder(
        model_size=config.get('model_size', 'base'), 
        max_tokens=config.get('max_tokens_per_modality', 100)
    )
    
    vietnamese_encoder = create_vietnamese_encoder(
        model_type="phobert",
        max_tokens=config.get('max_tokens_per_modality', 100)
    )
    
    # Create translator
    logging.info("Creating translator...")
    translator = create_en_vi_translator(
        model_type="opus",
        cache_size=config.get('translation', {}).get('cache_size', 1000)
    )
    
    # Create CCMT model
    logging.info("Creating CCMT model...")
    model = create_ccmt_model(
        task_type=config.get('task_type', 'classification'),
        num_classes=config.get('num_classes', 21),
        model_size=config.get('model_size', 'base')
    )
    
    return model, audio_encoder, english_encoder, vietnamese_encoder, translator


def create_datasets_and_loaders(config, audio_encoder, english_encoder, vietnamese_encoder, translator):
    """Create datasets and data loaders"""
    logging.info("Creating datasets...")
    
    # Create dataset splits
    train_dataset, val_dataset, test_dataset = create_dataset_splits(
        csv_path=config.csv_path,
        test_size=config.get('data', {}).get('test_size', 0.2),
        val_size=config.get('data', {}).get('val_size', 0.1),
        random_state=config.get('seed', 42),
        task_type=config.get('task_type', 'classification'),
        score_column=config.get('score_column', 'vocabulary'),
        translator=translator,
        cache_translations=config.get('processing', {}).get('cache_translations', True)
    )
    
    logging.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 4),
        audio_encoder=audio_encoder,
        english_encoder=english_encoder,
        vietnamese_encoder=vietnamese_encoder,
        device=get_device_config(config),
        pin_memory=config.get('pin_memory', True)
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load configuration
    config = load_configurations(args)
    
    # Setup directories
    setup_directories(config)
    
    # Setup logging
    log_file = os.path.join(config.output_dir, "logs", "training.log")
    setup_logging(
        log_level=config.get('logging', {}).get('level', 'INFO'),
        log_file=log_file
    )
    
    logging.info("Starting CCMT training...")
    logging.info(f"Output directory: {config.output_dir}")
    
    # Print configuration
    print_config(config)
    
    # Create models and processors
    model, audio_encoder, english_encoder, vietnamese_encoder, translator = create_models_and_processors(config)
    
    # Create datasets and loaders
    train_loader, val_loader, test_loader = create_datasets_and_loaders(
        config, audio_encoder, english_encoder, vietnamese_encoder, translator
    )
    
    # Create trainer
    logging.info("Creating trainer...")
    trainer = create_trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        config=config.to_dict()  # Convert CCMTConfig to dict for trainer
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_model(args.resume, load_optimizer=True)
    
    # Train model
    logging.info("Starting training...")
    history = trainer.train(
        num_epochs=config.get('num_epochs', 50),
        save_dir=config.output_dir
    )
    
    # Final evaluation on test set
    if test_loader:
        logging.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        logging.info("Test Results:")
        for metric, value in test_metrics.items():
            logging.info(f"  {metric}: {value:.4f}")
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model.pt")
    trainer.save_model(final_model_path, include_optimizer=False)
    logging.info(f"Final model saved to: {final_model_path}")
    
    logging.info("Training completed successfully!")
    return history


# Set start method to spawn
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    try:
        history = main()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)
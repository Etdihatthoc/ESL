#!/usr/bin/env python3
"""
Example training script for CCMT ESL grading model
Demonstrates how to use the complete training pipeline
"""

import os
import pandas as pd
import logging
from datetime import datetime

from models import (
    ESLCCMTModel,
    CCMTTrainer,
    TextProcessor,
    CCMTTrainingConfig,
    get_quick_test_config,
    save_config,
    create_esl_ccmt_model
)


def create_sample_data():
    """Create sample data for demonstration"""
    print("Creating sample data...")
    
    # Create sample training data
    train_data = {
        'text': [
            '"Hello, my name is John and I am studying English for three years."',
            '"I like to read books and watch movies in English to improve my skills."',
            '"Yesterday I went to the library and borrowed some English novels."',
            '"My favorite subject in school is mathematics because it is very interesting."',
            '"I want to travel to America next year to practice my English speaking."',
            '"The weather today is very nice and sunny, perfect for walking in the park."',
            '"I have been learning English since I was in elementary school."',
            '"My teacher always encourages us to speak English in class every day."',
        ] * 10,  # Repeat to have more samples
        'grammar': [6.5, 7.0, 5.5, 8.0, 6.0, 7.5, 8.5, 7.0] * 10,
        'question_type': [1, 2, 3, 1, 2, 3, 1, 2] * 10,
    }
    
    # Create sample validation and test data (smaller)
    val_test_data = {
        'text': [
            '"I am learning English to get better job opportunities in the future."',
            '"Reading English newspapers helps me understand current events better."',
            '"I practice speaking English with my friends every weekend."',
            '"Grammar is the most difficult part of learning English for me."',
        ] * 5,
        'grammar': [6.0, 7.5, 5.0, 8.0] * 5,
        'question_type': [1, 2, 3, 1] * 5,
    }
    
    # Create DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_test_data)
    test_df = pd.DataFrame(val_test_data)
    
    # Create data directory
    os.makedirs("./sample_data", exist_ok=True)
    
    # Save to CSV files
    train_path = "./sample_data/train.csv"
    val_path = "./sample_data/val.csv"
    test_path = "./sample_data/test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Sample data created:")
    print(f"  Train: {len(train_df)} samples -> {train_path}")
    print(f"  Val: {len(val_df)} samples -> {val_path}")
    print(f"  Test: {len(test_df)} samples -> {test_path}")
    
    return train_path, val_path, test_path


def setup_logging():
    """Setup basic logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f"{log_dir}/example_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def example_quick_training():
    """Example of quick training run"""
    print("=" * 60)
    print("CCMT ESL Grading - Quick Training Example")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    
    # Create sample data
    train_path, val_path, test_path = create_sample_data()
    
    # Get quick test configuration
    config = get_quick_test_config()
    config.train_path = train_path
    config.val_path = val_path
    config.test_path = test_path
    
    # Override some settings for this example
    config.training.epochs = 3
    config.training.batch_size = 1
    config.training.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    config.data.remove_low_content = False  # Don't filter sample data
    config.data.filter_scores = False
    
    print(f"\nUsing device: {config.training.device}")
    print(f"Training for {config.training.epochs} epochs with batch size {config.training.batch_size}")
    
    # Save configuration
    config_path = "./sample_data/training_config.json"
    save_config(config, config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Create text processor (optional - can be None for demo)
    try:
        print("\nInitializing text processor...")
        text_processor = TextProcessor(
            asr_model_name="openai/whisper-base",
            translation_model_name="Helsinki-NLP/opus-mt-en-vi",
            device="cpu"  # Use CPU to avoid multiprocessing CUDA issues
        )
        print("Text processor initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize text processor: {e}")
        print("Continuing without text processor...")
        text_processor = None
    
    # Create model
    print("\nCreating CCMT model...")
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
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Create trainer
    print("\nCreating trainer...")
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
        use_wandb=False  # Disable for example
    )
    
    print(f"Trainer created successfully")
    print(f"Training dataset size: {len(trainer.train_dataset)}")
    print(f"Validation dataset size: {len(trainer.val_dataset)}")
    print(f"Test dataset size: {len(trainer.test_dataset)}")
    
    # Train the model
    print("\n" + "="*40)
    print("Starting Training")
    print("="*40)
    
    try:
        trainer.train()
        print("\nTraining completed successfully!")
        
        # Test the model
        print("\n" + "="*40)
        print("Running Final Test")
        print("="*40)
        
        test_results = trainer.test()
        
        print(f"\nFinal Test Results:")
        print(f"  MSE: {test_results['test_mse']:.4f}")
        print(f"  MAE: {test_results['test_mae']:.4f}")
        print(f"  Correlation: {test_results['test_correlation']:.4f}")
        
        # Show some predictions
        results_df = test_results['results_df']
        print(f"\nSample predictions:")
        print(results_df.head(10).round(3))
        
        print(f"\nAll results saved to checkpoints directory")
        print(f"Check the following files:")
        print(f"  - Model: ./checkpoints/best_model.pth")
        print(f"  - Predictions: ./checkpoints/test_predictions.csv")
        print(f"  - Logs: ./logs/")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
    
    return True


def example_config_only():
    """Example of creating and saving different configurations"""
    print("=" * 60)
    print("Configuration Examples")
    print("=" * 60)
    
    # Create different configurations
    configs = {
        "quick_test": get_quick_test_config(),
        "full_training": get_full_training_config(),
        # "large_model": get_large_model_config(),  # Might be too large for demo
    }
    
    # Create configs directory
    os.makedirs("./configs", exist_ok=True)
    
    for name, config in configs.items():
        # Set dummy paths
        config.train_path = "./data/train.csv"
        config.val_path = "./data/val.csv" 
        config.test_path = "./data/test.csv"
        
        # Save configuration
        config_path = f"./configs/{name}_config.json"
        save_config(config, config_path)
        print(f"Saved {name} configuration to: {config_path}")
    
    print("\nYou can use these configurations with the main training script:")
    print("  python models/train_ccmt.py --config ./configs/quick_test_config.json")
    print("  python models/train_ccmt.py --config ./configs/full_training_config.json")


def main():
    """Main function to run examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CCMT Training Examples")
    parser.add_argument("--config_only", action="store_true", 
                       help="Only create configuration examples")
    parser.add_argument("--no_training", action="store_true",
                       help="Skip actual training (setup only)")
    
    args = parser.parse_args()
    
    if args.config_only:
        example_config_only()
    else:
        success = example_quick_training()
        if not success:
            return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
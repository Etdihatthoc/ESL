"""
train_classifier.py - Main training script for ESL binary classifier
"""

import argparse
import os
import torch
from transformers import AutoTokenizer
import pandas as pd

# Import our modules
from data import create_data_loaders
from model import ESLBinaryClassifier
from trainer import ESLBinaryTrainer, analyze_score_distribution
from utils import (
    set_seed, setup_logging, count_parameters, get_optimizer, 
    get_scheduler, calculate_dataset_stats, create_model_config,
    create_training_config, save_experiment_config, check_device_memory
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ESL Binary Classifier')
    
    # Data paths
    parser.add_argument('--train_path', type=str, default='./data/Full/Full_train.csv',
                       help='Path to training CSV file')
    parser.add_argument('--val_path', type=str, default='./data/Full/val_pro.csv',
                       help='Path to validation CSV file')
    parser.add_argument('--test_path', type=str, default='./data/Full/test_pro.csv',
                       help='Path to test CSV file')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default='Alibaba-NLP/gte-Qwen2-1.5B-instruct',
                       help='Pretrained model name')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for classifier')
    parser.add_argument('--pooling_dropout', type=float, default=0.3,
                       help='Dropout rate for pooling layer')
    parser.add_argument('--classifier_dropout', type=float, default=0.5,
                       help='Dropout rate for classifier')
    parser.add_argument('--avg_last_k', type=int, default=4,
                       help='Number of last layers to average')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio for scheduler')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                       choices=['cosine', 'linear', 'none'],
                       help='Type of learning rate scheduler')
    parser.add_argument('--optimizer_type', type=str, default='adamw',
                       choices=['adamw', 'adam'],
                       help='Type of optimizer')
    
    # Loss function
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use focal loss instead of weighted cross entropy')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                       help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss')
    
    # Data loading
    parser.add_argument('--use_balanced_sampling', action='store_true', default=True,
                       help='Use balanced sampling for training')
    parser.add_argument('--num_workers', type=int, default=40,
                       help='Number of data loading workers')
    
    # Output paths
    parser.add_argument('--output_dir', type=str, default='./results/binary_classifier',
                       help='Output directory for results')
    parser.add_argument('--model_save_path', type=str, default='./models/esl_binary_classifier_Qwen2.pth',
                       help='Path to save the trained model')
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--max_length', type=int, default=8192,
                       help='Maximum sequence length')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(
        log_dir=os.path.join(args.output_dir, 'logs'),
        experiment_name='esl_binary_classifier_Qwen2'
    )
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    check_device_memory()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        use_balanced_sampling=args.use_balanced_sampling
    )
    
    # Calculate dataset statistics
    calculate_dataset_stats(train_loader, val_loader, test_loader)
    
    # Create model
    logger.info("Creating model...")
    model_config = create_model_config(args)
    model = ESLBinaryClassifier(**model_config)
    #model = ESLBinaryClassifier.load('./models/esl_binary_classifier_3.5_7.5.pth')
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model created with {trainable_params:,} trainable parameters")
    
    # Create optimizer and scheduler
    num_training_steps = len(train_loader) * args.epochs
    optimizer = get_optimizer(
        model, 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer_type
    )
    
    scheduler = None
    if args.scheduler_type != 'none':
        scheduler = get_scheduler(
            optimizer,
            num_training_steps=num_training_steps,
            warmup_ratio=args.warmup_ratio,
            scheduler_type=args.scheduler_type
        )
    
    # Create training configuration
    training_config = create_training_config(args, num_training_steps)
    
    # Save experiment configuration
    experiment_config = {
        **model_config,
        **training_config,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'device': device,
        'seed': args.seed
    }
    
    config_save_path = os.path.join(args.output_dir, 'experiment_config.json')
    save_experiment_config(experiment_config, config_save_path)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = ESLBinaryTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        logger=logger
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the trained model
    logger.info("Saving trained model...")
    trainer.save_model(args.model_save_path)
    
    # Test the model
    logger.info("Testing the model...")
    test_results_path = os.path.join(args.output_dir, 'test_results.csv')
    test_metrics = trainer.test(output_path=test_results_path)
    
    # Analyze results
    logger.info("Analyzing results...")
    analyze_score_distribution(test_metrics['results_df'])
    
    # Final summary
    final_summary = f"""
    =============== TRAINING COMPLETED ===============
    Best Validation F1: {trainer.best_val_f1:.4f}
    Best Validation Accuracy: {trainer.best_val_acc:.4f}
    Test Accuracy: {test_metrics['accuracy']:.4f}
    Test F1-Score: {test_metrics['f1_score']:.4f}
    Test Precision: {test_metrics['precision']:.4f}
    Test Recall: {test_metrics['recall']:.4f}
    
    Model saved to: {args.model_save_path}
    Results saved to: {test_results_path}
    Config saved to: {config_save_path}
    ================================================
    """
    
    print(final_summary)
    logger.info(final_summary.replace('\n', ' '))
    
    return test_metrics


def inference_example():
    """
    Example function showing how to use the trained model for inference
    """
    # Load the trained model
    model_path = './models/esl_binary_classifier_Qwen2.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first by running the main() function")
        return
    
    model = ESLBinaryClassifier.load(model_path, device=device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-1.5B-instruct')
    
    # Example texts for classification
    example_texts = [
        "The following is a spoken English response by a non-native speaker. Classify the proficiency level based on the transcript below: [Question Type: Social Interaction: Answer several questions about familiar topics] I like to eat pizza and watch movies with my friends on weekend.",
        "The following is a spoken English response by a non-native speaker. Classify the proficiency level based on the transcript below: [Question Type: Topic Development: Present a given topic with supporting ideas and answer follow-up questions] In my opinion, environmental conservation represents one of the most critical challenges of our contemporary era. The multifaceted nature of this issue necessitates comprehensive approaches that integrate governmental policies, corporate responsibility, and individual behavioral modifications."
    ]
    
    print("=== INFERENCE EXAMPLE ===")
    
    with torch.no_grad():
        for i, text in enumerate(example_texts):
            # Tokenize
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors='pt'
            ).to(device)
            
            # Predict
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            
            predicted_group = outputs['predictions'].item()
            probabilities = outputs['probabilities'].cpu().numpy()[0]
            
            group_name = "Group 0 (3.5-6.5)" if predicted_group == 0 else "Group 1 (7.0-10.0)"
            
            print(f"\nExample {i+1}:")
            print(f"Text: {text[:100]}...")
            print(f"Predicted Group: {group_name}")
            print(f"Confidence: {probabilities[predicted_group]:.3f}")
            print(f"Probabilities: Group 0: {probabilities[0]:.3f}, Group 1: {probabilities[1]:.3f}")


if __name__ == "__main__":
    # Run training
    test_metrics = main()
    
    # Optionally run inference example
    print("\n" + "="*50)
    print("Would you like to see an inference example? (y/n)")
    user_input = input().strip().lower()
    if user_input in ['y', 'yes']:
        inference_example()
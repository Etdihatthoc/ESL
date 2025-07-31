#!/usr/bin/env python3
"""
Evaluation script for CCMT English Speaking Scoring
Usage: python scripts/evaluate.py --model_path outputs/final_model.pt --csv_path data/test_scores.csv
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
import pandas as pd

# Add project root to path and prioritize local modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # Insert at beginning to prioritize local modules

import torch
import numpy as np
from models import create_ccmt_model, create_audio_encoder, create_english_encoder, create_vietnamese_encoder, create_en_vi_translator
from data import SpeakingScoringDataset, create_dataloaders
from training import evaluate_model
from training.metrics import create_metrics_summary, ScoringMetrics
from utils import load_config, get_device_config


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate CCMT for English Speaking Scoring")
    
    # Model
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to base configuration file")
    
    # Data
    parser.add_argument("--csv_path", type=str, required=True,
                       help="Path to CSV file with test data")
    parser.add_argument("--score_column", type=str, default="vocabulary",
                       help="Column to use as target score")
    
    # Evaluation
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Evaluation batch size")
    parser.add_argument("--task_type", type=str, choices=["classification", "regression"], default="classification",
                       help="Task type")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save individual predictions to CSV")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cpu, cuda, auto)")
    
    return parser.parse_args()


def load_model_and_config(model_path: str, config_path: str, device: str):
    """Load trained model and configuration"""
    # Load checkpoint
    logging.info(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load configuration
    if Path(config_path).exists():
        config_obj = load_config(config_path)
        config = config_obj.to_dict() if hasattr(config_obj, 'to_dict') else config_obj
    else:
        config = {}
    
    # Get model config from checkpoint if available
    model_config = checkpoint.get('model_config', {})
    task_type = model_config.get('task_type', config.get('task_type', 'classification'))
    num_classes = model_config.get('num_classes', config.get('num_classes', 21))
    
    # Create model
    model = create_ccmt_model(
        task_type=task_type,
        num_classes=num_classes,
        model_size=config.get('model_size', 'base')
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logging.info(f"Model loaded successfully. Task: {task_type}, Classes: {num_classes}")
    
    return model, config, task_type, num_classes


def create_evaluation_components(config, device):
    """Create encoders and translator for evaluation"""
    # Create encoders
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
    translator = create_en_vi_translator(
        model_type="opus",
        cache_size=config.get('translation', {}).get('cache_size', 1000)
    )
    
    return audio_encoder, english_encoder, vietnamese_encoder, translator


def create_test_dataset_and_loader(csv_path: str, score_column: str, task_type: str, 
                                  audio_encoder, english_encoder, vietnamese_encoder, 
                                  translator, batch_size: int, device: str):
    """Create test dataset and data loader"""
    # Create dataset
    dataset = SpeakingScoringDataset(
        csv_path=csv_path,
        task_type=task_type,
        score_column=score_column,
        translator=translator,
        cache_translations=True
    )
    
    logging.info(f"Test dataset size: {len(dataset)}")
    
    # Create data loader
    _, _, test_loader = create_dataloaders(
        train_dataset=None,
        val_dataset=None,
        test_dataset=dataset,
        batch_size=batch_size,
        num_workers=2,
        audio_encoder=audio_encoder,
        english_encoder=english_encoder,
        vietnamese_encoder=vietnamese_encoder,
        device=device,
        pin_memory=False
    )
    
    return dataset, test_loader


def evaluate_and_collect_predictions(model, test_loader, device, task_type):
    """Evaluate model and collect predictions"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_scores = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Get inputs
            inputs = batch['ccmt_input'].to(device)
            targets = batch['targets'].to(device)
            scores = batch.get('scores', targets * 0.5).to(device)
            metadata = batch['metadata']
            
            # Forward pass
            predictions = model(inputs)
            
            # Collect predictions
            if task_type == "classification":
                # Convert logits to class predictions and scores
                pred_classes = torch.argmax(predictions, dim=-1)
                pred_scores = pred_classes.float() * 0.5
                all_predictions.extend(pred_classes.cpu().numpy())
            else:  # regression
                # Use direct predictions
                if predictions.dim() > 1:
                    predictions = predictions.squeeze(-1)
                pred_scores = predictions
                all_predictions.extend(predictions.cpu().numpy())
            
            all_targets.extend(targets.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_metadata.extend(metadata)
    
    return np.array(all_predictions), np.array(all_targets), np.array(all_scores), all_metadata


def compute_detailed_metrics(predictions, targets, scores, task_type):
    """Compute detailed evaluation metrics"""
    metrics_calculator = ScoringMetrics(task_type=task_type, num_classes=21)
    
    # Convert to tensors for metrics calculator
    pred_tensor = torch.tensor(predictions)
    target_tensor = torch.tensor(targets)
    score_tensor = torch.tensor(scores)
    
    metrics_calculator.update(pred_tensor, target_tensor, score_tensor)
    metrics = metrics_calculator.compute()
    
    # Add per-class metrics for classification
    if task_type == "classification":
        per_class_metrics = metrics_calculator.compute_per_class_metrics()
        metrics.update(per_class_metrics)
    
    return metrics


def save_predictions(predictions, targets, scores, metadata, output_path, task_type):
    """Save predictions to CSV file"""
    # Prepare data for CSV
    results_data = {
        'audio_path': [meta['audio_path'] for meta in metadata],
        'true_target': targets,
        'true_score': scores,
        'predicted': predictions
    }
    
    if task_type == "classification":
        results_data['predicted_score'] = predictions * 0.5
        results_data['predicted_class'] = predictions
    else:
        results_data['predicted_score'] = predictions
        results_data['predicted_class'] = np.round(predictions * 2).astype(int)
    
    # Add error metrics
    if task_type == "classification":
        pred_scores = predictions * 0.5
    else:
        pred_scores = predictions
    
    results_data['absolute_error'] = np.abs(pred_scores - scores)
    results_data['squared_error'] = (pred_scores - scores) ** 2
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to: {output_path}")


def save_evaluation_report(metrics, output_path):
    """Save evaluation report"""
    # Create detailed report
    report = {
        'evaluation_summary': metrics,
        'key_metrics': {
            'accuracy': metrics.get('accuracy', 0.0),
            'pearson_correlation': metrics.get('pearson', 0.0),
            'spearman_correlation': metrics.get('spearman', 0.0),
            'mae': metrics.get('mae', 0.0),
            'rmse': metrics.get('rmse', 0.0),
            'within_0.5_points': metrics.get('within_0.5', 0.0),
            'within_1.0_points': metrics.get('within_1.0', 0.0)
        }
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logging.info(f"Evaluation report saved to: {output_path}")


def print_evaluation_results(metrics):
    """Print evaluation results to console"""
    print("\n" + "="*60)
    print("CCMT EVALUATION RESULTS")
    print("="*60)
    
    # Main metrics
    if 'accuracy' in metrics:
        print(f"Accuracy:              {metrics['accuracy']:.4f}")
    if 'pearson' in metrics:
        print(f"Pearson Correlation:   {metrics['pearson']:.4f}")
    if 'spearman' in metrics:
        print(f"Spearman Correlation:  {metrics['spearman']:.4f}")
    if 'mae' in metrics:
        print(f"Mean Absolute Error:   {metrics['mae']:.4f}")
    if 'rmse' in metrics:
        print(f"Root Mean Squared Error: {metrics['rmse']:.4f}")
    
    print("\nRange-based Accuracy:")
    if 'within_0.5' in metrics:
        print(f"  Within ±0.5 points:  {metrics['within_0.5']:.4f}")
    if 'within_1.0' in metrics:
        print(f"  Within ±1.0 points:  {metrics['within_1.0']:.4f}")
    
    print("="*60)


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine device
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device)
    
    logging.info("Starting CCMT evaluation...")
    logging.info(f"Device: {device}")
    logging.info(f"Model: {args.model_path}")
    logging.info(f"Test data: {args.csv_path}")
    
    # Load model and configuration
    model, config, task_type, num_classes = load_model_and_config(
        args.model_path, args.config, device
    )
    
    # Override task type if specified
    if args.task_type:
        task_type = args.task_type
    
    # Create evaluation components
    audio_encoder, english_encoder, vietnamese_encoder, translator = create_evaluation_components(
        config, device
    )
    
    # Create test dataset and loader
    dataset, test_loader = create_test_dataset_and_loader(
        args.csv_path, args.score_column, task_type,
        audio_encoder, english_encoder, vietnamese_encoder, translator,
        args.batch_size, device
    )
    
    # Evaluate model and collect predictions
    logging.info("Running evaluation...")
    predictions, targets, scores, metadata = evaluate_and_collect_predictions(
        model, test_loader, device, task_type
    )
    
    # Compute metrics
    logging.info("Computing metrics...")
    metrics = compute_detailed_metrics(predictions, targets, scores, task_type)
    
    # Print results
    print_evaluation_results(metrics)
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, "predictions.csv")
        save_predictions(predictions, targets, scores, metadata, predictions_path, task_type)
    
    # Save evaluation report
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    save_evaluation_report(metrics, report_path)
    
    logging.info("Evaluation completed successfully!")
    
    return metrics


if __name__ == "__main__":
    try:
        metrics = main()
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)
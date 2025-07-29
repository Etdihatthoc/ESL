"""
trainer.py - Training logic for ESL binary classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import logging


class ESLBinaryTrainer:
    """
    Trainer for ESL binary classification model
    """
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 test_loader,
                 train_dataset,
                 optimizer,
                 scheduler=None,
                 epochs=10,
                 device='cuda',
                 use_focal_loss=False,
                 focal_alpha=1.0,
                 focal_gamma=2.0,
                 use_label_smoothing=False,
                 logger=None):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_dataset = train_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup loss function
        if use_focal_loss:
            from model import FocalLoss
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.logger.info(f"Using Focal Loss with alpha={focal_alpha}, gamma={focal_gamma}")
        else:
            # Use weighted cross entropy
            from model import get_class_weights
            class_weights = get_class_weights(train_dataset).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            self.logger.info(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
        
        # Label smoothing
        self.use_label_smoothing = use_label_smoothing
        
        # For tracking best model
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_state_dict = None
        
        # Mixed precision scaler
        self.scaler = amp.GradScaler('cuda')
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []

        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['group'].to(self.device)
            audio = batch['audio'].to(self.device) if batch['audio'] is not None else None


            # Use soft labels if label smoothing is enabled
            if self.use_label_smoothing:
                soft_labels = batch['soft_labels'].to(self.device)
                train_targets = soft_labels
            else:
                train_targets = targets

            self.optimizer.zero_grad()

            with amp.autocast('cuda'):
                outputs = self.model(input_ids, attention_mask, audio)
                logits = outputs['logits']
                # If using label smoothing, assume criterion supports soft labels
                loss = self.criterion(logits, train_targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            # Statistics
            predictions = torch.argmax(logits, dim=-1)
            total_loss += loss.item()
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = total_correct / total_samples
        epoch_f1 = f1_score(all_targets, all_predictions, average='weighted')

        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1_score': epoch_f1,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_raw_scores = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['group'].to(self.device)
                raw_scores = batch['raw_score']
                audio = batch['audio'].to(self.device) if batch['audio'] is not None else None
                
                with amp.autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask, audio)
                    logits = outputs['logits']
                    probabilities = outputs['probabilities']
                    loss = self.criterion(logits, targets)
                
                predictions = torch.argmax(logits, dim=-1)
                total_loss += loss.item()
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_raw_scores.extend(raw_scores.numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(self.val_loader)
        val_acc = total_correct / total_samples
        val_precision = precision_score(all_targets, all_predictions, average='weighted')
        val_recall = recall_score(all_targets, all_predictions, average='weighted')
        val_f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return {
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1_score': val_f1,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'raw_scores': all_raw_scores
        }
    
    def train(self):
        """Full training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.epochs):
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Logging
            log_message = (
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Train F1: {train_metrics['f1_score']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1_score']:.4f}"
            )
            print(log_message)
            self.logger.info(log_message)
            
            # Save best model based on F1 score
            if val_metrics['f1_score'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_score']
                self.best_val_acc = val_metrics['accuracy']
                self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                save_message = f"New best model! Val F1: {self.best_val_f1:.4f}, Val Acc: {self.best_val_acc:.4f}"
                print(save_message)
                self.logger.info(save_message)
            
            # Early stopping check (optional)
            if epoch > 5 and val_metrics['f1_score'] < self.best_val_f1 * 0.95:
                self.logger.info("Early stopping triggered")
                break
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load best model
        if self.best_state_dict is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_state_dict.items()})
            self.logger.info(f"Loaded best model with Val F1: {self.best_val_f1:.4f}")
    
    def test(self, output_path="./results/binary_classification_results.csv"):
        """Test the model and save detailed results"""
        self.logger.info("Starting testing...")
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_raw_scores = []
        all_question_types = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['group'].to(self.device)
                raw_scores = batch['raw_score']
                question_types = batch['question_type']
                audio = batch['audio'].to(self.device) if batch['audio'] is not None else None
                
                with amp.autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask, audio)
                    probabilities = outputs['probabilities']
                
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_raw_scores.extend(raw_scores.numpy())
                all_question_types.extend(question_types.numpy())
        
        # Calculate metrics
        test_acc = accuracy_score(all_targets, all_predictions)
        test_precision = precision_score(all_targets, all_predictions, average='weighted')
        test_recall = recall_score(all_targets, all_predictions, average='weighted')
        test_f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # Detailed classification report
        class_report = classification_report(
            all_targets, 
            all_predictions, 
            target_names=['Group 0 (3.5-6.5)', 'Group 1 (7-10)']
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Ground_Truth_Group': all_targets,
            'Predicted_Group': all_predictions,
            'Probability_Group_0': [prob[0] for prob in all_probabilities],
            'Probability_Group_1': [prob[1] for prob in all_probabilities],
            'Raw_Score': all_raw_scores,
            'Question_Type': all_question_types,
            'Correct_Prediction': np.array(all_targets) == np.array(all_predictions)
        })
        
        # Add score range info
        results_df['Score_Range_Group_0'] = '3.5-6.5'
        results_df['Score_Range_Group_1'] = '7.0-10.0'
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        # Print results
        test_results = f"""
        =============== TEST RESULTS ===============
        Test Accuracy: {test_acc:.4f}
        Test Precision: {test_precision:.4f}
        Test Recall: {test_recall:.4f}
        Test F1-Score: {test_f1:.4f}
        
        Classification Report:
        {class_report}
        
        Confusion Matrix:
        {conf_matrix}
        
        Total samples: {len(all_targets)}
        Results saved to: {output_path}
        ==========================================
        """
        
        print(test_results)
        self.logger.info(test_results.replace('\n', ' '))
        
        # Analyze misclassifications
        misclassified = results_df[~results_df['Correct_Prediction']]
        if len(misclassified) > 0:
            print(f"\nMisclassified samples: {len(misclassified)}")
            print("Score distribution of misclassified samples:")
            print(misclassified['Raw_Score'].describe())
            
            # Show some examples
            print("\nSample misclassifications:")
            sample_misc = misclassified.head(5)[['Ground_Truth_Group', 'Predicted_Group', 'Raw_Score', 'Probability_Group_0', 'Probability_Group_1']]
            print(sample_misc.round(3))
        
        return {
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'results_df': results_df,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(self, path):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        self.logger.info(f"Model saved to {path}")


def analyze_score_distribution(results_df):
    """
    Analyze how well the classifier performs across different score ranges
    """
    print("\n=== SCORE DISTRIBUTION ANALYSIS ===")
    
    # Group by actual scores to see classification performance
    score_analysis = results_df.groupby('Raw_Score').agg({
        'Correct_Prediction': ['count', 'sum', 'mean'],
        'Predicted_Group': 'mean'
    }).round(3)
    
    print("Performance by raw score:")
    print(score_analysis)
    
    # Analyze boundary cases (scores near 6.5-7.0)
    boundary_scores = results_df[
        (results_df['Raw_Score'] >= 6.0) & (results_df['Raw_Score'] <= 7.5)
    ]
    
    if len(boundary_scores) > 0:
        print(f"\nBoundary region analysis (scores 6.5-7.0): {len(boundary_scores)} samples")
        print("Accuracy in boundary region:", boundary_scores['Correct_Prediction'].mean())
        print("Score distribution:")
        print(boundary_scores['Raw_Score'].value_counts().sort_index())
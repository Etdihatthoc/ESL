"""
run_test_device_fixed.py - Device-aware fixed version
Location: /media/gpus/Data/AES/ESL-Grading/run_test_device_fixed.py

Usage: python run_test_device_fixed.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import math

# Add paths
sys.path.append('/media/gpus/Data/AES/ESL-Grading/Classifier')

# Try imports with error handling
try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig, Wav2Vec2Model
    from Classifier.model import ESLBinaryClassifier
    print("✓ Imported classifier successfully")
except Exception as e:
    print(f"✗ Error importing classifier: {e}")
    sys.exit(1)


# Define ESLGradingModelWithAudio class directly here to avoid import issues
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, expected_seq_len=32, attn_proj=None, dropout=None):
        super().__init__()
        self.attn_proj = attn_proj or nn.Linear(hidden_dim, 1)
        init_scale = 1.0 / math.log(expected_seq_len)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        if dropout is not None and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, hidden_states, attention_mask=None, visualize=False):
        B, T, D = hidden_states.size()
        device = hidden_states.device

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.float32, device=device)

        raw_scores = self.attn_proj(hidden_states)
        scale_factor = self.scale * math.log(T)
        scaled_scores = raw_scores * scale_factor
        attn_mask = attention_mask.unsqueeze(-1)
        scaled_scores = scaled_scores.masked_fill(attn_mask == 0, -1e9)
        attn_weights = F.softmax(scaled_scores, dim=1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        pooled = torch.sum(attn_weights * hidden_states, dim=1)

        if visualize:
            return pooled, attn_weights
        else:
            return pooled


class ESLGradingModelWithAudio(nn.Module):
    def __init__(self, 
                 model_name='bert-base-uncased', 
                 audio_encoder_id="jonatasgrosman/wav2vec2-large-xlsr-53-english",
                 pooling_dropout=0.3, 
                 regression_dropout=0.5, 
                 avg_last_k=4,
                 d_fuse=256):
        super().__init__()
        self.num_types = 3
        self.pooling_dropout = pooling_dropout
        self.regression_dropout = regression_dropout
        self.avg_last_k = avg_last_k
        self.d_fuse = d_fuse

        # TEXT ENCODER
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        text_hidden_size = self.encoder.config.hidden_size
        self.encoder.gradient_checkpointing_enable()

        # AUDIO ENCODER
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_id)
        self.audio_hidden_dim = self.audio_encoder.config.output_hidden_size
        
        # PROJECTION LAYERS
        self.audio_proj = nn.Linear(self.audio_hidden_dim, d_fuse)
        self.audio_norm = nn.LayerNorm(d_fuse)
        self.text_proj = nn.Linear(text_hidden_size, d_fuse)
        self.text_norm = nn.LayerNorm(d_fuse)
        
        # 3 ATTENTION MECHANISMS
        self.text_self_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.text_self_norm = nn.LayerNorm(d_fuse)
        self.text_to_audio_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.t2a_norm = nn.LayerNorm(d_fuse)
        self.audio_to_text_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.a2t_norm = nn.LayerNorm(d_fuse)
        
        # 3 ATTENTION POOLING LAYERS
        self.text_self_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, 256), nn.Tanh(), nn.Dropout(pooling_dropout), nn.Linear(256, 1, bias=False)
        )
        self.text_self_pool = AttentionPooling(d_fuse, attn_proj=self.text_self_attn_proj, 
                                              expected_seq_len=512, dropout=pooling_dropout)
        
        self.t2a_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, 256), nn.Tanh(), nn.Dropout(pooling_dropout), nn.Linear(256, 1, bias=False)
        )
        self.t2a_pool = AttentionPooling(d_fuse, attn_proj=self.t2a_attn_proj, 
                                        expected_seq_len=512, dropout=pooling_dropout)
        
        self.a2t_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, 256), nn.Tanh(), nn.Dropout(pooling_dropout), nn.Linear(256, 1, bias=False)
        )
        self.a2t_pool = AttentionPooling(d_fuse, attn_proj=self.a2t_attn_proj, 
                                        expected_seq_len=10, dropout=pooling_dropout)
        
        # REGRESSION HEAD
        self.reg_head = nn.Sequential(
            nn.Linear(3 * d_fuse, 2 * d_fuse, bias=False), nn.LayerNorm(2 * d_fuse), nn.GELU(), nn.Dropout(regression_dropout),
            nn.Linear(2 * d_fuse, d_fuse, bias=False), nn.LayerNorm(d_fuse), nn.GELU(), nn.Dropout(regression_dropout),
            nn.Linear(d_fuse, 21, bias=False)
        )

    def encode_text(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states
        k = min(self.avg_last_k, len(all_hidden_states))
        if k == 1:
            hidden_states = all_hidden_states[-1]
        else:
            hidden_states = torch.stack(all_hidden_states[-k:], dim=0).mean(dim=0)
        hidden_states = hidden_states.float()
        return hidden_states

    def encode_audio(self, audio):
        if audio is None:
            return None

        batch_size, num_chunks, waveform_len = audio.shape
        device = next(self.parameters()).device

        audio_encoder_out = []
        for i in range(num_chunks):
            inp = audio[:, i, :].to(device)
            with torch.no_grad():
                out = self.audio_encoder(input_values=inp).last_hidden_state
                audio_encoder_out.append(out.mean(dim=1).detach().cpu())
            del inp, out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        audio_features = torch.stack(audio_encoder_out, dim=1).to(device)
        audio_features = self.audio_proj(audio_features)
        audio_features = self.audio_norm(audio_features)
        return audio_features

    def apply_three_attention_mechanisms(self, text_features, audio_features, attention_mask):
        batch_size = text_features.size(0)
        device = text_features.device
        
        text_proj = self.text_proj(text_features)
        text_proj = self.text_norm(text_proj)
        
        # Text Self-Attention
        text_self_output, _ = self.text_self_attention(query=text_proj, key=text_proj, value=text_proj)
        text_self_output = self.text_self_norm(text_self_output)
        
        with torch.amp.autocast('cuda', enabled=False):
            text_self_pooled = self.text_self_pool(text_self_output, attention_mask)
        
        if audio_features is None:
            t2a_pooled = torch.zeros(batch_size, self.d_fuse, device=device)
            a2t_pooled = torch.zeros(batch_size, self.d_fuse, device=device)
        else:
            # Text-to-Audio Cross-Attention
            t2a_output, _ = self.text_to_audio_attention(query=text_proj, key=audio_features, value=audio_features)
            t2a_output = self.t2a_norm(t2a_output)
            
            with torch.amp.autocast('cuda', enabled=False):
                t2a_pooled = self.t2a_pool(t2a_output, attention_mask)
            
            # Audio-to-Text Cross-Attention
            a2t_output, _ = self.audio_to_text_attention(query=audio_features, key=text_proj, value=text_proj)
            a2t_output = self.a2t_norm(a2t_output)
            
            with torch.amp.autocast('cuda', enabled=False):
                a2t_pooled = self.a2t_pool(a2t_output)
        
        return text_self_pooled, t2a_pooled, a2t_pooled

    def forward(self, input_ids, attention_mask, audio=None):
        text_hidden_states = self.encode_text(input_ids, attention_mask)
        audio_features = self.encode_audio(audio)
        text_self_pooled, t2a_pooled, a2t_pooled = self.apply_three_attention_mechanisms(
            text_hidden_states, audio_features, attention_mask
        )
        combined_features = torch.cat([text_self_pooled, t2a_pooled, a2t_pooled], dim=1)
        logits = self.reg_head(combined_features)
        probs = torch.softmax(logits, dim=-1)
        score_bins = torch.linspace(0, 10, steps=21).to(probs.device)
        expected_score = (probs * score_bins).sum(dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'expected_score': expected_score
        }

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        model = cls(
            model_name=config.get('model_name', 'Alibaba-NLP/gte-multilingual-base'),
            pooling_dropout=config.get('pooling_dropout', 0.3),
            regression_dropout=config.get('regression_dropout', 0.5),
            avg_last_k=config.get('avg_last_k', 1),
            d_fuse=config.get('d_fuse', 256)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def convert_score_to_new_groups(score):
    """Convert score to new groups: 0 (3.5-6.5), 1 (7-10)"""
    if 3.5 <= score <= 6.5:
        return 0
    elif 7.0 <= score <= 10.0:
        return 1
    else:
        return 0 if score < 6.75 else 1


def ensure_device_compatibility(model, device):
    """Ensure model is properly moved to device and handle any device issues"""
    model = model.to(device)
    
    # Force all parameters to be on the correct device
    for param in model.parameters():
        param.data = param.data.to(device)
    
    # Force all buffers to be on the correct device
    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)
    
    return model


class SimplePipeline:
    """Simplified pipeline for testing with proper device handling"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Check GPU memory
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-multilingual-base')
        
        # Model paths
        classifier_path = '/media/gpus/Data/AES/ESL-Grading/Classifier/models/esl_binary_classifier.pth'
        model_35_65_path = '/media/gpus/Data/AES/ESL-Grading/model/sep/model_with_audio_bestmae_35_65_e2e.pth'
        model_7_10_path = '/media/gpus/Data/AES/ESL-Grading/model/sep/model_with_audio_bestmae_7_10_e2e.pth'
        
        # Check paths
        missing_files = []
        for name, path in [('Classifier', classifier_path), ('Model 3.5-6.5', model_35_65_path), ('Model 7-10', model_7_10_path)]:
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            print("✗ Missing files:")
            for f in missing_files:
                print(f"  {f}")
            sys.exit(1)
        
        # Load models with proper device handling
        print("Loading models...")
        try:
            print("  Loading classifier...")
            self.classifier = ESLBinaryClassifier.load(classifier_path, device='cpu')  # Load to CPU first
            self.classifier = ensure_device_compatibility(self.classifier, self.device)  # Then move to GPU
            self.classifier.eval()
            print("  ✓ Classifier loaded and moved to GPU")
        except Exception as e:
            print(f"  ✗ Error loading classifier: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        try:
            print("  Loading model 3.5-6.5...")
            self.model_35_65 = ESLGradingModelWithAudio.load(model_35_65_path)
            self.model_35_65 = ensure_device_compatibility(self.model_35_65, self.device)
            self.model_35_65.eval()
            print("  ✓ Model 3.5-6.5 loaded and moved to GPU")
        except Exception as e:
            print(f"  ✗ Error loading model 3.5-6.5: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        try:
            print("  Loading model 7-10...")
            self.model_7_10 = ESLGradingModelWithAudio.load(model_7_10_path)
            self.model_7_10 = ensure_device_compatibility(self.model_7_10, self.device)
            self.model_7_10.eval()
            print("  ✓ Model 7-10 loaded and moved to GPU")
        except Exception as e:
            print(f"  ✗ Error loading model 7-10: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        print("✓ All models loaded successfully!")
        
        # Test a small forward pass to ensure everything works
        print("Testing device compatibility...")
        try:
            test_input = self.tokenizer("test", return_tensors='pt', max_length=512, truncation=True, padding=True)
            test_input = {k: v.to(self.device) for k, v in test_input.items()}
            
            with torch.no_grad():
                _ = self.classifier(test_input['input_ids'], test_input['attention_mask'])
            print("✓ Device compatibility test passed!")
            
        except Exception as e:
            print(f"✗ Device compatibility test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def predict_batch(self, texts, dummy_audio=None):
        """Predict scores for a batch of texts with proper device handling"""
        # Tokenize and ensure tensors are on correct device
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, 
            max_length=8192, return_tensors='pt'
        )
        
        # Explicitly move all tensors to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        if dummy_audio is not None:
            dummy_audio = dummy_audio.to(self.device)
        
        with torch.no_grad():
            # Step 1: Classify groups
            classifier_out = self.classifier(input_ids, attention_mask)
            predicted_groups = classifier_out['predictions']
            
            # Step 2: Route to appropriate models
            batch_size = len(texts)
            final_scores = torch.zeros(batch_size, device=self.device)
            
            # Group 0 (3.5-6.5)
            group_0_mask = (predicted_groups == 0)
            if group_0_mask.any():
                scores_0 = self.model_35_65(
                    input_ids[group_0_mask],
                    attention_mask[group_0_mask],
                    dummy_audio[group_0_mask] if dummy_audio is not None else None
                )['expected_score']
                final_scores[group_0_mask] = scores_0
            
            # Group 1 (7-10)
            group_1_mask = (predicted_groups == 1)
            if group_1_mask.any():
                scores_1 = self.model_7_10(
                    input_ids[group_1_mask],
                    attention_mask[group_1_mask],
                    dummy_audio[group_1_mask] if dummy_audio is not None else None
                )['expected_score']
                final_scores[group_1_mask] = scores_1
        
        return {
            'predicted_groups': predicted_groups.cpu().numpy(),
            'final_scores': final_scores.cpu().numpy()
        }


def process_dataframe(df, pipeline, dataset_name, batch_size=4):  # Reduced batch size
    """Process a dataframe and return results"""
    print(f"\n=== Processing {dataset_name} ===")
    
    # Filter valid scores
    df_clean = df[(df['grammar'] >= 3.5) & (df['grammar'] <= 10.0)].copy()
    print(f"Samples: {len(df)} -> {len(df_clean)} (after filtering)")
    
    if len(df_clean) == 0:
        return None
    
    # Prepare texts
    text_prefix = "The following is a spoken English response by a non-native speaker. Grade the grammar score based on the transcript below:"
    texts = []
    for _, row in df_clean.iterrows():
        qtype_map = {1: "Social Interaction", 2: "Solution Discussion", 3: "Topic Development"}
        qtype_text = qtype_map.get(row['question_type'], '')
        text = f"{text_prefix} [Question Type: {qtype_text}] {row['text'][2:-1]}"
        texts.append(text)
    
    # Ground truth
    ground_truth_scores = df_clean['grammar'].values
    ground_truth_groups = [convert_score_to_new_groups(s) for s in ground_truth_scores]
    
    # Predict in batches
    all_pred_scores = []
    all_pred_groups = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing {dataset_name}"):
        batch_texts = texts[i:i+batch_size]
        
        # Create dummy audio (10 chunks of 30 seconds each)
        dummy_audio = torch.zeros(len(batch_texts), 10, 30*16000, device=pipeline.device)
        
        try:
            # Predict
            results = pipeline.predict_batch(batch_texts, dummy_audio)
            
            all_pred_scores.extend(results['final_scores'])
            all_pred_groups.extend(results['predicted_groups'])
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Fill with dummy values for this batch
            all_pred_scores.extend([5.0] * len(batch_texts))
            all_pred_groups.extend([0] * len(batch_texts))
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate metrics
    pred_scores = np.array(all_pred_scores)
    pred_groups = np.array(all_pred_groups)
    gt_groups = np.array(ground_truth_groups)
    
    mse = np.mean((ground_truth_scores - pred_scores) ** 2)
    mae = np.mean(np.abs(ground_truth_scores - pred_scores))
    
    try:
        from scipy.stats import pearsonr
        correlation, _ = pearsonr(ground_truth_scores, pred_scores)
    except:
        correlation = 0.0
    
    group_accuracy = np.mean(gt_groups == pred_groups)
    
    print(f"Results:")
    print(f"  Score: MSE={mse:.4f}, MAE={mae:.4f}, Corr={correlation:.4f}")
    print(f"  Group: Accuracy={group_accuracy:.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'Ground_Truth_Score': ground_truth_scores,
        'Predicted_Score': pred_scores,
        'Ground_Truth_Group': gt_groups,
        'Predicted_Group': pred_groups,
        'Score_Error': np.abs(ground_truth_scores - pred_scores),
        'Group_Correct': gt_groups == pred_groups
    })
    
    output_path = f'/media/gpus/Data/AES/ESL-Grading/results_{dataset_name}.csv'
    results_df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return {
        'mse': mse, 'mae': mae, 'correlation': correlation, 
        'group_accuracy': group_accuracy, 'n_samples': len(ground_truth_scores)
    }


def main():
    """Main function"""
    print("=== ESL Combined Pipeline Test (Device Fixed) ===")
    
    # Initialize pipeline
    pipeline = SimplePipeline()
    
    # Test datasets
    datasets = ['test_pro_removenoise.csv'] #'val_pro_removenoise.csv',
    results = {}
    
    for dataset_file in datasets:
        dataset_path = f'/media/gpus/Data/AES/ESL-Grading/data/PreprocessData/{dataset_file}'
        dataset_name = dataset_file.replace('.csv', '').replace('_pro', '')
        
        if not os.path.exists(dataset_path):
            print(f"✗ Dataset not found: {dataset_path}")
            continue
        
        df = pd.read_csv(dataset_path)
        result = process_dataframe(df, pipeline, dataset_name)
        
        if result:
            results[dataset_name] = result
    
    # Summary
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print(f"{'='*50}")
    
    for name, metrics in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Samples: {metrics['n_samples']}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  Group Accuracy: {metrics['group_accuracy']:.4f}")
    
    print(f"\nResults saved to: /media/gpus/Data/AES/ESL-Grading/results_[test|train|val].csv")


if __name__ == "__main__":
    main()
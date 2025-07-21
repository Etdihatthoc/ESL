import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from train import ESLDataset, ESLGradingModel, get_collate_fn
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr

def model_test(model, tokenizer, test_path, batch_size=32, device='cuda'):
    model.eval()

    # 1. Load test file
    test_df = pd.read_csv(test_path).reset_index(drop=True)
    test_dataset = ESLDataset(test_df)
    collate_fn = get_collate_fn(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    total_loss = 0.0
    total_mae = 0.0
    total_delta = 0.0
    total_count = 0

    binwise_mse = defaultdict(float)
    binwise_mae = defaultdict(float)
    binwise_delta = defaultdict(float)
    binwise_count = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            question_type = batch['question_type'].to(device)
            true_scores = batch['score'].to(device)

            out = model(input_ids, attention_mask)
            pred_scores = out['expected_score']

            mse_batch = F.mse_loss(pred_scores, true_scores, reduction='none')
            mae_batch = torch.abs(pred_scores - true_scores)
            delta_batch = pred_scores - true_scores

            total_loss += mse_batch.sum().item()
            total_mae += mae_batch.sum().item()
            total_delta += delta_batch.sum().item()
            total_count += input_ids.size(0)

            for i in range(input_ids.size(0)):
                score_bin = round(float(true_scores[i].item()) * 2) / 2  # bin size 0.5
                binwise_mse[score_bin] += mse_batch[i].item()
                binwise_mae[score_bin] += mae_batch[i].item()
                binwise_delta[score_bin] += delta_batch[i].item()
                binwise_count[score_bin] += 1

    print("\nPer-Score MSE / MAE / Avg Delta:")
    mse_list = []
    mae_list = []
    for b in sorted(binwise_count):
        n = binwise_count[b]
        if n > 0:
            mse_avg = binwise_mse[b] / n
            mae_avg = binwise_mae[b] / n
            delta_avg = binwise_delta[b] / n
            mse_list.append(mse_avg)
            mae_list.append(mae_avg)
            print(f"Score {b:.1f}: MSE = {mse_avg:.4f}, MAE = {mae_avg:.4f}, Avg Delta = {delta_avg:.4f}")

    print("\n---Summary---")
    print(f"Overall Test MSE: {total_loss / total_count:.4f}")
    print(f"Overall Test MAE: {total_mae / total_count:.4f}")
    print(f"Overall Test Avg Delta: {total_delta / total_count:.4f}")
    if mse_list and mae_list:
        print(f"Average MSE over all bins: {np.mean(mse_list):.4f}")
        print(f"Average MAE over all bins: {np.mean(mae_list):.4f}")

def run_examples(model, tokenizer, test_path, num_examples=64, batch_size=16, device='cuda'):
    # 1. Load test data and preserve audio_path
    sample_df = pd.read_csv(test_path).reset_index()
    bins = np.linspace(0, 10, num=21)
    sample_df['score_bin'] = pd.cut(sample_df['final'], bins=bins, labels=False)

    grouped = sample_df.groupby('score_bin')
    examples_df = grouped.apply(
        lambda x: x.sample(n=min(len(x), max(1, num_examples // len(bins))), random_state=42),
        include_groups=False
    ).reset_index(drop=True)

    remaining = num_examples - len(examples_df)
    if remaining > 0:
        extra = sample_df.sample(n=remaining, random_state=42)
        examples_df = pd.concat([examples_df, extra], ignore_index=True)

    original_indices = examples_df['index'].tolist()
    examples_df = examples_df.drop(columns=['index'])

    remaining = num_examples - len(examples_df)
    if remaining > 0:
        extra = sample_df.sample(n=remaining, random_state=42)
        examples_df = pd.concat([examples_df, extra], ignore_index=True)

    example_audio_paths = examples_df['audio_path'].tolist()

    # 2. Predict on sampled test examples using model.forward
    examples_dataset = ESLDataset(examples_df)
    collate_fn = get_collate_fn(tokenizer)
    examples_loader = DataLoader(examples_dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_true = []
    all_pred = []
    all_probs = []

    with torch.no_grad():
        for batch in examples_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            question_type = batch['question_type'].to(device)
            true_scores = batch['score'].to(device)

            out = model(input_ids, attention_mask)
            pred_scores = out['expected_score']
            probs = out['probs']

            all_true.extend(true_scores.cpu().numpy())
            all_pred.extend(pred_scores.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())


    # 3. Output results with audio_path
    original_test_df = pd.read_csv(test_path)
    print("\nSampled Examples (True vs Predicted):")
    for orig_idx, t, p, probs in zip(original_indices, all_true, all_pred, all_probs):
        audio_path = original_test_df.loc[orig_idx, 'audio_path']
        diff = t - p
        print(f"{audio_path}: True = {t:.2f}, Pred = {p:.2f}, Diff = {diff:.2f}, Probs = {probs}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ESLGradingModel.load("/media/gpus/Data/AES/best_avg.pth").to(device)
tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-multilingual-base')

train_path = "./data/train_pro.csv"
val_path = "./data/val_pro.csv"
test_path = "./data/test_pro.csv"

model_test(model, tokenizer, test_path, device=device)
# run_examples(model, tokenizer, test_path, device=device)
# visualize_pooling(model, tokenizer, sample_path=test_path, device=device)
# visualize_encoding(model, tokenizer, sample_path=test_path, device=device)
# visualize_encoding_distributions(model, tokenizer, [train_path, val_path, test_path], device=device)
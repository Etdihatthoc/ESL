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

def model_test(model, tokenizer, sample_path, test_path, batch_size=16, k=32, k_exclude=4, device='cuda'):
    model.eval()

    # 1. Sample k exemplars from sampling file
    sample_df = pd.read_csv(sample_path)
    bins = np.linspace(0, 10, num=21)
    sample_df['score_bin'] = pd.cut(sample_df['final'], bins=bins, labels=False)
    grouped = sample_df.groupby('score_bin')
    sampled_df = grouped.apply(lambda x: x.sample(n=min(len(x), max(1, k // len(bins))), random_state=42), include_groups=False)
    sampled_df = sampled_df.reset_index(drop=True)

    remaining_k = k - len(sampled_df)
    if remaining_k > 0:
        extra_samples = sample_df.sample(n=remaining_k, random_state=42)
        sampled_df = pd.concat([sampled_df, extra_samples], ignore_index=True)

    # 2. Load sampled exemplar embeddings
    sampled_dataset = ESLDataset(sampled_df)
    collate_fn = get_collate_fn(tokenizer)
    sampled_loader = DataLoader(sampled_dataset, batch_size=batch_size, collate_fn=collate_fn)

    ref_vecs_list = []
    score_ref_list = []
    with torch.no_grad():
        for sample_batch in sampled_loader:
            input_ids_ref = sample_batch['input_ids'].to(device)
            attention_mask_ref = sample_batch['attention_mask'].to(device)
            question_type_ref = sample_batch['question_type'].to(device)
            score_ref = sample_batch['score'].to(device)
            vecs = model.encode(input_ids_ref, attention_mask_ref, question_type_ref)
            ref_vecs_list.append(vecs)
            score_ref_list.append(score_ref)

    ref_vecs = torch.cat(ref_vecs_list, dim=0)  # [k, d]
    score_ref = torch.cat(score_ref_list, dim=0)  # [k]

    # 3. Evaluate on test file
    test_df = pd.read_csv(test_path).reset_index(drop=True)
    test_dataset = ESLDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    binwise_mse = defaultdict(float)
    binwise_mae = defaultdict(float)
    binwise_count = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            question_type = batch['question_type'].to(device)
            true_scores = batch['score'].to(device)

            batch_vecs = model.encode(input_ids, attention_mask, question_type)  # [B, d]
            deltas = model.reg_head(batch_vecs[:, None, :] - ref_vecs[None, :, :]).squeeze(-1)  # [B, k]
            pred_scores = (score_ref.unsqueeze(0) + deltas) # [B, k]

            # Trim top-k and bottom-k
            if pred_scores.size(1) > 2 * k_exclude:
                sorted_preds, _ = torch.sort(pred_scores, dim=1)
                trimmed = sorted_preds[:, k_exclude:-k_exclude]
                avg_pred_trimmed = trimmed.mean(dim=1).clamp(0, 10) 
            else:
                avg_pred_trimmed = pred_scores.mean(dim=1).clamp(0, 10) 

            mse_batch = F.mse_loss(avg_pred_trimmed, true_scores, reduction='none')
            mae_batch = torch.abs(avg_pred_trimmed - true_scores)

            total_loss += mse_batch.sum().item()
            total_mae += mae_batch.sum().item()
            total_count += input_ids.size(0)

            for i in range(input_ids.size(0)):
                score_bin = round(float(true_scores[i].item()) * 2) / 2  # bin size 0.5
                binwise_mse[score_bin] += mse_batch[i].item()
                binwise_mae[score_bin] += mae_batch[i].item()
                binwise_count[score_bin] += 1

    print(f"\nOverall Test MSE: {total_loss / total_count:.4f}")
    print(f"Overall Test MAE: {total_mae / total_count:.4f}")

    print("\nPer-Score MSE / MAE:")
    for b in sorted(binwise_count):
        n = binwise_count[b]
        if n > 0:
            mse_avg = binwise_mse[b] / n
            mae_avg = binwise_mae[b] / n
            print(f"Score {b:.1f}: MSE = {mse_avg:.4f}, MAE = {mae_avg:.4f}")

def run_examples(model, tokenizer, sample_path, test_path, num_examples=64, batch_size=16, k=32, k_exclude=4, device='cuda'):
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

    # 2. Sample k exemplars from sample_path
    sample_train_df = pd.read_csv(sample_path)
    sample_train_df['score_bin'] = pd.cut(sample_train_df['final'], bins=bins, labels=False)
    grouped_train = sample_train_df.groupby('score_bin')
    sampled_df = grouped_train.apply(
        lambda x: x.sample(n=min(len(x), max(1, k // len(bins))), random_state=42),
        include_groups=False
    ).reset_index(drop=True)

    remaining_k = k - len(sampled_df)
    if remaining_k > 0:
        extra_samples = sample_train_df.sample(n=remaining_k, random_state=42)
        sampled_df = pd.concat([sampled_df, extra_samples], ignore_index=True)

    # 3. Encode exemplar vectors
    sampled_dataset = ESLDataset(sampled_df)
    collate_fn = get_collate_fn(tokenizer)
    sampled_loader = DataLoader(sampled_dataset, batch_size=batch_size, collate_fn=collate_fn)

    ref_vecs_list = []
    score_ref_list = []
    with torch.no_grad():
        for sample_batch in sampled_loader:
            input_ids_ref = sample_batch['input_ids'].to(device)
            attention_mask_ref = sample_batch['attention_mask'].to(device)
            question_type_ref = sample_batch['question_type'].to(device)
            score_ref = sample_batch['score'].to(device)
            vecs = model.encode(input_ids_ref, attention_mask_ref, question_type_ref)
            ref_vecs_list.append(vecs)
            score_ref_list.append(score_ref)

    ref_vecs = torch.cat(ref_vecs_list, dim=0)
    score_ref = torch.cat(score_ref_list, dim=0)

    # 4. Predict on sampled test examples
    examples_dataset = ESLDataset(examples_df)
    examples_loader = DataLoader(examples_dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in examples_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            question_type = batch['question_type'].to(device)
            true_scores = batch['score'].to(device)

            batch_vecs = model.encode(input_ids, attention_mask, question_type)
            deltas = model.reg_head(batch_vecs[:, None, :] - ref_vecs[None, :, :]).squeeze(-1)
            pred_scores = (score_ref.unsqueeze(0) + deltas)

            if pred_scores.size(1) > 2 * k_exclude:
                sorted_preds, _ = torch.sort(pred_scores, dim=1)
                trimmed = sorted_preds[:, k_exclude:-k_exclude]
                avg_pred_trimmed = trimmed.mean(dim=1).clamp(0, 10) 
            else:
                avg_pred_trimmed = pred_scores.mean(dim=1).clamp(0, 10) 

            all_true.extend(true_scores.cpu().numpy())
            all_pred.extend(avg_pred_trimmed.cpu().numpy())

    # 5. Output results with audio_path
    original_test_df = pd.read_csv(test_path)
    print("\nSampled Examples (True vs Predicted):")
    for orig_idx, t, p in zip(original_indices, all_true, all_pred):
        audio_path = original_test_df.loc[orig_idx, 'audio_path']
        diff = t - p
        print(f"{audio_path}: True = {t:.2f}, Pred = {p:.2f}, Diff = {diff:.2f}")

def visualize_pooling(model, tokenizer, sample_path, device='cpu'):
    # Take some examples from sample_path, encode, and visualize attention scores in attention pooling
    df = pd.read_csv(sample_path).sample(n=1)
    dataset = ESLDataset(df)
    collate_fn = get_collate_fn(tokenizer)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            question_type = batch['question_type'].to(device)

            # Forward pass to get encoder outputs and attention scores
            # Assume model.encode returns (pooled, attn_scores) if visualize=True
            pooled, attn_scores = model.encode(input_ids, attention_mask, question_type, visualize=True)
            attn_scores = attn_scores.squeeze()  # Remove all singleton dimensions
            attn_scores = attn_scores.cpu().numpy()
            if attn_scores.ndim > 1:
                attn_scores = attn_scores.flatten()  # Ensure 1D

            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
            print(attn_scores)
            plt.figure(figsize=(40, 2))  # Very wide figure for readability
            plt.bar(range(len(attn_scores)), attn_scores)
            plt.xticks(range(len(tokens)), tokens, rotation=90)
            plt.title(f"Example Attention Scores")
            plt.xlabel("Token")
            plt.ylabel("Attention Score")
            plt.tight_layout()
            plt.savefig("fig.png")
            plt.close()

            print("Pooled Representation Vector", pooled)

def visualize_encoding(model, tokenizer, sample_path, num_examples=256, device='cpu'):
    # Take num_examples examples, encode them, and visualize scores and encodings in 2D

    df = pd.read_csv(sample_path).sample(n=num_examples)
    dataset = ESLDataset(df)
    collate_fn = get_collate_fn(tokenizer)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    model.eval()
    all_vecs = []
    all_scores = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            question_type = batch['question_type'].to(device)
            scores = batch['score'].cpu().numpy()
            vecs = model.encode(input_ids, attention_mask, question_type)
            if isinstance(vecs, tuple):  # If model.encode returns (vecs, attn_scores)
                vecs = vecs[0]
            all_vecs.append(vecs.cpu().numpy())
            all_scores.extend(scores)

    all_vecs = np.concatenate(all_vecs, axis=0)
    all_scores = np.array(all_scores)

    # Reduce to 2D
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(all_vecs)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], c=all_scores, cmap='viridis', s=60)
    plt.colorbar(scatter, label='Score')
    plt.title('2D Visualization of Encoded Representations')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.tight_layout()
    plt.savefig("encoding_2d.png")
    plt.show()

    # Compute pairwise distances in encoding and score space

    # Pairwise distances between encodings
    encoding_dists = squareform(pdist(all_vecs, metric='cosine'))
    # Pairwise absolute differences between scores
    score_dists = squareform(pdist(all_scores[:, None], metric='cityblock'))

    # Flatten upper triangle (excluding diagonal)
    triu_idx = np.triu_indices_from(encoding_dists, k=1)
    enc_flat = encoding_dists[triu_idx]
    score_flat = score_dists[triu_idx]

    # Correlation between encoding distance and score distance
    spearman_corr, _ = spearmanr(enc_flat, score_flat)
    pearson_corr, _ = pearsonr(enc_flat, score_flat)

    print(f"Spearman correlation between encoding and score distance: {spearman_corr:.4f}")
    print(f"Pearson correlation between encoding and score distance: {pearson_corr:.4f}")

    plt.figure(figsize=(6, 5))
    plt.hexbin(score_flat, enc_flat, gridsize=40, cmap='Blues', mincnt=1)
    plt.xlabel("Score Distance")
    plt.ylabel("Encoding Distance")
    plt.title("Encoding Distance vs. Score Distance")
    plt.colorbar(label="Pair Count")
    plt.tight_layout()
    plt.savefig("encoding_vs_score_distance.png")
    plt.show()

def visualize_encoding_distributions(model, tokenizer, paths, device='cpu'):
    all_vecs = []
    all_labels = []
    all_scores = []

    for path in paths:
        df = pd.read_csv(path)
        dataset = ESLDataset(df)
        collate_fn = get_collate_fn(tokenizer)
        loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        vecs = []
        scores = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Encoding file {path}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                question_type = batch['question_type'].to(device)
                score = batch['score'].cpu().numpy()
                out = model.encode(input_ids, attention_mask, question_type)
                if isinstance(out, tuple):
                    out = out[0]
                vecs.append(out.cpu().numpy())
                scores.extend(score)
        vecs = np.concatenate(vecs, axis=0)
        all_vecs.append(vecs)
        all_labels.extend([path] * len(vecs))
        all_scores.extend(scores)

    all_vecs_concat = np.concatenate(all_vecs, axis=0)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(all_vecs_concat)

    plt.figure(figsize=(8, 6))
    for path in paths:
        idx = all_labels == path
        plt.scatter(vecs_2d[idx, 0], vecs_2d[idx, 1], label=path, alpha=0.6, s=40)
    plt.legend()
    plt.title("2D PCA of Encodings by File")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig("encoding_distributions_pca.png")
    plt.show()

    # Optionally, visualize score distributions per file
    plt.figure(figsize=(8, 4))
    for path in paths:
        idx = all_labels == path
        sns.kdeplot(all_scores[idx], label=path, fill=True, alpha=0.3)
    plt.legend()
    plt.title("Score Distributions by File")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("score_distributions.png")
    plt.show()


device = "cuda" if torch.cuda.is_available() else "cpu"

model = ESLGradingModel.load("./model/model.pth").to(device)
tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-multilingual-base')

train_path = "./data/train_pro.csv"
val_path = "./data/val_pro.csv"
test_path = "./data/test_pro.csv"

# model_test(model, tokenizer, train_path, test_path, device=device)
# run_examples(model, tokenizer, train_path, test_path, device=device)
visualize_pooling(model, tokenizer, sample_path=test_path, device=device)
visualize_encoding(model, tokenizer, sample_path=test_path, device=device)
# visualize_encoding_distributions(model, tokenizer, [train_path, val_path, test_path], device=device)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torch.amp as amp
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
import random
from collections import deque, defaultdict
import os
import gc

# ----------------------
# Dataset
# ----------------------
import torch
from torch.utils.data import Dataset

class ESLDataset(Dataset):
    def __init__(self, dataframe):
        self.text_prefix = (
            "The following is a spoken English response by a non-native speaker. "
            "Grade the fluency, grammar, vocabulary, pronunciation, and content based on the transcript below:"
        )
        self.texts = [self.text_prefix + " " + t[2:-1] for t in dataframe['text'].tolist()]
        self.scores = dataframe['final'].astype(float).tolist()
        self.question_types = dataframe['question_type'].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'score': torch.tensor(self.scores[idx], dtype=torch.float32),
            'question_type': self.question_types[idx]
        }
    
def get_collate_fn(tokenizer, max_length=8192): # max_length <= context window of embedding model
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        scores = torch.stack([item['score'] for item in batch])
        question_types = torch.tensor([item['question_type'] for item in batch], dtype=torch.long)

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'score': scores,
            'question_type': question_types
        }
    return collate_fn


class GatedFF(nn.Module):
    def __init__(self, input_dim, output_dim=64, dropout=0.1, v_activation=nn.GELU(), g_activation=None, use_norm=True):
        super().__init__()
        self.value_proj = nn.Linear(input_dim, output_dim)
        self.gate_proj = nn.Linear(input_dim, output_dim)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(output_dim)
        self.v_activation = v_activation
        self.g_activation = g_activation
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        v = self.value_proj(hidden_states)                 # [B, T, output_dim]
        if self.v_activation is not None:
            v = self.v_activation(v)
        g = self.gate_proj(hidden_states)
        if self.g_activation is not None:
            g = self.g_activation(g)
        x = v * g                                          # gated interaction
        if self.use_norm:
            x = self.norm(x)                               # normalize gated output
        if self.dropout: 
            x = self.dropout(x)
        return x                                           # [B, T, output_dim]
    
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, expected_seq_len=32, attn_proj=None):
        super().__init__()
        self.attn_proj = attn_proj or nn.Linear(hidden_dim, 1)
        init_scale = 1.0 / math.log(expected_seq_len)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: [B, T, D]
        attention_mask: [B, T] (1 = keep, 0 = pad); optional
        """
        B, T, D = hidden_states.size()
        device = hidden_states.device

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.float32, device=device)

        raw_scores = self.attn_proj(hidden_states)  # [B, T, 1]

        scale_factor = self.scale * math.log(T)
        scaled_scores = raw_scores * scale_factor  # [B, T, 1]

        attn_mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
        scaled_scores = scaled_scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = F.softmax(scaled_scores, dim=1)  # [B, T, 1]
        pooled = torch.sum(attn_weights * hidden_states, dim=1)  # [B, D]

        return pooled

class ESLGradingModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout=0.3, avg_last_k=4, prefix_dim=8):
        super().__init__()
        self.num_types = 3  # question types: 1, 2, 3
        self.dropout = dropout
        self.avg_last_k = avg_last_k
        self.prefix_dim = prefix_dim

        # Load encoder and apply dropout
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # config.hidden_dropout_prob = dropout
        # config.attention_probs_dropout_prob = dropout
        config.output_hidden_states=True
        self.encoder = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        hidden_size = self.encoder.config.hidden_size
        self.encoder.gradient_checkpointing_enable()

        # Question type encoder: 1 x D vector per type
        self.type_encoder = nn.Embedding(self.num_types, prefix_dim*hidden_size)

        # Gated attention pooling
        self.attn_proj = nn.Sequential(
            GatedFF(hidden_size, output_dim=64, dropout=dropout, 
                    v_activation=None, g_activation=nn.Sigmoid()),
            nn.Linear(64, 1, bias=False)
        )
        self.attn_pool = AttentionPooling(hidden_size, attn_proj=self.attn_proj, expected_seq_len=512)

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size, bias=False),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.config.hidden_size, 256, bias=False),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, 64, bias=False),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(64, 1, bias=False),
            nn.Hardtanh(min_val=-10, max_val=10)
        )

    def encode(self, input_ids, attention_mask, question_type):
        batch_size = input_ids.size(0)
        device = input_ids.device
        hidden_size = self.encoder.config.hidden_size
        seq_len = input_ids.size(1)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states
        k = min(self.avg_last_k, len(all_hidden_states))
        if k == 1:
            hidden_states = all_hidden_states[-1]
        else:
            hidden_states = torch.stack(all_hidden_states[-k:], dim=0).mean(dim=0)
        hidden_states = hidden_states.float()

        with amp.autocast('cuda', enabled=False):
            # Type embeddings
            type_embeds = self.type_encoder(question_type - 1)
            type_embeds = type_embeds.view(batch_size, self.prefix_dim, hidden_size)

            hidden_states = torch.cat([type_embeds, hidden_states], dim=1)
            prefix_mask = torch.ones((batch_size, self.prefix_dim), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # Attention pooling
            pooled = self.attn_pool(hidden_states, attention_mask)

        return pooled

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'dropout': self.dropout,
                'model_name': self.encoder.config._name_or_path,
                'avg_last_k': self.avg_last_k
            }
        }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        model = cls(
            model_name=config.get('model_name', 'bert-base-uncased'),
            dropout=config['dropout'],
            avg_last_k=config.get('avg_last_k', 1)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

class WarmupInverseSquareScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = max(1, self.last_epoch)  # avoid div by zero
        
        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Inverse square decay after warmup
            decay_step = step - self.warmup_steps + 1
            decay_factor = 1.0 / (decay_step ** 0.5)
            return [base_lr * decay_factor for base_lr in self.base_lrs]  
        
def maybe_empty_cache(threshold=0.93):
    if torch.cuda.is_available():
        try:
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            if reserved / total > threshold:
                torch.cuda.empty_cache()
        except Exception:
            torch.cuda.empty_cache()

# ----------------------
# Training Runner Class
# ----------------------
class ESLTrainer:
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        model,
        tokenizer,
        batch_size=16,
        in_batch_pairs=128,
        batch_buffer_pairs=128,
        from_buffer_pairs=256,
        buffer_reencode_prob=1/16,
        buffer_size=1024,
        epochs=3,
        lr=2e-5,
        optimizer=None,
        scheduler=None
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.in_batch_pairs = in_batch_pairs
        self.batch_buffer_pairs = batch_buffer_pairs
        self.from_buffer_pairs = from_buffer_pairs
        self.buffer_reencode_prob = buffer_reencode_prob
        self.buffer_size = buffer_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()

        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = scheduler

        self.mem = deque(maxlen=self.buffer_size)

        self._prepare_data()

    def _prepare_data(self):
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)

        collate_fn = get_collate_fn(self.tokenizer)

        self.train_loader = DataLoader(
            ESLDataset(train_df),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        self.val_loader = DataLoader(
            ESLDataset(val_df),
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )

        self.test_loader = DataLoader(
            ESLDataset(test_df),
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )

    def validate(self, k=32):
        self.model.eval()

        # Load and sample k exemplars from train set
        train_df = pd.read_csv(self.train_path)
        scores = train_df['final']
        bins = np.linspace(0, 10, num=21)
        train_df['score_bin'] = pd.cut(scores, bins=bins, labels=False)
        grouped = train_df.groupby('score_bin')
        sampled_df = grouped.apply(lambda x: x.sample(n=min(len(x), max(1, k // len(bins))), random_state=42), include_groups=False)
        sampled_df = sampled_df.reset_index(drop=True)

        remaining_k = k - len(sampled_df)
        if remaining_k > 0:
            extra_samples = train_df.sample(n=remaining_k, random_state=42)
            sampled_df = pd.concat([sampled_df, extra_samples], ignore_index=True)

        sampled_dataset = ESLDataset(sampled_df)
        collate_fn = get_collate_fn(self.tokenizer)
        sampled_loader = DataLoader(sampled_dataset, batch_size=self.batch_size, collate_fn=collate_fn)

        ref_vecs_list = []
        score_ref_list = []
        with torch.no_grad():
            for sample_batch in sampled_loader:
                input_ids_ref = sample_batch['input_ids'].to(self.device)
                attention_mask_ref = sample_batch['attention_mask'].to(self.device)
                question_type_ref = sample_batch['question_type'].to(self.device)
                score_ref = sample_batch['score'].to(self.device)
                vecs = self.model.encode(input_ids_ref, attention_mask_ref, question_type_ref)
                ref_vecs_list.append(vecs)
                score_ref_list.append(score_ref)

        ref_vecs = torch.cat(ref_vecs_list, dim=0)
        score_ref = torch.cat(score_ref_list, dim=0)

        # Load validation set
        val_df = pd.read_csv(self.val_path).reset_index(drop=True)
        val_dataset = ESLDataset(val_df)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=collate_fn)

        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                question_type = batch['question_type'].to(self.device)
                true_scores = batch['score'].to(self.device)

                batch_vecs = self.model.encode(input_ids, attention_mask, question_type)
                deltas = self.model.reg_head(batch_vecs[:, None, :] - ref_vecs[None, :, :])
                deltas = deltas.squeeze(-1)
                pred_scores = (score_ref.unsqueeze(0) + deltas)

                # Simply average (no exclusion)
                avg_pred = pred_scores.mean(dim=1).clamp(0, 10)

                loss = F.mse_loss(avg_pred, true_scores, reduction='sum').item()
                total_loss += loss
                total_count += input_ids.size(0)

        torch.cuda.empty_cache()
        return total_loss / total_count

    def train(self):
        scaler = amp.GradScaler('cuda')
        best_val_loss = float('inf')
        best_state_dict = None

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            total_pairs = 0

            for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                question_type = batch['question_type'].to(self.device)
                scores = batch['score'].to(self.device)
                B = input_ids.size(0)
                if B < 2:
                    continue

                # full fp16-forward for encode and head
                with amp.autocast('cuda'):
                    vecs = self.model.encode(input_ids, attention_mask, question_type)

                    loss_sum = 0.0
                    pairs_count = 0

                    # 1) in-batch pairs
                    diffs = torch.abs(scores.unsqueeze(0) - scores.unsqueeze(1))
                    triu_mask = torch.triu(torch.ones_like(diffs, device=self.device), diagonal=1).bool()
                    valid_diffs = diffs[triu_mask]
                    alpha = 1.5
                    weights = (valid_diffs ** alpha) + 1  # adjust alpha here
                    weights = weights / weights.sum()

                    # Get all triu indices
                    idx_i_all, idx_j_all = torch.triu_indices(B, B, offset=1, device=self.device)
                    perm = torch.multinomial(weights, self.in_batch_pairs, replacement=(idx_i_all.size(0) < self.in_batch_pairs))
                    idx_i, idx_j = idx_i_all[perm], idx_j_all[perm]

                    vecs_a = vecs[idx_i]
                    vecs_b = vecs[idx_j]
                    true_diffs = (scores[idx_i] - scores[idx_j]).clamp(-10, 10)
                    pred_diffs = self.model.reg_head(vecs_a - vecs_b).squeeze(1)
                    loss_sum += self.criterion(pred_diffs, true_diffs) * idx_i.size(0)
                    pairs_count += idx_i.size(0)

                    # 2) batch-buffer pairs
                    if len(self.mem) > 0 and self.batch_buffer_pairs > 0:
                        # Prepare buffer vectors and scores
                        buffer_vecs, buffer_scores = zip(*random.sample(list(self.mem), min(len(self.mem), self.batch_buffer_pairs * 4)))
                        buffer_vecs = torch.stack(buffer_vecs).to(self.device)
                        buffer_scores = torch.tensor([s.item() for s in buffer_scores], device=self.device)

                        # Inverse frequency sampling for batch
                        batch_score_half = torch.round(scores * 2) / 2
                        unique, counts = batch_score_half.unique(return_counts=True)
                        freq = dict(zip(unique.tolist(), counts.tolist()))
                        inv_freq = {k: 1 / (v + 1e-5) for k, v in freq.items()}
                        weights_batch = torch.tensor([inv_freq[x.item()] for x in batch_score_half], device=self.device)
                        weights_batch = weights_batch / weights_batch.sum()

                        # Inverse frequency sampling for buffer
                        buffer_score_half = torch.round(buffer_scores * 2) / 2
                        unique_b, counts_b = buffer_score_half.unique(return_counts=True)
                        freq_b = dict(zip(unique_b.tolist(), counts_b.tolist()))
                        inv_freq_b = {k: 1 / (v + 1e-5) for k, v in freq_b.items()}
                        weights_buffer = torch.tensor([inv_freq_b[x.item()] for x in buffer_score_half], device=self.device)
                        weights_buffer = weights_buffer / weights_buffer.sum()

                        idx_batch = torch.multinomial(weights_batch, self.batch_buffer_pairs, replacement=(B < self.batch_buffer_pairs))
                        idx_buffer = torch.multinomial(weights_buffer, self.batch_buffer_pairs, replacement=(buffer_vecs.size(0) < self.batch_buffer_pairs))

                        vecs_a = vecs[idx_batch]
                        vecs_b = buffer_vecs[idx_buffer]
                        score_diffs = (scores[idx_batch] - buffer_scores[idx_buffer]).clamp(-10, 10)

                        pred_diffs = self.model.reg_head(vecs_a - vecs_b).squeeze(1)
                        loss_sum += self.criterion(pred_diffs, score_diffs) * self.batch_buffer_pairs
                        pairs_count += self.batch_buffer_pairs

                    # 3) buffer pairs
                    # Update buffer
                    vecs_cpu = vecs.detach().cpu()
                    scores_cpu = scores.detach().cpu()
                    for v, s in zip(vecs_cpu, scores_cpu):
                        self.mem.append((v, s))

                    # Sample buffer pairs
                    if len(self.mem) >= self.from_buffer_pairs:
                        score_buckets = defaultdict(list)
                        for v, s in self.mem:
                            # Round to nearest 0.5 instead of 1
                            score_half = round(s.item() * 2) / 2  # scores assumed in [0,10] in 0.5 increments
                            score_buckets[score_half].append((v, s))

                        # Compute bucket weights inversely proportional to frequency
                        all_counts = {k: len(vs) for k, vs in score_buckets.items()}
                        total = sum(all_counts.values())
                        raw_freqs = {k: count / total for k, count in all_counts.items()}
                        alpha = 0.5  # smooth inverse-freq
                        inv_freqs = {k: 1 / (raw_freqs[k] + 1e-5) for k in raw_freqs}
                        norm = sum(v ** alpha for v in inv_freqs.values())
                        probs = {k: (v ** alpha) / norm for k, v in inv_freqs.items()}

                        # Sample score buckets for anchors and partners
                        anchors, partners = [], []
                        for _ in range(self.from_buffer_pairs):
                            sk1 = random.choices(list(probs.keys()), weights=probs.values())[0]
                            sk2 = random.choices(list(probs.keys()), weights=probs.values())[0]

                            ex1 = random.choice(score_buckets[sk1])
                            ex2 = random.choice(score_buckets[sk2])
                            anchors.append(ex1)
                            partners.append(ex2)

                        vecs_a = torch.stack([x[0] for x in anchors]).to(self.device)
                        vecs_b = torch.stack([x[0] for x in partners]).to(self.device)
                        score_diffs = torch.tensor(
                            [x[1].item() - y[1].item() for x, y in zip(anchors, partners)],
                            device=self.device
                        ).clamp(-10, 10)

                        pred_diffs = self.model.reg_head(vecs_a - vecs_b).squeeze(1)
                        loss_sum += self.criterion(pred_diffs, score_diffs) * self.from_buffer_pairs
                        pairs_count += self.from_buffer_pairs

                    loss = loss_sum / pairs_count if pairs_count > 0 else torch.tensor(0.0, device=self.device)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()

                total_loss += loss.item() * pairs_count
                total_pairs += pairs_count
                
                maybe_empty_cache()

            avg_loss = total_loss / total_pairs
            print(f"Epoch {epoch+1}: Train Pairwise Loss = {avg_loss:.4f}")
            val_loss = self.validate()
            print(f"Epoch {epoch+1}: Validation Loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            
            torch.cuda.empty_cache()
            gc.collect()

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print("Loaded best model state from validation.")

    def test(self, k=32):
        self.model.eval()

        train_df = pd.read_csv(self.train_path)
        scores = train_df['final']

        bins = np.linspace(0, 10, num=21)
        train_df['score_bin'] = pd.cut(scores, bins=bins, labels=False)
        grouped = train_df.groupby('score_bin')
        sampled_df = grouped.apply(lambda x: x.sample(n=min(len(x), max(1, k // len(bins))), random_state=42), include_groups=False)
        sampled_df = sampled_df.reset_index(drop=True)

        remaining_k = k - len(sampled_df)
        if remaining_k > 0:
            extra_samples = train_df.sample(n=remaining_k, random_state=42)
            sampled_df = pd.concat([sampled_df, extra_samples], ignore_index=True)

        sampled_dataset = ESLDataset(sampled_df)
        collate_fn = get_collate_fn(self.tokenizer)
        sampled_loader = DataLoader(sampled_dataset, batch_size=self.batch_size, collate_fn=collate_fn)

        ref_vecs_list = []
        score_ref_list = []
        with torch.no_grad():
            for sample_batch in sampled_loader:
                input_ids_ref = sample_batch['input_ids'].to(self.device)
                attention_mask_ref = sample_batch['attention_mask'].to(self.device)
                question_type_ref = sample_batch['question_type'].to(self.device)
                score_ref = sample_batch['score'].to(self.device)
                vecs = self.model.encode(input_ids_ref, attention_mask_ref, question_type_ref)
                ref_vecs_list.append(vecs)
                score_ref_list.append(score_ref)

        ref_vecs = torch.cat(ref_vecs_list, dim=0)
        score_ref = torch.cat(score_ref_list, dim=0)

        test_loss = 0.0
        test_mae = 0.0
        count = 0

        for batch in self.test_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            question_type = batch['question_type'].to(self.device)
            true_scores = batch['score'].to(self.device)

            with torch.no_grad():
                batch_vecs = self.model.encode(input_ids, attention_mask, question_type)
                deltas = self.model.reg_head(batch_vecs[:, None, :] - ref_vecs[None, :, :])
                deltas = deltas.squeeze(-1)
                pred_scores = (score_ref.unsqueeze(0) + deltas)

            # Exclude top-k and bottom-k predictions, then compute mean
            k_exclude = 4
            if pred_scores.size(1) > 2 * k_exclude:
                sorted_preds, _ = torch.sort(pred_scores, dim=1)
                trimmed = sorted_preds[:, k_exclude:-k_exclude]
                avg_pred_trimmed = trimmed.mean(dim=1).clamp(0, 10)
            else:
                avg_pred_trimmed = pred_scores.mean(dim=1).clamp(0, 10)

            test_loss += F.mse_loss(avg_pred_trimmed, true_scores, reduction='sum').item()
            test_mae += torch.abs(avg_pred_trimmed - true_scores).sum().item()
            count += input_ids.size(0)

        avg_test_loss = test_loss / count
        avg_test_mae = test_mae / count

        print(f"Test MSE: {avg_test_loss:.4f}")
        print(f"Test MAE: {avg_test_mae:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    def get_test_loader(self):
        return self.test_loader
    
def get_param_groups(model, base_lr=1e-5, scale_lr=1e-3):
    special_params = []
    base_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'scale' in name or 'alpha' in name:
            special_params.append(param)
        else:
            base_params.append(param)

    return [
        {'params': base_params, 'lr': base_lr},
        {'params': special_params, 'lr': scale_lr}
    ]
    
if __name__ == "__main__":
    model = ESLGradingModel(model_name='Alibaba-NLP/gte-multilingual-base', dropout=0.3)
    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-multilingual-base')

    train_df = pd.read_csv("./data/train_pro.csv")
    batch_size = 16
    in_batch_pairs = 32
    batch_buffer_pairs = 64
    from_buffer_pairs = 256
    buffer_size = 1024
    epochs = 10
    steps_per_epoch = len(train_df) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = 1000

    param_groups = get_param_groups(model, base_lr=1e-4, scale_lr=1e-3)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    trainer = ESLTrainer(
        train_path="./data/train_pro.csv",
        test_path="./data/test_pro.csv",
        val_path="./data/val_pro.csv",
        model=model,
        tokenizer=tokenizer,
        epochs=epochs,
        batch_size=batch_size,
        in_batch_pairs=in_batch_pairs,
        batch_buffer_pairs=batch_buffer_pairs,
        from_buffer_pairs=from_buffer_pairs,
        buffer_size=buffer_size,
        optimizer=optimizer,
        scheduler=scheduler
    )

    trainer.train()
    trainer.test()
    trainer.model.save("./model/model.pth")
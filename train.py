import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torch.amp as amp
from scipy.stats import truncnorm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
import random
from collections import deque, defaultdict, Counter
import os
import gc
import nltk
nltk.download('stopwords')
from text_processing import ALL_STOPWORDS, is_low_content, replace_repeats, most_common_words
from transformers import Wav2Vec2Model
# ----------------------
# Dataset
# ----------------------
import torch
from torch.utils.data import Dataset

def clean_dataframe(df, remove_low_content=True):
    """
    Cleans the dataframe by processing the 'text' field:
    - Applies replace_repeats
    - Optionally removes rows with low content using is_low_content
    """
    # print(f"Rows before cleaning: {len(df)}")
    df = df.copy()
    df['text'] = df['text'].apply(lambda t: replace_repeats(t, k=2, tag="[REPEAT]"))
    if remove_low_content:
        mask = ~df['text'].apply(is_low_content)
        df = df[mask].reset_index(drop=True)
    # df = df[df['final'] >= 3].reset_index(drop=True) # for some testing
    # print(f"Rows after cleaning: {len(df)}")
    # print(df['final'].value_counts().sort_index())
    return df

class ESLDataset(Dataset):
    def __init__(self, dataframe, remove_low_content=True):
        dataframe = clean_dataframe(dataframe, remove_low_content)
        self.text_prefix = "The following is a spoken English response by a non-native speaker. Grade the fluency, grammar, vocabulary, pronunciation, and content based on the transcript below:"
        self.question_type_map = {
            1: "Answer some questions about you personally.",
            2: "Choose one of several options in a situation.",
            3: "Give your opinion about a topic."
        }
        self.question_types = dataframe['question_type'].astype(int).tolist()
        self.scores = dataframe['final'].astype(float).tolist()
        raw_texts = dataframe['text'].tolist()
        self.texts = [
            f"{self.text_prefix} [Question Type: {self.question_type_map.get(qtype, '')}] {t[2:-1]}"
            for t, qtype in zip(raw_texts, self.question_types)
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'score': torch.tensor(self.scores[idx], dtype=torch.float32),
            'question_type': self.question_types[idx]
        }

class InverseScoreSampler(Sampler):
    def __init__(self, dataset, alpha=0.5, replacement=True):
        self.dataset = dataset
        self.replacement = replacement
        self.alpha = alpha # 1 for inverse-frequency sampling, 0 for random sampling

        # Round scores to nearest 0.5 for binning
        binned_scores = [round(float(s) * 2) / 2 for s in dataset.scores]
        counter = Counter(binned_scores)

        # Compute inverse frequency weights
        freqs = np.array([counter[round(float(s) * 2) / 2] for s in dataset.scores], dtype=np.float32)
        self.weights = (1.0 / freqs) ** alpha
        self.weights /= self.weights.sum()  # Normalize to sum to 1

    def __iter__(self):
        n = len(self.dataset)
        indices = np.random.choice(
            np.arange(n), size=n, replace=self.replacement, p=self.weights
        )
        return iter(indices.tolist())

    def __len__(self):
        return len(self.dataset)
    
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

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        pooled = torch.sum(attn_weights * hidden_states, dim=1)  # [B, D]

        if visualize:
            return pooled, attn_weights
        else:
            return pooled

class ESLGradingModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', pooling_dropout=0.3, regression_dropout=0.5, avg_last_k=4):
        super().__init__()
        self.num_types = 3  # question types: 1, 2, 3
        self.pooling_dropout = pooling_dropout
        self.regression_dropout = regression_dropout
        self.avg_last_k = avg_last_k

        # Load encoder and apply dropout
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # config.hidden_dropout_prob = dropout
        # config.attention_probs_dropout_prob = dropout
        config.output_hidden_states=True
        self.encoder = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        hidden_size = self.encoder.config.hidden_size
        self.encoder.gradient_checkpointing_enable()

        # Gated attention pooling
        self.attn_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(), 
            nn.Dropout(pooling_dropout),
            nn.Linear(256, 1, bias=False)
        )
        self.attn_pool = AttentionPooling(hidden_size, attn_proj=self.attn_proj, expected_seq_len=512, dropout=pooling_dropout)

        
        # Thêm sau phần encoder hiện tại
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_hidden_dim = self.audio_encoder.config.output_hidden_size  # 768

        # Audio projection để match với text feature dimension
        self.audio_proj = nn.Linear(self.audio_hidden_dim, self.encoder.config.hidden_size)
        self.audio_norm = nn.LayerNorm(self.encoder.config.hidden_size)

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256, bias=False),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(regression_dropout),
            nn.Linear(256, 128, bias=False),
            nn.GELU(),
            nn.Dropout(regression_dropout),
            nn.Linear(128, 21, bias=False)  # Output 21 logits for classes 0.0 to 10.0
        )

    def encode(self, input_ids, attention_mask, visualize=False):
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
            pooled = self.attn_pool(hidden_states, attention_mask, visualize=visualize)

        return pooled
    
    def forward(self, input_ids, attention_mask):
        # Encode input
        pooled = self.encode(input_ids, attention_mask)

        # Get logits for 21 classes
        logits = self.reg_head(pooled)

        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Score bins from 0.0 to 10.0 in 0.5 increments
        score_bins = torch.linspace(0, 10, steps=21).to(probs.device)

        # Expected score (soft target regression)
        expected_score = (probs * score_bins).sum(dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'expected_score': expected_score
        }

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'pooling_dropout': self.pooling_dropout,
                'regression_dropout': self.regression_dropout,
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
            pooling_dropout=config.get('pooling_dropout', 0.3),
            regression_dropout=config.get('regression_dropout', 0.5),
            avg_last_k=config.get('avg_last_k', 1)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
        
def maybe_empty_cache(threshold=0.93):
    if torch.cuda.is_available():
        try:
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            if reserved / total > threshold:
                torch.cuda.empty_cache()
        except Exception:
            torch.cuda.empty_cache()

def selective_freeze_embedding_layer(model, tokenizer, unfrozen_words):
    """
    Freezes the embedding layer of a transformer model,
    but allows selected tokens (from unfrozen_words) to remain trainable.

    Args:
        model: Hugging Face transformer model (e.g., AutoModel)
        tokenizer: Corresponding tokenizer (e.g., AutoTokenizer)
        unfrozen_words: List or set of words to keep trainable
    """
    # Freeze the entire embedding layer
    embedding_layer = model.embeddings.word_embeddings
    embedding_layer.weight.requires_grad = True  # must stay True for masking
    for param in model.embeddings.parameters():
        param.requires_grad = True  # required for backward hook to work

    # Get token IDs of unfrozen words and all special tokens
    token_ids = set()
    for word in unfrozen_words:
        ids = tokenizer(word, add_special_tokens=False)['input_ids']
        token_ids.update(ids)

    # Add all special token IDs
    if hasattr(tokenizer, "all_special_ids"):
        token_ids.update(tokenizer.all_special_ids)
    else:
        # Fallback for tokenizers without all_special_ids
        for tok in tokenizer.all_special_tokens:
            ids = tokenizer(tok, add_special_tokens=False)['input_ids']
            token_ids.update(ids)

    vocab_size, hidden_size = embedding_layer.weight.shape
    grad_mask = torch.zeros(vocab_size, 1, device=embedding_layer.weight.device)
    for idx in token_ids:
        if idx < vocab_size:
            grad_mask[idx] = 1.0

    # Register gradient hook to zero out updates for frozen tokens
    def hook_fn(grad):
        # grad: [vocab_size, hidden_size]
        return grad * grad_mask

    embedding_layer.weight.register_hook(hook_fn)


def get_class_counts_from_dataframe(df, class_bins):
    """
    Returns counts for each class bin (length = len(class_bins))
    """
    class_to_index = {v: i for i, v in enumerate(class_bins)}
    indices = df['final'].map(class_to_index)
    counts = np.zeros(len(class_bins), dtype=int)
    for idx in indices:
        counts[idx] += 1
    return counts

def get_effective_number_weights(class_counts, beta=0.9999):
    """
    Implements Cui et al. (2019) class-balanced loss weights
    """
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.mean(weights)  # normalize to mean 1
    return torch.tensor(weights, dtype=torch.float32)

# ---- ESLTrainer ----
class ESLTrainer:
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        model,
        tokenizer,
        batch_size=16,
        epochs=3,
        lr=2e-5,
        optimizer=None,
        scheduler=None,
        std=0.3
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.std = std  # Gaussian smoothing std

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.criterion = nn.KLDivLoss(reduction='batchmean')  # use with log_softmax + soft targets

        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        self.scheduler = scheduler

        self._prepare_data()

    def _prepare_data(self):
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)

        collate_fn = get_collate_fn(self.tokenizer)

        sampling_alpha = 0.5    
        train_dataset = ESLDataset(train_df)
        train_sampler = InverseScoreSampler(train_dataset, alpha=sampling_alpha)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn
        )
        class_bins = [i * 0.5 for i in range(21)]  # 0.0 to 10.0 in steps of 0.5
        class_counts = get_class_counts_from_dataframe(train_df, class_bins)
        eff_class_counts = (class_counts + 1) ** (1 - sampling_alpha) # compensate for the sampler; plus one to avoid zeros
        self.loss_weights = get_effective_number_weights(eff_class_counts, beta=0.99).to(self.device)
        self.train_logits = np.log(eff_class_counts)

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

    def _create_soft_targets(self, scores, std=None):
        """
        Generate soft target distributions using a truncated Gaussian centered on each score.
        Slightly skews expected values to the center on edges (0.0 or 10.0).
        
        Args:
            scores (torch.Tensor): shape (B,), scalar scores in [0, 10]
            std (float): Gaussian standard deviation. Defaults to self.std.

        Returns:
            torch.Tensor: shape (B, 21), soft label distributions over 21 bins from 0 to 10.
        """
        if std is None:
            std = self.std

        # Convert to NumPy
        scores_np = scores.cpu().numpy()  # shape (B,)
        B = scores_np.shape[0]

        # Define bin centers: 0.0 to 10.0 in 0.5 steps (21 bins)
        bin_centers = np.linspace(0, 10, 21)  # shape (21,)

        # Prepare output array
        soft_labels = np.zeros((B, 21), dtype=np.float32)

        for i, score in enumerate(scores_np):
            # Scale std based on distance from center (5.0)
            scaled_std = std + random.uniform(-0.05, 0.05)

            # Define truncated Gaussian
            a = (0.0 - score) / scaled_std
            b = (10.0 - score) / scaled_std
            dist = truncnorm(a, b, loc=score, scale=scaled_std)

            # Get normalized probabilities
            probs = dist.pdf(bin_centers)
            probs /= probs.sum()  # Normalize

            soft_labels[i] = probs

        # Convert back to torch tensor
        return torch.from_numpy(soft_labels).to(scores.device)  # shape (B, 21)

    def train(self):
        scaler = amp.GradScaler('cuda')
        best_val_loss = float('inf')
        best_state_dict = None
        
        # Lambdas
        lambda_kl = 0.9
        lambda_mse = 0.1

        stopwords = ALL_STOPWORDS.union(most_common_words(pd.read_csv(self.train_path), 0.05))
        selective_freeze_embedding_layer(self.model.encoder, self.tokenizer, stopwords)

        for epoch in range(self.epochs):
            self.model.train()
            total_kl_loss = 0.0
            total_mse_loss = 0.0
            total_loss = 0.0
            total_batches = 0

            for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                question_type = batch['question_type'].to(self.device)
                true_scores = batch['score'].to(self.device)

                soft_targets = self._create_soft_targets(true_scores)  # (B, 21)

                with amp.autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask)
                
                target_indexes = (true_scores * 2).long().clamp(0, 20)  # 0.0->0, 0.5->1, ..., 10.0->20
                weights = self.loss_weights[target_indexes]

                # KL loss between predicted log probs and soft targets
                logits = outputs['logits']  # (B, 21)
                log_probs = F.log_softmax(logits, dim=-1)
                kl_loss_per_sample = F.kl_div(log_probs, soft_targets, reduction='none').sum(dim=-1) # (B,)
                weighted_kl_loss = (kl_loss_per_sample * weights).sum() / weights.sum()

                # MSE Loss, weighted so that points farther from center contribute more
                pred_scores = outputs['expected_score']  # (B,)
                mse_loss_per_sample = F.mse_loss(pred_scores, true_scores, reduction='none')  # (B,)
                weighted_mse = (mse_loss_per_sample * weights).sum() / weights.sum()
                # Combine losses
                loss = lambda_kl * weighted_kl_loss + lambda_mse * weighted_mse

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()

                total_kl_loss += weighted_kl_loss.item()
                total_mse_loss += weighted_mse.item()
                total_loss += loss.item()
                total_batches += 1

            avg_kl_loss = total_kl_loss / total_batches
            avg_mse_loss = total_mse_loss / total_batches
            avg_loss = total_loss / total_batches
            print(f"Epoch {epoch + 1}: Train KLDiv Loss = {avg_kl_loss:.4f}, Weighted MSE Loss = {avg_mse_loss:.4f}, Total Loss = {avg_loss:.4f}")

            val_w_loss, val_avg_loss = self.validate()
            print(f"Epoch {epoch + 1}: Validation MSE: weighted = {val_w_loss:.4f}, average = {val_avg_loss:.4f}")

            if val_w_loss < best_val_loss:
                best_val_loss = val_w_loss
                best_state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            elif val_w_loss > best_val_loss * 1.1:
                self.model.load_state_dict(best_state_dict)
                print("Current model is too bad; reloading best validation model.")

            torch.cuda.empty_cache()
            gc.collect()

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print("Loaded best model state from validation.")

    def validate(self, alpha=0.5):
        self.model.eval()
        total_loss = 0.0
        total_weight = 0.0  # use weights sum for normalization
        total_per_item_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                question_type = batch['question_type'].to(self.device)
                true_scores = batch['score'].to(self.device)  # (B,)

                with amp.autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask)
                    pred_scores = outputs['expected_score']  # (B,)

                # Calculate frequency of each score in batch
                unique_scores, counts = torch.unique(true_scores, return_counts=True)
                freq_map = {score.item(): count.item() for score, count in zip(unique_scores, counts)}

                # Compute inverse frequency weights with smoothing
                weights = torch.tensor(
                    [((1.0 / freq_map[score.item()]) ** alpha) for score in true_scores],
                    device=self.device
                )

                # Compute weighted MSE loss per example
                per_example_loss = (pred_scores - true_scores) ** 2
                weighted_loss = (weights * per_example_loss).sum().item()

                total_loss += weighted_loss
                total_weight += weights.sum().item()
                total_per_item_loss += per_example_loss.sum().item()
                total_count += input_ids.size(0)

        torch.cuda.empty_cache()
        weighted_avg = total_loss / total_weight if total_weight > 0 else 0.0
        per_item_avg = total_per_item_loss / total_count if total_count > 0 else 0.0
        return weighted_avg, per_item_avg

    def test(self):
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        count = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                question_type = batch['question_type'].to(self.device)
                true_scores = batch['score'].to(self.device)

                with amp.autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask)
                    pred_scores = outputs['expected_score']  # (B,)

                total_loss += F.mse_loss(pred_scores, true_scores, reduction='sum').item()
                total_mae += torch.abs(pred_scores - true_scores).sum().item()
                count += input_ids.size(0)

        avg_loss = total_loss / count
        avg_mae = total_mae / count

        print(f"Test MSE: {avg_loss:.4f}")
        print(f"Test MAE: {avg_mae:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    def get_test_loader(self):
        return self.test_loader
    
def get_param_groups(model, base_lr=1e-5, encoder_lr=1e-6, scale_lr=1e-3):
    special_params = []
    encoder_params = []
    base_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'scale' in name or 'alpha' in name:
            special_params.append(param)
        elif name.startswith('encoder.'):
            encoder_params.append(param)
        else:
            base_params.append(param)

    return [
        {'params': base_params, 'lr': base_lr},
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': special_params, 'lr': scale_lr}
    ]

if __name__ == "__main__":
    model = ESLGradingModel(model_name='Alibaba-NLP/gte-multilingual-base', pooling_dropout=0.3, regression_dropout=0.5)
    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-multilingual-base')

    train_df = pd.read_csv("./data/train_pro.csv")
    batch_size = 16
    epochs = 20
    steps_per_epoch = len(train_df) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = 1000

    param_groups = get_param_groups(model, base_lr=1e-4, encoder_lr=1e-5, scale_lr=1e-3)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=(total_steps - warmup_steps) / (4 * steps_per_epoch)
    )

    trainer = ESLTrainer(
        train_path="./data/train_pro.csv",
        test_path="./data/test_pro.csv",
        val_path="./data/val_pro.csv",
        model=model,
        tokenizer=tokenizer,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    trainer.train()
    trainer.test()
    trainer.model.save("./model/model.pth")
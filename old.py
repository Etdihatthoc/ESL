import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torch.amp as amp
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
from tqdm import tqdm
import os
import gc

# ----------------------
# Dataset
# ----------------------
import torch
from torch.utils.data import Dataset

class ESLDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe['text'].str[2:-1].tolist()
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
    def __init__(self, input_dim, output_dim=64, dropout=0.1, activation=nn.GELU(), use_norm=True):
        super().__init__()
        self.value_proj = nn.Linear(input_dim, output_dim)
        self.gate_proj = nn.Linear(input_dim, output_dim)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(output_dim)
        self.activation = activation
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        v = self.value_proj(hidden_states)                 # [B, T, output_dim]
        g = self.activation(self.gate_proj(hidden_states)) # [B, T, output_dim]
        x = v * g                                          # gated interaction
        if self.use_norm:
            x = self.norm(x)                               # normalize gated output
        if self.dropout: 
            x = self.dropout(x)
        return x                                           # [B, T, output_dim]

class ESLGradingModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout=0.3, avg_last_k=4, prefix_dim=8):
        super().__init__()
        self.num_types = 3  # question types: 1, 2, 3
        self.dropout = dropout
        self.avg_last_k = avg_last_k
        self.prefix_dim = prefix_dim

        # Load encoder and apply dropout
        self.encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True, trust_remote_code=True)
        hidden_size = self.encoder.config.hidden_size
        self.encoder.gradient_checkpointing_enable()

        # Question type encoder: 1 x D vector per type
        self.type_encoder = nn.Embedding(self.num_types, prefix_dim*hidden_size)

        # Gated attention pooling
        self.attn_proj = nn.Sequential(
            GatedFF(hidden_size, output_dim=64, dropout=dropout, activation=nn.Sigmoid()),
            nn.Linear(64, 1, bias=False)
        )
        self.scale = nn.Parameter(torch.tensor(0.2))

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            GatedFF(hidden_size, output_dim=256, dropout=dropout),
            GatedFF(256, output_dim=64, use_norm=False, dropout=dropout),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, question_type):
        batch_size = input_ids.size(0)
        device = input_ids.device
        seq_len = input_ids.size(1)
        hidden_size = self.encoder.config.hidden_size

        # Encode
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
            raw_scores = self.attn_proj(hidden_states)
            scale_factor = self.scale * torch.log(torch.tensor(seq_len, dtype=torch.float32, device=device)).detach()
            scaled_scores = raw_scores * scale_factor
            attn_mask = attention_mask.unsqueeze(-1)
            scaled_scores = scaled_scores.masked_fill(attn_mask == 0, -1e9)

            attn_weights = F.softmax(scaled_scores, dim=1)
            pooled = torch.sum(attn_weights * hidden_states, dim=1)

        out = self.head(pooled).squeeze(1)
        return torch.clamp(out, 0, 10)

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

def maybe_empty_cache(threshold=0.95):
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    usage_ratio = reserved / total
    if usage_ratio > threshold:
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
        epochs=3,
        lr=2e-5,
        optimizer=None,
        scheduler=None,
        pairwise_lambda=30,
        pairwise_margin=-1.5,
        max_pairs=128
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.pairwise_lambda = pairwise_lambda
        self.pairwise_margin = pairwise_margin
        self.max_pairs = max_pairs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.pairwise_loss_fn = nn.MarginRankingLoss(margin=self.pairwise_margin)

        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = scheduler

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

    def _compute_pairwise_loss(self, preds, scores):
        n = preds.size(0)
        if n < 2:
            return 0.0

        idx_i, idx_j = torch.triu_indices(n, n, offset=1).to(preds.device)
        score_diff = scores[idx_i] - scores[idx_j]
        # pred_diff = preds[idx_i] - preds[idx_j]
        rank_labels = torch.sign(score_diff).to(preds.device)  

        valid = (rank_labels != 0)
        idx_i, idx_j = idx_i[valid], idx_j[valid]
        rank_labels = rank_labels[valid]

        if rank_labels.numel() == 0:
            return 0.0

        if idx_i.numel() > self.max_pairs:
            indices = torch.randperm(idx_i.numel(), device=preds.device)[:self.max_pairs]
            idx_i = idx_i[indices]
            idx_j = idx_j[indices]
            rank_labels = rank_labels[indices]

        pred_i = preds[idx_i]
        pred_j = preds[idx_j]

        loss = self.pairwise_loss_fn(pred_i, pred_j, rank_labels)
        return loss * n / rank_labels.numel() # Balance the loss by number of pairs

    def train(self):
        scaler = amp.GradScaler('cuda')  # initialize GradScaler for mixed precision
        best_val_loss = float('inf')
        best_state_dict = None

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            mse_loss = 0.0
            pairwise_loss_value = 0.0

            for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                question_type = batch['question_type'].to(self.device)
                scores = batch['score'].to(self.device)

                self.optimizer.zero_grad()

                with amp.autocast('cuda'):  # automatic mixed precision context
                    outputs = self.model(input_ids, attention_mask, question_type)
                    loss = self.criterion(outputs, scores)
                    pairwise_loss = self._compute_pairwise_loss(outputs, scores)
                    total_loss = loss + self.pairwise_lambda * pairwise_loss

                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if self.scheduler is not None:
                    self.scheduler.step()

                train_loss += total_loss.item() * input_ids.size(0)
                mse_loss += loss.item() * input_ids.size(0)
                pairwise_loss_value += pairwise_loss.item() * input_ids.size(0)

                maybe_empty_cache()

            avg_train_loss = train_loss / len(self.train_loader.dataset)
            avg_mse_loss = mse_loss / len(self.train_loader.dataset)
            avg_pairwise_loss = pairwise_loss_value / len(self.train_loader.dataset)
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, MSE Loss = {avg_mse_loss:.4f}, Pairwise Loss = {avg_pairwise_loss:.4f}")

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    question_type = batch['question_type'].to(self.device)
                    scores = batch['score'].to(self.device)
                    with amp.autocast('cuda'):
                        outputs = self.model(input_ids, attention_mask, question_type)
                        loss = self.criterion(outputs, scores)
                    val_loss += loss.item() * input_ids.size(0)
                    val_mae += torch.abs(outputs - scores).sum().item()

            avg_val_loss = val_loss / len(self.val_loader.dataset)
            avg_val_mae = val_mae / len(self.val_loader.dataset)
            print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}, Val MAE = {avg_val_mae:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

            torch.cuda.empty_cache()
            gc.collect()

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print("Loaded best model parameters from validation.")

        # Testing loop (can remain mostly unchanged, optionally add autocast)
        self.model.eval()
        test_loss = 0.0
        test_mae = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                question_type = batch['question_type'].to(self.device)
                scores = batch['score'].to(self.device)
                with amp.autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask, question_type)
                    loss = self.criterion(outputs, scores)
                test_loss += loss.item() * input_ids.size(0)
                test_mae += torch.abs(outputs - scores).sum().item()
        avg_test_loss = test_loss / len(self.test_loader.dataset)
        avg_test_mae = test_mae / len(self.test_loader.dataset)
        print(f"Test Loss = {avg_test_loss:.4f}, Test MAE = {avg_test_mae:.4f}")

    def get_test_loader(self):
        return self.test_loader
      
if __name__ == "__main__":
    model = ESLGradingModel(model_name='Alibaba-NLP/gte-multilingual-base', dropout=0.3)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-multilingual-base')

    train_df = pd.read_csv("./data/train_pro.csv")
    batch_size = 16
    epochs = 15
    steps_per_epoch = len(train_df) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = 1000

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = WarmupInverseSquareScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps
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
        scheduler=scheduler
    )

    trainer.train()
    trainer.model.save("./model/model.pth")
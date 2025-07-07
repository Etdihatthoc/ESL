import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torch.amp as amp
from scipy.stats import truncnorm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig, Wav2Vec2Model, Wav2Vec2Processor
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
import random
from collections import deque, defaultdict, Counter
import os
import gc
import nltk
import asyncio
from text_processing import ALL_STOPWORDS, is_low_content, replace_repeats, most_common_words
from transformers import Wav2Vec2Model

import wandb
import logging
from datetime import datetime
# ----------------------
# Dataset
# ----------------------
import torch
from torch.utils.data import Dataset
import librosa

# ----------------------
# Audio Processing Functions (from your provided code)
# ----------------------

async def preprocess_audio_wav2vec(absolute_path, processor, sample_rate=16000, num_chunks=10, chunk_length_sec=30):
    """
    Asynchronously preprocess audio file for the Wav2Vec2 model.
    """
    try:
        loop = asyncio.get_event_loop()
        audio_tensor = await loop.run_in_executor(
            None,
            lambda: _process_audio_file(absolute_path, processor, sample_rate, num_chunks, chunk_length_sec)
        )
        return audio_tensor
    except Exception as e:
        print(f"Error in preprocessing audio: {str(e)}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

def _process_audio_file(absolute_path, processor, sample_rate=16000, num_chunks=10, chunk_length_sec=30):
    """Process a single audio file (non-async helper function)."""
    audio, sr = librosa.load(absolute_path, sr=sample_rate)
    audio_chunks = fixed_chunk_audio(audio, sr, num_chunks=num_chunks, chunk_length_sec=chunk_length_sec)
    
    chunk_samples = int(chunk_length_sec * sample_rate)
    processed_chunks = []
    
    for chunk in audio_chunks:
        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
        chunk_tensor = inputs.input_values.squeeze(0)
        
        if chunk_tensor.shape[0] < chunk_samples:
            pad_length = chunk_samples - chunk_tensor.shape[0]
            chunk_tensor = torch.nn.functional.pad(chunk_tensor, (0, pad_length), 'constant', 0)
        elif chunk_tensor.shape[0] > chunk_samples:
            chunk_tensor = chunk_tensor[:chunk_samples]
            
        processed_chunks.append(chunk_tensor)
    
    audio_tensor = torch.stack(processed_chunks)
    del audio, audio_chunks
    gc.collect()
    return audio_tensor

def fixed_chunk_audio(audio, sr, num_chunks=10, chunk_length_sec=30):
    """Cuts audio into exactly num_chunks with each chunk of length chunk_length_sec."""
    chunk_samples = int(chunk_length_sec * sr)
    audio_length = len(audio)
    if audio_length < chunk_samples:
        audio = np.pad(audio, (0, chunk_samples - audio_length), mode='constant')
        audio_length = len(audio)
    
    if num_chunks == 1:
        starts = [0]
    else:
        max_start = audio_length - chunk_samples
        starts = np.linspace(0, max_start, num_chunks, dtype=int)
    
    chunks = []
    for start in starts:
        end = start + chunk_samples
        chunk = audio[start:end]
        chunks.append(chunk)
    return chunks

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

class ESLDatasetWithAudio(Dataset):
    def __init__(self, dataframe, audio_processor=None, remove_low_content=True, num_chunks=10, chunk_length_sec=30):
        """
        Enhanced ESL Dataset that supports both text and audio.
        
        Args:
            dataframe: DataFrame with columns 'text', 'final', 'question_type', 'absolute_path'
            audio_processor: Wav2Vec2Processor instance
            remove_low_content: Whether to remove low content samples
            num_chunks: Number of audio chunks to extract
            chunk_length_sec: Length of each audio chunk in seconds
        """
        dataframe = clean_dataframe(dataframe, remove_low_content)
        self.audio_processor = audio_processor
        self.num_chunks = num_chunks
        self.chunk_length_sec = chunk_length_sec
        
        # Original text processing (unchanged)
        self.text_prefix = "The following is a spoken English response by a non-native speaker. Grade the grammar score based on the transcript below:"
        self.question_type_map = {
            1: "Social Interaction – Answer 3–6 questions about 1–2 familiar topics",
            2: "Solution Discussion – Choose one option from a situation and justify your choice",
            3: "Topic Development – Present a given topic with supporting ideas and answer follow-up questions"
        }
        
        self.question_types = dataframe['question_type'].astype(int).tolist()
        self.scores = dataframe['grammar'].astype(float).tolist()
        raw_texts = dataframe['text'].tolist()
        self.texts = [
            f"{self.text_prefix} [Question Type: {self.question_type_map.get(qtype, '')}] {t[2:-1]}"
            for t, qtype in zip(raw_texts, self.question_types)
        ]
        
        # Audio paths
        dataframe['absolute_path'] = dataframe['absolute_path'].str.replace('/mnt/son_usb/DATA_Vocal', '/media/gpus/Data/DATA_Vocal')
        self.absolute_paths = dataframe['absolute_path'].tolist() if 'absolute_path' in dataframe.columns else [None] * len(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            'text': self.texts[idx],
            'score': torch.tensor(self.scores[idx], dtype=torch.float32),
            'question_type': self.question_types[idx]
        }
        
        # Process audio if available
        if self.absolute_paths[idx] is not None and self.audio_processor is not None:
            try:
                audio_tensor = _process_audio_file(
                    self.absolute_paths[idx], 
                    self.audio_processor,
                    num_chunks=self.num_chunks,
                    chunk_length_sec=self.chunk_length_sec
                )
                item['audio'] = audio_tensor
                item['has_audio'] = True
            except Exception as e:
                print(f"Error processing audio {self.absolute_paths[idx]}: {e}")
                # Create dummy audio tensor if processing fails
                chunk_samples = int(self.chunk_length_sec * 16000)
                item['audio'] = torch.zeros(self.num_chunks, chunk_samples)
                item['has_audio'] = False
        else:
            # Create dummy audio tensor
            chunk_samples = int(self.chunk_length_sec * 16000)
            item['audio'] = torch.zeros(self.num_chunks, chunk_samples)
            item['has_audio'] = False
            
        return item

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

def get_collate_fn_with_audio(tokenizer, max_length=8192):
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        scores = torch.stack([item['score'] for item in batch])
        question_types = torch.tensor([item['question_type'] for item in batch], dtype=torch.long)
        
        # Text encoding
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Audio processing
        has_audio = [item.get('has_audio', False) for item in batch]
        if any(has_audio):
            audios = torch.stack([item['audio'] for item in batch])
        else:
            audios = None

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'score': scores,
            'question_type': question_types,
            'audio': audios,
            'has_audio': has_audio
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

class ESLGradingModelWithAudio(nn.Module):
    def __init__(self, 
                 model_name='bert-base-uncased', 
                 audio_encoder_id="facebook/wav2vec2-base-960h",
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

        # ========== ORIGINAL TEXT PIPELINE (UNCHANGED) ==========
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        text_hidden_size = self.encoder.config.hidden_size
        self.encoder.gradient_checkpointing_enable()

        # Original attention pooling (unchanged)
        self.attn_proj = nn.Sequential(
            nn.Linear(text_hidden_size, 256),
            nn.Tanh(), 
            nn.Dropout(pooling_dropout),
            nn.Linear(256, 1, bias=False)
        )
        self.attn_pool = AttentionPooling(text_hidden_size, attn_proj=self.attn_proj, expected_seq_len=512, dropout=pooling_dropout)

        # ========== NEW AUDIO PIPELINE ==========
        # Audio encoder
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_id)
        self.audio_hidden_dim = self.audio_encoder.config.output_hidden_size
        
        # Audio projection to common space
        self.audio_proj = nn.Linear(self.audio_hidden_dim, d_fuse)
        self.audio_norm = nn.LayerNorm(d_fuse)
        
        # Text projection to common space
        self.text_proj = nn.Linear(text_hidden_size, d_fuse)
        self.text_norm = nn.LayerNorm(d_fuse)
        
        # Cross-attention between audio and text
        self.audio_text_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.attention_norm = nn.LayerNorm(d_fuse)
        
        # ========== ENHANCED REGRESSION HEAD ==========
        # Now takes concatenated text features + fused audio-text features
        self.reg_head = nn.Sequential(
            nn.Linear(text_hidden_size + d_fuse, 256, bias=False),  # text_hidden_size + d_fuse
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(regression_dropout),
            nn.Linear(256, 128, bias=False),
            nn.GELU(),
            nn.Dropout(regression_dropout),
            nn.Linear(128, 21, bias=False)
        )

    def encode_text(self, input_ids, attention_mask, visualize=False):
        """Original text encoding pipeline (unchanged)"""
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states
        k = min(self.avg_last_k, len(all_hidden_states))
        if k == 1:
            hidden_states = all_hidden_states[-1]
        else:
            hidden_states = torch.stack(all_hidden_states[-k:], dim=0).mean(dim=0)
        hidden_states = hidden_states.float()

        with torch.amp.autocast('cuda', enabled=False):
            pooled_text = self.attn_pool(hidden_states, attention_mask, visualize=visualize)

        return pooled_text, hidden_states

    def encode_audio(self, audio):
        """
        Encode audio chunks using Wav2Vec2
        Args:
            audio: Tensor of shape (batch, num_chunks, waveform_len)
        Returns:
            audio_features: Tensor of shape (batch, num_chunks, d_fuse)
        """
        if audio is None:
            return None
            
        batch_size, num_chunks, waveform_len = audio.shape
        device = next(self.parameters()).device
        
        audio_encoder_out = []
        for i in range(num_chunks):
            inp = audio[:, i, :].to(device)
            with torch.no_grad():  # Freeze audio encoder gradients if needed
                out = self.audio_encoder(input_values=inp).last_hidden_state
                audio_encoder_out.append(out.mean(dim=1).detach().cpu())
            
            del inp, out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        audio_features = torch.stack(audio_encoder_out, dim=1).to(device)  # (batch, num_chunks, audio_hidden_dim)
        audio_features = self.audio_proj(audio_features)  # (batch, num_chunks, d_fuse)
        audio_features = self.audio_norm(audio_features)
        
        return audio_features

    def fuse_audio_text(self, text_features, audio_features):
        """
        Fuse audio and text features using cross-attention
        Args:
            text_features: Text hidden states (batch, seq_len, text_hidden_dim)
            audio_features: Audio features (batch, num_chunks, d_fuse)
        Returns:
            fused_features: Fused representation (batch, d_fuse)
        """
        if audio_features is None:
            # Return zero vector if no audio
            return torch.zeros(text_features.size(0), self.d_fuse, device=text_features.device)
        
        # Project text to common space
        text_proj = self.text_proj(text_features)  # (batch, seq_len, d_fuse)
        text_proj = self.text_norm(text_proj)
        
        # Cross-attention: use audio as query, text as key/value
        fused_output, _ = self.audio_text_attention(
            query=audio_features, 
            key=text_proj, 
            value=text_proj
        )
        fused_output = self.attention_norm(fused_output)
        
        # Pool across audio chunks
        fused_vector = fused_output.mean(dim=1)  # (batch, d_fuse)
        
        return fused_vector

    def forward(self, input_ids, attention_mask, audio=None):
        """
        Forward pass with both text and audio
        Args:
            input_ids: Text input ids
            attention_mask: Text attention mask
            audio: Audio tensor (batch, num_chunks, waveform_len) or None
        """
        # Text encoding (original pipeline)
        pooled_text, text_hidden_states = self.encode_text(input_ids, attention_mask)
        
        # Audio encoding (new)
        audio_features = self.encode_audio(audio)
        
        # Audio-text fusion (new)
        fused_features = self.fuse_audio_text(text_hidden_states, audio_features)
        
        # Concatenate text and fused features
        combined_features = torch.cat([pooled_text, fused_features], dim=1)
        
        # Final prediction
        logits = self.reg_head(combined_features)
        probs = torch.softmax(logits, dim=-1)
        score_bins = torch.linspace(0, 10, steps=21).to(probs.device)
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
                'avg_last_k': self.avg_last_k,
                'd_fuse': self.d_fuse
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
            avg_last_k=config.get('avg_last_k', 1),
            d_fuse=config.get('d_fuse', 256)
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
    indices = df['grammar'].map(class_to_index)
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
            collate_fn=collate_fn,
            num_workers=40,  
            pin_memory=True,
            persistent_workers=True
        )
        class_bins = [i * 0.5 for i in range(21)]  # 0.0 to 10.0 in steps of 0.5
        class_counts = get_class_counts_from_dataframe(train_df, class_bins)
        eff_class_counts = (class_counts + 1) ** (1 - sampling_alpha) # compensate for the sampler; plus one to avoid zeros
        self.loss_weights = get_effective_number_weights(eff_class_counts, beta=0.99).to(self.device)
        self.train_logits = np.log(eff_class_counts)

        self.val_loader = DataLoader(
            ESLDataset(val_df),
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=40,  
            pin_memory=True,
            persistent_workers=True
        )

        self.test_loader = DataLoader(
            ESLDataset(test_df),
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=40,  
            pin_memory=True,
            persistent_workers=True
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

class ESLTrainerWithAudio(ESLTrainer):
    def __init__(self, 
                 train_path,
                 val_path,
                 test_path,
                 model,
                 tokenizer,
                 audio_processor=None,
                 batch_size=16,
                 epochs=3,
                 lr=2e-5,
                 optimizer=None,
                 scheduler=None,
                 std=0.3, 
                 logger=None):
        
        self.audio_processor = audio_processor
        self.logger = logger or logging.getLogger(__name__)
        
        # Call parent init but override data preparation
        super().__init__(train_path, val_path, test_path, model, tokenizer, 
                        batch_size, epochs, lr, optimizer, scheduler, std)

    def _prepare_data(self):
        """Override data preparation to include audio"""
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)

        collate_fn = get_collate_fn_with_audio(self.tokenizer)

        sampling_alpha = 0.5    
        train_dataset = ESLDatasetWithAudio(train_df, self.audio_processor)
        train_sampler = InverseScoreSampler(train_dataset, alpha=sampling_alpha)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=40,  
            pin_memory=True,
            persistent_workers=True
        )
        
        # Calculate class weights (same as original)
        class_bins = [i * 0.5 for i in range(21)]
        class_counts = get_class_counts_from_dataframe(train_df, class_bins)
        eff_class_counts = (class_counts + 1) ** (1 - sampling_alpha)
        self.loss_weights = get_effective_number_weights(eff_class_counts, beta=0.99).to(self.device)
        self.train_logits = np.log(eff_class_counts)

        self.val_loader = DataLoader(
            ESLDatasetWithAudio(val_df, self.audio_processor),
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=40,  
            pin_memory=True,
            persistent_workers=True
        )

        self.test_loader = DataLoader(
            ESLDatasetWithAudio(test_df, self.audio_processor),
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=40,  
            pin_memory=True,
            persistent_workers=True
        )

    def train(self):
        """Override training loop to handle audio data"""
        scaler = amp.GradScaler('cuda')
        best_val_loss = float('inf')
        best_val_mae = float('inf')
        best_state_dict = None
        
        lambda_kl = 0.9
        lambda_mse = 0.1

        for epoch in range(self.epochs):
            self.model.train()
            total_kl_loss = 0.0
            total_mse_loss = 0.0
            total_loss = 0.0
            total_mae = 0.0
            total_batches = 0

            for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                audio = batch['audio'].to(self.device) if batch['audio'] is not None else None
                true_scores = batch['score'].to(self.device)

                soft_targets = self._create_soft_targets(true_scores)

                with amp.autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask, audio)
                
                target_indexes = (true_scores * 2).long().clamp(0, 20)
                weights = self.loss_weights[target_indexes]

                # Loss calculation (same as original)
                logits = outputs['logits']
                log_probs = F.log_softmax(logits, dim=-1)
                kl_loss_per_sample = F.kl_div(log_probs, soft_targets, reduction='none').sum(dim=-1)
                weighted_kl_loss = (kl_loss_per_sample * weights).sum() / weights.sum()

                pred_scores = outputs['expected_score']
                mse_loss_per_sample = F.mse_loss(pred_scores, true_scores, reduction='none')
                weighted_mse = (mse_loss_per_sample * weights).sum() / weights.sum()
                
                mae_loss = torch.abs(pred_scores - true_scores).mean()
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
                total_mae += mae_loss.item()
                total_batches += 1

            avg_kl_loss = total_kl_loss / total_batches
            avg_mse_loss = total_mse_loss / total_batches
            avg_loss = total_loss / total_batches
            avg_mae = total_mae / total_batches
            
            train_metrics = {
                "epoch": epoch + 1,
                "train_kl_loss": avg_kl_loss,
                "train_mse_loss": avg_mse_loss,
                "train_total_loss": avg_loss,
                "train_mae": avg_mae
            }
            
            log_message = f"Epoch {epoch + 1}: Train KLDiv Loss = {avg_kl_loss:.4f}, Weighted MSE Loss = {avg_mse_loss:.4f}, Total Loss = {avg_loss:.4f}, MAE = {avg_mae:.4f}"
            print(log_message)
            self.logger.info(log_message)

            val_w_loss, val_avg_loss, val_mae = self.validate()  
            
            val_log_message = f"Epoch {epoch + 1}: Validation MSE: weighted = {val_w_loss:.4f}, average = {val_avg_loss:.4f}, MAE = {val_mae:.4f}"
            print(val_log_message)
            self.logger.info(val_log_message)
            
            # THÊM VAL METRICS
            val_metrics = {
                "val_weighted_mse": val_w_loss,
                "val_avg_mse": val_avg_loss,
                "val_mae": val_mae
            }
            
            # LOG TO WANDB
            wandb.log({**train_metrics, **val_metrics})

            # SỬA LOGIC SAVE BEST MODEL DỰA TRÊN VAL MAE
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_loss = val_w_loss  # Keep this for backward compatibility
                best_state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                
                # SAVE BEST CHECKPOINT
                checkpoint_path = "./model/model_with_audio_bestmae.pth"  # THAY ĐỔI DÒNG NÀY
                os.makedirs("./model", exist_ok=True)
                self.model.save(checkpoint_path)
                
                save_message = f"Best model updated at epoch {epoch + 1} with VAL MAE: {val_mae:.4f} -> saved to {checkpoint_path}"  # SỬA MESSAGE
                print(save_message)
                self.logger.info(save_message)
                
                # THÊM LOG TO WANDB
                wandb.log({"best_val_mae": val_mae, "best_epoch": epoch + 1})
                
            elif val_mae > best_val_mae * 1.15:  # SỬA: DÙNG MAE THAY VÌ LOSS
                self.model.load_state_dict(best_state_dict)
                reload_message = "Current model is too bad; reloading best validation model."
                print(reload_message)
                self.logger.info(reload_message)

            torch.cuda.empty_cache()
            gc.collect()

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            final_message = f"Loaded best model state from validation with MAE: {best_val_mae:.4f}"
            print(final_message)
            self.logger.info(final_message)

    def validate(self, alpha=0.5):
        """Override validation to handle audio data"""
        self.model.eval()
        total_loss = 0.0
        total_weight = 0.0
        total_per_item_loss = 0.0
        total_mae = 0.0  # THÊM DÒNG NÀY
        total_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                audio = batch['audio'].to(self.device) if batch['audio'] is not None else None
                true_scores = batch['score'].to(self.device)

                with amp.autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask, audio)
                    pred_scores = outputs['expected_score']

                unique_scores, counts = torch.unique(true_scores, return_counts=True)
                freq_map = {score.item(): count.item() for score, count in zip(unique_scores, counts)}

                weights = torch.tensor(
                    [((1.0 / freq_map[score.item()]) ** alpha) for score in true_scores],
                    device=self.device
                )

                per_example_loss = (pred_scores - true_scores) ** 2
                weighted_loss = (weights * per_example_loss).sum().item()
                
                
                mae_batch = torch.abs(pred_scores - true_scores).sum().item()

                total_loss += weighted_loss
                total_weight += weights.sum().item()
                total_per_item_loss += per_example_loss.sum().item()
                total_mae += mae_batch 
                total_count += input_ids.size(0)

        torch.cuda.empty_cache()
        weighted_avg = total_loss / total_weight if total_weight > 0 else 0.0
        per_item_avg = total_per_item_loss / total_count if total_count > 0 else 0.0
        mae_avg = total_mae / total_count if total_count > 0 else 0.0 
        
        return weighted_avg, per_item_avg, mae_avg 

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
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Initialize wandb
    wandb.init(
        project="esl-audio-grading",
        name=f"audio_text_model_{timestamp}",
        config={
            "model_name": "Alibaba-NLP/gte-multilingual-base",
            "audio_encoder": "facebook/wav2vec2-base-960h",
            "batch_size": 6,
            "epochs": 5,
            "d_fuse": 256,
            "pooling_dropout": 0.3,
            "regression_dropout": 0.5
        }
    )
    
    # Initialize audio processor
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Initialize enhanced model
    model = ESLGradingModelWithAudio(
        model_name='Alibaba-NLP/gte-multilingual-base', 
        audio_encoder_id="facebook/wav2vec2-base-960h",
        pooling_dropout=0.3, 
        regression_dropout=0.5,
        d_fuse=256
    )
    
    # ====================
    # Load pretrained Wav2Vec2 encoder
    checkpoint_path = "/mnt/disk1/SonDinh/SonDinh/AES_project/speech-score-api_W2V/models/ckpt_pronunciation/ckpt_SWA_pronunciation_wav2vec2model.pth"
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract audio encoder weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter audio encoder weights
        audio_encoder_weights = {}
        for key, value in state_dict.items():
            if key.startswith('audio_encoder.'):
                # Remove 'audio_encoder.' prefix to match current model structure
                new_key = key[len('audio_encoder.'):]
                audio_encoder_weights[new_key] = value
        
        # Load weights into current model's audio encoder
        if audio_encoder_weights:
            model.audio_encoder.load_state_dict(audio_encoder_weights, strict=False)
            print(f"Successfully loaded pretrained Wav2Vec2 encoder from {checkpoint_path}")
            print(f"Loaded {len(audio_encoder_weights)} audio encoder parameters")
        else:
            print("No audio encoder weights found in checkpoint")
            
    except Exception as e:
        print(f"Error loading pretrained audio encoder: {e}")
        print("Continuing with randomly initialized Wav2Vec2 encoder")
    # ====================
    
    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-multilingual-base')

    # Setup training parameters
    train_df = pd.read_csv("./data/Full/train_pro.csv")
    batch_size = 6  # Reduced due to audio memory requirements
    epochs = 5
    steps_per_epoch = len(train_df) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = 500

    param_groups = get_param_groups(model, base_lr=1e-4, encoder_lr=1e-5, scale_lr=1e-3)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=(total_steps - warmup_steps) / (4 * steps_per_epoch)
    )

    # Initialize enhanced trainer
    trainer = ESLTrainerWithAudio(
        train_path="./data/Full/train_pro.csv",
        test_path="./data/Full/test_pro.csv",
        val_path="./data/Full/val_pro.csv",
        model=model,
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger
    )

    trainer.train()
    trainer.test()
    trainer.model.save("./model/model_with_audio_final.pth")
    wandb.finish()
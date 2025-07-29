"""
data.py - Dataset classes and data loading utilities for ESL score classification
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
from collections import Counter
from text_processing import ALL_STOPWORDS, is_low_content, replace_repeats, most_common_words
from scipy.stats import norm
import functools
import librosa
import audiomentations as A
import random
import asyncio
import os
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# wandb.login(key='072fb112587c6b4507f5ec59e575d234c3e22649', relogin=True)

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

def gaussian_soft_label(score, threshold=6.75, sigma=0.5):
    """
    Returns the soft label for a given raw score using a Gaussian distribution
    centered at the threshold.
    """
    z = (score - threshold) / sigma
    return norm.cdf(z)  # P(class 1)

def clean_dataframe(df, remove_low_content=True, filter_scores=True):
    """
    Cleans the dataframe by processing the 'text' field:
    - Applies replace_repeats
    - Optionally removes rows with low content using is_low_content
    """
    df = df.copy()
    df['text'] = df['text'].apply(lambda t: replace_repeats(t, k=2, tag="[REPEAT]"))
    
    if remove_low_content:
        mask = ~df['text'].apply(is_low_content)
        df = df[mask].reset_index(drop=True)
    
    if filter_scores:
        score_column = 'grammar' if 'grammar' in df.columns else 'final'
        # Keep scores between 3.5 and 10 for binary classification
        mask = (df[score_column] >= 3.5) & (df[score_column] <= 10)
        df = df[mask].reset_index(drop=True)
        print(f"After score filtering: {len(df)} samples")
        print(f"Score distribution: {df[score_column].value_counts().sort_index()}")
    
    return df


def convert_score_to_group(score):
    """
    Convert numerical score to group label:
    Group 0: scores 3.5-6.5
    Group 1: scores 7-10
    """
    if 3.5 <= score <= 6.5:
        return 0
    elif 7.0 <= score <= 10.0:
        return 1
    else:
        raise ValueError(f"Score {score} is outside valid range")


class ESLBinaryDataset(Dataset):
    """
    Dataset for binary classification of ESL scores into two groups,
    with audio support.
    """
    def __init__(self, dataframe, audio_processor=None, remove_low_content=True, 
                 num_chunks=10, chunk_length_sec=30, is_train=False):
        dataframe = clean_dataframe(dataframe, remove_low_content, filter_scores=True)
        self.audio_processor = audio_processor
        self.num_chunks = num_chunks
        self.chunk_length_sec = chunk_length_sec
        self.is_train = is_train

        self.text_prefix = "The following is a spoken English response by a non-native speaker. Classify the proficiency level based on the transcript below:"
        self.question_type_map = {
            1: "Social Interaction: Answer several questions about familiar topics",
            2: "Solution Discussion: Choose one option from a situation and justify your choice",
            3: "Topic Development: Present a given topic with supporting ideas and answer follow-up questions"
        }

        self.question_types = dataframe['question_type'].astype(int).tolist()
        raw_scores = dataframe['grammar'].astype(float).tolist()
        self.groups = [convert_score_to_group(score) for score in raw_scores]
        self.raw_scores = raw_scores
        self.soft_labels = [gaussian_soft_label(score) for score in raw_scores]

        raw_texts = dataframe['text'].tolist()
        self.texts = [
            f"{self.text_prefix} [Question Type: {self.question_type_map.get(qtype, '')}] {t[2:-1]}"
            for t, qtype in zip(raw_texts, self.question_types)
        ]

        # Audio paths
        if 'absolute_path' in dataframe.columns:
            dataframe['absolute_path'] = dataframe['absolute_path'].str.replace(
                "/mnt/son_usb/DATA_Vocal", "/media/gpus/Data/DATA_Vocal"
            )
            self.absolute_paths = dataframe['absolute_path'].tolist()
        else:
            self.absolute_paths = [None] * len(self.texts)
        
        # Audio augmentations (only used during training)
        if self.is_train:
            self.noise_aug = A.AddGaussianNoise(
                min_amplitude=0.001, 
                max_amplitude=0.015, 
                p=1.0
            )
            self.speed_aug = A.TimeStretch(
                min_rate=0.8, 
                max_rate=1.25, 
                p=1.0
            )
            self.pitch_aug = A.PitchShift(
                min_semitones=-4, 
                max_semitones=4, 
                p=1.0
            )

        group_counts = Counter(self.groups)
        print(f"Group distribution - Group 0 (3.5-6.5): {group_counts[0]}, Group 1 (7-10): {group_counts[1]}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            'text': self.texts[idx],
            'group': torch.tensor(self.groups[idx], dtype=torch.long),
            'raw_score': torch.tensor(self.raw_scores[idx], dtype=torch.float32),
            'soft_label': torch.tensor(self.soft_labels[idx], dtype=torch.float32),
            'question_type': self.question_types[idx]
        }

        # Process audio if available
        if self.absolute_paths[idx] is not None and self.audio_processor is not None:
            try:
                # Load raw audio first (before processing)
                audio, sr = librosa.load(self.absolute_paths[idx], sr=16000)
                
                # Apply augmentations during training
                if self.is_train:
                    if random.random() < 0.7:
                        audio = self.noise_aug(samples=audio, sample_rate=sr)
                    if random.random() < 0.7:
                        audio = self.speed_aug(samples=audio, sample_rate=sr)
                    if random.random() < 0.7:
                        audio = self.pitch_aug(samples=audio, sample_rate=sr)
                
                # Now process the augmented audio
                audio_chunks = fixed_chunk_audio(audio, sr, num_chunks=self.num_chunks, 
                                                chunk_length_sec=self.chunk_length_sec)
                
                chunk_samples = int(self.chunk_length_sec * sr)
                processed_chunks = []
                
                for chunk in audio_chunks:
                    inputs = self.audio_processor(chunk, sampling_rate=sr, return_tensors="pt")
                    chunk_tensor = inputs.input_values.squeeze(0)
                    
                    if chunk_tensor.shape[0] < chunk_samples:
                        pad_length = chunk_samples - chunk_tensor.shape[0]
                        chunk_tensor = torch.nn.functional.pad(chunk_tensor, (0, pad_length), 'constant', 0)
                    elif chunk_tensor.shape[0] > chunk_samples:
                        chunk_tensor = chunk_tensor[:chunk_samples]
                        
                    processed_chunks.append(chunk_tensor)
                
                audio_tensor = torch.stack(processed_chunks)
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


class BalancedSampler(Sampler):
    """
    Balanced sampler to handle class imbalance in binary classification
    """
    def __init__(self, dataset, replacement=True):
        self.dataset = dataset
        self.replacement = replacement
        
        # Count samples per group
        group_counts = Counter(dataset.groups)
        total_samples = len(dataset)
        
        # Calculate weights for each sample (inverse frequency)
        self.weights = []
        for group in dataset.groups:
            # Weight inversely proportional to group frequency
            weight = total_samples / (2 * group_counts[group])
            self.weights.append(weight)
        
        self.weights = np.array(self.weights, dtype=np.float32)
        self.weights /= self.weights.sum()  # Normalize
        
        print(f"Group 0 weight: {self.weights[dataset.groups.index(0)]:.4f}")
        print(f"Group 1 weight: {self.weights[dataset.groups.index(1)]:.4f}")

    def __iter__(self):
        n = len(self.dataset)
        indices = np.random.choice(
            np.arange(n), size=n, replace=self.replacement, p=self.weights
        )
        return iter(indices.tolist())

    def __len__(self):
        return len(self.dataset)


def binary_collate_fn(batch, tokenizer, max_length=8192):
    texts = [item['text'] for item in batch]
    groups = torch.stack([item['group'] for item in batch])
    raw_scores = torch.stack([item['raw_score'] for item in batch])
    soft_labels = torch.stack([item['soft_label'] for item in batch])
    question_types = torch.tensor([item['question_type'] for item in batch], dtype=torch.long)

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
        'group': groups,
        'raw_score': raw_scores,
        'soft_label': soft_labels,
        'question_type': question_types,
        'audio': audios,
        'has_audio': has_audio
    }


def create_data_loaders(train_path, val_path, test_path, tokenizer, batch_size=32, use_balanced_sampling=True):
    """
    Create data loaders for train, validation, and test sets
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    collate_fn = functools.partial(binary_collate_fn, tokenizer=tokenizer)
    
    # Create datasets
    train_dataset = ESLBinaryDataset(train_df)
    val_dataset = ESLBinaryDataset(val_df)
    test_dataset = ESLBinaryDataset(test_df)
    
    # Create samplers
    if use_balanced_sampling:
        train_sampler = BalancedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=40,
            pin_memory=True,
            persistent_workers=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=40,
            pin_memory=True,
            persistent_workers=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=40,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=40,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset
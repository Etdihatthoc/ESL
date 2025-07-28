"""
data.py - Dataset classes and data loading utilities for ESL score classification
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
from collections import Counter
from text_processing import ALL_STOPWORDS, is_low_content, replace_repeats, most_common_words


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
    Dataset for binary classification of ESL scores into two groups
    """
    def __init__(self, dataframe, remove_low_content=True):
        dataframe = clean_dataframe(dataframe, remove_low_content, filter_scores=True)
        
        self.text_prefix = "The following is a spoken English response by a non-native speaker. Classify the proficiency level based on the transcript below:"
        self.question_type_map = {
            1: "Social Interaction: Answer several questions about familiar topics",
            2: "Solution Discussion: Choose one option from a situation and justify your choice",
            3: "Topic Development: Present a given topic with supporting ideas and answer follow-up questions"
        }
        
        self.question_types = dataframe['question_type'].astype(int).tolist()
        
        # Convert scores to binary groups
        raw_scores = dataframe['grammar'].astype(float).tolist()
        self.groups = [convert_score_to_group(score) for score in raw_scores]
        self.raw_scores = raw_scores  # Keep original scores for analysis
        
        # Process texts
        raw_texts = dataframe['text'].tolist()
        self.texts = [
            f"{self.text_prefix} [Question Type: {self.question_type_map.get(qtype, '')}] {t[2:-1]}"
            for t, qtype in zip(raw_texts, self.question_types)
        ]
        
        # Print group distribution
        group_counts = Counter(self.groups)
        print(f"Group distribution - Group 0 (3.5-6.5): {group_counts[0]}, Group 1 (7-10): {group_counts[1]}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'group': torch.tensor(self.groups[idx], dtype=torch.long),
            'raw_score': torch.tensor(self.raw_scores[idx], dtype=torch.float32),
            'question_type': self.question_types[idx]
        }


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


def get_binary_collate_fn(tokenizer, max_length=8192):
    """
    Collate function for binary classification
    """
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        groups = torch.stack([item['group'] for item in batch])
        raw_scores = torch.stack([item['raw_score'] for item in batch])
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
            'group': groups,
            'raw_score': raw_scores,
            'question_type': question_types
        }
    
    return collate_fn


def create_data_loaders(train_path, val_path, test_path, tokenizer, batch_size=32, use_balanced_sampling=True):
    """
    Create data loaders for train, validation, and test sets
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    collate_fn = get_binary_collate_fn(tokenizer)
    
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
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset
"""
Data samplers for ESL grading training
"""

import torch
import numpy as np
from torch.utils.data import Sampler
from collections import Counter
from typing import Iterator, List


class InverseScoreSampler(Sampler):
    """
    Sampler that uses inverse frequency weighting to balance score distribution
    """
    def __init__(self, dataset, alpha: float = 0.5, replacement: bool = True):
        """
        Args:
            dataset: Dataset with 'scores' attribute
            alpha: Sampling power (1 for inverse-frequency, 0 for random)
            replacement: Whether to sample with replacement
        """
        self.dataset = dataset
        self.replacement = replacement
        self.alpha = alpha

        # Round scores to nearest 0.5 for binning
        binned_scores = [round(float(s) * 2) / 2 for s in dataset.scores]
        counter = Counter(binned_scores)
        
        print(f"Score distribution before sampling:")
        for score in sorted(counter.keys()):
            print(f"  Score {score}: {counter[score]} samples")

        # Compute inverse frequency weights
        freqs = np.array([counter[round(float(s) * 2) / 2] for s in dataset.scores], dtype=np.float32)
        self.weights = (1.0 / freqs) ** alpha
        self.weights /= self.weights.sum()  # Normalize to sum to 1
        
        # Print effective sampling distribution
        effective_dist = Counter()
        for i, score in enumerate(binned_scores):
            effective_dist[score] += self.weights[i]
        
        print(f"Effective sampling distribution (alpha={alpha}):")
        for score in sorted(effective_dist.keys()):
            print(f"  Score {score}: {effective_dist[score]:.4f}")

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        indices = np.random.choice(
            np.arange(n), size=n, replace=self.replacement, p=self.weights
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return len(self.dataset)


class StratifiedBatchSampler(Sampler):
    """
    Sampler that ensures each batch has diverse score representation
    """
    def __init__(self, dataset, batch_size: int, drop_last: bool = True):
        """
        Args:
            dataset: Dataset with 'scores' attribute
            batch_size: Size of each batch
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by score bins
        self.score_to_indices = {}
        for idx, score in enumerate(dataset.scores):
            binned_score = round(float(score) * 2) / 2
            if binned_score not in self.score_to_indices:
                self.score_to_indices[binned_score] = []
            self.score_to_indices[binned_score].append(idx)
        
        self.scores = list(self.score_to_indices.keys())
        self.num_batches = len(dataset) // batch_size if drop_last else (len(dataset) + batch_size - 1) // batch_size
        
        print(f"StratifiedBatchSampler: {len(self.scores)} unique scores, {self.num_batches} batches")

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each score group
        shuffled_indices = {}
        for score, indices in self.score_to_indices.items():
            shuffled_indices[score] = np.random.permutation(indices).tolist()
        
        # Create stratified batches
        for batch_idx in range(self.num_batches):
            batch = []
            
            # Try to include samples from different scores
            scores_cycle = iter(np.random.permutation(self.scores))
            
            while len(batch) < self.batch_size:
                try:
                    score = next(scores_cycle)
                    if shuffled_indices[score]:
                        batch.append(shuffled_indices[score].pop())
                except StopIteration:
                    # If we've cycled through all scores, start again
                    scores_cycle = iter(np.random.permutation(self.scores))
                    # If no more samples available, break
                    if not any(shuffled_indices.values()):
                        break
            
            if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                yield batch

    def __len__(self) -> int:
        return self.num_batches


class WeightedRandomSampler(Sampler):
    """
    Simple weighted random sampler with optional replacement
    """
    def __init__(self, weights: List[float], num_samples: int, replacement: bool = True):
        """
        Args:
            weights: Sampling weights for each sample
            num_samples: Number of samples to draw
            replacement: Whether to sample with replacement
        """
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(self.weights, self.num_samples, self.replacement)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def create_balanced_sampler(dataset, 
                          strategy: str = "inverse_score",
                          alpha: float = 0.5,
                          batch_size: int = 32) -> Sampler:
    """
    Factory function to create different types of samplers
    
    Args:
        dataset: Dataset with scores attribute
        strategy: "inverse_score", "stratified", "weighted_random", or "random"
        alpha: Alpha parameter for inverse score sampling
        batch_size: Batch size for stratified sampling
        
    Returns:
        Sampler instance
    """
    if strategy == "inverse_score":
        return InverseScoreSampler(dataset, alpha=alpha)
    elif strategy == "stratified":
        return StratifiedBatchSampler(dataset, batch_size=batch_size)
    elif strategy == "weighted_random":
        # Calculate inverse frequency weights
        binned_scores = [round(float(s) * 2) / 2 for s in dataset.scores]
        counter = Counter(binned_scores)
        weights = [(1.0 / counter[round(float(s) * 2) / 2]) ** alpha for s in dataset.scores]
        return WeightedRandomSampler(weights, len(dataset))
    else:  # random
        return torch.utils.data.RandomSampler(dataset)


def analyze_sampling_distribution(sampler: Sampler, dataset, num_epochs: int = 1):
    """
    Analyze the distribution of samples produced by a sampler
    """
    print(f"\nAnalyzing sampling distribution for {num_epochs} epoch(s)...")
    
    sample_counts = Counter()
    score_counts = Counter()
    
    for epoch in range(num_epochs):
        for idx in sampler:
            sample_counts[idx] += 1
            score = round(float(dataset.scores[idx]) * 2) / 2
            score_counts[score] += 1
    
    # Print sample frequency distribution
    print("Sample frequency distribution:")
    freq_dist = Counter(sample_counts.values())
    for freq, count in sorted(freq_dist.items()):
        print(f"  {count} samples seen {freq} time(s)")
    
    # Print score distribution
    print("Score distribution in sampled data:")
    total_samples = sum(score_counts.values())
    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        percentage = count / total_samples * 100
        print(f"  Score {score}: {count} samples ({percentage:.1f}%)")
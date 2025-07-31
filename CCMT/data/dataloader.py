"""
Custom DataLoader and collate functions for CCMT multimodal data
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class CCMTCollator:
    """
    Custom collate function for CCMT multimodal data
    Handles batching of audio, English text, Vietnamese text, and targets
    """
    
    def __init__(
        self,
        audio_encoder: Optional[Any] = None,
        english_encoder: Optional[Any] = None,
        vietnamese_encoder: Optional[Any] = None,
        max_audio_length: int = 480000,  # 30 seconds at 16kHz
        pad_audio: bool = True,
        return_raw_text: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize collator
        
        Args:
            audio_encoder: Audio encoder model (Wav2Vec2, etc.)
            english_encoder: English text encoder (BERT, etc.)
            vietnamese_encoder: Vietnamese text encoder (PhoBERT, etc.)
            max_audio_length: Maximum audio length in samples
            pad_audio: Whether to pad audio to max_audio_length
            return_raw_text: Whether to return raw text in addition to encoded
            device: Device to place tensors on
        """
        self.audio_encoder = audio_encoder
        self.english_encoder = english_encoder
        self.vietnamese_encoder = vietnamese_encoder
        self.max_audio_length = max_audio_length
        self.pad_audio = pad_audio
        self.return_raw_text = return_raw_text
        self.device = torch.device(device)
        
        # Set encoders to eval mode
        if self.audio_encoder:
            self.audio_encoder.eval()
        if self.english_encoder:
            self.english_encoder.eval()
        if self.vietnamese_encoder:
            self.vietnamese_encoder.eval()
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples
        
        Args:
            batch: List of sample dictionaries from dataset
            
        Returns:
            Batched data dictionary
        """
        batch_size = len(batch)
        
        # Separate data by modality
        audio_data = [sample['audio'] for sample in batch]
        english_texts = [sample['english_text'] for sample in batch]
        vietnamese_texts = [sample['vietnamese_text'] for sample in batch]
        targets = [sample['target'] for sample in batch]
        scores = [sample['score'] for sample in batch]
        metadata = [sample['metadata'] for sample in batch]
        
        # Process audio
        audio_batch = self._collate_audio(audio_data)
        
        # Process text
        english_batch = self._collate_text(english_texts, self.english_encoder, "english")
        vietnamese_batch = self._collate_text(vietnamese_texts, self.vietnamese_encoder, "vietnamese")
        
        # Process targets
        targets_batch = torch.tensor(targets, dtype=torch.long if isinstance(targets[0], int) else torch.float)
        scores_batch = torch.tensor(scores, dtype=torch.float)
        
        # Create result dictionary
        result = {
            'targets': targets_batch.to(self.device),
            'scores': scores_batch.to(self.device),
            'metadata': metadata,
            'batch_size': batch_size
        }
        
        # Add audio data
        if audio_batch is not None:
            result['audio'] = audio_batch.to(self.device)
            if self.audio_encoder:
                result['audio_encoded'] = self._encode_audio(audio_batch)
        
        # Add text data
        if english_batch is not None:
            result['english_text'] = english_batch
            if self.return_raw_text:
                result['english_text_raw'] = english_texts
        
        if vietnamese_batch is not None:
            result['vietnamese_text'] = vietnamese_batch
            if self.return_raw_text:
                result['vietnamese_text_raw'] = vietnamese_texts
        
        # Create CCMT input if all encoders available
        if all([self.audio_encoder, self.english_encoder, self.vietnamese_encoder]):
            ccmt_input = self._create_ccmt_input(result)
            result['ccmt_input'] = ccmt_input
        
        return result
    
    def _collate_audio(self, audio_data: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Collate audio data into batch tensor"""
        if not audio_data or audio_data[0] is None:
            return None
        
        batch_size = len(audio_data)
        
        if self.pad_audio:
            # Pad all audio to same length
            padded_audio = torch.zeros(batch_size, self.max_audio_length)
            
            for i, audio in enumerate(audio_data):
                if audio is not None:
                    audio_len = min(len(audio), self.max_audio_length)
                    padded_audio[i, :audio_len] = audio[:audio_len]
            
            return padded_audio
        else:
            # Stack audio (assumes same length)
            try:
                return torch.stack(audio_data)
            except Exception as e:
                logger.warning(f"Failed to stack audio, using padding: {e}")
                return self._collate_audio_with_padding(audio_data)
    
    def _collate_audio_with_padding(self, audio_data: List[torch.Tensor]) -> torch.Tensor:
        """Fallback audio collation with padding"""
        max_len = max(len(audio) for audio in audio_data if audio is not None)
        batch_size = len(audio_data)
        
        padded_audio = torch.zeros(batch_size, max_len)
        
        for i, audio in enumerate(audio_data):
            if audio is not None:
                audio_len = min(len(audio), max_len)
                padded_audio[i, :audio_len] = audio[:audio_len]
        
        return padded_audio
    
    def _collate_text(
        self, 
        texts: List[str], 
        encoder: Optional[Any], 
        modality: str
    ) -> Optional[torch.Tensor]:
        """Collate and optionally encode text data"""
        if not texts or not encoder:
            return None
        
        try:
            # Tokenize texts
            if hasattr(encoder, 'tokenize_texts'):
                tokenized = encoder.tokenize_texts(texts)
            else:
                # Fallback for different encoder interfaces
                tokenized = encoder.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
            
            # Move to device
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            # Encode if encoder available
            with torch.no_grad():
                if hasattr(encoder, 'forward'):
                    encoded = encoder(**tokenized)
                else:
                    # Alternative encoding method
                    encoded = encoder.encode_texts(texts)
                    if isinstance(encoded, torch.Tensor):
                        encoded = encoded.to(self.device)
            
            return encoded
            
        except Exception as e:
            logger.error(f"Failed to encode {modality} text: {e}")
            return None
    
    def _encode_audio(self, audio_batch: torch.Tensor) -> torch.Tensor:
        """Encode audio batch using audio encoder"""
        try:
            with torch.no_grad():
                audio_encoded = self.audio_encoder(audio_batch.to(self.device))
            return audio_encoded
        except Exception as e:
            logger.error(f"Failed to encode audio: {e}")
            # Return dummy encoding
            batch_size = audio_batch.shape[0]
            return torch.zeros(batch_size, 100, 768, device=self.device)
    
    def _create_ccmt_input(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """Create concatenated input for CCMT model"""
        try:
            # Get encoded features
            english_features = batch_data.get('english_text')
            vietnamese_features = batch_data.get('vietnamese_text')  
            audio_features = batch_data.get('audio_encoded')
            
            if None in [english_features, vietnamese_features, audio_features]:
                logger.warning("Missing modality features for CCMT input")
                # Return dummy input
                batch_size = batch_data['batch_size']
                return torch.zeros(batch_size, 300, 768, device=self.device)
            
            # Concatenate features: [English, Vietnamese, Audio]
            ccmt_input = torch.cat([
                english_features,
                vietnamese_features,
                audio_features
            ], dim=1)  # Concatenate along sequence dimension
            
            return ccmt_input
            
        except Exception as e:
            logger.error(f"Failed to create CCMT input: {e}")
            # Return dummy input
            batch_size = batch_data['batch_size']
            return torch.zeros(batch_size, 300, 768, device=self.device)


class CCMTDataLoader(DataLoader):
    """
    Custom DataLoader for CCMT with additional functionality
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        # CCMT-specific parameters
        audio_encoder: Optional[Any] = None,
        english_encoder: Optional[Any] = None,
        vietnamese_encoder: Optional[Any] = None,
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize CCMT DataLoader
        
        Args:
            dataset: CCMT dataset
            audio_encoder: Audio encoder for on-the-fly encoding
            english_encoder: English text encoder
            vietnamese_encoder: Vietnamese text encoder
            device: Device for computations
            **kwargs: Additional DataLoader arguments
        """
        # Create collate function if not provided
        if collate_fn is None:
            collate_fn = CCMTCollator(
                audio_encoder=audio_encoder,
                english_encoder=english_encoder,
                vietnamese_encoder=vietnamese_encoder,
                device=device
            )
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            **kwargs
        )
        
        self.device = torch.device(device)
        self.audio_encoder = audio_encoder
        self.english_encoder = english_encoder
        self.vietnamese_encoder = vietnamese_encoder
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get statistics about batches"""
        batch_sizes = []
        audio_lengths = []
        text_lengths = []
        
        for batch in self:
            batch_sizes.append(batch['batch_size'])
            
            if 'audio' in batch:
                audio_lengths.extend([torch.sum(audio != 0).item() for audio in batch['audio']])
            
            if 'english_text_raw' in batch:
                text_lengths.extend([len(text.split()) for text in batch['english_text_raw']])
            
            # Only process a few batches for statistics
            if len(batch_sizes) >= 10:
                break
        
        stats = {
            'avg_batch_size': np.mean(batch_sizes),
            'avg_audio_length': np.mean(audio_lengths) if audio_lengths else 0,
            'avg_text_length': np.mean(text_lengths) if text_lengths else 0,
        }
        
        return stats


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    audio_encoder: Optional[Any] = None,
    english_encoder: Optional[Any] = None,
    vietnamese_encoder: Optional[Any] = None,
    device: str = "cpu",
    pin_memory: bool = True,
    **kwargs
) -> Tuple[CCMTDataLoader, Optional[CCMTDataLoader], Optional[CCMTDataLoader]]:
    """
    Create train/validation/test dataloaders
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        audio_encoder: Audio encoder
        english_encoder: English text encoder
        vietnamese_encoder: Vietnamese text encoder
        device: Device for data
        pin_memory: Whether to pin memory
        **kwargs: Additional DataLoader arguments
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    
    # Shared parameters for all loaders
    shared_params = {
        'num_workers': num_workers,
        'audio_encoder': audio_encoder,
        'english_encoder': english_encoder,
        'vietnamese_encoder': vietnamese_encoder,
        'device': device,
        'pin_memory': pin_memory and torch.cuda.is_available(),
        'multiprocessing_context': 'spawn',  # Add this line
        **kwargs
    }
    
    # Create train loader
    train_loader = CCMTDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **shared_params
    )
    
    # Create validation loader
    val_loader = None
    if val_dataset is not None:
        val_loader = CCMTDataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **shared_params
        )
    
    # Create test loader
    test_loader = None
    if test_dataset is not None:
        test_loader = CCMTDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **shared_params
        )
    
    logger.info(f"Created dataloaders: train_batches={len(train_loader)}, "
                f"val_batches={len(val_loader) if val_loader else 0}, "
                f"test_batches={len(test_loader) if test_loader else 0}")
    
    return train_loader, val_loader, test_loader


class BalancedBatchSampler:
    """
    Balanced batch sampler for imbalanced datasets
    Ensures each batch has roughly equal representation of classes
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_classes: int = 21,
        samples_per_class: Optional[int] = None
    ):
        """
        Initialize balanced sampler
        
        Args:
            dataset: Dataset with classification targets
            batch_size: Desired batch size
            num_classes: Number of classes
            samples_per_class: Samples per class per batch (default: batch_size // num_classes)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class or max(1, batch_size // num_classes)
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset[idx]
            target = sample['target']
            if isinstance(target, torch.Tensor):
                target = target.item()
            self.class_indices[target].append(idx)
        
        # Calculate number of batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = min_class_size // self.samples_per_class
        
        logger.info(f"BalancedBatchSampler: {self.num_batches} batches, "
                    f"{self.samples_per_class} samples per class")
    
    def __iter__(self):
        """Generate balanced batches"""
        for _ in range(self.num_batches):
            batch_indices = []
            
            for class_idx in range(self.num_classes):
                if class_idx in self.class_indices:
                    class_indices = self.class_indices[class_idx]
                    selected = random.sample(class_indices, 
                                           min(self.samples_per_class, len(class_indices)))
                    batch_indices.extend(selected)
            
            # Shuffle within batch
            random.shuffle(batch_indices)
            
            # Truncate to exact batch size if needed
            if len(batch_indices) > self.batch_size:
                batch_indices = batch_indices[:self.batch_size]
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches


def create_balanced_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    num_classes: int = 21,
    **dataloader_kwargs
) -> CCMTDataLoader:
    """
    Create dataloader with balanced sampling
    
    Args:
        dataset: Classification dataset
        batch_size: Batch size
        num_classes: Number of classes
        **dataloader_kwargs: Additional DataLoader arguments
    
    Returns:
        Balanced CCMTDataLoader
    """
    batch_sampler = BalancedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        num_classes=num_classes
    )
    
    return CCMTDataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        **dataloader_kwargs
    )


# Memory management utilities
class DataLoaderMemoryManager:
    """Utilities for managing memory usage in dataloaders"""
    
    @staticmethod
    def estimate_memory_usage(
        batch_size: int,
        audio_length: int = 480000,  # 30s at 16kHz
        sequence_length: int = 100,
        feature_dim: int = 768,
        num_modalities: int = 3
    ) -> Dict[str, float]:
        """
        Estimate memory usage for a batch
        
        Returns memory usage in MB
        """
        # Audio: batch_size * audio_length * 4 bytes (float32)
        audio_memory = batch_size * audio_length * 4 / (1024**2)
        
        # Text features: batch_size * sequence_length * feature_dim * 4 bytes * num_modalities
        text_memory = batch_size * sequence_length * feature_dim * 4 * num_modalities / (1024**2)
        
        # Additional overhead (gradients, intermediate values)
        overhead = (audio_memory + text_memory) * 0.5
        
        total_memory = audio_memory + text_memory + overhead
        
        return {
            'audio_memory_mb': audio_memory,
            'text_memory_mb': text_memory,
            'overhead_mb': overhead,
            'total_memory_mb': total_memory
        }
    
    @staticmethod
    def recommend_batch_size(
        available_memory_gb: float = 8.0,
        audio_length: int = 480000,
        sequence_length: int = 100,
        feature_dim: int = 768
    ) -> int:
        """Recommend batch size based on available memory"""
        available_memory_mb = available_memory_gb * 1024
        
        # Leave 20% buffer for other operations
        usable_memory_mb = available_memory_mb * 0.8
        
        # Estimate memory per sample
        single_sample_memory = DataLoaderMemoryManager.estimate_memory_usage(
            batch_size=1,
            audio_length=audio_length,
            sequence_length=sequence_length,
            feature_dim=feature_dim
        )['total_memory_mb']
        
        recommended_batch_size = int(usable_memory_mb / single_sample_memory)
        
        # Ensure minimum batch size of 1 and maximum reasonable size
        recommended_batch_size = max(1, min(recommended_batch_size, 64))
        
        logger.info(f"Recommended batch size: {recommended_batch_size} "
                    f"(estimated {single_sample_memory:.2f} MB per sample)")
        
        return recommended_batch_size


# Example usage and testing
if __name__ == "__main__":
    print("Testing CCMT DataLoader...")
    
    # This would require actual models and data to test properly
    # For now, just test the memory estimation
    memory_usage = DataLoaderMemoryManager.estimate_memory_usage(batch_size=16)
    print(f"Estimated memory usage for batch_size=16: {memory_usage}")
    
    recommended_batch_size = DataLoaderMemoryManager.recommend_batch_size()
    print(f"Recommended batch size: {recommended_batch_size}")
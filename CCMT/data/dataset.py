"""
CCMT Dataset classes for English speaking scoring task
"""

import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import logging
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CCMTDataset(Dataset):
    """
    Base CCMT dataset for multimodal (audio + text) data
    """
    
    def __init__(
        self,
        data_path: Union[str, pd.DataFrame],
        audio_processor: Optional[Any] = None,
        text_processor: Optional[Any] = None,
        translator: Optional[Any] = None,
        augmentation: Optional[Any] = None,
        task_type: str = "classification",  # or "regression"
        max_audio_length: float = 30.0,  # seconds
        cache_translations: bool = True,
        preload_audio: bool = False
    ):
        """
        Initialize CCMT dataset
        
        Args:
            data_path: Path to CSV file or pandas DataFrame
            audio_processor: Audio preprocessing pipeline
            text_processor: Text preprocessing pipeline  
            translator: English to Vietnamese translator
            augmentation: Data augmentation pipeline
            task_type: "classification" or "regression"
            max_audio_length: Maximum audio length in seconds
            cache_translations: Whether to cache translations
            preload_audio: Whether to preload audio data (memory intensive)
        """
        self.task_type = task_type
        self.max_audio_length = max_audio_length
        self.cache_translations = cache_translations
        self.preload_audio = preload_audio
        
        # Load data
        if isinstance(data_path, str):
            self.data = pd.read_csv(data_path)
        elif isinstance(data_path, pd.DataFrame):
            self.data = data_path.copy()
        else:
            raise ValueError(f"data_path must be str or DataFrame, got {type(data_path)}")
        self.data['absolute_path'] = self.data['absolute_path'].str.replace("/mnt/son_usb/DATA_Vocal","/media/gpus/Data/DATA_Vocal")
        # Validate required columns
        required_columns = ['absolute_path', 'text']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Set up processors
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.translator = translator
        self.augmentation = augmentation
        
        # Translation cache
        self.translation_cache = {} if cache_translations else None
        
        # Preloaded audio data
        self.audio_cache = {} if preload_audio else None
        
        # Initialize score mapping for classification
        if task_type == "classification":
            self._init_score_mapping()
        
        # Preload data if requested
        if preload_audio:
            self._preload_audio_data()
        
        logger.info(f"Initialized dataset with {len(self.data)} samples")
    
    def _init_score_mapping(self):
        """Initialize score to class mapping for classification"""
        # Scores: 0, 0.5, 1.0, 1.5, ..., 10.0 = 21 classes
        self.score_to_class = {i * 0.5: i for i in range(21)}
        self.class_to_score = {i: i * 0.5 for i in range(21)}
    
    def _preload_audio_data(self):
        """Preload all audio data into memory"""
        logger.info("Preloading audio data...")
        for idx in range(len(self.data)):
            audio_path = self.data.iloc[idx]['absolute_path']
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                self.audio_cache[idx] = (waveform, sample_rate)
            except Exception as e:
                logger.warning(f"Failed to load audio {audio_path}: {e}")
                self.audio_cache[idx] = None
        logger.info("Audio preloading complete")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample
        
        Returns:
            Dictionary with keys:
            - audio: torch.Tensor
            - english_text: str
            - vietnamese_text: str  
            - score: float (original score)
            - target: int (class) or float (regression target)
            - metadata: dict with additional info
        """
        try:
            row = self.data.iloc[idx]
            
            # Load audio
            audio = self._load_audio(idx, row['absolute_path'])
            
            # Process text
            english_text = self._process_english_text(row['text'])
            vietnamese_text = self._get_vietnamese_translation(english_text)
            
            # Get target
            target = self._get_target(row)
            
            # Create sample
            sample = {
                'audio': audio,
                'english_text': english_text,
                'vietnamese_text': vietnamese_text,
                'score': self._get_score(row),
                'target': target,
                'metadata': {
                    'index': idx,
                    'audio_path': row['absolute_path'],
                    'vocabulary': row.get('vocabulary', None),
                    'grammar': row.get('grammar', None), 
                    'content': row.get('content', None)
                }
            }
            
            # Apply augmentation if available
            if self.augmentation is not None:
                sample = self.augmentation(sample)
            
            return sample
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return dummy sample to avoid breaking training
            return self._get_dummy_sample(idx)
    
    def _load_audio(self, idx: int, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio"""
        try:
            # Use cached audio if available
            if self.audio_cache is not None and idx in self.audio_cache:
                cached_data = self.audio_cache[idx]
                if cached_data is not None:
                    waveform, sample_rate = cached_data
                else:
                    return torch.zeros(1, int(16000 * self.max_audio_length))
            else:
                # Load audio from file
                waveform, sample_rate = torchaudio.load(audio_path)
            
            # Process audio if processor available
            if self.audio_processor is not None:
                waveform = self.audio_processor(waveform, sample_rate)
            else:
                # Basic processing
                waveform = self._basic_audio_processing(waveform, sample_rate)
            
            return waveform
            
        except Exception as e:
            logger.warning(f"Failed to load audio {audio_path}: {e}")
            # Return silent audio
            return torch.zeros(1, int(16000 * self.max_audio_length))
    
    def _basic_audio_processing(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Basic audio processing when no processor is provided"""
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Truncate or pad to max length
        max_samples = int(16000 * self.max_audio_length)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform.squeeze(0)  # Remove channel dimension
    
    def _process_english_text(self, text: str) -> str:
        """Process English text"""
        if self.text_processor is not None:
            return self.text_processor.process_english(text)
        else:
            # Basic cleaning
            return text.strip()
    
    def _get_vietnamese_translation(self, english_text: str) -> str:
        """Get Vietnamese translation of English text"""
        if self.translator is None:
            return english_text  # Return original if no translator
        
        # Check cache first
        if self.translation_cache is not None and english_text in self.translation_cache:
            return self.translation_cache[english_text]
        
        try:
            vietnamese_text = self.translator.translate_single(english_text)
            
            # Cache translation
            if self.translation_cache is not None:
                self.translation_cache[english_text] = vietnamese_text
            
            return vietnamese_text
            
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return english_text  # Return original on failure
    
    def _get_score(self, row: pd.Series) -> float:
        """Extract score from row"""
        # Try different possible score columns
        score_columns = ['score', 'vocabulary', 'grammar', 'content']
        for col in score_columns:
            if col in row and pd.notna(row[col]):
                return float(row[col])
        
        # Default score if none found
        logger.warning(f"No score found in row, using default 5.0")
        return 5.0
    
    def _get_target(self, row: pd.Series) -> Union[int, float]:
        """Get target for training (class index or regression value)"""
        score = self._get_score(row)
        
        if self.task_type == "classification":
            # Convert score to class index
            # Clamp score to valid range [0, 10]
            score = max(0.0, min(10.0, score))
            # Round to nearest 0.5
            rounded_score = round(score * 2) / 2
            return self.score_to_class[rounded_score]
        else:  # regression
            # Clamp to [0, 10] range
            return max(0.0, min(10.0, score))
    
    def _get_dummy_sample(self, idx: int) -> Dict[str, Any]:
        """Create dummy sample for error cases"""
        return {
            'audio': torch.zeros(int(16000 * self.max_audio_length)),
            'english_text': "Error loading sample",
            'vietnamese_text': "Lỗi tải mẫu",
            'score': 5.0,
            'target': 10 if self.task_type == "classification" else 5.0,  # Middle class/score
            'metadata': {
                'index': idx,
                'audio_path': "ERROR",
                'vocabulary': None,
                'grammar': None,
                'content': None
            }
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data"""
        if self.task_type != "classification":
            raise ValueError("Class weights only available for classification tasks")
        
        # Count samples per class
        class_counts = torch.zeros(21)
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            target = self._get_target(row)
            class_counts[target] += 1
        
        # Calculate weights (inverse frequency)
        total_samples = class_counts.sum()
        class_weights = total_samples / (21 * class_counts + 1e-8)  # Add epsilon to avoid division by zero
        
        return class_weights
    
    def get_score_distribution(self) -> Dict[str, Any]:
        """Get distribution of scores in dataset"""
        scores = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            score = self._get_score(row)
            scores.append(score)
        
        scores = np.array(scores)
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'distribution': scores.tolist()
        }


class SpeakingScoringDataset(CCMTDataset):
    """
    Specialized dataset for English speaking scoring task
    """
    
    def __init__(
        self,
        csv_path: str,
        audio_processor: Optional[Any] = None,
        text_processor: Optional[Any] = None,
        translator: Optional[Any] = None,
        augmentation: Optional[Any] = None,
        task_type: str = "classification",
        score_column: str = "vocabulary",  # Which column to use as target
        max_audio_length: float = 30.0,
        **kwargs
    ):
        """
        Initialize speaking scoring dataset
        
        Args:
            csv_path: Path to CSV with columns: absolute_path, vocabulary, grammar, content, text
            score_column: Which column to use as target score
            **kwargs: Additional arguments passed to CCMTDataset
        """
        self.score_column = score_column
        
        super().__init__(
            data_path=csv_path,
            audio_processor=audio_processor,
            text_processor=text_processor,
            translator=translator,
            augmentation=augmentation,
            task_type=task_type,
            max_audio_length=max_audio_length,
            **kwargs
        )
        
        # Validate score column exists
        if score_column not in self.data.columns:
            raise ValueError(f"Score column '{score_column}' not found in data")
        
        logger.info(f"Using '{score_column}' as target score")
    
    def _get_score(self, row: pd.Series) -> float:
        """Extract score from specified column"""
        if pd.isna(row[self.score_column]):
            logger.warning(f"NaN score in column '{self.score_column}', using default 5.0")
            return 5.0
        return float(row[self.score_column])


def create_dataset_splits(
    csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify_column: Optional[str] = None,
    **dataset_kwargs
) -> Tuple[SpeakingScoringDataset, SpeakingScoringDataset, SpeakingScoringDataset]:
    """
    Create train/validation/test splits from CSV data
    
    Args:
        csv_path: Path to CSV file
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining data)
        random_state: Random seed
        stratify_column: Column to stratify splits on
        **dataset_kwargs: Arguments passed to dataset constructor
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Load data
    data = pd.read_csv(csv_path)
    
    # Create stratification labels if requested
    stratify = None
    if stratify_column and stratify_column in data.columns:
        # Bin continuous scores for stratification
        stratify = pd.cut(data[stratify_column], bins=5, labels=False)
    
    # First split: train+val vs test
    train_val_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    # Second split: train vs val
    if val_size > 0:
        # Adjust validation size relative to remaining data
        adjusted_val_size = val_size / (1 - test_size)
        
        # Re-stratify if needed
        stratify_train_val = None
        if stratify is not None:
            train_val_indices = train_val_data.index
            stratify_train_val = stratify[train_val_indices]
        
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=stratify_train_val
        )
    else:
        train_data = train_val_data
        val_data = pd.DataFrame()  # Empty validation set
    
    # Create datasets
    train_dataset = SpeakingScoringDataset(train_data, **dataset_kwargs)
    
    if len(val_data) > 0:
        val_dataset = SpeakingScoringDataset(val_data, **dataset_kwargs)
    else:
        val_dataset = None
    
    test_dataset = SpeakingScoringDataset(test_data, **dataset_kwargs)
    
    logger.info(f"Created splits: train={len(train_dataset)}, "
                f"val={len(val_dataset) if val_dataset else 0}, "
                f"test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def save_dataset_info(dataset: CCMTDataset, save_path: str):
    """Save dataset information and statistics"""
    info = {
        'dataset_size': len(dataset),
        'task_type': dataset.task_type,
        'score_distribution': dataset.get_score_distribution(),
        'data_columns': list(dataset.data.columns),
    }
    
    if dataset.task_type == "classification":
        info['class_weights'] = dataset.get_class_weights().tolist()
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Dataset info saved to {save_path}")


def load_dataset_from_config(config: Dict[str, Any]) -> CCMTDataset:
    """Load dataset from configuration dictionary"""
    dataset_type = config.get('dataset_type', 'SpeakingScoringDataset')
    
    if dataset_type == 'SpeakingScoringDataset':
        return SpeakingScoringDataset(**config['dataset_params'])
    elif dataset_type == 'CCMTDataset':
        return CCMTDataset(**config['dataset_params'])
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    # Example usage
    print("Testing dataset loading...")
    
    # Create dummy CSV for testing
    dummy_data = pd.DataFrame({
        'absolute_path': ['/path/to/audio1.wav', '/path/to/audio2.wav'],
        'text': ['Hello how are you today', 'The weather is nice'],
        'vocabulary': [7.5, 8.0],
        'grammar': [7.0, 8.5],
        'content': [6.5, 7.5]
    })
    
    try:
        dataset = SpeakingScoringDataset(
            dummy_data,
            task_type="classification",
            score_column="vocabulary"
        )
        
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        print(f"Score distribution: {dataset.get_score_distribution()}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
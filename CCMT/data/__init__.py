"""
CCMT Data Package
Contains dataset, dataloader, preprocessing, and augmentation utilities
"""

from .dataset import (
    CCMTDataset,
    SpeakingScoringDataset,
    create_dataset_splits,
    save_dataset_info,
    load_dataset_from_config
)
from .dataloader import (
    CCMTDataLoader,
    CCMTCollator,
    create_dataloaders
)
from .preprocessing import (
    AudioPreprocessor,
    TextPreprocessor,
    MultimodalPreprocessor,
    PreprocessingPipeline
)
from .augmentation import (
    AudioAugmentation,
    TextAugmentation,
    MultimodalAugmentation,
    create_augmentation_pipeline
)

__all__ = [
    # Dataset classes
    'CCMTDataset',
    'SpeakingScoringDataset', 
    'create_dataset_splits',
    'save_dataset_info',
    'load_dataset_from_config',
    
    # DataLoader classes
    'CCMTDataLoader',
    'CCMTCollator',
    'create_dataloaders',
    
    # Preprocessing classes
    'AudioPreprocessor',
    'TextPreprocessor', 
    'MultimodalPreprocessor',
    'PreprocessingPipeline',
    
    # Augmentation classes
    'AudioAugmentation',
    'TextAugmentation',
    'MultimodalAugmentation',
    'create_augmentation_pipeline'
]
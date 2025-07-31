"""
Data augmentation strategies for CCMT multimodal data
"""

import torch
import torchaudio
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from abc import ABC, abstractmethod
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class BaseAugmentation(ABC):
    """Base class for data augmentation"""
    
    def __init__(self, probability: float = 0.5):
        """
        Initialize augmentation
        
        Args:
            probability: Probability of applying this augmentation
        """
        self.probability = probability
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Apply augmentation to data"""
        pass
    
    def should_apply(self) -> bool:
        """Check if augmentation should be applied"""
        return random.random() < self.probability


class AudioAugmentation:
    """
    Audio augmentation techniques for speech data
    """
    
    def __init__(
        self,
        noise_prob: float = 0.3,
        noise_factor: float = 0.05,
        pitch_shift_prob: float = 0.3,
        pitch_shift_range: Tuple[float, float] = (-2.0, 2.0),
        speed_change_prob: float = 0.3,
        speed_range: Tuple[float, float] = (0.9, 1.1),
        volume_change_prob: float = 0.4,
        volume_range: Tuple[float, float] = (0.7, 1.3),
        time_masking_prob: float = 0.2,
        time_mask_ratio: float = 0.05,
        freq_masking_prob: float = 0.2,
        freq_mask_ratio: float = 0.1,
        reverb_prob: float = 0.1,
        sample_rate: int = 16000
    ):
        """
        Initialize audio augmentation
        
        Args:
            noise_prob: Probability of adding noise
            noise_factor: Noise intensity factor
            pitch_shift_prob: Probability of pitch shifting
            pitch_shift_range: Range of pitch shift in semitones
            speed_change_prob: Probability of speed change
            speed_range: Range of speed multipliers
            volume_change_prob: Probability of volume change
            volume_range: Range of volume multipliers
            time_masking_prob: Probability of time masking
            time_mask_ratio: Ratio of time to mask
            freq_masking_prob: Probability of frequency masking
            freq_mask_ratio: Ratio of frequencies to mask
            reverb_prob: Probability of adding reverb
            sample_rate: Audio sample rate
        """
        self.noise_prob = noise_prob
        self.noise_factor = noise_factor
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.speed_change_prob = speed_change_prob
        self.speed_range = speed_range
        self.volume_change_prob = volume_change_prob
        self.volume_range = volume_range
        self.time_masking_prob = time_masking_prob
        self.time_mask_ratio = time_mask_ratio
        self.freq_masking_prob = freq_masking_prob
        self.freq_mask_ratio = freq_mask_ratio
        self.reverb_prob = reverb_prob
        self.sample_rate = sample_rate
        
        # Pre-computed transforms for efficiency
        self.pitch_shift_transforms = {}
        self.speed_change_transforms = {}
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply audio augmentations
        
        Args:
            audio: Audio tensor (samples,)
            
        Returns:
            Augmented audio tensor
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension
        
        # Apply augmentations in sequence
        audio = self._add_noise(audio)
        audio = self._change_volume(audio)
        audio = self._change_speed(audio)
        audio = self._shift_pitch(audio)
        audio = self._apply_time_masking(audio)
        audio = self._add_reverb(audio)
        
        return audio.squeeze(0)  # Remove channel dimension
    
    def _add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add random noise to audio"""
        if random.random() < self.noise_prob:
            noise = torch.randn_like(audio) * self.noise_factor
            # Scale noise relative to audio RMS
            audio_rms = torch.sqrt(torch.mean(audio**2))
            noise = noise * audio_rms
            audio = audio + noise
        return audio
    
    def _change_volume(self, audio: torch.Tensor) -> torch.Tensor:
        """Change audio volume"""
        if random.random() < self.volume_change_prob:
            volume_factor = random.uniform(*self.volume_range)
            audio = audio * volume_factor
        return audio
    
    def _change_speed(self, audio: torch.Tensor) -> torch.Tensor:
        """Change audio speed without changing pitch"""
        if random.random() < self.speed_change_prob:
            speed_factor = random.uniform(*self.speed_range)
            
            try:
                # Use torchaudio's speed perturbation
                if speed_factor not in self.speed_change_transforms:
                    self.speed_change_transforms[speed_factor] = torchaudio.transforms.Speed(
                        orig_freq=self.sample_rate, factor=speed_factor
                    )
                
                transform = self.speed_change_transforms[speed_factor]
                audio = transform(audio)
                
            except Exception as e:
                logger.warning(f"Speed change failed: {e}")
        
        return audio
    
    def _shift_pitch(self, audio: torch.Tensor) -> torch.Tensor:
        """Shift audio pitch"""
        if random.random() < self.pitch_shift_prob:
            pitch_shift = random.uniform(*self.pitch_shift_range)
            
            try:
                # Simple pitch shifting using resampling
                # This is approximate but fast
                pitch_factor = 2**(pitch_shift/12.0)
                new_sample_rate = int(self.sample_rate * pitch_factor)
                
                # Resample to change pitch
                resampler = torchaudio.transforms.Resample(
                    orig_freq=self.sample_rate,
                    new_freq=new_sample_rate
                )
                audio_resampled = resampler(audio)
                
                # Resample back to original rate
                resampler_back = torchaudio.transforms.Resample(
                    orig_freq=new_sample_rate,
                    new_freq=self.sample_rate
                )
                audio = resampler_back(audio_resampled)
                
            except Exception as e:
                logger.warning(f"Pitch shift failed: {e}")
        
        return audio
    
    def _apply_time_masking(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply time masking (zero out random time segments)"""
        if random.random() < self.time_masking_prob:
            seq_len = audio.shape[1]
            mask_len = int(seq_len * self.time_mask_ratio)
            
            if mask_len > 0:
                mask_start = random.randint(0, seq_len - mask_len)
                audio[:, mask_start:mask_start + mask_len] = 0
        
        return audio
    
    def _add_reverb(self, audio: torch.Tensor) -> torch.Tensor:
        """Add simple reverb effect"""
        if random.random() < self.reverb_prob:
            try:
                # Simple reverb using delay and decay
                delay_samples = random.randint(800, 1600)  # 50-100ms at 16kHz
                decay_factor = random.uniform(0.1, 0.3)
                
                # Create delayed version
                delayed = torch.zeros_like(audio)
                if delay_samples < audio.shape[1]:
                    delayed[:, delay_samples:] = audio[:, :-delay_samples] * decay_factor
                
                # Mix with original
                audio = audio + delayed
                
            except Exception as e:
                logger.warning(f"Reverb failed: {e}")
        
        return audio
    
    def apply_spec_augment(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram"""
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension
        
        # Frequency masking
        if random.random() < self.freq_masking_prob:
            freq_mask = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=int(spectrogram.shape[1] * self.freq_mask_ratio)
            )
            spectrogram = freq_mask(spectrogram)
        
        # Time masking
        if random.random() < self.time_masking_prob:
            time_mask = torchaudio.transforms.TimeMasking(
                time_mask_param=int(spectrogram.shape[2] * self.time_mask_ratio)
            )
            spectrogram = time_mask(spectrogram)
        
        return spectrogram.squeeze(0)  # Remove batch dimension


class TextAugmentation:
    """
    Text augmentation techniques for speech transcripts
    """
    
    def __init__(
        self,
        synonym_replacement_prob: float = 0.2,
        random_insertion_prob: float = 0.1,
        random_swap_prob: float = 0.1,
        random_deletion_prob: float = 0.1,
        back_translation_prob: float = 0.1,
        paraphrasing_prob: float = 0.1,
        typo_injection_prob: float = 0.05,
        max_changes: int = 3
    ):
        """
        Initialize text augmentation
        
        Args:
            synonym_replacement_prob: Probability of synonym replacement
            random_insertion_prob: Probability of random word insertion
            random_swap_prob: Probability of random word swapping
            random_deletion_prob: Probability of random word deletion
            back_translation_prob: Probability of back-translation
            paraphrasing_prob: Probability of paraphrasing
            typo_injection_prob: Probability of injecting typos
            max_changes: Maximum number of changes per text
        """
        self.synonym_replacement_prob = synonym_replacement_prob
        self.random_insertion_prob = random_insertion_prob
        self.random_swap_prob = random_swap_prob
        self.random_deletion_prob = random_deletion_prob
        self.back_translation_prob = back_translation_prob
        self.paraphrasing_prob = paraphrasing_prob
        self.typo_injection_prob = typo_injection_prob
        self.max_changes = max_changes
        
        # Common synonyms for frequent words
        self.synonyms = {
            'good': ['great', 'excellent', 'nice', 'fine', 'wonderful'],
            'bad': ['poor', 'terrible', 'awful', 'horrible'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'mini', 'compact'],
            'happy': ['glad', 'joyful', 'cheerful', 'pleased'],
            'sad': ['unhappy', 'sorrowful', 'depressed', 'gloomy'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'gradual', 'leisurely'],
            'easy': ['simple', 'effortless', 'straightforward'],
            'hard': ['difficult', 'challenging', 'tough', 'complex']
        }
        
        # Common filler words for insertion
        self.filler_words = ['um', 'uh', 'like', 'you know', 'I mean', 'actually', 'basically']
        
        # Common typo patterns
        self.typo_patterns = [
            ('th', 't'),   # "the" -> "te"
            ('ing', 'in'), # "running" -> "runnin"
            ('er', 'a'),   # "better" -> "betta"
            ('you', 'u'),  # "you" -> "u"
            ('are', 'r'),  # "are" -> "r"
        ]
    
    def __call__(self, text: str) -> str:
        """
        Apply text augmentations
        
        Args:
            text: Input text string
            
        Returns:
            Augmented text string
        """
        if not text or len(text.strip()) == 0:
            return text
        
        words = text.split()
        if len(words) == 0:
            return text
        
        # Apply augmentations
        augmented_words = words.copy()
        changes_made = 0
        
        # Synonym replacement
        if random.random() < self.synonym_replacement_prob and changes_made < self.max_changes:
            augmented_words = self._synonym_replacement(augmented_words)
            changes_made += 1
        
        # Random insertion
        if random.random() < self.random_insertion_prob and changes_made < self.max_changes:
            augmented_words = self._random_insertion(augmented_words)
            changes_made += 1
        
        # Random swap
        if random.random() < self.random_swap_prob and changes_made < self.max_changes:
            augmented_words = self._random_swap(augmented_words)
            changes_made += 1
        
        # Random deletion
        if random.random() < self.random_deletion_prob and changes_made < self.max_changes:
            augmented_words = self._random_deletion(augmented_words)
            changes_made += 1
        
        # Typo injection
        if random.random() < self.typo_injection_prob and changes_made < self.max_changes:
            augmented_words = self._inject_typos(augmented_words)
            changes_made += 1
        
        augmented_text = ' '.join(augmented_words)
        
        # Back-translation (more complex, requires external service)
        if random.random() < self.back_translation_prob:
            augmented_text = self._back_translate(augmented_text)
        
        return augmented_text
    
    def _synonym_replacement(self, words: List[str]) -> List[str]:
        """Replace words with synonyms"""
        new_words = words.copy()
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self.synonyms:
                synonym = random.choice(self.synonyms[word_lower])
                # Preserve original case
                if word.isupper():
                    synonym = synonym.upper()
                elif word.istitle():
                    synonym = synonym.capitalize()
                new_words[i] = synonym
                break  # Only replace one word
        return new_words
    
    def _random_insertion(self, words: List[str]) -> List[str]:
        """Insert random filler words"""
        if len(words) == 0:
            return words
        
        insert_pos = random.randint(0, len(words))
        filler = random.choice(self.filler_words)
        
        new_words = words[:insert_pos] + [filler] + words[insert_pos:]
        return new_words
    
    def _random_swap(self, words: List[str]) -> List[str]:
        """Swap two random words"""
        if len(words) < 2:
            return words
        
        idx1, idx2 = random.sample(range(len(words)), 2)
        new_words = words.copy()
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return new_words
    
    def _random_deletion(self, words: List[str]) -> List[str]:
        """Delete a random word"""
        if len(words) <= 1:
            return words
        
        delete_idx = random.randint(0, len(words) - 1)
        new_words = words[:delete_idx] + words[delete_idx + 1:]
        return new_words
    
    def _inject_typos(self, words: List[str]) -> List[str]:
        """Inject common typos"""
        if len(words) == 0:
            return words
        
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]
        
        # Apply typo pattern
        for pattern, replacement in self.typo_patterns:
            if pattern in word.lower():
                new_word = word.lower().replace(pattern, replacement, 1)
                # Preserve original case
                if word.isupper():
                    new_word = new_word.upper()
                elif word.istitle():
                    new_word = new_word.capitalize()
                
                new_words = words.copy()
                new_words[word_idx] = new_word
                return new_words
        
        return words
    
    def _back_translate(self, text: str) -> str:
        """
        Back-translation augmentation
        Note: This is a placeholder - real implementation would use translation services
        """
        # Placeholder implementation
        # In practice, you would:
        # 1. Translate English -> Vietnamese
        # 2. Translate Vietnamese -> English
        # This often produces paraphrases
        
        return text  # Return original for now
    
    def augment_with_asr_errors(self, text: str) -> str:
        """Simulate ASR transcription errors"""
        # Common ASR error patterns
        asr_errors = [
            ('ing', 'in'),
            ('the', 'a'),
            ('to', 'too'),
            ('there', 'their'),
            ('have', 'of'),
            ('going to', 'gonna'),
            ('want to', 'wanna')
        ]
        
        augmented = text
        for original, error in asr_errors:
            if random.random() < 0.1:  # 10% chance for each error
                augmented = re.sub(r'\b' + original + r'\b', error, augmented, count=1)
        
        return augmented


class MultimodalAugmentation:
    """
    Coordinated augmentation for multimodal data
    Ensures consistent augmentation across audio and text modalities
    """
    
    def __init__(
        self,
        audio_augmentation: Optional[AudioAugmentation] = None,
        text_augmentation: Optional[TextAugmentation] = None,
        sync_augmentation: bool = True,
        preserve_duration: bool = True
    ):
        """
        Initialize multimodal augmentation
        
        Args:
            audio_augmentation: Audio augmentation pipeline
            text_augmentation: Text augmentation pipeline
            sync_augmentation: Whether to synchronize augmentations
            preserve_duration: Whether to preserve audio duration
        """
        self.audio_aug = audio_augmentation or AudioAugmentation()
        self.text_aug = text_augmentation or TextAugmentation()
        self.sync_augmentation = sync_augmentation
        self.preserve_duration = preserve_duration
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply multimodal augmentation to sample
        
        Args:
            sample: Sample dictionary with 'audio', 'english_text', 'vietnamese_text', etc.
            
        Returns:
            Augmented sample dictionary
        """
        augmented_sample = sample.copy()
        
        # Apply audio augmentation
        if 'audio' in sample and sample['audio'] is not None:
            original_audio = sample['audio']
            augmented_audio = self.audio_aug(original_audio)
            
            # Ensure duration preservation if required
            if self.preserve_duration and len(augmented_audio) != len(original_audio):
                # Truncate or pad to match original length
                if len(augmented_audio) > len(original_audio):
                    augmented_audio = augmented_audio[:len(original_audio)]
                else:
                    padding = len(original_audio) - len(augmented_audio)
                    augmented_audio = torch.nn.functional.pad(augmented_audio, (0, padding))
            
            augmented_sample['audio'] = augmented_audio
        
        # Apply text augmentation
        if 'english_text' in sample and sample['english_text']:
            augmented_sample['english_text'] = self.text_aug(sample['english_text'])
        
        # Update Vietnamese translation if needed
        if self.sync_augmentation and 'vietnamese_text' in sample:
            # In practice, you might want to re-translate the augmented English text
            # For now, we'll apply similar augmentation to Vietnamese text
            if sample.get('vietnamese_text') and sample['vietnamese_text'] != sample.get('english_text', ''):
                # Apply text augmentation to Vietnamese as well
                augmented_sample['vietnamese_text'] = self.text_aug(sample['vietnamese_text'])
        
        return augmented_sample
    
    def augment_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply augmentation to a batch of samples"""
        return [self(sample) for sample in batch]


class AugmentationPipeline:
    """
    Flexible augmentation pipeline with multiple strategies
    """
    
    def __init__(
        self,
        augmentations: List[BaseAugmentation],
        apply_all: bool = False,
        max_augmentations: int = 3
    ):
        """
        Initialize augmentation pipeline
        
        Args:
            augmentations: List of augmentation strategies
            apply_all: Whether to apply all augmentations or sample randomly
            max_augmentations: Maximum number of augmentations to apply
        """
        self.augmentations = augmentations
        self.apply_all = apply_all
        self.max_augmentations = max_augmentations
    
    def __call__(self, data: Any) -> Any:
        """Apply augmentation pipeline"""
        if self.apply_all:
            # Apply all augmentations
            for aug in self.augmentations:
                if aug.should_apply():
                    data = aug(data)
        else:
            # Randomly sample augmentations
            num_augs = min(self.max_augmentations, len(self.augmentations))
            selected_augs = random.sample(self.augmentations, 
                                        random.randint(1, num_augs))
            
            for aug in selected_augs:
                if aug.should_apply():
                    data = aug(data)
        
        return data


def create_augmentation_pipeline(
    audio_config: Optional[Dict[str, Any]] = None,
    text_config: Optional[Dict[str, Any]] = None,
    multimodal_config: Optional[Dict[str, Any]] = None
) -> MultimodalAugmentation:
    """
    Create augmentation pipeline from configuration
    
    Args:
        audio_config: Audio augmentation configuration
        text_config: Text augmentation configuration
        multimodal_config: Multimodal augmentation configuration
        
    Returns:
        Configured MultimodalAugmentation
    """
    # Create individual augmentations
    audio_aug = AudioAugmentation(**(audio_config or {}))
    text_aug = TextAugmentation(**(text_config or {}))
    
    # Create multimodal augmentation
    multimodal_aug = MultimodalAugmentation(
        audio_augmentation=audio_aug,
        text_augmentation=text_aug,
        **(multimodal_config or {})
    )
    
    return multimodal_aug


def create_training_augmentation() -> MultimodalAugmentation:
    """Create standard training augmentation pipeline"""
    audio_config = {
        'noise_prob': 0.3,
        'pitch_shift_prob': 0.2,
        'speed_change_prob': 0.2,
        'volume_change_prob': 0.4,
        'time_masking_prob': 0.1
    }
    
    text_config = {
        'synonym_replacement_prob': 0.1,
        'random_insertion_prob': 0.05,
        'typo_injection_prob': 0.05,
        'max_changes': 2
    }
    
    return create_augmentation_pipeline(audio_config, text_config)


def create_light_augmentation() -> MultimodalAugmentation:
    """Create light augmentation for validation/testing"""
    audio_config = {
        'noise_prob': 0.1,
        'volume_change_prob': 0.2,
        'time_masking_prob': 0.05
    }
    
    text_config = {
        'synonym_replacement_prob': 0.05,
        'max_changes': 1
    }
    
    return create_augmentation_pipeline(audio_config, text_config)


# Example usage and testing
if __name__ == "__main__":
    print("Testing augmentation pipeline...")
    
    # Test audio augmentation
    audio_aug = AudioAugmentation()
    dummy_audio = torch.randn(16000)  # 1 second of audio
    augmented_audio = audio_aug(dummy_audio)
    print(f"Audio shape: {dummy_audio.shape} -> {augmented_audio.shape}")
    
    # Test text augmentation
    text_aug = TextAugmentation()
    sample_text = "Hello, how are you doing today? I think this is a good example."
    augmented_text = text_aug(sample_text)
    print(f"Original: {sample_text}")
    print(f"Augmented: {augmented_text}")
    
    # Test multimodal augmentation
    multimodal_aug = create_training_augmentation()
    sample = {
        'audio': dummy_audio,
        'english_text': sample_text,
        'vietnamese_text': "Xin chào, bạn có khỏe không?"
    }
    
    augmented_sample = multimodal_aug(sample)
    print("Multimodal augmentation completed successfully")
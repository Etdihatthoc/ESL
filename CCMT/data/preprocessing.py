"""
Audio and text preprocessing pipeline for CCMT
"""

import torch
import torchaudio
import numpy as np
import re
import string
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from pathlib import Path
import librosa
from collections import Counter
import unicodedata

# Import your existing text processing utilities
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from text_processing import (
        tokenize, count_content_words, is_low_content, 
        replace_repeats, most_common_words, ALL_STOPWORDS
    )
except ImportError:
    # Fallback implementations if text_processing not available
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def replace_repeats(text: str, k: int = 3, tag: str = "") -> str:
        return text  # Simplified fallback
    
    def is_low_content(text: str, threshold: int = 5) -> bool:
        return len(tokenize(text)) < threshold
    
    ALL_STOPWORDS = set()

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Audio preprocessing pipeline for CCMT
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        max_length: float = 30.0,  # seconds
        normalize: bool = True,
        remove_silence: bool = False,
        silence_threshold: float = 0.01,
        apply_filter: bool = False,
        filter_low_freq: float = 80.0,
        filter_high_freq: float = 8000.0,
        device: str = "cpu"
    ):
        """
        Initialize audio preprocessor
        
        Args:
            sample_rate: Target sample rate
            max_length: Maximum audio length in seconds
            normalize: Whether to normalize audio
            remove_silence: Whether to remove silence
            silence_threshold: Threshold for silence detection
            apply_filter: Whether to apply frequency filtering
            filter_low_freq: Low frequency cutoff for filter
            filter_high_freq: High frequency cutoff for filter
            device: Device for computations
        """
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_samples = int(max_length * sample_rate)
        self.normalize = normalize
        self.remove_silence = remove_silence
        self.silence_threshold = silence_threshold
        self.apply_filter = apply_filter
        self.filter_low_freq = filter_low_freq
        self.filter_high_freq = filter_high_freq
        self.device = torch.device(device)
        
        # Pre-compute resampler for common sample rates
        self.resamplers = {}
        common_rates = [8000, 22050, 44100, 48000]
        for rate in common_rates:
            if rate != sample_rate:
                self.resamplers[rate] = torchaudio.transforms.Resample(
                    orig_freq=rate, new_freq=sample_rate
                )
    
    def __call__(
        self, 
        audio: Union[str, torch.Tensor], 
        original_sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Process audio data
        
        Args:
            audio: Audio file path or tensor
            original_sample_rate: Original sample rate if audio is tensor
            
        Returns:
            Processed audio tensor (samples,)
        """
        # Load audio if path provided
        if isinstance(audio, str):
            waveform, orig_sr = self._load_audio(audio)
        else:
            waveform = audio
            orig_sr = original_sample_rate or self.sample_rate
        
        # Convert to mono
        waveform = self._to_mono(waveform)
        
        # Resample if necessary
        waveform = self._resample(waveform, orig_sr)
        
        # Apply frequency filtering
        if self.apply_filter:
            waveform = self._apply_bandpass_filter(waveform)
        
        # Remove silence
        if self.remove_silence:
            waveform = self._remove_silence(waveform)
        
        # Normalize
        if self.normalize:
            waveform = self._normalize(waveform)
        
        # Truncate or pad to target length
        waveform = self._adjust_length(waveform)
        
        return waveform
    
    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            return waveform, sample_rate
        except Exception as e:
            logger.warning(f"Failed to load audio {audio_path} with torchaudio: {e}")
            try:
                # Fallback to librosa
                waveform, sample_rate = librosa.load(audio_path, sr=None)
                waveform = torch.from_numpy(waveform).unsqueeze(0)
                return waveform, sample_rate
            except Exception as e2:
                logger.error(f"Failed to load audio {audio_path}: {e2}")
                # Return silent audio
                return torch.zeros(1, self.max_samples), self.sample_rate
    
    def _to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel audio to mono"""
        if waveform.shape[0] > 1:
            # Average channels
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform
    
    def _resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate"""
        if orig_sr != self.sample_rate:
            if orig_sr in self.resamplers:
                resampler = self.resamplers[orig_sr]
            else:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sr, new_freq=self.sample_rate
                )
                self.resamplers[orig_sr] = resampler
            
            waveform = resampler(waveform)
        
        return waveform
    
    def _apply_bandpass_filter(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply bandpass filter to remove unwanted frequencies"""
        try:
            # Simple bandpass filter using torchaudio
            nyquist = self.sample_rate / 2
            low_norm = self.filter_low_freq / nyquist
            high_norm = self.filter_high_freq / nyquist
            
            # Apply high-pass filter
            if low_norm > 0:
                highpass = torchaudio.transforms.Highpass(
                    sample_rate=self.sample_rate, 
                    cutoff_freq=self.filter_low_freq
                )
                waveform = highpass(waveform)
            
            # Apply low-pass filter
            if high_norm < 1:
                lowpass = torchaudio.transforms.Lowpass(
                    sample_rate=self.sample_rate,
                    cutoff_freq=self.filter_high_freq
                )
                waveform = lowpass(waveform)
            
        except Exception as e:
            logger.warning(f"Failed to apply filter: {e}")
        
        return waveform
    
    def _remove_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """Remove silence from audio"""
        try:
            # Simple energy-based silence removal
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.01 * self.sample_rate)    # 10ms hop
            
            # Calculate energy for each frame
            waveform_np = waveform.squeeze().numpy()
            frames = librosa.util.frame(waveform_np, frame_length=frame_length, 
                                      hop_length=hop_length, axis=0)
            energy = np.mean(frames**2, axis=0)
            
            # Find non-silent frames
            non_silent = energy > self.silence_threshold
            
            # Expand back to sample level
            non_silent_samples = np.repeat(non_silent, hop_length)
            
            # Truncate to original length
            non_silent_samples = non_silent_samples[:waveform.shape[1]]
            
            # Apply mask
            if np.any(non_silent_samples):
                waveform = waveform[:, non_silent_samples]
            
        except Exception as e:
            logger.warning(f"Failed to remove silence: {e}")
        
        return waveform
    
    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio"""
        # RMS normalization
        rms = torch.sqrt(torch.mean(waveform**2))
        if rms > 1e-8:  # Avoid division by zero
            waveform = waveform / (rms + 1e-8)
        
        # Peak normalization
        peak = torch.max(torch.abs(waveform))
        if peak > 1.0:
            waveform = waveform / peak
        
        return waveform
    
    def _adjust_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """Adjust audio length to target"""
        current_length = waveform.shape[1]
        
        if current_length > self.max_samples:
            # Truncate (random crop during training, center crop during inference)
            if self.training:
                start = torch.randint(0, current_length - self.max_samples + 1, (1,)).item()
                waveform = waveform[:, start:start + self.max_samples]
            else:
                start = (current_length - self.max_samples) // 2
                waveform = waveform[:, start:start + self.max_samples]
        
        elif current_length < self.max_samples:
            # Pad with zeros
            padding = self.max_samples - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform.squeeze(0)  # Remove channel dimension
    
    def set_training_mode(self, training: bool):
        """Set training mode for different preprocessing behavior"""
        self.training = training
    
    def get_audio_stats(self, waveform: torch.Tensor) -> Dict[str, float]:
        """Get statistics about audio"""
        return {
            'length_seconds': len(waveform) / self.sample_rate,
            'rms': torch.sqrt(torch.mean(waveform**2)).item(),
            'peak': torch.max(torch.abs(waveform)).item(),
            'zero_crossing_rate': torch.mean(
                (waveform[1:] * waveform[:-1] < 0).float()
            ).item()
        }


class TextPreprocessor:
    """
    Text preprocessing pipeline for CCMT
    """
    
    def __init__(
        self,
        max_length: int = 512,
        remove_repeats: bool = True,
        repeat_threshold: int = 3,
        remove_low_content: bool = True,
        content_threshold: int = 5,
        normalize_unicode: bool = True,
        remove_special_chars: bool = True,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        language: str = "en"  # "en" or "vi"
    ):
        """
        Initialize text preprocessor
        
        Args:
            max_length: Maximum text length in characters
            remove_repeats: Whether to remove repeated phrases
            repeat_threshold: Threshold for repeat detection
            remove_low_content: Whether to filter low-content text
            content_threshold: Minimum content words threshold
            normalize_unicode: Whether to normalize unicode
            remove_special_chars: Whether to remove special characters
            lowercase: Whether to convert to lowercase
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            language: Language code for language-specific processing
        """
        self.max_length = max_length
        self.remove_repeats = remove_repeats
        self.repeat_threshold = repeat_threshold
        self.remove_low_content = remove_low_content
        self.content_threshold = content_threshold
        self.normalize_unicode = normalize_unicode
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.language = language
        
        # Compile regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.special_chars_pattern = re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]')
        
        # Language-specific patterns
        if language == "vi":
            # Vietnamese-specific cleaning patterns
            self.vietnamese_patterns = {
                'tone_marks': re.compile(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]'),
                'repeated_chars': re.compile(r'(.)\1{2,}')  # 3+ repeated characters
            }
    
    def __call__(self, text: str) -> str:
        """
        Process text
        
        Args:
            text: Input text string
            
        Returns:
            Processed text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = self._basic_cleaning(text)
        
        # Language-specific processing
        if self.language == "vi":
            text = self._process_vietnamese(text)
        else:
            text = self._process_english(text)
        
        # Remove repeats
        if self.remove_repeats:
            text = replace_repeats(text, k=self.repeat_threshold, tag="[REPEAT]")
        
        # Filter low content
        if self.remove_low_content and is_low_content(text, self.content_threshold):
            logger.warning(f"Low content text detected: {text[:50]}...")
            return "[LOW_CONTENT]"
        
        # Final cleanup
        text = self._final_cleanup(text)
        
        return text
    
    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub('[URL]', text)
        
        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub('[EMAIL]', text)
        
        # Normalize unicode
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove special characters
        if self.remove_special_chars:
            text = self.special_chars_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _process_english(self, text: str) -> str:
        """English-specific processing"""
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove ASR artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed annotations
        text = re.sub(r'\b(um|uh|hmm|ah|oh)\b', '', text, flags=re.IGNORECASE)
        
        # Fix common ASR errors
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)  # Remove immediate word repetitions
        
        return text
    
    def _process_vietnamese(self, text: str) -> str:
        """Vietnamese-specific processing"""
        # Convert to lowercase (preserving tone marks)
        if self.lowercase:
            text = text.lower()
        
        # Remove repeated characters (common in informal Vietnamese)
        if hasattr(self, 'vietnamese_patterns'):
            text = self.vietnamese_patterns['repeated_chars'].sub(r'\1\1', text)
        
        # Remove ASR artifacts
        text = re.sub(r'\[.*?\]', '', text)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final text cleanup"""
        # Truncate to max length
        if len(text) > self.max_length:
            text = text[:self.max_length]
            # Try to break at word boundary
            last_space = text.rfind(' ')
            if last_space > self.max_length * 0.8:  # If space is reasonably close to end
                text = text[:last_space]
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_english(self, text: str) -> str:
        """Process English text specifically"""
        original_language = self.language
        self.language = "en"
        result = self(text)
        self.language = original_language
        return result
    
    def process_vietnamese(self, text: str) -> str:
        """Process Vietnamese text specifically"""
        original_language = self.language
        self.language = "vi"
        result = self(text)
        self.language = original_language
        return result
    
    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """Get statistics about text"""
        tokens = tokenize(text)
        return {
            'length_chars': len(text),
            'length_words': len(tokens),
            'content_words': count_content_words(text),
            'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0,
            'unique_words': len(set(tokens)),
            'is_low_content': is_low_content(text, self.content_threshold)
        }


class MultimodalPreprocessor:
    """
    Coordinated preprocessing for multimodal data
    """
    
    def __init__(
        self,
        audio_processor: Optional[AudioPreprocessor] = None,
        text_processor: Optional[TextPreprocessor] = None,
        translator: Optional[Any] = None,
        cache_translations: bool = True
    ):
        """
        Initialize multimodal preprocessor
        
        Args:
            audio_processor: Audio preprocessing pipeline
            text_processor: Text preprocessing pipeline
            translator: Translation model
            cache_translations: Whether to cache translations
        """
        self.audio_processor = audio_processor or AudioPreprocessor()
        self.text_processor = text_processor or TextPreprocessor()
        self.translator = translator
        self.translation_cache = {} if cache_translations else None
    
    def __call__(
        self, 
        audio: Union[str, torch.Tensor],
        text: str,
        audio_sample_rate: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process multimodal sample
        
        Args:
            audio: Audio file path or tensor
            text: English text
            audio_sample_rate: Original audio sample rate
            
        Returns:
            Dictionary with processed data
        """
        # Process audio
        processed_audio = self.audio_processor(audio, audio_sample_rate)
        audio_stats = self.audio_processor.get_audio_stats(processed_audio)
        
        # Process English text
        processed_english = self.text_processor.process_english(text)
        english_stats = self.text_processor.get_text_stats(processed_english)
        
        # Get Vietnamese translation
        vietnamese_text = self._get_translation(processed_english)
        vietnamese_stats = None
        if vietnamese_text != processed_english:  # Only compute stats if actually translated
            vietnamese_stats = self.text_processor.get_text_stats(vietnamese_text)
        
        return {
            'audio': processed_audio,
            'english_text': processed_english,
            'vietnamese_text': vietnamese_text,
            'audio_stats': audio_stats,
            'english_stats': english_stats,
            'vietnamese_stats': vietnamese_stats,
            'metadata': {
                'audio_length': audio_stats['length_seconds'],
                'english_length': english_stats['length_words'],
                'vietnamese_length': vietnamese_stats['length_words'] if vietnamese_stats else 0,
                'translation_used': vietnamese_text != processed_english
            }
        }
    
    def _get_translation(self, english_text: str) -> str:
        """Get Vietnamese translation with caching"""
        if not self.translator:
            return english_text
        
        # Check cache
        if self.translation_cache is not None and english_text in self.translation_cache:
            return self.translation_cache[english_text]
        
        try:
            vietnamese_text = self.translator.translate_single(english_text)
            
            # Cache result
            if self.translation_cache is not None:
                self.translation_cache[english_text] = vietnamese_text
            
            return vietnamese_text
            
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return english_text


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for CCMT dataset
    """
    
    def __init__(
        self,
        audio_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        translator: Optional[Any] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize preprocessing pipeline
        
        Args:
            audio_config: Audio preprocessing configuration
            text_config: Text preprocessing configuration
            translator: Translation model
            batch_size: Batch size for processing
            num_workers: Number of parallel workers
            cache_dir: Directory for caching preprocessed data
        """
        # Initialize processors
        self.audio_processor = AudioPreprocessor(**(audio_config or {}))
        self.text_processor = TextPreprocessor(**(text_config or {}))
        self.multimodal_processor = MultimodalPreprocessor(
            self.audio_processor, self.text_processor, translator
        )
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def process_dataset(
        self, 
        dataset: Any,
        save_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Process entire dataset
        
        Args:
            dataset: Dataset to process
            save_stats: Whether to save processing statistics
            
        Returns:
            Processing statistics
        """
        logger.info(f"Processing dataset with {len(dataset)} samples...")
        
        stats = {
            'processed_samples': 0,
            'failed_samples': 0,
            'audio_stats': [],
            'text_stats': [],
            'processing_times': []
        }
        
        import time
        start_time = time.time()
        
        for idx in range(len(dataset)):
            try:
                sample_start = time.time()
                
                # Get raw sample
                raw_sample = dataset[idx]
                
                # Process sample
                processed = self.multimodal_processor(
                    raw_sample['audio'],
                    raw_sample['english_text']
                )
                
                # Update statistics
                stats['processed_samples'] += 1
                stats['audio_stats'].append(processed['audio_stats'])
                stats['text_stats'].append(processed['english_stats'])
                stats['processing_times'].append(time.time() - sample_start)
                
                if idx % 100 == 0:
                    logger.info(f"Processed {idx+1}/{len(dataset)} samples")
                
            except Exception as e:
                logger.error(f"Failed to process sample {idx}: {e}")
                stats['failed_samples'] += 1
        
        total_time = time.time() - start_time
        stats['total_processing_time'] = total_time
        stats['avg_processing_time'] = np.mean(stats['processing_times'])
        
        logger.info(f"Dataset processing complete: {stats['processed_samples']} successful, "
                    f"{stats['failed_samples']} failed, {total_time:.2f}s total")
        
        # Save statistics
        if save_stats and self.cache_dir:
            stats_path = self.cache_dir / 'preprocessing_stats.json'
            import json
            with open(stats_path, 'w') as f:
                json.dump({k: v for k, v in stats.items() if k != 'processing_times'}, f, indent=2)
        
        return stats


# Factory functions
def create_audio_preprocessor(config: Dict[str, Any]) -> AudioPreprocessor:
    """Create audio preprocessor from config"""
    return AudioPreprocessor(**config)


def create_text_preprocessor(config: Dict[str, Any]) -> TextPreprocessor:
    """Create text preprocessor from config"""
    return TextPreprocessor(**config)


def create_multimodal_preprocessor(
    audio_config: Dict[str, Any],
    text_config: Dict[str, Any],
    translator: Optional[Any] = None
) -> MultimodalPreprocessor:
    """Create multimodal preprocessor from configs"""
    audio_processor = create_audio_preprocessor(audio_config)
    text_processor = create_text_preprocessor(text_config)
    return MultimodalPreprocessor(audio_processor, text_processor, translator)


# Example usage
if __name__ == "__main__":
    print("Testing preprocessing pipeline...")
    
    # Test audio preprocessing
    audio_processor = AudioPreprocessor()
    
    # Test text preprocessing
    text_processor = TextPreprocessor()
    sample_text = "Hello, how are you today? Um, I think the weather is really really nice."
    processed_text = text_processor(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed_text}")
    
    # Test multimodal preprocessing
    multimodal_processor = MultimodalPreprocessor(audio_processor, text_processor)
    print("Multimodal processor created successfully")
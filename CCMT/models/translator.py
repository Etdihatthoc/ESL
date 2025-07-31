"""
English to Vietnamese translation module for CCMT pipeline
"""

import torch
import torch.nn as nn
from transformers import (
    MarianMTModel, MarianTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from typing import List, Optional, Dict, Union
import logging
from functools import lru_cache
import time

# Set up logging
logger = logging.getLogger(__name__)


class BaseTranslator(nn.Module):
    """Base class for translation models"""
    
    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        batch_size: int = 8,
        cache_size: int = 1000
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Set model to eval mode by default
        self.model.eval()
        
        # Translation cache to avoid re-translating identical texts
        self._cache = {}
        self.cache_size = cache_size
    
    def _add_to_cache(self, source: str, translation: str):
        """Add translation to cache with size limit"""
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[source] = translation
    
    def _get_from_cache(self, source: str) -> Optional[str]:
        """Get translation from cache"""
        return self._cache.get(source)
    
    def translate_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        **generation_kwargs
    ) -> List[str]:
        """
        Translate a batch of texts
        
        Args:
            texts: List of source texts
            use_cache: Whether to use translation cache
            **generation_kwargs: Additional generation parameters
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        # Check cache first
        if use_cache:
            cached_translations = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cached = self._get_from_cache(text)
                if cached is not None:
                    cached_translations.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached_translations = []
        
        # Translate uncached texts
        if uncached_texts:
            new_translations = self._translate_batch_impl(uncached_texts, **generation_kwargs)
            
            # Add to cache
            if use_cache:
                for text, translation in zip(uncached_texts, new_translations):
                    self._add_to_cache(text, translation)
        else:
            new_translations = []
        
        # Combine cached and new translations
        result = [''] * len(texts)
        
        # Fill in cached translations
        for idx, translation in cached_translations:
            result[idx] = translation
        
        # Fill in new translations
        for idx, translation in zip(uncached_indices, new_translations):
            result[idx] = translation
        
        return result
    
    def _translate_batch_impl(self, texts: List[str], **generation_kwargs) -> List[str]:
        """
        Actual implementation of batch translation
        Override in subclasses
        """
        raise NotImplementedError
    
    def translate_single(self, text: str, use_cache: bool = True, **generation_kwargs) -> str:
        """
        Translate a single text
        
        Args:
            text: Source text
            use_cache: Whether to use translation cache
            **generation_kwargs: Additional generation parameters
        Returns:
            Translated text
        """
        return self.translate_batch([text], use_cache=use_cache, **generation_kwargs)[0]
    
    def clear_cache(self):
        """Clear translation cache"""
        self._cache.clear()


class EnglishToVietnameseTranslator(BaseTranslator):
    """
    English to Vietnamese translator using various translation models
    """
    
    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-vi",
        max_length: int = 512,
        batch_size: int = 8,
        cache_size: int = 1000,
        device: Optional[str] = None
    ):
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            cache_size=cache_size
        )
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        logger.info(f"Initialized translator {model_name} on {self.device}")
    
    def _translate_batch_impl(self, texts: List[str], **generation_kwargs) -> List[str]:
        """
        Implement batch translation using the loaded model
        
        Args:
            texts: List of English texts
            **generation_kwargs: Generation parameters
        Returns:
            List of Vietnamese translations
        """
        # Default generation parameters
        default_kwargs = {
            "max_length": self.max_length,
            "num_beams": 4,
            "early_stopping": True,
            "do_sample": False
        }
        default_kwargs.update(generation_kwargs)
        
        # Process in batches to avoid memory issues
        all_translations = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize input
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate translations
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **default_kwargs
                )
            
            # Decode translations
            batch_translations = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            
            all_translations.extend(batch_translations)
        
        return all_translations
    
    def translate_with_multiple_candidates(
        self,
        texts: List[str],
        num_candidates: int = 3,
        **generation_kwargs
    ) -> List[List[str]]:
        """
        Generate multiple translation candidates for each text
        
        Args:
            texts: List of English texts
            num_candidates: Number of translation candidates
            **generation_kwargs: Generation parameters
        Returns:
            List of lists of translation candidates
        """
        generation_kwargs.update({
            "num_return_sequences": num_candidates,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9
        })
        
        all_candidates = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize input
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate translations
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    **generation_kwargs
                )
            
            # Decode translations
            batch_translations = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            
            # Group by input text
            for j in range(len(batch_texts)):
                candidates = batch_translations[j*num_candidates:(j+1)*num_candidates]
                all_candidates.append(candidates)
        
        return all_candidates


class FallbackTranslator:
    """
    Fallback translator using multiple translation services/models
    """
    
    def __init__(self, translators: List[BaseTranslator]):
        self.translators = translators
        self.active_translator_idx = 0
    
    def translate_batch(self, texts: List[str], **kwargs) -> List[str]:
        """
        Try translating with primary translator, fallback to others if failed
        """
        for i, translator in enumerate(self.translators):
            try:
                translations = translator.translate_batch(texts, **kwargs)
                if i != self.active_translator_idx:
                    logger.info(f"Switched to translator {i}: {translator.model_name}")
                    self.active_translator_idx = i
                return translations
            except Exception as e:
                logger.warning(f"Translator {i} failed: {e}")
                if i == len(self.translators) - 1:
                    # All translators failed, return original texts
                    logger.error("All translators failed, returning original texts")
                    return texts
                continue
        
        return texts


# Factory functions for easy creation
def create_en_vi_translator(
    model_type: str = "opus",  # "opus", "t5", "marian"
    device: Optional[str] = None,
    cache_size: int = 1000
) -> EnglishToVietnameseTranslator:
    """
    Factory function to create English-Vietnamese translators
    
    Args:
        model_type: Type of translation model
        device: Device to run model on
        cache_size: Size of translation cache
    Returns:
        Configured translator
    """
    model_configs = {
        "opus": "Helsinki-NLP/opus-mt-en-vi",
        "t5": "google/flan-t5-base",  # Would need special prompting
        "marian": "Helsinki-NLP/opus-mt-en-vi",  # Same as opus for now
        "multilingual": "facebook/mbart-large-50-many-to-many-mmt"
    }
    
    model_name = model_configs.get(model_type, "Helsinki-NLP/opus-mt-en-vi")
    
    return EnglishToVietnameseTranslator(
        model_name=model_name,
        device=device,
        cache_size=cache_size
    )


def create_fallback_translator(
    model_types: List[str] = ["opus", "multilingual"],
    device: Optional[str] = None
) -> FallbackTranslator:
    """
    Create a fallback translator with multiple models
    
    Args:
        model_types: List of model types to use as fallbacks
        device: Device to run models on
    Returns:
        FallbackTranslator instance
    """
    translators = []
    for model_type in model_types:
        try:
            translator = create_en_vi_translator(model_type, device)
            translators.append(translator)
        except Exception as e:
            logger.warning(f"Failed to load translator {model_type}: {e}")
    
    if not translators:
        raise RuntimeError("No translators could be loaded")
    
    return FallbackTranslator(translators)


# Convenience function for quick translation
@lru_cache(maxsize=None)
def get_default_translator() -> EnglishToVietnameseTranslator:
    """Get default translator (cached)"""
    return create_en_vi_translator("opus")


def quick_translate(text: str, use_default: bool = True) -> str:
    """
    Quick translation function for single texts
    
    Args:
        text: English text to translate
        use_default: Whether to use cached default translator
    Returns:
        Vietnamese translation
    """
    if use_default:
        translator = get_default_translator()
    else:
        translator = create_en_vi_translator()
    
    return translator.translate_single(text)
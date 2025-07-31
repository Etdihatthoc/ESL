"""
Translation utility functions for CCMT
"""

import json
import pickle
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import logging
import hashlib

logger = logging.getLogger(__name__)


def batch_translate(texts: List[str], translator: Any, batch_size: int = 32) -> List[str]:
    """
    Translate texts in batches for efficiency
    
    Args:
        texts: List of texts to translate
        translator: Translation model/service
        batch_size: Number of texts per batch
    
    Returns:
        List of translated texts
    """
    if not texts:
        return []
    
    translations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            if hasattr(translator, 'translate_batch'):
                batch_translations = translator.translate_batch(batch)
            else:
                # Fallback to single translations
                batch_translations = [translator.translate_single(text) for text in batch]
        
            translations.extend(batch_translations)
            
        except Exception as e:
            logger.warning(f"Batch translation failed: {e}")
            # Fallback to original texts
            translations.extend(batch)
    
    return translations


def validate_translation_quality(original: str, translated: str, min_similarity: float = 0.1) -> bool:
    """
    Basic validation of translation quality
    
    Args:
        original: Original text
        translated: Translated text
        min_similarity: Minimum similarity threshold
    
    Returns:
        True if translation seems valid
    """
    if not original or not translated:
        return False
    
    # Check if translation is too similar (might be untranslated)
    if original.lower() == translated.lower():
        return False
    
    # Check if translation is too short compared to original
    if len(translated) < len(original) * 0.3:
        return False
    
    # Check for obvious translation artifacts
    artifacts = ['[UNK]', '<unk>', 'TRANSLATE_ERROR', '###']
    if any(artifact in translated for artifact in artifacts):
        return False
    
    return True


def cache_translations(cache_file: str = "translation_cache.json"):
    """
    Decorator to cache translations to file
    
    Args:
        cache_file: Path to cache file
    
    Returns:
        Decorator function
    """
    def decorator(translate_func):
        def wrapper(text: str, *args, **kwargs):
            # Load existing cache
            cache_path = Path(cache_file)
            cache = {}
            if cache_path.exists():
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load translation cache: {e}")
            
            # Create cache key
            cache_key = _create_cache_key(text, args, kwargs)
            
            # Check cache
            if cache_key in cache:
                return cache[cache_key]
            
            # Translate
            translation = translate_func(text, *args, **kwargs)
            
            # Save to cache
            cache[cache_key] = translation
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to save translation cache: {e}")
            
            return translation
        
        return wrapper
    return decorator


def _create_cache_key(text: str, args: tuple, kwargs: dict) -> str:
    """Create a hash key for caching"""
    key_data = {
        'text': text,
        'args': args,
        'kwargs': {k: v for k, v in kwargs.items() if k != 'translator'}
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def translation_fallback(primary_translator: Any, fallback_translator: Any):
    """
    Create a fallback translation function
    
    Args:
        primary_translator: Primary translation service
        fallback_translator: Fallback translation service
    
    Returns:
        Translation function with fallback
    """
    def translate_with_fallback(text: str) -> str:
        try:
            # Try primary translator
            translation = primary_translator.translate_single(text)
            
            # Validate translation
            if validate_translation_quality(text, translation):
                return translation
            else:
                raise ValueError("Translation quality check failed")
                
        except Exception as e:
            logger.warning(f"Primary translation failed: {e}, trying fallback")
            
            try:
                # Try fallback translator
                translation = fallback_translator.translate_single(text)
                return translation
            except Exception as e2:
                logger.error(f"Fallback translation also failed: {e2}")
                return text  # Return original text as last resort
    
    return translate_with_fallback


def load_translation_cache(cache_file: str) -> Dict[str, str]:
    """
    Load translation cache from file
    
    Args:
        cache_file: Path to cache file
    
    Returns:
        Translation cache dictionary
    """
    cache_path = Path(cache_file)
    
    if not cache_path.exists():
        return {}
    
    try:
        if cache_file.endswith('.json'):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif cache_file.endswith('.pkl'):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            logger.error(f"Unsupported cache file format: {cache_file}")
            return {}
    
    except Exception as e:
        logger.error(f"Failed to load translation cache: {e}")
        return {}


def save_translation_cache(cache: Dict[str, str], cache_file: str):
    """
    Save translation cache to file
    
    Args:
        cache: Translation cache dictionary
        cache_file: Path to cache file
    """
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if cache_file.endswith('.json'):
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        elif cache_file.endswith('.pkl'):
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f)
        else:
            logger.error(f"Unsupported cache file format: {cache_file}")
    
    except Exception as e:
        logger.error(f"Failed to save translation cache: {e}")


def merge_translation_caches(*cache_files: str) -> Dict[str, str]:
    """
    Merge multiple translation cache files
    
    Args:
        cache_files: Paths to cache files
    
    Returns:
        Merged cache dictionary
    """
    merged_cache = {}
    
    for cache_file in cache_files:
        cache = load_translation_cache(cache_file)
        merged_cache.update(cache)
        logger.info(f"Loaded {len(cache)} translations from {cache_file}")
    
    logger.info(f"Merged cache contains {len(merged_cache)} translations")
    return merged_cache


def clean_translation_cache(cache: Dict[str, str], min_length: int = 3) -> Dict[str, str]:
    """
    Clean translation cache by removing invalid entries
    
    Args:
        cache: Translation cache
        min_length: Minimum text length
    
    Returns:
        Cleaned cache
    """
    cleaned_cache = {}
    removed_count = 0
    
    for original, translation in cache.items():
        if (len(original) >= min_length and 
            len(translation) >= min_length and
            validate_translation_quality(original, translation)):
            cleaned_cache[original] = translation
        else:
            removed_count += 1
    
    logger.info(f"Cleaned cache: removed {removed_count} invalid entries, "
                f"{len(cleaned_cache)} entries remaining")
    
    return cleaned_cache


def estimate_translation_cost(texts: List[str], cost_per_char: float = 0.000020) -> float:
    """
    Estimate translation cost based on character count
    
    Args:
        texts: List of texts to translate
        cost_per_char: Cost per character (Google Translate pricing)
    
    Returns:
        Estimated cost in USD
    """
    total_chars = sum(len(text) for text in texts)
    estimated_cost = total_chars * cost_per_char
    
    return estimated_cost


def split_long_text_for_translation(text: str, max_length: int = 5000) -> List[str]:
    """
    Split long text into chunks for translation APIs with length limits
    
    Args:
        text: Long text to split
        max_length: Maximum length per chunk
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    # Try to split at sentence boundaries
    sentences = text.replace('.', '.\n').replace('!', '!\n').replace('?', '?\n').split('\n')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def combine_translated_chunks(chunks: List[str]) -> str:
    """
    Combine translated text chunks back into single text
    
    Args:
        chunks: List of translated chunks
    
    Returns:
        Combined text
    """
    return " ".join(chunks)


def detect_translation_language(text: str) -> str:
    """
    Simple detection of text language for validation
    
    Args:
        text: Text to analyze
    
    Returns:
        Language code ('en', 'vi', or 'unknown')
    """
    if not text:
        return 'unknown'
    
    # Vietnamese specific characters
    vietnamese_chars = 'àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵđ'
    
    # Count Vietnamese-specific characters
    vietnamese_count = sum(1 for char in text.lower() if char in vietnamese_chars)
    total_alpha = sum(1 for char in text if char.isalpha())
    
    if total_alpha == 0:
        return 'unknown'
    
    vietnamese_ratio = vietnamese_count / total_alpha
    
    if vietnamese_ratio > 0.05:  # 5% threshold for Vietnamese detection
        return 'vi'
    else:
        return 'en'


def create_translation_report(cache: Dict[str, str]) -> Dict[str, Any]:
    """
    Create a report about translation cache
    
    Args:
        cache: Translation cache
    
    Returns:
        Report dictionary
    """
    if not cache:
        return {'total_translations': 0}
    
    # Analyze languages
    source_languages = [detect_translation_language(text) for text in cache.keys()]
    target_languages = [detect_translation_language(text) for text in cache.values()]
    
    # Calculate statistics
    total_chars_source = sum(len(text) for text in cache.keys())
    total_chars_target = sum(len(text) for text in cache.values())
    
    avg_length_source = total_chars_source / len(cache) if cache else 0
    avg_length_target = total_chars_target / len(cache) if cache else 0
    
    report = {
        'total_translations': len(cache),
        'total_chars_source': total_chars_source,
        'total_chars_target': total_chars_target,
        'avg_length_source': avg_length_source,
        'avg_length_target': avg_length_target,
        'source_language_distribution': {lang: source_languages.count(lang) for lang in set(source_languages)},
        'target_language_distribution': {lang: target_languages.count(lang) for lang in set(target_languages)}
    }
    
    return report
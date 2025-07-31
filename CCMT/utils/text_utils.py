"""
Enhanced text processing utilities for CCMT
Builds upon existing text_processing.py functions
"""

import re
import string
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import Counter
import numpy as np

# Import existing text processing functions
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from text_processing import (
        tokenize, count_content_words, is_low_content, 
        replace_repeats, most_common_words, ALL_STOPWORDS
    )
except ImportError:
    # Fallback implementations
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())
    
    def count_content_words(text: str) -> int:
        return len([w for w in tokenize(text) if len(w) > 2])
    
    def is_low_content(text: str, threshold: int = 5) -> bool:
        return count_content_words(text) < threshold
    
    def replace_repeats(text: str, k: int = 3, tag: str = "") -> str:
        return text
    
    ALL_STOPWORDS = set()


def clean_transcript(text: str, remove_brackets: bool = True, remove_repeats: bool = True) -> str:
    """
    Clean ASR transcript using existing functions
    
    Args:
        text: Raw transcript text
        remove_brackets: Remove bracketed annotations like [laugh]
        remove_repeats: Remove repeated phrases
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove bracketed annotations
    if remove_brackets:
        text = re.sub(r'\[.*?\]', '', text)
    
    # Remove repeated phrases
    if remove_repeats:
        text = replace_repeats(text, k=3, tag="")
    
    # Basic cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_text(text: str, lowercase: bool = True, remove_punctuation: bool = False) -> str:
    """
    Normalize text for consistent processing
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation marks
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_content_words(text: str, min_length: int = 3) -> List[str]:
    """
    Extract meaningful content words from text
    
    Args:
        text: Input text
        min_length: Minimum word length
    
    Returns:
        List of content words
    """
    tokens = tokenize(text)
    content_words = [
        word for word in tokens 
        if len(word) >= min_length and word not in ALL_STOPWORDS
    ]
    return content_words


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using word overlap
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(extract_content_words(text1))
    words2 = set(extract_content_words(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def batch_process_texts(texts: List[str], clean: bool = True, normalize: bool = True) -> List[str]:
    """
    Process a batch of texts efficiently
    
    Args:
        texts: List of input texts
        clean: Apply cleaning
        normalize: Apply normalization
    
    Returns:
        List of processed texts
    """
    processed = []
    
    for text in texts:
        if clean:
            text = clean_transcript(text)
        if normalize:
            text = normalize_text(text)
        processed.append(text)
    
    return processed


def get_text_statistics(text: str) -> Dict[str, Any]:
    """
    Get comprehensive text statistics
    
    Args:
        text: Input text
    
    Returns:
        Dictionary of statistics
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'content_word_count': 0,
            'avg_word_length': 0.0,
            'is_low_content': True
        }
    
    tokens = tokenize(text)
    content_word_count = count_content_words(text)
    
    return {
        'char_count': len(text),
        'word_count': len(tokens),
        'content_word_count': content_word_count,
        'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0.0,
        'is_low_content': is_low_content(text)
    }


def detect_language_simple(text: str) -> str:
    """
    Simple language detection for English vs Vietnamese
    
    Args:
        text: Input text
    
    Returns:
        'en' for English, 'vi' for Vietnamese, 'unknown' for unclear
    """
    if not text:
        return 'unknown'
    
    # Vietnamese specific characters
    vietnamese_chars = 'àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵđ'
    
    # Count Vietnamese characters
    vi_char_count = sum(1 for char in text.lower() if char in vietnamese_chars)
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return 'unknown'
    
    vi_ratio = vi_char_count / total_chars
    
    if vi_ratio > 0.1:  # More than 10% Vietnamese characters
        return 'vi'
    else:
        return 'en'


def preprocess_for_scoring(text: str) -> str:
    """
    Preprocess text specifically for English speaking scoring
    
    Args:
        text: Raw transcript text
    
    Returns:
        Preprocessed text ready for scoring
    """
    # Apply all cleaning steps
    text = clean_transcript(text, remove_brackets=True, remove_repeats=True)
    text = normalize_text(text, lowercase=True, remove_punctuation=False)
    
    # Remove filler words common in speech
    filler_words = ['um', 'uh', 'hmm', 'ah', 'oh', 'like', 'you know']
    pattern = r'\b(' + '|'.join(filler_words) + r')\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def validate_transcript_quality(text: str, min_words: int = 5, max_repeat_ratio: float = 0.3) -> bool:
    """
    Validate if transcript is good enough for training
    
    Args:
        text: Transcript text
        min_words: Minimum number of words required
        max_repeat_ratio: Maximum allowed repetition ratio
    
    Returns:
        True if transcript passes quality checks
    """
    if not text or is_low_content(text, threshold=min_words):
        return False
    
    # Check repetition ratio
    tokens = tokenize(text)
    if len(tokens) == 0:
        return False
    
    # Count repeated tokens
    token_counts = Counter(tokens)
    repeated_tokens = sum(count - 1 for count in token_counts.values() if count > 1)
    repeat_ratio = repeated_tokens / len(tokens)
    
    return repeat_ratio <= max_repeat_ratio


def truncate_text_smart(text: str, max_length: int) -> str:
    """
    Truncate text smartly at word boundaries
    
    Args:
        text: Input text
        max_length: Maximum character length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Find last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If space is reasonably close
        return truncated[:last_space]
    else:
        return truncated


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences
    
    Args:
        text: Input text
    
    Returns:
        List of sentences
    """
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def merge_short_sentences(sentences: List[str], min_length: int = 10) -> List[str]:
    """
    Merge very short sentences with adjacent ones
    
    Args:
        sentences: List of sentences
        min_length: Minimum sentence length in characters
    
    Returns:
        List of merged sentences
    """
    if not sentences:
        return []
    
    merged = []
    current = sentences[0]
    
    for i in range(1, len(sentences)):
        if len(current) < min_length:
            current = current + ' ' + sentences[i]
        else:
            merged.append(current)
            current = sentences[i]
    
    merged.append(current)
    return merged
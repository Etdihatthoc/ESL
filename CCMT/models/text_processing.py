"""
Text processing utilities for ESL grading
This module contains the text cleaning and processing functions from the original codebase
"""

import re
import pandas as pd
import numpy as np
from typing import List, Set
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import nltk, provide fallback if not available
try:
    import nltk
    from nltk.corpus import stopwords
    
    # Download stopwords if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    
    NLTK_STOPWORDS: Set[str] = set(stopwords.words("english"))
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not available. Using basic stopwords list.")
    print("For full functionality, install with: pip install nltk")
    # Fallback stopwords list
    NLTK_STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', 'should', 'now'
    }
    NLTK_AVAILABLE = False

# Extra "fillers" commonly seen in ASR noise, plus common punctuation
EXTRA_FILLERS = {
    "um", "uh", "hmm", "mmm", "ah", "oh", 
    "…", "...", "--",  # variants of ellipsis
    ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "\"", "'", "/", "\\" # punctuation
}

ALL_STOPWORDS = NLTK_STOPWORDS.union(EXTRA_FILLERS)


def tokenize(text: str) -> List[str]:
    """
    Extract lowercased alphabetic tokens and bracketed tokens (like [laugh]).
    """
    # capture word‑characters and bracketed annotations
    toks = re.findall(r"\[[^\]]+\]|[A-Za-z]+", text)
    return [t.lower() for t in toks]


def count_content_words(text: str, stopwords: Set[str] = ALL_STOPWORDS) -> int:
    """
    Count tokens not in the stopword set.
    """
    toks = tokenize(text)
    return sum(1 for t in toks if t not in stopwords)


def is_low_content(transcript: str, threshold: int = 5) -> bool:
    """
    Flag transcripts with fewer than `threshold` content words.
    """
    return count_content_words(transcript) < threshold


def replace_repeats(text: str, k: int = 3, tag: str = "") -> str:
    """
    Scans text for any contiguous token-sequence that repeats > k times,
    keeps the first k copies, and replaces the rest with `tag`.
    Preserves whitespace and structure.

    Args:
        text: the input string (tokens are split on whitespace).
        k: the maximum number of repeats to keep.
        tag: what to put in place of the surplus repeats (default delete).

    Returns:
        A new string with over-repeated substrings collapsed.
    """
    # Tokenize including whitespace
    tokens = re.findall(r'\S+|\s+', text)

    # Build non-whitespace token list and mapping to original tokens
    non_ws_tokens = []
    idx_map = []  # map from non-ws-token index to tokens index
    for idx, tok in enumerate(tokens):
        if not tok.isspace():
            idx_map.append(idx)
            non_ws_tokens.append(tok)

    n = len(non_ws_tokens)
    i = 0
    out_tokens = []
    token_idx = 0

    while i < n:
        replaced = False
        max_L = (n - i) // k

        for L in range(1, max_L + 1):
            seq = non_ws_tokens[i : i + L]
            count = 1
            while i + (count + 1) * L <= n and non_ws_tokens[i + count * L : i + (count + 1) * L] == seq:
                count += 1
            
            if count > k:
                # Copy first k repeats (L * k tokens) using the original indices
                start_token_idx = idx_map[i]
                end_token_idx = idx_map[i + L * k - 1] + 1

                out_tokens.extend(tokens[start_token_idx:end_token_idx])
                if tag:
                    out_tokens.append(" " + tag + " ")

                # Move pointers
                i += count * L
                # Find new token_idx: go to end of last skipped non-ws token
                token_idx = idx_map[i] if i < len(idx_map) else len(tokens)
                replaced = True
                break

        if not replaced:
            # Copy next token and any whitespace after
            if token_idx < len(tokens):
                # Copy the next non-whitespace token
                out_tokens.append(tokens[token_idx])
                token_idx += 1
                i += 1

                # Copy all whitespace tokens that follow
                while token_idx < len(tokens) and tokens[token_idx].isspace():
                    out_tokens.append(tokens[token_idx])
                    token_idx += 1
    
    if token_idx < len(tokens):
        out_tokens.extend(tokens[token_idx:])

    return "".join(out_tokens)


def most_common_words(df, proportion=0.1, verbose=False):
    """
    Return the most common `proportion` of words in df['text'], sorted descending by TF-IDF score.
    Each word is counted at most once per row.
    Removes punctuation and stopwords.
    """
    try:
        vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(df['text'])

        # Compute mean TF-IDF for each word across all docs
        means = tfidf_matrix.mean(axis=0).A1  # convert to flat array
        vocab = vectorizer.get_feature_names_out()

        tfidf_scores = list(zip(vocab, means))
        tfidf_scores.sort(key=lambda x: x[1], reverse=True)

        n_show = max(1, int(len(tfidf_scores) * proportion))
        top_words = tfidf_scores[:n_show]

        if verbose:
            for word, score in top_words:
                print(f"{word}: {score:.4f}")

        return [word for word, _ in top_words]
    
    except ImportError:
        print("Warning: scikit-learn not available for TF-IDF analysis")
        print("For full functionality, install with: pip install scikit-learn")
        # Fallback: return some common English words that might be important
        return ['good', 'bad', 'very', 'really', 'think', 'know', 'like', 'want', 'need', 'make']
    except Exception as e:
        print(f"Warning: Error in TF-IDF analysis: {e}")
        return []


def clean_dataframe(df, remove_low_content=True, filter_scores=True):
    """
    Cleans the dataframe by processing the 'text' field:
    - Applies replace_repeats
    - Optionally removes rows with low content using is_low_content
    - Optionally filters scores
    """
    print(f"Rows before cleaning: {len(df)}")
    df = df.copy()
    
    # Apply repeat replacement
    df['text'] = df['text'].apply(lambda t: replace_repeats(t, k=2, tag="[REPEAT]"))
    
    # Remove low content samples
    if remove_low_content:
        mask = ~df['text'].apply(is_low_content)
        df = df[mask].reset_index(drop=True)
        print(f"Rows after low-content filtering: {len(df)}")
    
    # Filter scores if requested
    if filter_scores:
        score_column = 'grammar' if 'grammar' in df.columns else 'final'
        if score_column in df.columns:
            # Keep scores >= 3
            mask = (df[score_column] >= 3)
            df = df[mask].reset_index(drop=True)
            print(f"After score filtering: {len(df)} samples")
            print(f"Score distribution: {df[score_column].value_counts().sort_index()}")
        else:
            print(f"Warning: No score column found for filtering")
    
    print(f"Rows after cleaning: {len(df)}")
    return df


# Additional utility functions for text analysis
def analyze_text_length_distribution(df, text_column='text'):
    """Analyze the distribution of text lengths"""
    if text_column not in df.columns:
        print(f"Column '{text_column}' not found in dataframe")
        return
    
    # Calculate text statistics
    text_lengths = df[text_column].apply(lambda x: len(tokenize(x)))
    content_word_counts = df[text_column].apply(count_content_words)
    
    print("Text Length Analysis:")
    print(f"  Total samples: {len(df)}")
    print(f"  Token length - Mean: {text_lengths.mean():.1f}, Median: {text_lengths.median():.1f}")
    print(f"  Token length - Min: {text_lengths.min()}, Max: {text_lengths.max()}")
    print(f"  Content words - Mean: {content_word_counts.mean():.1f}, Median: {content_word_counts.median():.1f}")
    print(f"  Low-content samples (<5 words): {(content_word_counts < 5).sum()}")


def get_text_quality_stats(df, text_column='text'):
    """Get statistics about text quality"""
    if text_column not in df.columns:
        print(f"Column '{text_column}' not found in dataframe")
        return {}
    
    stats = {}
    
    # Basic stats
    stats['total_samples'] = len(df)
    stats['empty_texts'] = df[text_column].isna().sum()
    stats['very_short'] = df[text_column].apply(lambda x: len(str(x).strip()) < 10).sum()
    
    # Content analysis
    content_words = df[text_column].apply(count_content_words)
    stats['low_content'] = (content_words < 5).sum()
    stats['avg_content_words'] = content_words.mean()
    
    # Token analysis
    token_counts = df[text_column].apply(lambda x: len(tokenize(x)))
    stats['avg_tokens'] = token_counts.mean()
    stats['max_tokens'] = token_counts.max()
    
    return stats


if __name__ == "__main__":
    # Test the functions
    print("Testing text processing functions...")
    
    # Test replace_repeats
    text = "Thank you. Thank you. Thank you. Thank you. Thank you."
    cleaned = replace_repeats(text, 2, "[REPEAT]")
    print(f"Original: {text}")
    print(f"Cleaned: {cleaned}")
    print(f"Is low content: {is_low_content(cleaned)}")
    
    # Test with sample dataframe
    sample_data = {
        'text': [
            '"Hello, how are you today?"',
            '"Good good good good good good."',
            '"I like English very much and I study it every day."',
            '"Um, uh, well..."'
        ],
        'grammar': [7.5, 3.0, 8.0, 2.5]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"\nOriginal dataframe: {len(df)} samples")
    
    cleaned_df = clean_dataframe(df, remove_low_content=True, filter_scores=True)
    print(f"Cleaned dataframe: {len(cleaned_df)} samples")
    
    # Print quality stats
    stats = get_text_quality_stats(df)
    print(f"\nText quality stats: {stats}")
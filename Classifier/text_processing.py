import re
from typing import List, Set

import nltk
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Base English stop‑words (includes pronouns, articles, auxiliaries…)
NLTK_STOPWORDS: Set[str] = set(stopwords.words("english"))

# Extra “fillers” commonly seen in ASR noise, plus common punctuation
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
    Print the most common `proportion` of words in df['text'], sorted descending by document frequency.
    Each word is counted at most once per row.
    Removes punctuation and stopwords.
    """

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

if __name__ == "__main__":
    text = "Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. Thank you."
    print(replace_repeats(text, 2, "[REPEAT]"))
    print(is_low_content(replace_repeats(text, 2, "[REPEAT]")))

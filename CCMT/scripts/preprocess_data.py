#!/usr/bin/env python3
"""
Data preprocessing script for CCMT English Speaking Scoring
Usage: python scripts/preprocess_data.py --csv_path data/scores.csv --output_dir preprocessed_data/
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path and prioritize local modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # Insert at beginning to prioritize local modules

from utils.text_utils import (
    clean_transcript, normalize_text, preprocess_for_scoring,
    validate_transcript_quality, get_text_statistics, batch_process_texts
)
from utils.audio_utils import (
    load_audio_safe, validate_audio_file, calculate_audio_features,
    batch_process_audio
)
from utils.translation_utils import (
    batch_translate, validate_translation_quality, save_translation_cache,
    estimate_translation_cost, create_translation_report
)
from models import create_en_vi_translator


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Preprocess data for CCMT training")
    
    # Input/Output
    parser.add_argument("--csv_path", type=str, required=True,
                       help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, default="./preprocessed_data",
                       help="Output directory for preprocessed data")
    
    # Processing options
    parser.add_argument("--validate_audio", action="store_true",
                       help="Validate audio files")
    parser.add_argument("--process_text", action="store_true", 
                       help="Process and clean text transcripts")
    parser.add_argument("--translate_text", action="store_true",
                       help="Translate English text to Vietnamese")
    parser.add_argument("--compute_statistics", action="store_true",
                       help="Compute dataset statistics")
    
    # Audio processing
    parser.add_argument("--min_audio_duration", type=float, default=0.5,
                       help="Minimum audio duration in seconds")
    parser.add_argument("--max_audio_duration", type=float, default=60.0,
                       help="Maximum audio duration in seconds")
    
    # Text processing
    parser.add_argument("--min_text_length", type=int, default=5,
                       help="Minimum text length in words")
    parser.add_argument("--clean_text", action="store_true",
                       help="Apply text cleaning")
    
    # Translation
    parser.add_argument("--translation_model", type=str, default="opus",
                       help="Translation model to use")
    parser.add_argument("--translation_batch_size", type=int, default=32,
                       help="Batch size for translation")
    parser.add_argument("--cache_translations", action="store_true",
                       help="Cache translations to file")
    
    # Filtering
    parser.add_argument("--remove_invalid", action="store_true",
                       help="Remove invalid samples from dataset")
    
    # Misc
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Process only a sample of the data")
    
    return parser.parse_args()


def load_and_validate_csv(csv_path: str, sample_size: int = None):
    """Load and validate CSV data"""
    logging.info(f"Loading CSV from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} samples")
    
    # Check required columns
    required_columns = ['absolute_path', 'text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Sample data if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logging.info(f"Sampled {len(df)} samples for processing")
    
    # Basic validation
    logging.info("Validating CSV data...")
    valid_mask = df['absolute_path'].notna() & df['text'].notna()
    invalid_count = (~valid_mask).sum()
    
    if invalid_count > 0:
        logging.warning(f"Found {invalid_count} samples with missing data")
        df = df[valid_mask].reset_index(drop=True)
    
    # Check score columns
    score_columns = ['vocabulary', 'grammar', 'content']
    available_score_columns = [col for col in score_columns if col in df.columns]
    logging.info(f"Available score columns: {available_score_columns}")
    
    return df


def validate_audio_files(df: pd.DataFrame, min_duration: float, max_duration: float):
    """Validate audio files in the dataset"""
    logging.info("Validating audio files...")
    
    valid_audio = []
    audio_stats = []
    
    for idx, row in tqdm(df.iterrows(), desc="Validating audio", total=len(df)):
        audio_path = row['absolute_path']
        
        # Check if file exists
        if not Path(audio_path).exists():
            valid_audio.append(False)
            audio_stats.append({
                'valid': False,
                'error': 'File not found',
                'duration': 0.0
            })
            continue
        
        # Validate audio file
        is_valid = validate_audio_file(
            audio_path, 
            min_duration=min_duration,
            max_duration=max_duration
        )
        valid_audio.append(is_valid)
        
        # Get audio features if valid
        if is_valid:
            try:
                audio = load_audio_safe(audio_path)
                if audio is not None:
                    features = calculate_audio_features(audio)
                    audio_stats.append({
                        'valid': True,
                        'duration': features['duration'],
                        'rms': features['rms'],
                        'peak': features['peak']
                    })
                else:
                    audio_stats.append({
                        'valid': False,
                        'error': 'Failed to load',
                        'duration': 0.0
                    })
                    valid_audio[-1] = False
            except Exception as e:
                audio_stats.append({
                    'valid': False,
                    'error': str(e),
                    'duration': 0.0
                })
                valid_audio[-1] = False
        else:
            audio_stats.append({
                'valid': False,
                'error': 'Duration out of range',
                'duration': 0.0
            })
    
    df['audio_valid'] = valid_audio
    valid_count = sum(valid_audio)
    logging.info(f"Valid audio files: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)")
    
    return df, audio_stats


def process_text_data(df: pd.DataFrame, clean_text: bool, min_text_length: int):
    """Process and clean text data"""
    logging.info("Processing text data...")
    
    processed_texts = []
    text_valid = []
    text_stats = []
    
    for text in tqdm(df['text'], desc="Processing text"):
        # Clean text if requested
        if clean_text:
            processed_text = preprocess_for_scoring(text)
        else:
            processed_text = text.strip()
        
        # Validate text quality
        is_valid = validate_transcript_quality(
            processed_text,
            min_words=min_text_length
        )
        
        # Get text statistics
        stats = get_text_statistics(processed_text)
        
        processed_texts.append(processed_text)
        text_valid.append(is_valid)
        text_stats.append(stats)
    
    df['processed_text'] = processed_texts
    df['text_valid'] = text_valid
    
    valid_count = sum(text_valid)
    logging.info(f"Valid text samples: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)")
    
    return df, text_stats


def translate_texts(df: pd.DataFrame, model_name: str, batch_size: int, cache_translations: bool, output_dir: str):
    """Translate English texts to Vietnamese"""
    logging.info("Translating texts to Vietnamese...")
    
    # Create translator
    translator = create_en_vi_translator(model_type=model_name)
    
    # Get texts to translate
    texts_to_translate = df['processed_text'].fillna('').tolist()
    
    # Estimate cost
    cost = estimate_translation_cost(texts_to_translate)
    logging.info(f"Estimated translation cost: ${cost:.4f}")
    
    # Translate in batches
    vietnamese_texts = batch_translate(
        texts_to_translate,
        translator,
        batch_size=batch_size
    )
    
    # Validate translations
    translation_quality = []
    for orig, trans in zip(texts_to_translate, vietnamese_texts):
        quality = validate_translation_quality(orig, trans)
        translation_quality.append(quality)
    
    df['vietnamese_text'] = vietnamese_texts
    df['translation_valid'] = translation_quality
    
    valid_translations = sum(translation_quality)
    logging.info(f"Valid translations: {valid_translations}/{len(df)} ({valid_translations/len(df)*100:.1f}%)")
    
    # Save translation cache if requested
    if cache_translations:
        cache_file = os.path.join(output_dir, "translation_cache.json")
        if hasattr(translator, '_cache'):
            save_translation_cache(translator._cache, cache_file)
            logging.info(f"Translation cache saved to: {cache_file}")
    
    return df


def compute_dataset_statistics(df: pd.DataFrame, audio_stats: list, text_stats: list):
    """Compute comprehensive dataset statistics"""
    logging.info("Computing dataset statistics...")
    
    stats = {
        'total_samples': len(df),
        'audio_statistics': {},
        'text_statistics': {},
        'score_statistics': {}
    }
    
    # Audio statistics
    if audio_stats:
        valid_audio_stats = [s for s in audio_stats if s.get('valid', False)]
        if valid_audio_stats:
            durations = [s['duration'] for s in valid_audio_stats]
            rms_values = [s.get('rms', 0) for s in valid_audio_stats]
            
            stats['audio_statistics'] = {
                'valid_files': len(valid_audio_stats),
                'avg_duration': float(np.mean(durations)),
                'min_duration': float(np.min(durations)),
                'max_duration': float(np.max(durations)),
                'avg_rms': float(np.mean(rms_values)),
                'total_hours': float(np.sum(durations) / 3600)
            }
    
    # Text statistics
    if text_stats:
        valid_text_stats = [s for s in text_stats if not s.get('is_low_content', True)]
        if valid_text_stats:
            char_counts = [s['char_count'] for s in valid_text_stats]
            word_counts = [s['word_count'] for s in valid_text_stats]
            
            stats['text_statistics'] = {
                'valid_texts': len(valid_text_stats),
                'avg_char_count': float(np.mean(char_counts)),
                'avg_word_count': float(np.mean(word_counts)),
                'min_word_count': int(np.min(word_counts)),
                'max_word_count': int(np.max(word_counts))
            }
    
    # Score statistics
    score_columns = ['vocabulary', 'grammar', 'content']
    for col in score_columns:
        if col in df.columns:
            scores = df[col].dropna()
            if len(scores) > 0:
                stats['score_statistics'][col] = {
                    'count': len(scores),
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max()),
                    'distribution': scores.value_counts().sort_index().to_dict()
                }
    
    return stats


def filter_invalid_samples(df: pd.DataFrame, remove_invalid: bool):
    """Filter out invalid samples if requested"""
    if not remove_invalid:
        return df
    
    logging.info("Filtering invalid samples...")
    
    # Create overall validity mask
    validity_columns = ['audio_valid', 'text_valid', 'translation_valid']
    validity_mask = pd.Series([True] * len(df))
    
    for col in validity_columns:
        if col in df.columns:
            validity_mask &= df[col]
    
    valid_df = df[validity_mask].reset_index(drop=True)
    removed_count = len(df) - len(valid_df)
    
    logging.info(f"Removed {removed_count} invalid samples. Remaining: {len(valid_df)}")
    
    return valid_df


def save_results(df: pd.DataFrame, stats: dict, output_dir: str):
    """Save preprocessing results"""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save processed dataset
    output_csv = os.path.join(output_dir, "preprocessed_data.csv")
    df.to_csv(output_csv, index=False)
    logging.info(f"Preprocessed dataset saved to: {output_csv}")
    
    # Save statistics
    stats_file = os.path.join(output_dir, "preprocessing_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Statistics saved to: {stats_file}")
    
    # Save summary report
    summary_file = os.path.join(output_dir, "preprocessing_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("CCMT Data Preprocessing Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total samples: {stats['total_samples']}\n")
        
        if 'audio_statistics' in stats and stats['audio_statistics']:
            f.write(f"Valid audio files: {stats['audio_statistics']['valid_files']}\n")
            f.write(f"Total audio hours: {stats['audio_statistics']['total_hours']:.2f}\n")
        
        if 'text_statistics' in stats and stats['text_statistics']:
            f.write(f"Valid text samples: {stats['text_statistics']['valid_texts']}\n")
            f.write(f"Average words per text: {stats['text_statistics']['avg_word_count']:.1f}\n")
        
        if 'score_statistics' in stats:
            f.write("\nScore distributions:\n")
            for score_type, score_stats in stats['score_statistics'].items():
                f.write(f"  {score_type}: {score_stats['count']} samples, "
                       f"mean={score_stats['mean']:.2f}, std={score_stats['std']:.2f}\n")
    
    logging.info(f"Summary report saved to: {summary_file}")


def main():
    """Main preprocessing function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_file = os.path.join(args.output_dir, "preprocessing.log")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=log_file)
    
    logging.info("Starting data preprocessing...")
    logging.info(f"Input CSV: {args.csv_path}")
    logging.info(f"Output directory: {args.output_dir}")
    
    # Load and validate CSV
    df = load_and_validate_csv(args.csv_path, args.sample_size)
    
    # Initialize statistics
    audio_stats = []
    text_stats = []
    
    # Validate audio files
    if args.validate_audio:
        df, audio_stats = validate_audio_files(
            df, args.min_audio_duration, args.max_audio_duration
        )
    
    # Process text data
    if args.process_text:
        df, text_stats = process_text_data(
            df, args.clean_text, args.min_text_length
        )
    
    # Translate texts
    if args.translate_text:
        df = translate_texts(
            df, args.translation_model, args.translation_batch_size,
            args.cache_translations, args.output_dir
        )
    
    # Compute statistics
    if args.compute_statistics:
        stats = compute_dataset_statistics(df, audio_stats, text_stats)
    else:
        stats = {'total_samples': len(df)}
    
    # Filter invalid samples
    df = filter_invalid_samples(df, args.remove_invalid)
    
    # Update final stats
    stats['final_samples'] = len(df)
    
    # Save results
    save_results(df, stats, args.output_dir)
    
    logging.info("Data preprocessing completed successfully!")
    
    return df, stats


if __name__ == "__main__":
    try:
        df, stats = main()
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        sys.exit(1)
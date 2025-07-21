import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa
import numpy as np
import os
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, Wav2Vec2Processor, AutoConfig
from .text_processor import TextProcessor
import warnings
warnings.filterwarnings("ignore")


class ESLCCMTDataset(Dataset):
    """
    Dataset for ESL grading using CCMT architecture
    Handles audio, English text, and Vietnamese text modalities
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 # Tokenizers
                 english_tokenizer_name: str = "bert-base-uncased",
                 vietnamese_tokenizer_name: str = "vinai/phobert-base-v2",
                 # Audio processor
                 audio_processor_name: str = "facebook/wav2vec2-base-960h",
                 # Text processing
                 text_processor: Optional[TextProcessor] = None,
                 # Audio processing parameters
                 num_audio_chunks: int = 10,
                 chunk_length_sec: int = 30,
                 sample_rate: int = 16000,
                 # Text parameters
                 max_text_length: int = 512,
                 # Data filtering
                 remove_low_content: bool = True,
                 filter_scores: bool = True,
                 # Augmentation
                 is_train: bool = False,
                 # Caching
                 use_text_cache: bool = True):
        
        super().__init__()
        
        # Store parameters
        self.num_audio_chunks = num_audio_chunks
        self.chunk_length_sec = chunk_length_sec
        self.sample_rate = sample_rate
        self.max_text_length = max_text_length
        self.is_train = is_train
        self.use_text_cache = use_text_cache
        
        # Initialize tokenizers
        self.english_tokenizer = AutoTokenizer.from_pretrained(english_tokenizer_name)
        self.vietnamese_tokenizer = AutoTokenizer.from_pretrained(vietnamese_tokenizer_name)
        
        # Get model-specific max lengths
        self.english_max_length = self._get_model_max_length(english_tokenizer_name, max_text_length)
        self.vietnamese_max_length = self._get_model_max_length(vietnamese_tokenizer_name, max_text_length)
        
        print(f"English tokenizer max length: {self.english_max_length}")
        print(f"Vietnamese tokenizer max length: {self.vietnamese_max_length}")
        
        # Initialize audio processor
        self.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_name)
        
        # Initialize text processor for ASR and translation
        if text_processor is not None:
            self.text_processor = text_processor
        else:
            try:
                self.text_processor = TextProcessor()
            except Exception as e:
                print(f"Warning: Could not initialize TextProcessor: {e}")
                print("Text processing will be limited")
                self.text_processor = None
        
        # Clean and process dataframe
        self.dataframe = self._clean_dataframe(dataframe, remove_low_content, filter_scores)
        
        # Build question type mapping
        self.question_type_map = {
            1: "Answer some questions about you personally.",
            2: "Choose one of several options in a situation.", 
            3: "Give your opinion about a topic."
        }
        
        # Process data
        self._prepare_data()
        
        # Text cache for processed transcripts and translations
        self.text_cache = {} if use_text_cache else None
        
    def _get_model_max_length(self, model_name, fallback_length):
        """Get the maximum sequence length for a specific model"""
        try:
            config = AutoConfig.from_pretrained(model_name)
            if hasattr(config, 'max_position_embeddings'):
                # Leave buffer for special tokens
                max_len = min(config.max_position_embeddings - 2, fallback_length)
                return max_len
            else:
                return fallback_length
        except Exception as e:
            print(f"Warning: Could not get max length for {model_name}: {e}")
            return fallback_length
        
    def _clean_dataframe(self, df, remove_low_content, filter_scores):
        """Clean dataframe similar to original implementation"""
        try:
            from .text_processing import clean_dataframe  # Import from our text_processing module
            return clean_dataframe(df, remove_low_content, filter_scores)
        except ImportError as e:
            print(f"Warning: Could not import text_processing module: {e}")
            print("Using original dataframe without cleaning")
            return df
        except Exception as e:
            print(f"Warning: Error in data cleaning: {e}, using original dataframe")
            return df
    
    def _prepare_data(self):
        """Prepare data lists"""
        # Get basic data
        self.question_types = self.dataframe['question_type'].astype(int).tolist()
        self.scores = self.dataframe['grammar'].astype(float).tolist()
        
        # Process audio paths
        if 'absolute_path' in self.dataframe.columns:
            # Update paths if needed (adapt to your system)
            self.dataframe['absolute_path'] = self.dataframe['absolute_path'].str.replace(
                "/mnt/son_usb/DATA_Vocal", "/media/gpus/Data/DATA_Vocal"
            )
            self.audio_paths = self.dataframe['absolute_path'].tolist()
        else:
            self.audio_paths = [None] * len(self.question_types)
        
        # If text is provided in dataframe, use it as ground truth English text
        if 'text' in self.dataframe.columns:
            raw_texts = self.dataframe['text'].tolist()
            # Remove quotation marks from text
            self.ground_truth_texts = [t[2:-1] if t.startswith('""') and t.endswith('""') else t 
                                      for t in raw_texts]
        else:
            self.ground_truth_texts = [None] * len(self.question_types)
    
    def _process_audio_file(self, audio_path: str) -> torch.Tensor:
        """Process audio file to tensor chunks"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Apply augmentations if training
            if self.is_train:
                audio = self._apply_audio_augmentations(audio, sr)
            
            # Create fixed chunks
            audio_chunks = self._fixed_chunk_audio(audio, sr)
            
            # Process chunks with audio processor
            chunk_samples = int(self.chunk_length_sec * self.sample_rate)
            processed_chunks = []
            
            for chunk in audio_chunks:
                inputs = self.audio_processor(chunk, sampling_rate=self.sample_rate, return_tensors="pt")
                chunk_tensor = inputs.input_values.squeeze(0)
                
                # Ensure fixed length
                if chunk_tensor.shape[0] < chunk_samples:
                    pad_length = chunk_samples - chunk_tensor.shape[0]
                    chunk_tensor = torch.nn.functional.pad(chunk_tensor, (0, pad_length), 'constant', 0)
                elif chunk_tensor.shape[0] > chunk_samples:
                    chunk_tensor = chunk_tensor[:chunk_samples]
                
                processed_chunks.append(chunk_tensor)
            
            audio_tensor = torch.stack(processed_chunks)
            return audio_tensor
            
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            # Return dummy audio tensor
            chunk_samples = int(self.chunk_length_sec * self.sample_rate)
            return torch.zeros(self.num_audio_chunks, chunk_samples)
    
    def _fixed_chunk_audio(self, audio, sr):
        """Create fixed number of audio chunks"""
        chunk_samples = int(self.chunk_length_sec * sr)
        audio_length = len(audio)
        
        if audio_length < chunk_samples:
            audio = np.pad(audio, (0, chunk_samples - audio_length), mode='constant')
            audio_length = len(audio)
        
        if self.num_audio_chunks == 1:
            starts = [0]
        else:
            max_start = max(0, audio_length - chunk_samples)
            starts = np.linspace(0, max_start, self.num_audio_chunks, dtype=int)
        
        chunks = []
        for start in starts:
            end = start + chunk_samples
            chunk = audio[start:end]
            chunks.append(chunk)
        
        return chunks
    
    def _apply_audio_augmentations(self, audio, sr):
        """Apply audio augmentations during training"""
        # Simple augmentations - you can expand this
        import random
        
        # Add noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.005, audio.shape)
            audio = audio + noise
        
        # Volume change
        if random.random() < 0.3:
            volume_factor = random.uniform(0.8, 1.2)
            audio = audio * volume_factor
        
        return np.clip(audio, -1.0, 1.0)
    
    def _get_bilingual_text(self, idx: int) -> Tuple[str, str]:
        """Get English and Vietnamese text for the sample"""
        cache_key = f"text_{idx}"
        
        # Check cache first
        if self.text_cache is not None and cache_key in self.text_cache:
            return self.text_cache[cache_key]
        
        # Use ground truth text if available, otherwise transcribe from audio
        if self.ground_truth_texts[idx] is not None:
            english_text = self.ground_truth_texts[idx]
        else:
            # Transcribe from audio
            audio_path = self.audio_paths[idx]
            if audio_path and os.path.exists(audio_path) and self.text_processor is not None:
                english_text = self.text_processor.transcribe_audio(audio_path, language="en")
            else:
                english_text = "Sample English text for demonstration"  # Fallback for demo
        
        # Translate to Vietnamese
        if self.text_processor is not None:
            vietnamese_text = self.text_processor.translate_to_vietnamese(english_text)
        else:
            # Simple fallback for demo
            vietnamese_text = "Văn bản tiếng Việt mẫu để trình diễn"
        
        # Add question type context
        qtype = self.question_types[idx]
        question_context = self.question_type_map.get(qtype, "")
        
        # Format texts with context
        english_text_formatted = f"[Question Type: {question_context}] {english_text}"
        vietnamese_text_formatted = f"[Loại câu hỏi: {question_context}] {vietnamese_text}"
        
        result = (english_text_formatted, vietnamese_text_formatted)
        
        # Cache result
        if self.text_cache is not None:
            self.text_cache[cache_key] = result
        
        return result
    
    def _tokenize_text(self, text: str, tokenizer, is_vietnamese: bool = False):
        """Tokenize text with specified tokenizer"""
        # Use appropriate max length based on language
        max_length = self.vietnamese_max_length if is_vietnamese else self.english_max_length
        
        encoded = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        # Get basic info
        score = torch.tensor(self.scores[idx], dtype=torch.float32)
        question_type = self.question_types[idx]
        
        # Process audio
        audio_path = self.audio_paths[idx]
        if audio_path and os.path.exists(audio_path):
            try:
                audio_chunks = self._process_audio_file(audio_path)
                has_audio = True
            except Exception as e:
                print(f"Failed to process audio {audio_path}: {e}")
                chunk_samples = int(self.chunk_length_sec * self.sample_rate)
                audio_chunks = torch.zeros(self.num_audio_chunks, chunk_samples)
                has_audio = False
        else:
            chunk_samples = int(self.chunk_length_sec * self.sample_rate)
            audio_chunks = torch.zeros(self.num_audio_chunks, chunk_samples)
            has_audio = False
        
        # Get bilingual text
        english_text, vietnamese_text = self._get_bilingual_text(idx)
        
        # Tokenize texts
        english_tokens = self._tokenize_text(english_text, self.english_tokenizer, is_vietnamese=False)
        vietnamese_tokens = self._tokenize_text(vietnamese_text, self.vietnamese_tokenizer, is_vietnamese=True)
        
        return {
            'audio_chunks': audio_chunks,
            'english_input_ids': english_tokens['input_ids'],
            'english_attention_mask': english_tokens['attention_mask'],
            'vietnamese_input_ids': vietnamese_tokens['input_ids'],
            'vietnamese_attention_mask': vietnamese_tokens['attention_mask'],
            'score': score,
            'question_type': question_type,
            'has_audio': has_audio,
            'english_text': english_text,
            'vietnamese_text': vietnamese_text
        }


def get_ccmt_collate_fn():
    """Collate function for CCMT dataset"""
    def collate_fn(batch):
        # Stack tensors
        audio_chunks = torch.stack([item['audio_chunks'] for item in batch])
        english_input_ids = torch.stack([item['english_input_ids'] for item in batch])
        english_attention_mask = torch.stack([item['english_attention_mask'] for item in batch])
        vietnamese_input_ids = torch.stack([item['vietnamese_input_ids'] for item in batch])
        vietnamese_attention_mask = torch.stack([item['vietnamese_attention_mask'] for item in batch])
        scores = torch.stack([item['score'] for item in batch])
        question_types = torch.tensor([item['question_type'] for item in batch], dtype=torch.long)
        has_audio = [item['has_audio'] for item in batch]
        
        return {
            'audio_chunks': audio_chunks,
            'english_input_ids': english_input_ids,
            'english_attention_mask': english_attention_mask,
            'vietnamese_input_ids': vietnamese_input_ids,
            'vietnamese_attention_mask': vietnamese_attention_mask,
            'score': scores,
            'question_type': question_types,
            'has_audio': has_audio
        }
    
    return collate_fn
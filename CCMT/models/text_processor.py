import torch
import librosa
import asyncio
from typing import List, Optional, Union, Tuple
from transformers import AutoProcessor, pipeline
import warnings
warnings.filterwarnings("ignore")


class TextProcessor:
    """
    Handles ASR transcription and translation for CCMT pipeline
    """
    def __init__(self, 
                 asr_model_name="openai/whisper-base",
                 translation_model_name="Helsinki-NLP/opus-mt-en-vi",
                 device="cpu"):  # Use CPU by default to avoid multiprocessing issues
        self.device = device
        self.asr_model_name = asr_model_name
        self.translation_model_name = translation_model_name
        
        try:
            # Initialize ASR pipeline - use CPU to avoid multiprocessing CUDA issues
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=asr_model_name,
                device=-1,  # Always use CPU for multiprocessing compatibility
                return_timestamps=False
            )
            print(f"ASR model {asr_model_name} loaded successfully on CPU")
        except Exception as e:
            print(f"Warning: Could not load ASR model {asr_model_name}: {e}")
            self.asr_pipeline = None
        
        try:
            # Initialize translation pipeline - use CPU to avoid multiprocessing CUDA issues
            self.translation_pipeline = pipeline(
                "translation",
                model=translation_model_name,
                device=-1,  # Always use CPU for multiprocessing compatibility
            )
            print(f"Translation model {translation_model_name} loaded successfully on CPU")
        except Exception as e:
            print(f"Warning: Could not load translation model {translation_model_name}: {e}")
            self.translation_pipeline = None
        
    def transcribe_audio(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe audio file to text using ASR
        
        Args:
            audio_path: path to audio file
            language: language code (default: "en" for English)
            
        Returns:
            transcribed text
        """
        if self.asr_pipeline is None:
            return "ASR model not available"
            
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Transcribe using Whisper
            result = self.asr_pipeline(
                audio,
                generate_kwargs={
                    "language": language,
                    "task": "transcribe"
                }
            )
            
            transcript = result["text"].strip()
            return transcript
            
        except Exception as e:
            print(f"Error transcribing audio {audio_path}: {e}")
            return "Transcription failed"
    
    def translate_to_vietnamese(self, english_text: str) -> str:
        """
        Translate English text to Vietnamese
        
        Args:
            english_text: input English text
            
        Returns:
            translated Vietnamese text
        """
        if self.translation_pipeline is None:
            return f"Vietnamese translation of: {english_text}"
            
        try:
            if not english_text or english_text.strip() == "":
                return ""
            
            # Handle long texts by truncating or chunking
            max_input_length = 400  # Safe limit for translation model
            
            if len(english_text.split()) > max_input_length:
                print(f"Warning: Text too long ({len(english_text.split())} words), truncating to {max_input_length} words")
                words = english_text.split()[:max_input_length]
                english_text = " ".join(words)
            
            # Translate using Helsinki-NLP model with explicit max_length
            result = self.translation_pipeline(
                english_text, 
                max_length=512,  # Set explicit max_length
                truncation=True  # Enable truncation as backup
            )
            vietnamese_text = result[0]["translation_text"].strip()
            
            return vietnamese_text
            
        except Exception as e:
            print(f"Error translating text (length: {len(english_text)} chars): {e}")
            return f"Vietnamese translation of: {english_text[:100]}..."  # Fallback with truncation
    
    def process_audio_to_bilingual_text(self, audio_path: str) -> Tuple[str, str]:
        """
        Process audio file to get both English transcript and Vietnamese translation
        
        Args:
            audio_path: path to audio file
            
        Returns:
            tuple of (english_text, vietnamese_text)
        """
        # Step 1: Transcribe audio to English
        english_text = self.transcribe_audio(audio_path, language="en")
        
        # Step 2: Translate English to Vietnamese
        vietnamese_text = self.translate_to_vietnamese(english_text)
        
        return english_text, vietnamese_text
    
    def batch_process_audio(self, audio_paths: List[str]) -> List[Tuple[str, str]]:
        """
        Process multiple audio files in batch
        
        Args:
            audio_paths: list of audio file paths
            
        Returns:
            list of tuples (english_text, vietnamese_text)
        """
        results = []
        for audio_path in audio_paths:
            english_text, vietnamese_text = self.process_audio_to_bilingual_text(audio_path)
            results.append((english_text, vietnamese_text))
        
        return results


class AsyncTextProcessor(TextProcessor):
    """
    Async version of TextProcessor for better performance with large batches
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def async_transcribe_audio(self, audio_path: str, language: str = "en") -> str:
        """Async version of transcribe_audio"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe_audio, audio_path, language)
    
    async def async_translate_to_vietnamese(self, english_text: str) -> str:
        """Async version of translate_to_vietnamese"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.translate_to_vietnamese, english_text)
    
    async def async_process_audio_to_bilingual_text(self, audio_path: str) -> Tuple[str, str]:
        """Async version of process_audio_to_bilingual_text"""
        # Transcribe audio
        english_text = await self.async_transcribe_audio(audio_path, language="en")
        
        # Translate to Vietnamese
        vietnamese_text = await self.async_translate_to_vietnamese(english_text)
        
        return english_text, vietnamese_text
    
    async def async_batch_process_audio(self, audio_paths: List[str]) -> List[Tuple[str, str]]:
        """Process multiple audio files asynchronously"""
        tasks = [
            self.async_process_audio_to_bilingual_text(path) 
            for path in audio_paths
        ]
        results = await asyncio.gather(*tasks)
        return results
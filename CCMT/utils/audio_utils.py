"""
Audio processing utilities for CCMT
"""

import torch
import torchaudio
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def load_audio_safe(audio_path: str, sample_rate: int = 16000) -> Optional[torch.Tensor]:
    """
    Safely load audio file with error handling
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
    
    Returns:
        Audio tensor or None if loading fails
    """
    try:
        waveform, orig_sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)  # Remove channel dimension
        
    except Exception as e:
        logger.warning(f"Failed to load audio {audio_path}: {e}")
        return None


def normalize_audio(waveform: torch.Tensor, method: str = "rms") -> torch.Tensor:
    """
    Normalize audio waveform
    
    Args:
        waveform: Input audio tensor
        method: Normalization method ("rms", "peak", "standard")
    
    Returns:
        Normalized audio tensor
    """
    if waveform.numel() == 0:
        return waveform
    
    if method == "rms":
        rms = torch.sqrt(torch.mean(waveform**2))
        if rms > 1e-8:
            waveform = waveform / (rms + 1e-8)
    
    elif method == "peak":
        peak = torch.max(torch.abs(waveform))
        if peak > 1e-8:
            waveform = waveform / peak
    
    elif method == "standard":
        mean = torch.mean(waveform)
        std = torch.std(waveform)
        if std > 1e-8:
            waveform = (waveform - mean) / std
    
    return waveform


def calculate_audio_features(waveform: torch.Tensor, sample_rate: int = 16000) -> Dict[str, float]:
    """
    Calculate basic audio features
    
    Args:
        waveform: Audio tensor
        sample_rate: Sample rate
    
    Returns:
        Dictionary of audio features
    """
    if waveform.numel() == 0:
        return {
            'duration': 0.0,
            'rms': 0.0,
            'peak': 0.0,
            'zero_crossing_rate': 0.0,
            'spectral_centroid': 0.0
        }
    
    # Basic time-domain features
    duration = len(waveform) / sample_rate
    rms = torch.sqrt(torch.mean(waveform**2)).item()
    peak = torch.max(torch.abs(waveform)).item()
    
    # Zero crossing rate
    zero_crossings = torch.sum((waveform[1:] * waveform[:-1]) < 0).float()
    zcr = zero_crossings / len(waveform) if len(waveform) > 1 else 0.0
    
    # Simple spectral centroid approximation
    try:
        # Compute FFT
        fft = torch.fft.fft(waveform)
        magnitude = torch.abs(fft[:len(fft)//2])
        freqs = torch.linspace(0, sample_rate/2, len(magnitude))
        
        # Spectral centroid
        if torch.sum(magnitude) > 0:
            spectral_centroid = torch.sum(freqs * magnitude) / torch.sum(magnitude)
        else:
            spectral_centroid = 0.0
        
        spectral_centroid = spectral_centroid.item()
        
    except Exception:
        spectral_centroid = 0.0
    
    return {
        'duration': duration,
        'rms': rms,
        'peak': peak,
        'zero_crossing_rate': zcr.item() if isinstance(zcr, torch.Tensor) else zcr,
        'spectral_centroid': spectral_centroid
    }


def detect_speech_segments(waveform: torch.Tensor, 
                          threshold: float = 0.01,
                          min_segment_length: float = 0.1,
                          sample_rate: int = 16000) -> List[Tuple[int, int]]:
    """
    Detect speech segments in audio using energy-based approach
    
    Args:
        waveform: Audio tensor
        threshold: Energy threshold for speech detection
        min_segment_length: Minimum segment length in seconds
        sample_rate: Sample rate
    
    Returns:
        List of (start, end) sample indices for speech segments
    """
    if waveform.numel() == 0:
        return []
    
    # Frame-based analysis
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)    # 10ms hop
    
    # Compute frame energy
    frames = []
    for i in range(0, len(waveform) - frame_length + 1, hop_length):
        frame = waveform[i:i + frame_length]
        energy = torch.mean(frame**2)
        frames.append(energy.item())
    
    # Find speech frames
    speech_frames = [i for i, energy in enumerate(frames) if energy > threshold]
    
    if not speech_frames:
        return []
    
    # Group consecutive frames into segments
    segments = []
    start_frame = speech_frames[0]
    end_frame = speech_frames[0]
    
    for frame in speech_frames[1:]:
        if frame == end_frame + 1:  # Consecutive frame
            end_frame = frame
        else:  # Gap found, save current segment and start new one
            # Convert to sample indices
            start_sample = start_frame * hop_length
            end_sample = min((end_frame + 1) * hop_length + frame_length, len(waveform))
            
            # Check minimum length
            if (end_sample - start_sample) / sample_rate >= min_segment_length:
                segments.append((start_sample, end_sample))
            
            start_frame = frame
            end_frame = frame
    
    # Add final segment
    start_sample = start_frame * hop_length
    end_sample = min((end_frame + 1) * hop_length + frame_length, len(waveform))
    if (end_sample - start_sample) / sample_rate >= min_segment_length:
        segments.append((start_sample, end_sample))
    
    return segments


def trim_silence(waveform: torch.Tensor, 
                threshold: float = 0.01,
                frame_length: int = 400,
                hop_length: int = 160) -> torch.Tensor:
    """
    Trim silence from beginning and end of audio
    
    Args:
        waveform: Audio tensor
        threshold: Energy threshold
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
    
    Returns:
        Trimmed audio tensor
    """
    if waveform.numel() == 0:
        return waveform
    
    # Compute frame energies
    energies = []
    for i in range(0, len(waveform) - frame_length + 1, hop_length):
        frame = waveform[i:i + frame_length]
        energy = torch.mean(frame**2)
        energies.append(energy.item())
    
    if not energies:
        return waveform
    
    # Find first and last frames above threshold
    first_speech = next((i for i, e in enumerate(energies) if e > threshold), 0)
    last_speech = next((i for i, e in enumerate(reversed(energies)) if e > threshold), 0)
    last_speech = len(energies) - 1 - last_speech
    
    # Convert to sample indices
    start_sample = first_speech * hop_length
    end_sample = min((last_speech + 1) * hop_length + frame_length, len(waveform))
    
    return waveform[start_sample:end_sample]


def pad_or_truncate_audio(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Pad or truncate audio to target length
    
    Args:
        waveform: Audio tensor
        target_length: Target length in samples
    
    Returns:
        Audio tensor of target length
    """
    current_length = len(waveform)
    
    if current_length == target_length:
        return waveform
    
    elif current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        return torch.nn.functional.pad(waveform, (0, padding))
    
    else:
        # Truncate (center crop)
        start = (current_length - target_length) // 2
        return waveform[start:start + target_length]


def batch_process_audio(audio_paths: List[str], 
                       sample_rate: int = 16000,
                       target_length: Optional[int] = None,
                       normalize: bool = True) -> List[Optional[torch.Tensor]]:
    """
    Process a batch of audio files
    
    Args:
        audio_paths: List of audio file paths
        sample_rate: Target sample rate
        target_length: Target length in samples
        normalize: Whether to normalize audio
    
    Returns:
        List of processed audio tensors (None for failed loads)
    """
    processed_audio = []
    
    for path in audio_paths:
        # Load audio
        waveform = load_audio_safe(path, sample_rate)
        
        if waveform is not None:
            # Normalize if requested
            if normalize:
                waveform = normalize_audio(waveform)
            
            # Adjust length if specified
            if target_length is not None:
                waveform = pad_or_truncate_audio(waveform, target_length)
        
        processed_audio.append(waveform)
    
    return processed_audio


def validate_audio_file(audio_path: str, 
                       min_duration: float = 0.5,
                       max_duration: float = 60.0,
                       sample_rate: int = 16000) -> bool:
    """
    Validate if audio file meets requirements
    
    Args:
        audio_path: Path to audio file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        sample_rate: Expected sample rate
    
    Returns:
        True if audio file is valid
    """
    try:
        # Load audio info without loading full file
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate
        
        # Check duration
        if duration < min_duration or duration > max_duration:
            return False
        
        return True
        
    except Exception:
        return False


def create_audio_chunks(waveform: torch.Tensor, 
                       chunk_length: int,
                       overlap: int = 0) -> List[torch.Tensor]:
    """
    Split audio into chunks with optional overlap
    
    Args:
        waveform: Audio tensor
        chunk_length: Length of each chunk in samples
        overlap: Overlap between chunks in samples
    
    Returns:
        List of audio chunks
    """
    if len(waveform) <= chunk_length:
        return [waveform]
    
    chunks = []
    start = 0
    step = chunk_length - overlap
    
    while start < len(waveform):
        end = min(start + chunk_length, len(waveform))
        chunk = waveform[start:end]
        
        # Pad last chunk if necessary
        if len(chunk) < chunk_length:
            chunk = pad_or_truncate_audio(chunk, chunk_length)
        
        chunks.append(chunk)
        start += step
        
        if end >= len(waveform):
            break
    
    return chunks


def merge_audio_chunks(chunks: List[torch.Tensor], overlap: int = 0) -> torch.Tensor:
    """
    Merge audio chunks back into single waveform
    
    Args:
        chunks: List of audio chunks
        overlap: Overlap between chunks in samples
    
    Returns:
        Merged audio tensor
    """
    if not chunks:
        return torch.tensor([])
    
    if len(chunks) == 1:
        return chunks[0]
    
    # Calculate total length
    chunk_length = len(chunks[0])
    step = chunk_length - overlap
    total_length = step * (len(chunks) - 1) + len(chunks[-1])
    
    # Merge chunks
    merged = torch.zeros(total_length)
    
    for i, chunk in enumerate(chunks):
        start = i * step
        end = start + len(chunk)
        
        if overlap > 0 and i > 0:
            # Apply fade in/out for overlapping regions
            fade_length = min(overlap, len(chunk))
            fade_in = torch.linspace(0, 1, fade_length)
            fade_out = torch.linspace(1, 0, fade_length)
            
            # Fade out previous chunk
            if start < len(merged):
                overlap_end = min(start + fade_length, len(merged))
                merged[start:overlap_end] *= fade_out[:overlap_end-start]
            
            # Fade in current chunk
            chunk_faded = chunk.clone()
            chunk_faded[:fade_length] *= fade_in
            
            merged[start:end] += chunk_faded
        else:
            merged[start:end] = chunk
    
    return merged
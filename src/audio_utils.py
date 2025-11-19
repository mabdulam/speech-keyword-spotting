"""
Audio utility functions for loading, preprocessing, and normalizing audio files.

These utilities ensure consistent audio format across the dataset:
- Mono channel
- Fixed sample rate (16kHz)
- Fixed duration (1 second)
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple
from pathlib import Path

from src.config import SAMPLE_RATE, N_SAMPLES, DURATION


def load_audio(filepath: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to the target sample rate.

    Args:
        filepath: Path to the audio file
        sr: Target sample rate (default: 16000 Hz)

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        # Load audio file and convert to mono
        # librosa automatically resamples to target sr
        audio, sample_rate = librosa.load(filepath, sr=sr, mono=True)
        return audio, sample_rate
    except Exception as e:
        raise IOError(f"Failed to load audio file {filepath}: {str(e)}")


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to have zero mean and unit variance.

    This normalization helps with model training stability by ensuring
    consistent amplitude ranges across different recordings.

    Args:
        audio: Input audio signal

    Returns:
        Normalized audio signal
    """
    # Avoid division by zero for silent clips
    std = np.std(audio)
    if std < 1e-8:
        return audio

    normalized = (audio - np.mean(audio)) / std
    return normalized


def pad_or_truncate(audio: np.ndarray, target_length: int = N_SAMPLES) -> np.ndarray:
    """
    Pad or truncate audio to a fixed length.

    This is necessary because:
    1. Neural networks require fixed-size inputs
    2. Different recordings may have slightly different lengths
    3. Ensures consistent feature dimensions

    Padding strategy: Add zeros to the end
    Truncation strategy: Take the first N samples

    Args:
        audio: Input audio signal
        target_length: Desired length in samples (default: 16000 for 1 second at 16kHz)

    Returns:
        Audio signal with exactly target_length samples
    """
    current_length = len(audio)

    if current_length < target_length:
        # Pad with zeros at the end
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
    elif current_length > target_length:
        # Truncate to target length
        audio = audio[:target_length]

    return audio


def preprocess_audio(filepath: str) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single audio file.

    Steps:
    1. Load audio at target sample rate
    2. Ensure mono channel
    3. Pad or truncate to fixed duration
    4. Normalize amplitude

    Args:
        filepath: Path to the audio file

    Returns:
        Preprocessed audio array of shape (N_SAMPLES,)
    """
    # Load audio
    audio, sr = load_audio(filepath)

    # Ensure fixed length
    audio = pad_or_truncate(audio)

    # Normalize
    audio = normalize_audio(audio)

    return audio


def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """
    Add Gaussian noise to audio for data augmentation.

    This can help improve model robustness to noisy environments.

    Args:
        audio: Input audio signal
        noise_factor: Standard deviation of noise to add

    Returns:
        Audio with added noise
    """
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented


def time_shift(audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
    """
    Randomly shift audio in time for data augmentation.

    Args:
        audio: Input audio signal
        shift_max: Maximum shift as fraction of total length

    Returns:
        Time-shifted audio
    """
    shift = np.random.randint(-int(len(audio) * shift_max),
                             int(len(audio) * shift_max))
    return np.roll(audio, shift)


def save_audio(filepath: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    """
    Save audio to a file.

    Args:
        filepath: Output file path
        audio: Audio data to save
        sr: Sample rate
    """
    sf.write(filepath, audio, sr)


if __name__ == "__main__":
    # Simple test/demo
    import matplotlib.pyplot as plt

    # Create a simple sine wave for testing
    duration = 1.0
    freq = 440  # A4 note
    t = np.linspace(0, duration, N_SAMPLES)
    test_audio = np.sin(2 * np.pi * freq * t)

    # Test preprocessing functions
    print(f"Original shape: {test_audio.shape}")

    # Test padding
    short_audio = test_audio[:8000]
    padded = pad_or_truncate(short_audio)
    print(f"Padded shape: {padded.shape}")

    # Test truncation
    long_audio = np.concatenate([test_audio, test_audio])
    truncated = pad_or_truncate(long_audio)
    print(f"Truncated shape: {truncated.shape}")

    # Test normalization
    normalized = normalize_audio(test_audio)
    print(f"Normalized mean: {np.mean(normalized):.6f}, std: {np.std(normalized):.6f}")

    print("Audio utilities tests passed!")

"""
Feature extraction module for audio data.

This module focuses on MFCC (Mel-Frequency Cepstral Coefficients) extraction,
which is a standard feature representation for speech recognition tasks.

MFCCs work by:
1. Converting the audio signal to the frequency domain
2. Mapping frequencies to the mel scale (which matches human perception)
3. Taking the discrete cosine transform of log mel-scale energies
4. Keeping the first N coefficients (typically 13-40)

MFCCs capture the spectral envelope of speech while discarding fine pitch details,
making them robust for speech recognition.
"""

import numpy as np
import librosa
from typing import Tuple

from src.config import N_MFCC, N_FFT, HOP_LENGTH, SAMPLE_RATE, N_MELS


def extract_mfcc(audio: np.ndarray,
                 sr: int = SAMPLE_RATE,
                 n_mfcc: int = N_MFCC,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Extract MFCC features from audio signal.

    Args:
        audio: Audio time series
        sr: Sample rate
        n_mfcc: Number of MFCCs to extract
        n_fft: FFT window size
        hop_length: Number of samples between successive frames

    Returns:
        MFCC feature matrix of shape (n_mfcc, time_frames)
        Each column represents the MFCCs for one time frame
    """
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=N_MELS
    )

    return mfccs


def mfcc_to_summary_vector(mfcc: np.ndarray) -> np.ndarray:
    """
    Convert 2D MFCC matrix to a 1D summary vector.

    This is useful for simple classifiers (like MLPs) that need fixed-size
    feature vectors. We compute statistics over the time dimension:
    - Mean of each MFCC coefficient across time
    - Standard deviation of each MFCC coefficient across time

    This captures both the average spectral characteristics (mean) and
    their variability over time (std).

    Args:
        mfcc: MFCC matrix of shape (n_mfcc, time_frames)

    Returns:
        Summary vector of shape (2 * n_mfcc,)
        First half: means, second half: stds
    """
    # Compute statistics along the time axis (axis=1)
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)

    # Concatenate mean and std into a single vector
    summary = np.concatenate([mean, std])

    return summary


def extract_melspectrogram(audio: np.ndarray,
                          sr: int = SAMPLE_RATE,
                          n_fft: int = N_FFT,
                          hop_length: int = HOP_LENGTH,
                          n_mels: int = N_MELS) -> np.ndarray:
    """
    Extract mel-spectrogram features.

    Mel-spectrograms are an alternative to MFCCs and work well with CNNs
    as they provide a 2D time-frequency representation.

    Args:
        audio: Audio time series
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Number of samples between frames
        n_mels: Number of mel bands

    Returns:
        Log mel-spectrogram of shape (n_mels, time_frames)
    """
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec


def extract_delta_features(mfcc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute delta (velocity) and delta-delta (acceleration) features.

    These capture the temporal dynamics of the MFCCs and can improve
    recognition accuracy by providing information about how features
    change over time.

    Args:
        mfcc: MFCC matrix of shape (n_mfcc, time_frames)

    Returns:
        Tuple of (delta, delta_delta), each of shape (n_mfcc, time_frames)
    """
    # First-order differences (velocity)
    delta = librosa.feature.delta(mfcc, order=1)

    # Second-order differences (acceleration)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    return delta, delta_delta


def normalize_features(features: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance.

    Args:
        features: Feature matrix
        epsilon: Small constant to prevent division by zero

    Returns:
        Normalized features
    """
    mean = np.mean(features)
    std = np.std(features)

    if std < epsilon:
        return features - mean

    normalized = (features - mean) / std
    return normalized


def augment_features_with_deltas(mfcc: np.ndarray) -> np.ndarray:
    """
    Augment MFCC features with delta and delta-delta features.

    This triples the feature dimensionality but can significantly improve
    model performance by capturing temporal dynamics.

    Args:
        mfcc: MFCC matrix of shape (n_mfcc, time_frames)

    Returns:
        Augmented features of shape (3 * n_mfcc, time_frames)
    """
    delta, delta_delta = extract_delta_features(mfcc)

    # Stack along the feature dimension
    augmented = np.vstack([mfcc, delta, delta_delta])

    return augmented


if __name__ == "__main__":
    # Test feature extraction with synthetic data
    from src.audio_utils import N_SAMPLES

    # Create a simple test signal
    duration = 1.0
    freq = 440
    t = np.linspace(0, duration, N_SAMPLES)
    test_audio = np.sin(2 * np.pi * freq * t)

    # Extract MFCC features
    mfcc = extract_mfcc(test_audio)
    print(f"MFCC shape: {mfcc.shape}")
    print(f"Expected: ({N_MFCC}, ~62) frames")

    # Create summary vector
    summary = mfcc_to_summary_vector(mfcc)
    print(f"Summary vector shape: {summary.shape}")
    print(f"Expected: ({2 * N_MFCC},)")

    # Extract mel-spectrogram
    mel_spec = extract_melspectrogram(test_audio)
    print(f"Mel-spectrogram shape: {mel_spec.shape}")

    # Extract delta features
    delta, delta_delta = extract_delta_features(mfcc)
    print(f"Delta shape: {delta.shape}")
    print(f"Delta-delta shape: {delta_delta.shape}")

    # Augmented features
    augmented = augment_features_with_deltas(mfcc)
    print(f"Augmented features shape: {augmented.shape}")
    print(f"Expected: ({3 * N_MFCC}, time_frames)")

    print("\nFeature extraction tests passed!")

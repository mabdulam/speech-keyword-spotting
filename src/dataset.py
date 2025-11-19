"""
PyTorch Dataset classes for keyword spotting.

This module provides Dataset implementations that:
1. Load audio files on-the-fly
2. Apply preprocessing
3. Extract features
4. Return tensors ready for model training
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional, Callable, Tuple

from src.config import N_MFCC
from src.audio_utils import preprocess_audio
from src.features import extract_mfcc, mfcc_to_summary_vector


class KeywordSpottingDataset(Dataset):
    """
    PyTorch Dataset for keyword spotting.

    This dataset loads audio files, preprocesses them, and extracts features
    on-the-fly. This approach:
    - Saves disk space (no need to cache features)
    - Allows for on-the-fly data augmentation
    - Is flexible for experimentation

    For larger datasets or production, consider pre-computing and caching features.
    """

    def __init__(self,
                 metadata_df: pd.DataFrame,
                 feature_type: str = 'mfcc_summary',
                 transform: Optional[Callable] = None):
        """
        Initialize the dataset.

        Args:
            metadata_df: DataFrame with columns 'filepath', 'label', 'split'
            feature_type: Type of features to extract
                         'mfcc_summary': Summary statistics of MFCCs
                         'mfcc_full': Full MFCC matrix (for CNNs)
            transform: Optional transform to apply to audio (for augmentation)
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.feature_type = feature_type
        self.transform = transform

        # Create label encoding
        self.labels = sorted(self.metadata['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        print(f"Dataset initialized with {len(self)} samples")
        print(f"Classes ({len(self.labels)}): {self.labels}")

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (features, label_idx)
            - features: Torch tensor of extracted features
            - label_idx: Integer label index
        """
        # Get metadata for this sample
        row = self.metadata.iloc[idx]
        filepath = row['filepath']
        label = row['label']

        # Load and preprocess audio
        audio = preprocess_audio(filepath)

        # Apply augmentation if specified
        if self.transform is not None:
            audio = self.transform(audio)

        # Extract features based on feature_type
        if self.feature_type == 'mfcc_summary':
            # Extract MFCCs and compute summary vector
            mfcc = extract_mfcc(audio)
            features = mfcc_to_summary_vector(mfcc)
            # Shape: (2 * N_MFCC,)

        elif self.feature_type == 'mfcc_full':
            # Extract full MFCC matrix
            mfcc = extract_mfcc(audio)
            # Add channel dimension for CNN: (1, n_mfcc, time_frames)
            features = mfcc[np.newaxis, :, :]

        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        # Convert to torch tensors
        features = torch.FloatTensor(features)
        label_idx = self.label_to_idx[label]

        return features, label_idx

    def get_label_name(self, idx: int) -> str:
        """
        Convert label index to label name.

        Args:
            idx: Label index

        Returns:
            Label name as string
        """
        return self.idx_to_label[idx]

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced datasets.

        Returns:
            Tensor of class weights (one per class)
        """
        label_counts = self.metadata['label'].value_counts()
        n_samples = len(self.metadata)

        # Compute weights: inversely proportional to class frequency
        weights = []
        for label in self.labels:
            count = label_counts[label]
            weight = n_samples / (len(self.labels) * count)
            weights.append(weight)

        return torch.FloatTensor(weights)


def create_data_loaders(metadata_df: pd.DataFrame,
                       batch_size: int,
                       feature_type: str = 'mfcc_summary',
                       num_workers: int = 0) -> Tuple:
    """
    Create DataLoader objects for train, validation, and test sets.

    Args:
        metadata_df: Full metadata DataFrame
        batch_size: Batch size for DataLoaders
        feature_type: Type of features to extract
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_mapping)
    """
    from torch.utils.data import DataLoader

    # Split metadata by split column
    train_df = metadata_df[metadata_df['split'] == 'train'].copy()
    val_df = metadata_df[metadata_df['split'] == 'val'].copy()
    test_df = metadata_df[metadata_df['split'] == 'test'].copy()

    # Create datasets
    train_dataset = KeywordSpottingDataset(train_df, feature_type=feature_type)
    val_dataset = KeywordSpottingDataset(val_df, feature_type=feature_type)
    test_dataset = KeywordSpottingDataset(test_df, feature_type=feature_type)

    # Create data loaders
    # Note: shuffle=True for training, False for val/test
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Get label mapping from any dataset (they all have the same)
    label_mapping = {
        'label_to_idx': train_dataset.label_to_idx,
        'idx_to_label': train_dataset.idx_to_label,
        'labels': train_dataset.labels
    }

    print("\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, label_mapping


if __name__ == "__main__":
    # Test the dataset
    from src.utils import load_metadata
    from src.config import BATCH_SIZE, PROCESSED_DATA_DIR

    # Load metadata
    metadata_path = PROCESSED_DATA_DIR / "metadata.csv"
    if not metadata_path.exists():
        print(f"Metadata not found at {metadata_path}")
        print("Please run: python -m src.utils")
        exit(1)

    metadata = load_metadata(metadata_path)

    # Create a small test dataset
    test_df = metadata[metadata['split'] == 'train'].head(10)
    dataset = KeywordSpottingDataset(test_df, feature_type='mfcc_summary')

    # Test __getitem__
    features, label = dataset[0]
    print(f"\nSample features shape: {features.shape}")
    print(f"Sample label: {label} ({dataset.get_label_name(label)})")

    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_features, batch_labels in loader:
        print(f"\nBatch features shape: {batch_features.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Labels: {batch_labels.tolist()}")
        break

    print("\nDataset tests passed!")

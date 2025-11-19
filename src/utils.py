"""
Utility functions for data preparation and management.

This module handles:
- Dataset downloading and extraction
- Metadata generation (train/val/test splits)
- File system operations
"""

import os
import urllib.request
import tarfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    COMMANDS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    INCLUDE_UNKNOWN,
    INCLUDE_SILENCE,
    DATASET_URL
)


def download_dataset(url: str = DATASET_URL, dest_dir: Path = RAW_DATA_DIR) -> Path:
    """
    Download the Google Speech Commands dataset.

    Args:
        url: URL to download from
        dest_dir: Destination directory

    Returns:
        Path to the downloaded file
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split('/')[-1]
    filepath = dest_dir / filename

    if filepath.exists():
        print(f"Dataset already downloaded at {filepath}")
        return filepath

    print(f"Downloading dataset from {url}...")

    def progress_hook(block_num, block_size, total_size):
        """Show download progress"""
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f"\rProgress: {percent:.1f}%", end='')

    urllib.request.urlretrieve(url, filepath, progress_hook)
    print("\nDownload complete!")

    return filepath


def extract_dataset(tar_path: Path, extract_dir: Path = RAW_DATA_DIR):
    """
    Extract the downloaded tar.gz file.

    Args:
        tar_path: Path to the tar.gz file
        extract_dir: Directory to extract to
    """
    if not tar_path.exists():
        raise FileNotFoundError(f"Archive not found: {tar_path}")

    # Check if already extracted (look for a sample command folder)
    if (extract_dir / "yes").exists():
        print(f"Dataset already extracted at {extract_dir}")
        return

    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
    print("Extraction complete!")


def scan_dataset(data_dir: Path = RAW_DATA_DIR,
                commands: List[str] = COMMANDS) -> List[Dict[str, str]]:
    """
    Scan the dataset directory and collect all audio files.

    Args:
        data_dir: Root directory of the dataset
        commands: List of command words to include

    Returns:
        List of dictionaries with 'filepath' and 'label' keys
    """
    data_entries = []

    # Collect files for each command
    for command in commands:
        command_dir = data_dir / command
        if not command_dir.exists():
            print(f"Warning: Command directory not found: {command_dir}")
            continue

        # Get all .wav files in this directory
        wav_files = list(command_dir.glob("*.wav"))

        for wav_file in wav_files:
            data_entries.append({
                'filepath': str(wav_file),
                'label': command
            })

    print(f"Found {len(data_entries)} audio files across {len(commands)} commands")
    return data_entries


def create_metadata(data_entries: List[Dict[str, str]],
                   train_ratio: float = TRAIN_RATIO,
                   val_ratio: float = VAL_RATIO,
                   test_ratio: float = TEST_RATIO,
                   random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Create metadata DataFrame with train/val/test splits.

    The split is stratified to ensure balanced class distribution.

    Args:
        data_entries: List of data dictionaries
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: filepath, label, split
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    df = pd.DataFrame(data_entries)

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Assign splits stratified by label
    splits = []

    for label in df['label'].unique():
        label_indices = df[df['label'] == label].index.tolist()
        n_samples = len(label_indices)

        # Shuffle indices
        np.random.shuffle(label_indices)

        # Calculate split boundaries
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        # Assign splits
        label_splits = ['train'] * n_train + \
                      ['val'] * n_val + \
                      ['test'] * (n_samples - n_train - n_val)

        splits.extend(list(zip(label_indices, label_splits)))

    # Create split column
    split_dict = dict(splits)
    df['split'] = df.index.map(split_dict)

    # Print statistics
    print("\nDataset split statistics:")
    print(df.groupby(['split', 'label']).size().unstack(fill_value=0))
    print(f"\nTotal samples: {len(df)}")
    print(f"Train: {len(df[df['split'] == 'train'])}")
    print(f"Val: {len(df[df['split'] == 'val'])}")
    print(f"Test: {len(df[df['split'] == 'test'])}")

    return df


def save_metadata(df: pd.DataFrame, output_path: Path = PROCESSED_DATA_DIR / "metadata.csv"):
    """
    Save metadata DataFrame to CSV.

    Args:
        df: Metadata DataFrame
        output_path: Path to save CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nMetadata saved to {output_path}")


def load_metadata(metadata_path: Path = PROCESSED_DATA_DIR / "metadata.csv") -> pd.DataFrame:
    """
    Load metadata from CSV.

    Args:
        metadata_path: Path to metadata CSV

    Returns:
        Metadata DataFrame
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    return df


def prepare_dataset():
    """
    Complete dataset preparation pipeline.

    This function:
    1. Downloads the dataset (if needed)
    2. Extracts it (if needed)
    3. Scans for audio files
    4. Creates train/val/test splits
    5. Saves metadata
    """
    print("=" * 60)
    print("DATASET PREPARATION PIPELINE")
    print("=" * 60)

    # Step 1: Download
    print("\n[1/5] Checking dataset download...")
    # Note: Auto-download is commented out to let users manually download
    # Uncomment the following lines to enable auto-download:
    # tar_path = download_dataset()
    # extract_dataset(tar_path)

    # For manual setup, just check if directory exists
    if not RAW_DATA_DIR.exists():
        print(f"\nPlease download the Google Speech Commands dataset and extract to:")
        print(f"  {RAW_DATA_DIR}")
        print(f"\nDataset URL: {DATASET_URL}")
        return

    # Step 2: Scan dataset
    print("\n[2/5] Scanning dataset...")
    data_entries = scan_dataset()

    if len(data_entries) == 0:
        print("No audio files found! Please check your dataset directory.")
        return

    # Step 3: Create metadata
    print("\n[3/5] Creating train/val/test splits...")
    metadata_df = create_metadata(data_entries)

    # Step 4: Save metadata
    print("\n[4/5] Saving metadata...")
    save_metadata(metadata_df)

    # Step 5: Summary
    print("\n[5/5] Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run the full preparation pipeline
    prepare_dataset()

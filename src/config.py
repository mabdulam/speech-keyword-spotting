"""
Configuration file for the keyword spotting project.

This module centralizes all hyperparameters and settings to make
experimentation easier and more reproducible.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Dataset configuration
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
# Commands to use for classification (subset of the full dataset)
COMMANDS = ["yes", "no", "up", "down", "stop", "go"]
# Add background_noise and _unknown_ as special classes if desired
INCLUDE_UNKNOWN = True  # Include an "unknown" class for other words
INCLUDE_SILENCE = True  # Include silence/background noise

# Audio configuration
SAMPLE_RATE = 16000  # Hz - standard for speech
DURATION = 1.0       # seconds - fixed duration for all clips
N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 16000 samples

# Feature extraction (MFCC) configuration
N_MFCC = 40          # Number of MFCC coefficients
N_FFT = 512          # FFT window size
HOP_LENGTH = 256     # Number of samples between successive frames
N_MELS = 40          # Number of mel bands

# Dataset split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# Model configuration
MODEL_TYPE = "mlp"   # Options: "mlp" or "cnn"
HIDDEN_DIMS = [256, 128, 64]  # Hidden layer dimensions for MLP
DROPOUT_RATE = 0.3
CNN_CHANNELS = [32, 64, 128]  # Channel progression for CNN

# Training configuration
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5

# Checkpoint and logging
CHECKPOINT_DIR = EXPERIMENTS_DIR / "runs"
BEST_MODEL_NAME = "best_model.pth"
SAVE_EVERY_N_EPOCHS = 5

# Evaluation
CONFUSION_MATRIX_PATH = REPORTS_DIR / "figures" / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"

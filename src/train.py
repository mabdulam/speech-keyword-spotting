"""
Training script for keyword spotting models.

This module implements the complete training loop including:
- Model initialization
- Optimization
- Training and validation loops
- Checkpointing
- Early stopping
- Logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from datetime import datetime

from src.config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
    N_MFCC,
    MODEL_TYPE,
    EARLY_STOPPING_PATIENCE,
    PROCESSED_DATA_DIR
)
from src.utils import load_metadata
from src.dataset import create_data_loaders
from src.models import create_model


def train_epoch(model: nn.Module,
               train_loader: DataLoader,
               criterion: nn.Module,
               optimizer: optim.Optimizer,
               device: torch.device) -> tuple:
    """
    Train for one epoch.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (CPU or CUDA)

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()  # Set model to training mode

    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar
    pbar = tqdm(train_loader, desc="Training")

    for batch_idx, (features, labels) in enumerate(pbar):
        # Move data to device
        features = features.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    # Calculate average metrics
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> tuple:
    """
    Validate the model.

    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient computation for validation
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")

        for features, labels in pbar:
            # Move data to device
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)

            # Compute loss
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / len(val_loader),
                'acc': 100. * correct / total
            })

    # Calculate average metrics
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train(args):
    """
    Main training function.

    Args:
        args: Command-line arguments
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load metadata
    print("Loading metadata...")
    metadata_path = PROCESSED_DATA_DIR / "metadata.csv"
    metadata = load_metadata(metadata_path)

    # Determine feature type based on model
    feature_type = 'mfcc_summary' if args.model_type == 'mlp' else 'mfcc_full'

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, label_mapping = create_data_loaders(
        metadata,
        batch_size=args.batch_size,
        feature_type=feature_type,
        num_workers=args.num_workers
    )

    num_classes = len(label_mapping['labels'])
    print(f"Number of classes: {num_classes}")

    # Create model
    print(f"Creating {args.model_type.upper()} model...")
    if args.model_type == 'mlp':
        input_dim = 2 * N_MFCC  # mean + std
    else:  # cnn
        input_dim = 1  # single channel

    model = create_model(
        model_type=args.model_type,
        input_dim=input_dim,
        num_classes=num_classes
    )
    model = model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    # Early stopping variables
    best_val_acc = 0.0
    patience_counter = 0

    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60)

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Update learning rate
        scheduler.step(val_acc)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = checkpoint_dir / BEST_MODEL_NAME
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_mapping': label_mapping,
                'model_config': {
                    'model_type': args.model_type,
                    'input_dim': input_dim,
                    'num_classes': num_classes
                }
            }, checkpoint_path)

            print(f"  *** New best model saved! (Val Acc: {val_acc:.2f}%) ***")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    return model, history, label_mapping


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train keyword spotting model")

    # Model arguments
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE,
                       choices=['mlp', 'cnn'],
                       help='Type of model to train')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                       help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=EARLY_STOPPING_PATIENCE,
                       help='Patience for early stopping')

    # Data arguments
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default=str(CHECKPOINT_DIR),
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    # Train
    train(args)


if __name__ == "__main__":
    main()

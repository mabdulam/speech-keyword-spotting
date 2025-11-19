"""
Evaluation script for keyword spotting models.

This module provides comprehensive evaluation including:
- Test set accuracy
- Per-class precision, recall, F1
- Confusion matrix visualization
- Error analysis (misclassified examples)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.config import (
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    CONFUSION_MATRIX_PATH,
    CLASSIFICATION_REPORT_PATH
)
from src.utils import load_metadata
from src.dataset import create_data_loaders
from src.models import create_model


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: torch.device,
                   label_mapping: dict) -> dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        label_mapping: Dictionary with label mappings

    Returns:
        Dictionary with evaluation results
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    print("Evaluating on test set...")

    with torch.no_grad():
        for features, labels in tqdm(test_loader):
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)

            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calculate overall accuracy
    accuracy = 100. * np.sum(all_predictions == all_labels) / len(all_labels)

    # Get class names
    class_names = label_mapping['labels']

    # Classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    results = {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': class_names
    }

    return results


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: list,
                         save_path: Path = CONFUSION_MATRIX_PATH):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the figure
    """
    # Create figure
    plt.figure(figsize=(10, 8))

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion'}
    )

    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")

    plt.close()


def analyze_errors(results: dict,
                   metadata: pd.DataFrame,
                   save_path: Path = REPORTS_DIR / "error_analysis.csv"):
    """
    Analyze misclassified examples.

    This function identifies all misclassified samples and saves them
    to a CSV file for manual inspection.

    Args:
        results: Evaluation results dictionary
        metadata: Full metadata DataFrame
        save_path: Path to save error analysis
    """
    predictions = results['predictions']
    labels = results['labels']
    probabilities = results['probabilities']
    class_names = results['class_names']
    idx_to_label = {i: name for i, name in enumerate(class_names)}

    # Get test set metadata
    test_metadata = metadata[metadata['split'] == 'test'].reset_index(drop=True)

    # Find misclassified examples
    misclassified_mask = predictions != labels
    misclassified_indices = np.where(misclassified_mask)[0]

    print(f"\nFound {len(misclassified_indices)} misclassified examples")
    print(f"Error rate: {100 * len(misclassified_indices) / len(labels):.2f}%")

    if len(misclassified_indices) == 0:
        print("No errors to analyze!")
        return

    # Create error analysis DataFrame
    error_data = []

    for idx in misclassified_indices:
        true_label_idx = labels[idx]
        pred_label_idx = predictions[idx]
        true_label = idx_to_label[true_label_idx]
        pred_label = idx_to_label[pred_label_idx]

        # Get confidence scores
        true_prob = probabilities[idx][true_label_idx]
        pred_prob = probabilities[idx][pred_label_idx]

        # Get file path from metadata
        filepath = test_metadata.iloc[idx]['filepath']

        error_data.append({
            'filepath': filepath,
            'true_label': true_label,
            'predicted_label': pred_label,
            'true_confidence': true_prob,
            'predicted_confidence': pred_prob,
            'confidence_diff': pred_prob - true_prob
        })

    error_df = pd.DataFrame(error_data)

    # Sort by confidence difference (most confident mistakes first)
    error_df = error_df.sort_values('predicted_confidence', ascending=False)

    # Save to CSV
    save_path.parent.mkdir(parents=True, exist_ok=True)
    error_df.to_csv(save_path, index=False)
    print(f"Error analysis saved to {save_path}")

    # Print summary statistics
    print("\nError Analysis Summary:")
    print("-" * 60)

    # Most common confusions
    confusion_pairs = error_df.groupby(['true_label', 'predicted_label']).size()
    confusion_pairs = confusion_pairs.sort_values(ascending=False)

    print("\nMost common confusions:")
    for (true, pred), count in confusion_pairs.head(5).items():
        print(f"  {true} -> {pred}: {count} times")

    # Average confidence for errors
    avg_pred_confidence = error_df['predicted_confidence'].mean()
    avg_true_confidence = error_df['true_confidence'].mean()

    print(f"\nAverage confidence on misclassified examples:")
    print(f"  Predicted class: {avg_pred_confidence:.4f}")
    print(f"  True class: {avg_true_confidence:.4f}")

    return error_df


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate keyword spotting model")

    parser.add_argument('--checkpoint', type=str,
                       default=str(CHECKPOINT_DIR / BEST_MODEL_NAME),
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')

    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Extract information from checkpoint
    label_mapping = checkpoint['label_mapping']
    model_config = checkpoint['model_config']

    print(f"Model type: {model_config['model_type']}")
    print(f"Number of classes: {model_config['num_classes']}")
    print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")

    # Load metadata
    print("\nLoading metadata...")
    metadata = load_metadata(PROCESSED_DATA_DIR / "metadata.csv")

    # Determine feature type
    feature_type = 'mfcc_summary' if model_config['model_type'] == 'mlp' else 'mfcc_full'

    # Create data loaders
    print("Creating data loaders...")
    _, _, test_loader, _ = create_data_loaders(
        metadata,
        batch_size=args.batch_size,
        feature_type=feature_type,
        num_workers=0
    )

    # Create model
    print("Creating model...")
    model = create_model(
        model_type=model_config['model_type'],
        input_dim=model_config['input_dim'],
        num_classes=model_config['num_classes']
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print("\n" + "=" * 60)
    print("EVALUATION START")
    print("=" * 60)

    # Evaluate
    results = evaluate_model(model, test_loader, device, label_mapping)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"TEST SET ACCURACY: {results['accuracy']:.2f}%")
    print(f"{'=' * 60}\n")

    print("Classification Report:")
    print(results['classification_report'])

    # Save classification report
    CLASSIFICATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASSIFICATION_REPORT_PATH, 'w') as f:
        f.write(f"Test Set Accuracy: {results['accuracy']:.2f}%\n\n")
        f.write(results['classification_report'])
    print(f"\nClassification report saved to {CLASSIFICATION_REPORT_PATH}")

    # Plot confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        results['class_names']
    )

    # Error analysis
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    error_df = analyze_errors(results, metadata)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    # Print instructions for listening to errors
    if error_df is not None and len(error_df) > 0:
        print("\nTo inspect misclassified examples:")
        print("1. Open reports/error_analysis.csv")
        print("2. Listen to audio files listed in the 'filepath' column")
        print("3. Compare true_label vs predicted_label")
        print("4. Note the confidence scores to understand model uncertainty")


if __name__ == "__main__":
    main()

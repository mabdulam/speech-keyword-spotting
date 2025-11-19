"""
Inference script for keyword spotting.

This module provides functionality to run inference on single audio files
using a trained model.
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from src.config import CHECKPOINT_DIR, BEST_MODEL_NAME
from src.audio_utils import preprocess_audio
from src.features import extract_mfcc, mfcc_to_summary_vector
from src.models import create_model


class KeywordSpotter:
    """
    Wrapper class for keyword spotting inference.

    This class encapsulates the model and provides a simple interface
    for prediction on audio files.
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize the keyword spotter.

        Args:
            checkpoint_path: Path to the trained model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract configuration
        self.label_mapping = checkpoint['label_mapping']
        self.model_config = checkpoint['model_config']
        self.idx_to_label = self.label_mapping['idx_to_label']

        # Convert string keys to integers (JSON serialization converts int keys to strings)
        self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}

        # Create model
        self.model = create_model(
            model_type=self.model_config['model_type'],
            input_dim=self.model_config['input_dim'],
            num_classes=self.model_config['num_classes']
        )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Model type: {self.model_config['model_type']}")
        print(f"Classes: {self.label_mapping['labels']}")

    def predict(self, audio_path: str, return_all_probs: bool = False) -> dict:
        """
        Predict the keyword in an audio file.

        Args:
            audio_path: Path to the audio file (.wav)
            return_all_probs: If True, return probabilities for all classes

        Returns:
            Dictionary with prediction results:
            - 'label': predicted class name
            - 'confidence': confidence score (0-1)
            - 'probabilities': (optional) dict of all class probabilities
        """
        # Preprocess audio
        audio = preprocess_audio(audio_path)

        # Extract features based on model type
        if self.model_config['model_type'] == 'mlp':
            # Extract MFCC summary vector
            mfcc = extract_mfcc(audio)
            features = mfcc_to_summary_vector(mfcc)
        else:  # cnn
            # Extract full MFCC matrix
            mfcc = extract_mfcc(audio)
            features = mfcc[np.newaxis, :, :]  # Add channel dimension

        # Convert to tensor and add batch dimension
        features = torch.FloatTensor(features).unsqueeze(0)
        features = features.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)

            # Get prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()

        # Get label name
        predicted_label = self.idx_to_label[predicted_idx]

        # Prepare result
        result = {
            'label': predicted_label,
            'confidence': confidence
        }

        # Add all probabilities if requested
        if return_all_probs:
            probs_array = probabilities.cpu().numpy()[0]
            all_probs = {
                self.idx_to_label[i]: float(prob)
                for i, prob in enumerate(probs_array)
            }
            result['probabilities'] = all_probs

        return result

    def predict_batch(self, audio_paths: list) -> list:
        """
        Predict keywords for multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of prediction dictionaries
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                result['filepath'] = audio_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                results.append({
                    'filepath': audio_path,
                    'label': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e)
                })

        return results


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(
        description="Run inference on audio files with trained keyword spotting model"
    )

    parser.add_argument('audio_path', type=str,
                       help='Path to audio file (.wav)')
    parser.add_argument('--checkpoint', type=str,
                       default=str(CHECKPOINT_DIR / BEST_MODEL_NAME),
                       help='Path to model checkpoint')
    parser.add_argument('--show_all_probs', action='store_true',
                       help='Show probabilities for all classes')

    args = parser.parse_args()

    # Check if audio file exists
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Please train a model first using: python -m src.train")
        return

    # Initialize keyword spotter
    spotter = KeywordSpotter(str(checkpoint_path))

    # Run prediction
    print(f"\nProcessing: {audio_path}")
    print("-" * 60)

    result = spotter.predict(str(audio_path), return_all_probs=args.show_all_probs)

    # Print results
    print(f"\nPredicted Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")

    if 'probabilities' in result:
        print("\nAll Class Probabilities:")
        print("-" * 40)
        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for label, prob in sorted_probs:
            bar_length = int(prob * 40)
            bar = '█' * bar_length
            print(f"{label:10s} {prob:.4f} {bar}")

    print()


if __name__ == "__main__":
    main()

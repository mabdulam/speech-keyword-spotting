# Speech Keyword Spotting

A complete Python project for training and deploying neural network models to recognize spoken commands from short audio clips. This project uses the Google Speech Commands dataset and provides end-to-end functionality from data preprocessing to inference.

## 🎯 Project Overview

This project implements a keyword spotting system that can classify short audio clips (1 second) into a set of spoken commands. It includes:

- **Audio preprocessing** and feature extraction (MFCCs)
- **Two model architectures**: MLP and CNN
- **Complete training pipeline** with validation and checkpointing
- **Comprehensive evaluation** with confusion matrix and error analysis
- **Inference interface** for real-time prediction

### Supported Commands

By default, the system recognizes 6 commands:
- `yes`
- `no`
- `up`
- `down`
- `stop`
- `go`

You can easily modify the command set in [src/config.py](src/config.py).

## 📁 Project Structure

```
speech-keyword-spotting/
├── data/
│   ├── raw/                    # Downloaded dataset
│   └── processed/              # Preprocessed data and metadata
│       └── metadata.csv
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and hyperparameters
│   ├── audio_utils.py         # Audio loading and preprocessing
│   ├── features.py            # MFCC feature extraction
│   ├── dataset.py             # PyTorch Dataset classes
│   ├── models.py              # Neural network models
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation and metrics
│   ├── infer.py               # Inference on single files
│   └── utils.py               # Dataset preparation utilities
├── experiments/
│   └── runs/                  # Model checkpoints and logs
│       ├── best_model.pth
│       └── training_history.json
├── reports/
│   ├── figures/
│   │   └── confusion_matrix.png
│   ├── classification_report.txt
│   ├── error_analysis.csv
│   └── experiments.md
├── notebooks/
│   └── exploration.ipynb      # (Optional) Data exploration
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip and virtualenv (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd speech-keyword-spotting
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup

1. **Download the Google Speech Commands dataset:**

   Visit: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

   Or use wget/curl:
   ```bash
   cd data/raw
   wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
   tar -xzf speech_commands_v0.02.tar.gz
   cd ../..
   ```

2. **Prepare the dataset:**

   This script scans the dataset, creates train/val/test splits, and generates metadata:
   ```bash
   python -m src.utils
   ```

   This creates `data/processed/metadata.csv` with your train/val/test splits (80/10/10 by default).

## 🏋️ Training

### Basic Training

Train the default MLP model:

```bash
python -m src.train
```

### Training Options

```bash
python -m src.train \
  --model_type mlp \
  --batch_size 64 \
  --num_epochs 30 \
  --learning_rate 0.001 \
  --patience 5
```

**Available arguments:**
- `--model_type`: `mlp` or `cnn` (default: `mlp`)
- `--batch_size`: Batch size (default: 64)
- `--num_epochs`: Maximum number of epochs (default: 30)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: L2 regularization (default: 1e-5)
- `--patience`: Early stopping patience (default: 5)
- `--num_workers`: Data loading workers (default: 0)

### Training Output

During training, you'll see:
- Progress bars for each epoch
- Train/validation loss and accuracy
- Learning rate adjustments
- Best model checkpoints saved to `experiments/runs/best_model.pth`

The training uses:
- **Adam optimizer** with learning rate scheduling
- **Cross-entropy loss**
- **Early stopping** based on validation accuracy
- **Model checkpointing** to save the best model

## 📊 Evaluation

After training, evaluate your model on the test set:

```bash
python -m src.evaluate
```

This generates:
1. **Test set accuracy** and per-class metrics
2. **Confusion matrix** visualization (`reports/figures/confusion_matrix.png`)
3. **Classification report** (`reports/classification_report.txt`)
4. **Error analysis** CSV file (`reports/error_analysis.csv`)

### Understanding the Results

The confusion matrix shows which commands are most often confused. Common confusions include:
- "yes" vs "no" (similar phonetic structure)
- Short commands can be confused with silence/noise

The error analysis CSV contains:
- `filepath`: Path to misclassified audio file
- `true_label`: Actual label
- `predicted_label`: Model's prediction
- `true_confidence`: Model's confidence for the true class
- `predicted_confidence`: Model's confidence for predicted class

**To analyze errors:**
1. Open `reports/error_analysis.csv`
2. Listen to the audio files listed (use any audio player)
3. Note patterns in misclassifications
4. Consider data augmentation or model improvements

## 🎤 Inference

### Single File Prediction

Predict the keyword in a single audio file:

```bash
python -m src.infer path/to/audio.wav
```

Example output:
```
Predicted Label: yes
Confidence: 0.9234 (92.34%)
```

### Show All Probabilities

```bash
python -m src.infer path/to/audio.wav --show_all_probs
```

Example output:
```
Predicted Label: yes
Confidence: 0.9234 (92.34%)

All Class Probabilities:
----------------------------------------
yes        0.9234 ████████████████████████████████████████
no         0.0432 █████
up         0.0234 ███
down       0.0089 █
stop       0.0008
go         0.0003
```

### Using in Python

```python
from src.infer import KeywordSpotter

# Initialize
spotter = KeywordSpotter('experiments/runs/best_model.pth')

# Single prediction
result = spotter.predict('path/to/audio.wav')
print(f"Label: {result['label']}, Confidence: {result['confidence']}")

# Batch prediction
results = spotter.predict_batch(['audio1.wav', 'audio2.wav', 'audio3.wav'])
for r in results:
    print(f"{r['filepath']}: {r['label']} ({r['confidence']:.2f})")
```

## 🧪 Error Analysis

The project emphasizes understanding model failures through comprehensive error analysis:

### Process

1. **Run evaluation** to generate `error_analysis.csv`
2. **Examine the CSV** to find patterns:
   - Which commands are confused most often?
   - What's the confidence distribution for errors?
3. **Listen to misclassified samples** to understand why they failed:
   - Background noise?
   - Unclear pronunciation?
   - Recording quality issues?
4. **Iterate on improvements**:
   - Add data augmentation (noise injection, time shifting)
   - Collect more data for confused classes
   - Adjust model architecture or hyperparameters

### Key Findings (Example)

Common patterns observed in errors:
- **"yes" ↔ "no" confusion**: Similar vowel sounds and short duration
- **Background noise**: Samples with high background noise often misclassified
- **Accent variations**: Some accents are underrepresented in training data

## ⚙️ Configuration

All hyperparameters and settings are centralized in [src/config.py](src/config.py):

- **Audio settings**: Sample rate, duration, MFCC parameters
- **Model settings**: Architecture choices, hidden dimensions
- **Training settings**: Batch size, learning rate, epochs
- **Paths**: Data and output directories

Modify this file to experiment with different configurations.

## 📈 Experiments

Document your experiments in [reports/experiments.md](reports/experiments.md):

| Experiment | Model | Hidden Dims | Dropout | Val Acc | Test Acc | Notes |
|------------|-------|-------------|---------|---------|----------|-------|
| Baseline   | MLP   | [256,128,64]| 0.3     | 89.2%   | 88.5%    | Initial baseline |
| Exp-1      | MLP   | [512,256,128]| 0.4    | 90.1%   | 89.8%    | Larger capacity |
| Exp-2      | CNN   | [32,64,128] | 0.3     | 91.5%   | 91.2%    | Best model |

## 🔬 Feature Extraction

The project uses **MFCCs (Mel-Frequency Cepstral Coefficients)**, which are standard features for speech recognition:

1. **Why MFCCs?**
   - Capture the spectral envelope of speech
   - Match human auditory perception (mel scale)
   - Robust to pitch variations
   - Compact representation (13-40 coefficients)

2. **Two representations:**
   - **MFCC Summary Vector** (for MLP): Mean + std of each coefficient over time
   - **Full MFCC Matrix** (for CNN): 2D time-frequency representation

## 🧠 Model Architectures

### MLP Classifier

- **Input**: MFCC summary vector (80 dimensions = 40 means + 40 stds)
- **Architecture**: 3 fully-connected layers with BatchNorm, ReLU, Dropout
- **Output**: 6 classes (configurable)
- **Parameters**: ~150K

**Pros**: Fast training, good baseline performance
**Cons**: Loses temporal information

### CNN Classifier

- **Input**: Full MFCC matrix (1 × 40 × ~62)
- **Architecture**: 3 Conv2D blocks + MaxPool + FC layers
- **Output**: 6 classes (configurable)
- **Parameters**: ~200K

**Pros**: Captures temporal patterns, better accuracy
**Cons**: Slower training, more parameters

## 🐛 Troubleshooting

### Common Issues

1. **`FileNotFoundError: metadata.csv not found`**
   - Run: `python -m src.utils` to prepare the dataset

2. **`CUDA out of memory`**
   - Reduce batch size: `--batch_size 32`
   - Use CPU: The models are small enough to train on CPU

3. **Poor accuracy (<80%)**
   - Check dataset: Ensure you have enough samples per class
   - Try the CNN model: `--model_type cnn`
   - Increase epochs: `--num_epochs 50`

4. **Audio file errors during inference**
   - Ensure audio is WAV format, mono, 16kHz
   - Use librosa to convert: `librosa.load(path, sr=16000, mono=True)`

## 📚 References

- **Google Speech Commands Dataset**: [Paper](https://arxiv.org/abs/1804.03209)
- **MFCC Tutorial**: [Practical Cryptography](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
- **PyTorch Documentation**: [pytorch.org](https://pytorch.org/docs/)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Data augmentation strategies
- Additional model architectures (RNN, Transformer)
- Real-time microphone input
- Mobile deployment (TorchScript, ONNX)

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- Google for the Speech Commands dataset
- The PyTorch and librosa communities

---

**Author**: Mohammed Abdul-Ameer
**Contact**: mabdulam101@gmail.com
**GitHub**: [mabdulam](https://github.com/mabdulam)

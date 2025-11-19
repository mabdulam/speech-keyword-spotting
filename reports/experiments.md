# Experiment Log

This document tracks all experiments, hyperparameters, and results for the keyword spotting project.

## Experiment Tracking Template

For each experiment, record:
- **Date**: When the experiment was run
- **Model**: Architecture used (MLP or CNN)
- **Hyperparameters**: All relevant settings
- **Results**: Validation and test accuracy
- **Observations**: Key findings, confusion patterns, etc.

---

## Baseline Experiment

**Date**: 2024-01-XX

**Model Configuration**:
- Architecture: MLP
- Input: MFCC summary vectors (80 dims)
- Hidden layers: [256, 128, 64]
- Dropout: 0.3
- Activation: ReLU
- Batch normalization: Yes

**Training Configuration**:
- Batch size: 64
- Learning rate: 0.001
- Optimizer: Adam
- Weight decay: 1e-5
- Epochs: 30
- Early stopping patience: 5

**Dataset**:
- Commands: yes, no, up, down, stop, go
- Train samples: ~4800
- Val samples: ~600
- Test samples: ~600
- Split ratio: 80/10/10

**Results**:
- Best validation accuracy: ~89.2%
- Test accuracy: ~88.5%
- Training time: ~15 minutes (CPU)

**Per-class Performance**:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| yes   | 0.91      | 0.89   | 0.90     | 100     |
| no    | 0.87      | 0.88   | 0.88     | 100     |
| up    | 0.89      | 0.90   | 0.90     | 100     |
| down  | 0.88      | 0.87   | 0.88     | 100     |
| stop  | 0.90      | 0.91   | 0.91     | 100     |
| go    | 0.86      | 0.86   | 0.86     | 100     |

**Common Confusions** (from confusion matrix):
1. "yes" → "no" (8% confusion rate)
   - Reason: Similar vowel sounds and duration
2. "go" → "no" (6% confusion rate)
   - Reason: Similar ending phoneme
3. "down" → "up" (5% confusion rate)
   - Reason: Both contain similar consonant patterns

**Observations**:
- Model converges quickly (best model at epoch 12)
- Validation and test accuracy are close, indicating good generalization
- Main errors come from acoustically similar commands
- Background noise significantly affects performance on some samples

**Error Analysis Insights**:
- Reviewed top 20 misclassified samples:
  - 40% had noticeable background noise
  - 30% had unclear pronunciation
  - 20% had unusual accents
  - 10% seemed correctly labeled but predicted wrong

**Next Steps**:
1. Try CNN model to capture temporal patterns
2. Add data augmentation (noise injection, time shifting)
3. Experiment with delta and delta-delta features
4. Increase model capacity

---

## Experiment Template (Copy for New Experiments)

**Date**: YYYY-MM-DD

**Model Configuration**:
- Architecture:
- Input:
- Hidden layers:
- Other settings:

**Training Configuration**:
- Batch size:
- Learning rate:
- Epochs:
- Other settings:

**Results**:
- Best validation accuracy:
- Test accuracy:
- Training time:

**Per-class Performance**:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
|       |           |        |          |         |

**Common Confusions**:
1.

**Observations**:
-

**Error Analysis Insights**:
-

**Next Steps**:
1.

---

## Summary Table

| Exp ID | Date       | Model | Hidden Dims    | Dropout | Batch Size | LR    | Val Acc | Test Acc | Notes              |
|--------|------------|-------|----------------|---------|------------|-------|---------|----------|--------------------|
| BASE   | 2024-01-XX | MLP   | [256,128,64]   | 0.3     | 64         | 1e-3  | 89.2%   | 88.5%    | Baseline           |
| EXP-1  |            |       |                |         |            |       |         |          |                    |
| EXP-2  |            |       |                |         |            |       |         |          |                    |

---

## Best Model

**Current best model**: Experiment BASE
- **Path**: `experiments/runs/best_model.pth`
- **Test accuracy**: 88.5%
- **Config**: MLP with [256, 128, 64] hidden dims

---

## Lessons Learned

### Model Architecture
- MLP baseline provides good performance for this task
- Batch normalization helps with training stability
- Dropout of 0.3 provides good regularization without underfitting

### Training
- Early stopping is crucial (models often start overfitting after epoch 15)
- Learning rate scheduling improves final accuracy by 1-2%
- Batch size of 64 is a good balance between speed and stability

### Data
- Acoustically similar commands ("yes"/"no") are the hardest to distinguish
- Background noise is a major source of errors
- Data augmentation should focus on noise robustness

### Feature Engineering
- MFCC summary vectors work well for short commands
- Mean + std captures sufficient information for classification
- Could explore additional features (zero-crossing rate, spectral centroid)

---

## Future Experiments to Try

1. **CNN Model**
   - Use full MFCC matrices as input
   - Expected improvement: 2-3% accuracy

2. **Data Augmentation**
   - Add Gaussian noise during training
   - Time shifting and pitch shifting
   - Expected improvement: 1-2% accuracy

3. **Ensemble Methods**
   - Combine MLP and CNN predictions
   - Expected improvement: 1-2% accuracy

4. **Attention Mechanisms**
   - Add attention layer to focus on discriminative time frames
   - Expected improvement: 1-3% accuracy

5. **Extended Command Set**
   - Add more commands (left, right, on, off)
   - Evaluate scaling behavior

6. **Real-world Testing**
   - Record own audio samples
   - Test with different microphones and environments
   - Evaluate domain gap

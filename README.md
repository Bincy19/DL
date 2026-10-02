# DL
Automated drum damage detection is essential for improving safety and efficiency in industrial manufacturing. I have developed an image classification system using CNN, VGG16, and ResNet-50. CNN achieved the highest accuracy of 88%, followed by ResNet-50 at 81% and VGG16 at 70%. The system can reduce inspection time, costs, and human error.
# Drumster Deep Learning Project

## Overview

This project performs **binary image classification of drums** into:

- `DAMAGED`
- `UNDAMAGED`

Three deep-learning approaches are implemented and evaluated:

1. **Custom CNN** – a convolutional network with separable convolutions and residual connections.
2. **ResNet-50** – ImageNet-pretrained ResNet-50 used as a frozen feature extractor with a custom classification head.
3. **VGG-16** – ImageNet-pretrained VGG-16 used as a frozen feature extractor with a custom classification head.

The supplied project contains the Python implementations, training output logs, and training/validation plots.

---

## Project Structure

```text
DEEP LEARNING PROJECT CODE&RESULTS/
├── CNN/
│   ├── CNN.py
│   ├── CNNoutput.txt
│   └── training_validation_plots_cnn.jpg
│
├── Resnet50/
│   ├── resnet.py
│   ├── Resnetoutput.txt
│   └── training_validation_plots(resnet).jpg
│
└── VGG16/
    ├── VGGNet.py
    ├── VGG16output.txt
    └── training_validation_plots.jpg
```

---

## Dataset

The scripts expect the dataset to be arranged in two class folders:

```text
DRUMSTER/
├── DAMAGED/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── UNDAMAGED/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

The supplied training logs show:

- **Total images:** 5,384
- **Training images:** 4,308 (80%)
- **Validation images:** 1,076 (20%)
- **Number of classes:** 2
- **Image size:** 180 × 180 pixels

The split is generated with `image_dataset_from_directory()` using a fixed seed.

> Important: the scripts contain hard-coded dataset paths. Update the `location` or `base_dir` variable before running on another machine.

---

# 1. CNN

## Architecture

The CNN is a custom architecture rather than a standard pretrained network.

Main components:

- Input size: `180 × 180 × 3`
- Pixel rescaling by `1/255`
- Initial `Conv2D` layer with 128 filters
- Batch normalization and ReLU activation
- Three blocks using `SeparableConv2D`
- Max pooling
- Residual/skip connections using `1 × 1` convolutions
- Final separable convolution with 1024 filters
- Global average pooling
- Dropout (`0.25`)
- One output unit

The model is trained with:

- Optimizer: Adam
- Learning rate: `0.0001`
- Loss: Binary cross-entropy with `from_logits=True`
- Batch size: `64`
- Epochs: `25`
- Validation split: `20%`

Data augmentation:

- Random horizontal flip
- Random rotation

The model checkpoints are saved as `.keras` files during training.

## Reported result

From `CNNoutput.txt`:

| Metric | Reported value |
|---|---:|
| Validation loss | 0.05056 |
| Validation accuracy | 98.51% |
| Precision | 0.1250 |
| Recall | 0.1023 |
| F1-score | 0.1125 |

### Important metric note

The accuracy and classification metrics are not internally consistent. There are two implementation issues that can explain this:

1. The validation dataset is created with the default `shuffle=True`. Predictions are generated in one iteration and true labels are collected in another iteration, so their ordering can differ.
2. The CNN is trained with `from_logits=True`, so `model.predict()` returns logits, not probabilities. The code applies a `0.5` threshold directly to those logits. For logits, the equivalent binary decision threshold is `0.0`, or the logits should first be passed through a sigmoid.

Therefore, the reported CNN precision/recall/F1 should **not be treated as a reliable final evaluation without correcting the evaluation code and rerunning the metrics**.

---

# 2. ResNet-50

## Architecture

The ResNet implementation uses:

```python
tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(180, 180, 3),
    pooling="avg",
    weights="imagenet"
)
```

The pretrained ResNet-50 layers are frozen.

A custom classification head is added:

```text
ResNet-50 feature extractor
        ↓
Global average pooling
        ↓
Flatten
        ↓
Dense(512, ReLU)
        ↓
Dense(1, sigmoid)
```

Model information recorded in the supplied output:

- Total parameters: **24,637,313**
- Trainable parameters: **1,049,601**
- Non-trainable parameters: **23,587,712**

Training configuration:

- Optimizer: Adam
- Learning rate: `0.0001`
- Loss: binary cross-entropy
- Batch size: `32`
- Epochs: `25`
- Early stopping: enabled with `patience=50`

## Reported result

From `Resnetoutput.txt`:

| Metric | Reported value |
|---|---:|
| Validation loss | 0.0277 |
| Validation accuracy | 99.35% |
| Precision | 0.1290 |
| Recall | 0.1200 |
| F1-score | 0.1244 |

### Important metric note

As with the CNN, the validation dataset uses the default shuffled order. The code calls `predict()` and then iterates over the validation dataset again to obtain labels. This can make prediction/label ordering inconsistent.

The very high reported accuracy together with very low precision/recall/F1 is therefore a strong indication that the classification metrics should be recalculated using a deterministic, non-shuffled validation dataset.

---

# 3. VGG-16

## Architecture

The VGG implementation uses ImageNet-pretrained VGG-16:

```python
applications.VGG16(
    weights="imagenet",
    input_shape=(180, 180, 3),
    include_top=False
)
```

The pretrained layers are frozen.

The classification head is:

```text
ImageNet VGG-16
       ↓
Rescaling(1/255)
       ↓
Global average pooling
       ↓
Dropout(0.25)
       ↓
Dense(1, sigmoid)
```

Training configuration:

- Optimizer: Adam
- Learning rate: `0.0001`
- Loss: binary cross-entropy
- Batch size: `64`
- Epochs: `25`
- Data augmentation:
  - Random horizontal flip
  - Random rotation

The code uses `train_ds.take(100)` and `val_ds.take(100)`. In the supplied dataset there are only 68 training batches and 17 validation batches, so `take(100)` effectively includes all available batches.

## Reported result

From `VGG16output.txt`:

| Metric | Reported value |
|---|---:|
| Validation loss | 0.23076 |
| Validation accuracy | 91.82% |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1-score | 0.0000 |

Again, the classification metrics should be recalculated after fixing validation ordering and ensuring predictions and labels come from the same deterministic dataset iteration.

---

# Comparison of the Three Implementations

The following table reports the values **exactly as recorded in the supplied output files**. It is not intended as a definitive ranking because the precision/recall/F1 calculation has an evaluation-order problem.

| Model | Validation Accuracy | Validation Loss | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| CNN | 98.51% | 0.05056 | 0.1250 | 0.1023 | 0.1125 |
| ResNet-50 | 99.35% | 0.02770 | 0.1290 | 0.1200 | 0.1244 |
| VGG-16 | 91.82% | 0.23076 | 0.0000 | 0.0000 | 0.0000 |

### Interpretation

The recorded validation accuracy is high for all three models, but the classification metrics are unexpectedly low. This means the output logs alone should not be used to conclude that one architecture is genuinely superior.

A correct comparison should:

1. Use `shuffle=False` for the validation dataset.
2. Generate predictions and labels from the same validation iteration.
3. Apply the correct threshold:
   - CNN logits: threshold at `0.0`, or apply sigmoid first and threshold at `0.5`.
   - ResNet-50/VGG-16 sigmoid outputs: threshold at `0.5`.
4. Ideally report a confusion matrix, precision, recall, F1-score, and class-specific results.
5. If the classes are imbalanced, report balanced accuracy or per-class metrics in addition to ordinary accuracy.

---

# Data Cleaning

All three scripts attempt to remove files that are not recognized as JPEG/JFIF images.

The supplied logs report:

```text
Deleted 0 images.
```

This means no files were removed during the recorded runs.

> Caution: the current cleaning logic checks for the `JFIF` marker and can delete files directly from the dataset. Always keep a backup of the original dataset before running the scripts.

---

# Installation

## 1. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Models

Update the dataset path in each script.

For the standalone scripts, the original path is similar to:

```python
location = "/home/anbi23bd/DRUMSTER/DRUMSTER"
```

Change it to the location of your dataset.

Then run:

```bash
python CNN/CNN.py
```

```bash
python Resnet50/resnet.py
```

```bash
python VGG16/VGGNet.py
```

The scripts train the models, print validation results, calculate classification metrics, and save training/validation plots.

---

# Recommended Evaluation Fix

Before using the models for a formal report, the validation pipeline should be changed to deterministic ordering:

```python
val_ds = tf.keras.utils.image_dataset_from_directory(
    location,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=(180, 180),
    batch_size=64,
    shuffle=False
)
```

Then calculate labels and predictions from the same dataset:

```python
y_true = np.concatenate([y.numpy() for x, y in val_ds])

y_pred_prob = model.predict(val_ds)

# For sigmoid-output models:
y_pred = (y_pred_prob.ravel() >= 0.5).astype(int)
```

For the CNN, because the model uses:

```python
BinaryCrossentropy(from_logits=True)
```

use:

```python
y_pred_prob = tf.sigmoid(model.predict(val_ds)).numpy().ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)
```

Then calculate:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

This produces a more trustworthy model comparison.

---

# Reproducibility Notes

The original scripts use fixed random seeds for dataset splitting, but complete reproducibility may still depend on:

- TensorFlow version
- Keras version
- Python version
- CPU/GPU hardware
- CUDA/cuDNN versions when GPU training is used
- TensorFlow random operations
- Dataset file ordering

For a formal experiment, record the Python, TensorFlow, CUDA, and GPU versions together with the dataset version.

---

# Conclusion

This project demonstrates three approaches to drum-damage image classification:

- A custom CNN learns task-specific visual features from scratch.
- ResNet-50 uses transfer learning from an ImageNet-pretrained network.
- VGG-16 also uses ImageNet transfer learning with a lightweight binary classification head.

The supplied logs show high validation accuracy, but the precision/recall/F1 values reveal an evaluation-pipeline problem. The reported values should therefore be considered **recorded experimental outputs rather than validated final performance metrics** until the validation ordering and CNN logit threshold issues are corrected and the models are reevaluated.


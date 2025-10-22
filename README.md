# Parkinson's Disease Activity Recognition using Dual-Stream Transformer

This repository contains code for Human Activity Recognition (HAR) using smartwatch accelerometer data from both wrists to classify activities in healthy individuals and Parkinson's disease patients.

## Overview

The project implements a dual-stream transformer architecture that processes synchronized left and right wrist accelerometer data to recognize different activities. The model uses attention mechanisms to capture temporal patterns and cross-wrist coordination features.

## Dataset

The code uses the PADS (Parkinson's Disease Smartwatch) dataset, which contains:
- Accelerometer data from both left and right wrists
- Multiple activity types
- Data from both healthy subjects and Parkinson's disease patients
- 6-axis sensor data (likely 3-axis accelerometer + 3-axis gyroscope)

## Project Structure
```
├── data_preprocessing.py    # Converts raw .txt files to CSV format
├── classifier.py           # Main training script with dual-stream transformer
├── EDA.py
├── data_viz.ipynb
├── two_stream_transformer.pt
├── two_stream_transformer_hc_pd.pt                   # Exploratory data analysis and visualization
└── README.md
```


## Requirements
```
pandas
numpy
torch
scikit-learn
matplotlib
seaborn
```

Install dependencies:
```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn
```

## Usage

### 1. Data Preprocessing

First, convert raw sensor data files to a unified CSV format:
```python
python data_preprocessing.py
```

This script:
- Reads all `.txt` files from the dataset directory
- Extracts subject ID and activity labels from filenames
- Combines all data into a single `all_subjects.csv` file

**Configuration:**
- Update `ROOT` path to point to your dataset location
- Default expects 7 columns of sensor data

### 2. Model Training

Train the dual-stream transformer model:
```python
python classifier.py
```

**Key Features:**
- **Dual-stream architecture**: Separate transformer encoders for left and right wrist data
- **Automatic downsampling**: Reduces 2048-length sequences to 1024 for efficiency
- **Patch-based processing**: Reduces sequence length by averaging patches (default: patch_len=8)
- **Subject-aware splitting**: Prevents data leakage by splitting on subject IDs
- **Balanced sampling**: Matches healthy and Parkinson's subjects (70 each by default)

**Model Hyperparameters:**
```python
SEQ_LEN = 1024        # Sequence length after downsampling
PATCH_LEN = 8         # Temporal patch size (reduces to 128 timesteps)
EMBED_DIM = 128       # Transformer embedding dimension
FF_DIM = 256          # Feedforward layer dimension
NUM_HEADS = 4         # Number of attention heads
NUM_LAYERS = 4        # Number of transformer layers
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-3             # Learning rate
```

**Outputs:**
- `validation_results.csv`: Per-sample predictions with subject IDs and conditions
- `val_classification_results_hc_pd.csv`: Per-class precision, recall, F1-score
- Console output with classification reports and confusion matrices

### 3. Exploratory Data Analysis

Analyze the dataset and model results:
```python
python EDA.py
```

This script provides:
- Distribution plots for demographic variables (age, BMI, height, weight)
- Condition group distributions (Healthy vs Parkinson's vs Other)
- Confusion matrices with custom visualization
- Correlation analysis
- Missing value reports

**Update paths in the script:**
- `main_path`: Root directory of the dataset
- CSV file path for validation results

## Model Architecture

### Dual-Stream Transformer
```
Left Wrist Data  ─┐
                  ├──> StreamEncoder ──┐
                  │                    │
Right Wrist Data  │                    ├──> Concatenate ──> Classification Head│                    │
                 ─┘──> StreamEncoder ──┘
```

**StreamEncoder Components:**
1. **Linear Projection**: Maps input features to embedding dimension
2. **Positional Encoding**: Adds temporal position information
3. **CLS Token**: Learnable token prepended to sequence
4. **Transformer Encoder**: Multi-head self-attention layers
5. **Layer Normalization**: Normalizes CLS token output

**Key Design Choices:**
- Separate encoders allow learning wrist-specific patterns
- CLS token aggregates sequence information
- Concatenated representations capture cross-wrist coordination
- Pre-normalization (norm_first=True) for training stability

## Data Processing Pipeline

1. **Loading**: Read CSV with subject metadata and sensor readings
2. **Filtering**: Select healthy and Parkinson's subjects
3. **Balancing**: Match number of subjects per condition
4. **Pairing**: Align left and right wrist sequences by subject and activity
5. **Downsampling**: Reduce 2048 → 1024 timesteps if needed
6. **Normalization**: Z-score normalization using training set statistics
7. **Patching**: Optional temporal averaging to reduce sequence length

## Evaluation Metrics

The model outputs comprehensive evaluation including:
- **Overall accuracy**
- **Per-class metrics**: Precision, Recall, F1-Score, Support
- **Condition-specific evaluation**: Separate reports for Healthy and Parkinson's subjects
- **Confusion matrices**: Both overall and per-condition
- **Classification reports**: Detailed sklearn metrics

## Configuration

### Key Paths to Update

In `classifier.py`:
```python
main_path = 'YOUR_PATH_HERE/pads-parkinsons-disease-smartwatch-dataset-1.0.0/'
path = 'filtered_all_subjects_with_features.csv'
```

In `data_preprocessing.py`:
```python
ROOT = Path("YOUR_PATH_HERE/movement/timeseries/")
```

In `EDA.py`:
```python
main_path = 'YOUR_PATH_HERE/'
```

### Feature Selection

The model uses 6 sensor features:
```python
feature_cols = ["v2","v3","v4","v5","v6","v7"]
```

Note: `v1` appears to be excluded (possibly timestamp or redundant feature).


```
PADS - Parkinson's Disease Smartwatch Dataset
[Add proper citation when available]
```

## Model Description

two_stream_transformer: trained only on healthy data//
two_stream_transformer_hc_pd: trained using both//
two_stream_transformer_hc3: iterative trained model//
two_stream_transformerh: trained on PD

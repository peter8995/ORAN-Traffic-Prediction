# ORAN-Traffic-Prediction

This repository contains models and pipelines for predicting network traffic across different 5G O-RAN slices: **eMBB** (Enhanced Mobile Broadband), **mMTC** (Massive Machine Type Communications), and **uRLLC** (Ultra-Reliable Low-Latency Communications). 

The project leverages sequence-based time-series forecasting, utilizing a combination of **Vision Transformers (ViT)** for feature extraction and **Recurrent Neural Networks (LSTM / BiLSTM)** for temporal sequence modeling.

## 📂 Project Structure

- **`Dataset/`**: Contains raw and processed CSV data for the three network slices. Specifically, this incorporates data from the **TRACTOR dataset** (e.g., `Dataset/Tractor/`).
- **`model1/`**: Implements a unified multi-task predictive model.
- **`model2/`**: Implements independent predictive models for each network slice (MSE Loss).
- **`model3/`**: Implements independent predictive models, using tailored loss functions (e.g., Quantile Loss for mMTC).
- **`model4/`**: Implements a dynamic Colosseum ORAN dataset model with directory range parsing.
- **`model5/`**: Implements the SpikeAwareLSTM model for traffic prediction with dual-head spike detection.
- **`model6/`**: Anti-overfitting variant of model5 with ViT backbone freezing, dropout regularization, weight decay, early stopping, and pos_weight clamping.
- **`model_detect_peak/`**: Spike predictability baselines using Autoencoder and Isolation Forest (unsupervised anomaly detection).

## 📊 Datasets

This project utilizes data traces from comprehensive Open RAN research datasets:

### 1. Colosseum ORAN Coloran Dataset
A data-driven NextG Cellular Networks dataset supported by the U.S. NSF and ONR. It provides raw O-RAN KPI data collections.
- **Source**: [wineslab/colosseum-oran-coloran-dataset](https://github.com/wineslab/colosseum-oran-coloran-dataset)

### 2. TRACTOR Dataset
The "Traffic Analysis and Classification Tool for Open RAN" (TRACTOR) dataset originates from Northeastern University's GENESYS Lab. It provides comprehensive 5G traces and logs of O-RAN KPIs. The dataset is used to train machine learning models for user traffic classification in the near-RT RIC.
- **Source**: [genesys-neu/TRACTOR](https://github.com/genesys-neu/TRACTOR/tree/main)

## 🧠 Models Overview

### Model 1: Unified Multi-Task Model (`model1/`)
A single, joint model that predicts future traffic for all three slices simultaneously, as well as the aggregate global traffic.
- **Architecture**: Separate feature extraction backbones per slice (ViT + LSTM for eMBB, ViT + BiLSTM for mMTC and uRLLC). The extracted features are concatenated and fed into a **Multi-head Attention** mechanism to capture cross-slice dependencies before making final predictions.
- **Loss**: A combination of Mean Squared Error (MSE) and Huber Loss (Smooth L1) across the slice and global outputs.

### Model 2: Independent Slice Models (`model2/`)
Three separate models trained independently for each traffic slice.
- **Architecture**: Each slice model has its own backbone (similar to `model1`) and a linear regression head. There is no cross-slice attention.
- **Loss**: All three models are trained using standard Mean Squared Error (MSE) loss.

### Model 3: Tailored Independent Models (`model3/`)
Similar to `model2`, this approaches the problem using three independent models but optimizes them with slice-specific loss functions to better handle distinct traffic profiles.
- **Loss**: 
  - **eMBB** & **uRLLC**: Trained with Mean Squared Error (MSE) loss.
  - **mMTC**: Trained using **Quantile Loss** (e.g., q=0.7) to better account for the high variance, burstiness, and outlier characteristics typical of Machine Type Communications.

### Model 4: Dynamic Colosseum Model (`model4/`)
A model designed to integrate dynamically with the Colosseum ORAN Coloran dataset, supporting directory-based range training (e.g., `tr0-4`).
- **Data Extensibility**: Incorporates customized command-line argument parsing to handle variable directory ranges during training.

### Model 5: SpikeAwareLSTM Model (`model5/`)
A multi-task learning model inspired by the SpikeAwareLSTM architecture from ["Proactive AI-and-RAN Workload Orchestration in O-RAN Architectures for 6G Networks"](https://doi.org/10.1109/OJCOMS.2025.3608700), integrating anomaly (spike) detection directly within the prediction pipeline.
- **Architecture**: Dual-head output from shared ViT+LSTM+Attention backbone — one head for resource demand regression and another for spike probability classification (via `BCEWithLogitsLoss`).
- **Loss**: Composite loss `L = L_MSE + λ_detect × L_BCE`, where `λ_detect` (default 0.5) controls the trade-off between prediction accuracy and spike detection. Gradient clipping (`max_norm=1.0`) is applied for training stability.
- **Spike Detection**: Adaptive threshold labeling based on the paper's formula: `τ = Q_0.9 × (1 + ψ × ξ_m / ξ_max)`, with global `xi_max` computed across all training files via `fit_spike_params()`. Labels are generated on raw (unscaled) target values before MinMaxScaler transformation.
- **Data Processing**: Global `MinMaxScaler` fitted on all training files for consistent scaling. `DataProcessor` returns `(X, y, spike_labels)` triplets.
- **Evaluation**: Reports Precision, Recall, F1, Accuracy, FP Rate, and FN Rate for spike detection. Generates a dual-subplot visualization (demand prediction + spike detection overlay).

### Model 6: Anti-Overfitting SpikeAwareLSTM (`model6/`)
Built on top of Model 5, this model addresses the severe overfitting problem observed in Model 5's training (validation loss diverging after ~75 epochs) by applying multiple regularization techniques.
- **Architecture**: Same dual-head ViT+LSTM+Attention backbone as Model 5, with the following modifications:
  - **ViT Backbone Freezing**: All ViT encoder layers are frozen except `conv_proj`, `pos_embedding`, and the last 2 transformer blocks, significantly reducing trainable parameters.
  - **Dropout**: `Dropout(0.1)` added to FF layers, spike head, and attention (`dropout=0.1`).
  - **Spike Head**: Shares gradient flow with the backbone (no gradient detachment), allowing end-to-end multi-task learning.
- **Training**:
  - `Adam` optimizer with `weight_decay` (default `1e-4`) for L2 regularization, applied only to trainable parameters.
  - **Early stopping** with configurable `patience` (default 30 epochs) and automatic best model restoration.
  - `pos_weight` for `BCEWithLogitsLoss` clamped to `--max_pos_weight` (default `5.0`) to prevent extreme class imbalance from causing all-positive spike predictions.
- **Loss**: Same composite loss as Model 5: `L = L_MSE + λ_detect × L_BCE`, with `λ_detect` default `1.0`.

### Spike Predictability Baselines (`model_detect_peak/`)
Independent unsupervised/anomaly detection baselines to evaluate the inherent predictability of traffic spikes, without the full ViT+LSTM pipeline.
- **Models**: Autoencoder (PyTorch) and Isolation Forest (scikit-learn).
- **Input**: 17 features per timestep — 11 original O-RAN KPIs + 6 rolling temporal features (rolling mean/std/max over m=120, rolling mean/Q0.9 over L=1200, first-order difference). Sliding window of `seq_len=15` steps, flattened to a 255-dim vector.
- **Rolling Features Rationale**: The spike ground truth is defined by long-range rolling statistics (L=1200 ~5min, m=120 ~30s), but the sliding window only covers ~3.75s. Rolling features compress long-term trends into each timestep, bridging this time-scale gap.
- **Spike Labels**: Same adaptive threshold formula as model5/6: `τ = Q_0.9 × (1 + ψ × ξ_m / ξ_max)`, computed on raw (unscaled) target values.
- **Autoencoder**: Trains on all data, flags samples with reconstruction error above a percentile threshold (default P88) as spikes.
- **Isolation Forest**: Unsupervised anomaly detection with configurable contamination rate (default 0.11). Includes automatic threshold sweep to find the best F1.
- **Evaluation**: Reports Precision, Recall, F1, Accuracy, FP/FN Rate. Generates score distribution plots and spike detection timeline visualizations.

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- torchvision
- scikit-learn (MinMaxScaler, IsolationForest)
- pandas, numpy, matplotlib

### Data Preparation
Ensure the datasets are located in the `Dataset/` directory. The scripts are configured to look for CSV files representing the slices (e.g., `embb_11_18.csv`, `mmtc_11_18.csv`, `urll_11_18.csv`) within subdirectories like `Dataset/Tractor/Trial7/Raw/`.

### Training
For models 1-3, navigate to the model directory and run directly:
```bash
cd model1
python train.py
```

For models 4-6, use CLI arguments to specify dataset directories and hyperparameters:
```bash
cd model5
python train.py --train_dirs tr0-4 tr10 --test_dirs tr5-6 \
  --slice_type embb \
  --epochs 500 \
  --batch_size 1024 \
  --learning_rate 1e-6 \
  --sequence_length 15 \
  --val_split 0.2 \
  --lambda_detect 0.5   # model5 only: spike detection loss weight
```

Model 6 adds additional arguments on top of model 5:
```bash
cd model6
python train.py --train_dirs tr0-4 tr10 --test_dirs tr5-6 \
  --slice_type embb \
  --epochs 500 \
  --batch_size 1024 \
  --learning_rate 1e-4 \
  --sequence_length 15 \
  --val_split 0.2 \
  --lambda_detect 1.0 \
  --weight_decay 1e-4 \
  --patience 30 \
  --max_pos_weight 5.0
```
Directory range notation: `tr0-4` expands to `tr0, tr1, tr2, tr3, tr4`.

For spike detection baselines:
```bash
cd model_detect_peak

# Autoencoder
python autoencoder_detect.py --train_dirs tr0-4 tr10 --test_dirs tr5-6 \
  --slice_type embb --epochs 100 --sequence_length 15

# Isolation Forest
python isolation_forest_detect.py --train_dirs tr0-4 tr10 --test_dirs tr5-6 \
  --slice_type embb --sequence_length 15
```

### Results
During and after training, the scripts will generate evaluation metrics and save visualization plots in a `results/` folder within the respective model's directory:
- **Loss curve**: `{slice_type}_multi_dir_loss_curve.png`
- **Prediction comparison**: `{slice_type}_multi_dir_prediction.png`
- **Spike detection** (model5/6): `{slice_type}_spike_detection.png` — dual-subplot with demand prediction (top) and ground truth vs predicted spike overlay (bottom)
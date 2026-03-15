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
A model designed for improved O-RAN traffic prediction by explicitly accounting for traffic spikes.
- **Architecture**: Incorporates a `SpikeAwareLSTM` model with a dual-head output structure: one head for resource demand regression and another for spike probability classification.
- **Loss**: Computes a composite loss function combining standard regression loss with classification loss for spike detection.
- **Data Processing**: Includes custom data processing logic (`DataProcessor`) to automatically calculate, label, and integrate traffic spikes dynamically based on historical data averages.

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- pandas, numpy, matplotlib

### Data Preparation
Ensure the datasets are located in the `Dataset/` directory. The scripts are configured to look for CSV files representing the slices (e.g., `embb_11_18.csv`, `mmtc_11_18.csv`, `urll_11_18.csv`) within subdirectories like `Dataset/Tractor/Trial7/Raw/`.

### Training
To train any of the models, navigate to the specific model's directory and run `train.py`:
```bash
cd model1
python train.py
```

### Results
During and after training, the scripts will generate evaluation metrics and save visualization plots (prediction comparisons and loss curves) in a newly created `results/` folder within the respective model's directory.
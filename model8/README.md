# Project Overview

## Files

### 1. DataProcessor_v2.py
-  Multi-slice spatial-temporal matrix loading
-  Sequence extraction with flexible horizon
-  Global scaler fitting (train-only)
-  Comprehensive logging & error handling

### 2. model_v2.py
-  TinyViT for KPI sequence encoding
-  Per-slice ViT-LSTM/BiLSTM blocks
-  Hybrid architecture with attention fusion
-  Multi-horizon prediction ready

### 3. train_eval_v2.py
-  Smooth constraint regularization
-  Multi-loss training (regression + smoothness)
-  7+ publication-quality visualizations
-  Per-slice performance tables (CSV + LaTeX)
-  Error analysis & residual plots
-  Scatter plots with R² scores
-  Full training pipeline with early stopping

---

## 📊 Visualizations Generated

1. Per-slice predictions
2. Combined predictions (all slices)
3. Error distributions (histograms)
4. Error box plots
5. Residuals over time
6. Training history curves
7. Scatter plots (perfect prediction diagonal)

# 會使用codex來review程式碼
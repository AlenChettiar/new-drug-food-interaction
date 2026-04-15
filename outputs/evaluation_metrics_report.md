# Drug-Food Interaction Risk Assessment — Full Evaluation Report

## 1. Training Architecture
- **Model**: Soft-Voting Ensemble (XGBoost + RandomForest + GradientBoosting)
- **Task**: Multiclass Classification (0: Neutral, 1: Moderate, 2: Critical)
- **Preprocessing**: BorderlineSMOTE | RFE(20) | StandardScaler | Morgan FP (256-bit)
- **Split Strategy**: Leave-One-Drug-Out (LODO) for unseen evaluation
- **Training Sample Size**: 248 pairs
- **Validation Sample Size**: 62 pairs
- **Unseen Held-Out Size (LODO)**: 80 pairs

## 2. Classification Metrics by Split

### 🟢 Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9758 |
| Precision | 0.9796 |
| Recall | 0.9758 |
| F1 | 0.9770 |
| Roc_Auc | 0.9950 |

### 🟡 Validation Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9194 |
| Precision | 0.8745 |
| Recall | 0.9194 |
| F1 | 0.8963 |
| Roc_Auc | 0.9119 |

### 🔴 Unseen Test Set (LODO)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8375 |
| Precision | 0.7793 |
| Recall | 0.8375 |
| F1 | 0.7916 |
| Roc_Auc | 0.8768 |

## 3. Regression Metrics (% Bioavailability Change)

| Split | RMSE | MAE | R² |
|-------|------|-----|-----|
| Train | 2.8597 | 1.4940 | 0.9696 |
| Validation | 8.7305 | 4.4405 | 0.6691 |
| Unseen (LODO) | 15.3322 | 8.5298 | 0.3453 |

## 4. Visual Dashboards

### Evaluation Dashboard (Combined)
![Dashboard](./evaluation_dashboard.png)

### Confusion Matrices (Validation vs Unseen)
![Confusion Matrices](./confusion_matrices_val_unseen.png)

### Per-Class Metrics (Precision / Recall / F1)
![Per-Class Metrics](./per_class_metrics.png)

### ROC Curves (Critical vs Rest)
![ROC Curves](./roc_curves.png)

### Per-Class ROC Curves (One-vs-Rest)
![Per-Class ROC](./per_class_roc_curves.png)

### Metrics Radar Chart
![Radar Chart](./metrics_radar_chart.png)

### Class Distribution (Actual vs Predicted)
![Class Distribution](./class_distribution_comparison.png)

### Feature Importance (XGBoost sub-estimator)
![Feature Importance](./feature_importance.png)

### Predicted vs Actual (Regression)
![Predicted vs Actual](./predicted_vs_actual.png)

### Regression Residuals
![Residuals](./regression_residuals.png)

### Accuracy Comparison (Train/Val/Unseen)
![Accuracy](./accuracy_comparison.png)

### SHAP Interpretability
![SHAP](./shap_analysis.png)


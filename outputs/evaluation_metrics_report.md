# Drug-Food Interaction Risk Assessment — Full Evaluation Report

## 1. Training Architecture
- **Model**: Soft-Voting Ensemble (XGBoost + RandomForest + GradientBoosting)
- **Task**: Multiclass Classification (0: Neutral, 1: Moderate, 2: Critical)
- **Preprocessing**: BorderlineSMOTE | RFE(20) | StandardScaler | Morgan FP (256-bit)
- **Split Strategy**: Leave-One-Drug-Out (LODO) for unseen evaluation
- **Training Sample Size**: 283 pairs
- **Validation Sample Size**: 71 pairs
- **Unseen Held-Out Size (LODO)**: 84 pairs

## 2. Classification Metrics by Split

### 🟢 Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9541 |
| Precision | 0.9539 |
| Recall | 0.9541 |
| F1 | 0.9538 |
| Roc_Auc | 0.9949 |

### 🟡 Validation Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.7887 |
| Precision | 0.7881 |
| Recall | 0.7887 |
| F1 | 0.7790 |
| Roc_Auc | 0.8884 |

### 🔴 Unseen Test Set (LODO)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8810 |
| Precision | 0.8789 |
| Recall | 0.8810 |
| F1 | 0.8792 |
| Roc_Auc | 0.9528 |

## 3. Regression Metrics (% Bioavailability Change)

| Split | RMSE | MAE | R² |
|-------|------|-----|-----|
| Train | 6.2604 | 4.0180 | 0.9552 |
| Validation | 22.4861 | 17.3290 | 0.4091 |
| Unseen (LODO) | 20.5662 | 15.9350 | 0.4416 |

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


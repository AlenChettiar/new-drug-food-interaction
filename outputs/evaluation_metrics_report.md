# Drug-Food Interaction Risk Assessment — Full Evaluation Report

## 1. Training Architecture
- **Model**: Soft-Voting Ensemble (XGBoost + RandomForest + GradientBoosting)
- **Task**: Multiclass Classification (0: Neutral, 1: Moderate, 2: Critical)
- **Preprocessing**: BorderlineSMOTE | RFE(20) | StandardScaler | Morgan FP (256-bit)
- **Split Strategy**: Leave-One-Drug-Out (LODO) for unseen evaluation
- **Training Sample Size**: 120 pairs
- **Validation Sample Size**: 24 pairs
- **Unseen Held-Out Size (LODO)**: 52 pairs

## 2. Classification Metrics by Split

### 🟢 Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9750 |
| Precision | 0.9760 |
| Recall | 0.9750 |
| F1 | 0.9748 |
| Roc_Auc | 0.9992 |

### 🟡 Validation Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.8333 |
| Precision | 0.8472 |
| Recall | 0.8333 |
| F1 | 0.8232 |
| Roc_Auc | 0.9321 |

### 🔴 Unseen Test Set (LODO)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8077 |
| Precision | 0.8063 |
| Recall | 0.8077 |
| F1 | 0.7969 |
| Roc_Auc | 0.9342 |

## 3. Regression Metrics (% Bioavailability Change)

| Split | RMSE | MAE | R² |
|-------|------|-----|-----|
| Train | 2.4532 | 1.6287 | 0.9911 |
| Validation | 19.0861 | 14.0486 | 0.3555 |
| Unseen (LODO) | 18.7340 | 14.0220 | 0.4998 |

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


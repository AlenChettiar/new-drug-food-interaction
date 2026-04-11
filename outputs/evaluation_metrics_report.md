# Drug-Food Interaction Risk Assessment - Evaluation Report

## 1. Training Architecture
- **Task**: Multiclass Classification (0: Neutral, 1: Moderate, 2: Critical)
- **Training Sample Size**: 212 pairs
- **Validation Sample Size**: 54 pairs
- **Unseen Held-Out Size**: 50 pairs

## 2. Model Metrics by Split

### 🟢 Training Set Metrics
- **Accuracy**: 0.8160
- **Precision (Weighted)**: 0.8407
- **Recall (Weighted)**: 0.8160
- **F1 Score**: 0.8051

### 🟡 LODCO Validation Set Metrics
- **Accuracy**: 0.6852
- **Precision (Weighted)**: 0.7179
- **Recall (Weighted)**: 0.6852
- **F1 Score**: 0.6502

### 🔴 Unseen Test Set Metrics
- **Accuracy**: 0.7400
- **Precision (Weighted)**: 0.7159
- **Recall (Weighted)**: 0.7400
- **F1 Score**: 0.7142

## 3. Visual Dashboards

### Confusion Matrix (Validation)
![Confusion Matrix](./confusion_matrix.png)

### Accuracy Comparison
![Accuracy Comparison](./accuracy_comparison.png)

### ROC Curves
![ROC Curves](./roc_curves.png)

### Feature Importance
![Feature Importance](./feature_importance.png)


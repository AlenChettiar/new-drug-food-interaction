# ML Model Benchmarking Report — Real-World FDA Drug-Food Interactions

> Dataset: `clinical_interaction_real.csv` | Split: Leave-One-Drug-Out (LODO)
> Preprocessing: BorderlineSMOTE | RFE(20) | StandardScaler | Morgan FP (256-bit)

## Performance Summary

| Model                            |   Train Acc |   Val Acc |   Unseen Acc |   Val F1 |   Unseen F1 |   Val AUC |   Unseen AUC |
|:---------------------------------|------------:|----------:|-------------:|---------:|------------:|----------:|-------------:|
| Decision Tree                    |      0.8992 |    0.9355 |       0.85   |   0.9124 |      0.8357 |    0.7551 |       0.8136 |
| LightGBM                         |      0.9839 |    0.9355 |       0.8375 |   0.92   |      0.8187 |    0.8415 |       0.8733 |
| XGBoost                          |      0.9435 |    0.9355 |       0.85   |   0.9124 |      0.8306 |    0.8806 |       0.8617 |
| Soft-Voting Ensemble (XGB+RF+GB) |      0.9476 |    0.9355 |       0.8375 |   0.9124 |      0.7916 |    0.8578 |       0.8563 |
| Logistic Regression              |      0.9194 |    0.9194 |       0.825  |   0.9117 |      0.7759 |    0.8818 |       0.8872 |
| Random Forest                    |      0.9234 |    0.9194 |       0.8375 |   0.8963 |      0.7916 |    0.8468 |       0.8497 |
| Support Vector Machine           |      0.9435 |    0.871  |       0.825  |   0.8623 |      0.7759 |    0.8552 |       0.8562 |
| Naive Bayes                      |      0.8226 |    0.7742 |       0.725  |   0.784  |      0.6736 |    0.645  |       0.554  |
| AdaBoost                         |      0.5282 |    0.6452 |       0.6625 |   0.7001 |      0.6871 |    0.9349 |       0.8988 |
| K-Nearest Neighbors              |      0.7903 |    0.6129 |       0.6    |   0.6841 |      0.6385 |    0.5527 |       0.7801 |

## Winner
**Decision Tree** achieved the highest validation accuracy at `0.9355` and unseen accuracy at `0.8500`.

## Outputs
- `accuracy_comparison_all_models.png` — Bar chart per model across Train/Val/Unseen
- `f1_comparison_all_models.png` — F1 Score comparison
- `performance_heatmap_all_models.png` — All-metric heatmap
- `roc_auc_scatter_val_vs_unseen.png` — Generalisation scatter plot
- `<model>_metrics.txt` — Per-model text report
- `model_benchmarks.csv` — Full numeric table

| run | family | test_accuracy | test_precision | test_recall | test_f1 | test_roc_auc | test_pr_auc | cv_roc_auc_mean | overfit_gap_roc_auc | fit_seconds | predict_ms_per_1k_rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgb_shallow_weighted | xgboost | 0.730 | 0.495 | **0.832** | **0.621** | **0.841** | **0.668** | 0.846 | 0.066 | 0.298 | 6.490 |
| rf_shallow_balanced | random_forest | 0.733 | 0.499 | 0.782 | 0.609 | 0.835 | 0.652 | **0.848** | 0.068 | 0.373 | 33.059 |
| logreg_baseline | logistic_regression | **0.804** | **0.651** | 0.569 | 0.607 | 0.838 | 0.665 | 0.845 | 0.012 | 0.022 | 5.206 |
| logreg_l1_balanced | logistic_regression | 0.717 | 0.480 | 0.798 | 0.600 | 0.835 | 0.656 | 0.844 | 0.014 | 0.046 | 5.446 |
| xgb_deep_unweighted | xgboost | 0.772 | 0.578 | 0.523 | 0.549 | 0.801 | 0.584 | 0.819 | 0.199 | 0.852 | 7.466 |
| rf_deep_unweighted | random_forest | 0.778 | 0.599 | 0.496 | 0.543 | 0.807 | 0.574 | 0.824 | 0.193 | 0.116 | 20.188 |

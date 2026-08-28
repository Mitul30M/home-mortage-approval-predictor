# Day 6 Stronger Models - Summary

- Feature ablation (Step 0): full 103 vs trimmed 78 (adjusted: force-kept one level of fully-emptied family ['negative']). Gap=0.0004 -> FEATURES_DAY6 = 78 (trimmed).
- Imbalance: XGB/LGBM scale_pos_weight=0.3396; CatBoost auto_class_weights=Balanced. Early stopping=50 on X_val (AUC).
- CatBoost: run on one-hot FEATURES_DAY6 (native-categorical reconstruction deferred - LTI/applicant_age/*_missing derived in Day 4, not persisted).
- X_val metrics:
          roc_auc  pr_auc  precision  recall      f1
LR         0.8012  0.9127     0.8869  0.7225  0.7963
DT         0.8051  0.9045     0.8873  0.7327  0.8026
RF         0.8732  0.9417     0.8569  0.9652  0.9078
XGB        0.8851  0.9495     0.9124  0.8418  0.8757
LGBM       0.8823  0.9482     0.9118  0.8363  0.8724
CatBoost   0.8794  0.9470     0.9112  0.8311  0.8693

- Best model on X_val: XGB (ROC-AUC 0.8851).
- Single X_test look (XGB): ROC-AUC=0.8854, PR-AUC=0.9495, P=0.9125, R=0.8421, F1=0.8759.
- Val vs Test ROC-AUC gap: +0.0002.
- Open questions for Day 7:
  1. Which hyperparameters need Optuna: depth/leaves, lr, reg (reg_alpha/lambda, min_child_weight), subsample/colsample?
  2. Subsampling strategy for Optuna at this row count (budget)?
  3. Does XGB importance disagree with Day5 RF in a way that changes Day8 SHAP focus?
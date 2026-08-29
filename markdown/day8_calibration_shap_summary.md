# Day 8 - Calibration & SHAP Summary

- Model: Day 7 tuned XGBoost (20-trial Optuna). Reproduced val ROC-AUC=0.8930, test ROC-AUC=0.8933.
- Calibration: applied Platt (logistic) scaling on X_val only; RAW model retained for SHAP. Business output read as calibrated P(approve).
- Operating threshold (X_val Fbeta=0.3 precision-weighted) = 0.858.
  X_val : TPR=0.7347 FPR=0.1456 FNR=0.2653 P=0.9369 R=0.7347 F1=0.8236
  X_test: TPR=0.7357 FPR=0.1458 FNR=0.2643 P=0.9370 R=0.7357 F1=0.8242
- SHAP: TreeExplainer on raw model, stratified sample n=50000 (X_val).
- Top features by |SHAP|: loan_purpose_1, debt_to_income_ratio_missing, loan_to_value_ratio, income, property_value
- Race-proxy (tract_minority_population_percent): mean SHAP by derived_race computed (see proxy_shap_by_race.png).
- Open Q for Day 9: does proxy SHAP translate into subgroup error/calibration disparity? Use 0.5 or business threshold? Other geo-adjacent proxies?
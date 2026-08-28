# Day 7 Hyperparameter Tuning - Summary

- Tuning target: XGBoost; objective = PR-AUC on X_val; trials = 10; trial subsample = 1000000.
- scale_pos_weight fixed = 0.3396. Feature set = 78 (from day7_model_features.csv).
- Best PR-AUC (X_val) = 0.9494. Best params: {"max_depth": 10, "learning_rate": 0.029393392247576217, "n_estimators": 3000, "subsample": 0.86637030299845, "colsample_bytree": 0.7602638701835958, "min_child_weight": 6, "reg_lambda": 0.012341225700071089, "reg_alpha": 0.06889775400590788}
- Day 6 untuned XGBoost (X_val): n/a
- Tuned XGBoost X_val: ROC-AUC=0.8958 PR-AUC=0.9543 P=0.9147 R=0.8620 F1=0.8876
- Tuned XGBoost X_test (single look): ROC-AUC=0.8961 PR-AUC=0.9544 P=0.9148 R=0.8623 F1=0.8878
- Note: X_test was used only for this final evaluation (tuning used X_val only).
- Artifacts: day7_best_xgb_params.json, day7_tuned_metrics.csv, day7_optuna_progress.csv, figures/day7/*.png.
- Next: Day 8 SHAP on tuned model; Day 9 fairness across derived_race/ethnicity/sex.
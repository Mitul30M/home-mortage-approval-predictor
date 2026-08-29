# Day 7 Hyperparameter Tuning - Summary

- Tuning target: XGBoost; objective = PR-AUC on X_val; trials = 20; trial subsample = 1000000.
- scale_pos_weight fixed = 0.3396. Feature set = 78 (from day7_model_features.csv).
- Best PR-AUC (X_val) = 0.9496. Best params: {"max_depth": 13, "learning_rate": 0.008616500012778569, "n_estimators": 2500, "subsample": 0.9944859888550582, "colsample_bytree": 0.5952111558253492, "min_child_weight": 20, "reg_lambda": 0.418806703627406, "reg_alpha": 0.12419627835407367}- Day 6 untuned XGBoost (X_val): n/a
- Tuned XGBoost X_val: ROC-AUC=0.8930 PR-AUC=0.9531 P=0.9142 R=0.8565 F1=0.8844
- Tuned XGBoost X_test (single look): ROC-AUC=0.8933 PR-AUC=0.9532 P=0.9143 R=0.8571 F1=0.8848
- Note: X_test was used only for this final evaluation (tuning used X_val only).
- Artifacts: day7_best_xgb_params.json, day7_tuned_metrics.csv, day7_optuna_progress.csv, figures/day7/*.png.
- Next: Day 8 SHAP on tuned model; Day 9 fairness across derived_race/ethnicity/sex.
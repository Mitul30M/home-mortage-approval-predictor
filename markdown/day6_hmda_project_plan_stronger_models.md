# HMDA 2024 Mortgage Lending Project — Day 6

**Inputs from Day 5 (`5_baseline_models.ipynb`):**
- `X_train` (~5.95M) / `X_val` (661,199) / `X_test` (1,652,997) — three-way split, `X_test` never touched yet
- `val_demographics_lookup` — row-aligned to `X_val`
- Baseline metrics on `X_val`: LR 0.8012 / DT 0.8051 / **RF 0.8732** ROC-AUC (RF also leads PR-AUC 0.9417, recall 0.9652)
- `day5_feature_selection.csv` — 77 keep / 26 drop, all core continuous features retained, drops are rare one-hot levels + one redundant `_missing` flag
- `day5_cluster_profile.csv` — 6 clusters, approval rate 62.1%–86.2% vs. 74.65% overall

**Day 6 focus:** move from bagged (Random Forest) to boosted ensembles — XGBoost, LightGBM, CatBoost — with early stopping on `X_val`, a real ablation test of the Day 5 feature-selection recommendation (rather than applying it on faith), and exactly one honest look at `X_test` at the very end for the single model chosen to carry forward.

---

## Step 0 — Ablate the Day 5 Feature-Selection Recommendation

Before trusting the "77 keep / 26 drop" table on the models it wasn't derived from, test it.

**Achievable:**
- First, close yesterday's open item: confirm none of the 26 drop candidates fully empties a one-hot family (i.e. every categorical group still has at least one surviving level after the drops). If any family *would* be fully dropped, decide explicitly whether that's acceptable (implicit reference category) or whether that feature should be force-kept.
- Train one quick LightGBM (or XGBoost — pick whichever is fastest to iterate with, doesn't need to be the "real" model yet) on the **full 103-feature set**, and a second one on the **77-feature trimmed set**, same hyperparameters, same early stopping on `X_val`.
- Compare ROC-AUC and PR-AUC between the two. If the gap is within noise (~0.001–0.002), proceed with the trimmed set for faster iteration through the rest of Day 6 and Day 7's tuning. If the full set meaningfully outperforms, keep all 103 features and treat the Day 5 table as informative-but-not-actioned.
- Document the decision either way — this is the actual resolution to the "domain knowledge vs. computational necessity" question from earlier in the week.

**Deliverable:** a one-paragraph ablation result + the final feature set (`FEATURES_DAY6`) used for every model below.

---

## Step 1 — Shared Setup for Boosted Models

**Achievable:**
- Reuse Day 5's `evaluate_model` function unchanged — same metrics (ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix), same evaluation partition (`X_val`) for every model in this notebook, so the comparison stays apples-to-apples with the Day 5 baselines.
- Decide and document the imbalance-handling approach **per library**, since each one exposes it differently:
  - **XGBoost:** `scale_pos_weight = (neg_count / pos_count)` on `y_train`
  - **LightGBM:** either `is_unbalance=True` or an explicit `scale_pos_weight`, not both — pick one and note why
  - **CatBoost:** `auto_class_weights='Balanced'` or explicit `class_weights=[...]`
- Set a consistent early-stopping protocol across all three: same `eval_set=(X_val, y_val)`, same `eval_metric` (AUC or logloss — pick one and use it everywhere), same `early_stopping_rounds` (e.g. 50), so "how many rounds each model needed" is itself a comparable number.
- Note the RF headline numbers (0.8732 ROC-AUC, 0.9652 recall) as the bar to beat — but also flag explicitly that RF's very high recall may be partly a class-weighting artifact, not a free lunch; PR-AUC and precision matter just as much for a fair comparison, not ROC-AUC alone.

**Deliverable:** shared `eval_set`, `eval_metric`, and per-library imbalance-handling documented in one place before any model is fit.

---

## Step 2 — XGBoost Baseline (Untuned)

**Achievable:**
- Fit `XGBClassifier` with sensible defaults (moderate `max_depth`, `n_estimators` high with early stopping doing the real work, `learning_rate` ~0.05–0.1) on `FEATURES_DAY6`, `scale_pos_weight` from Step 1.
- Evaluate on `X_val` with the shared function.
- Extract `feature_importances_` (gain-based, not just split-count — specify `importance_type='gain'`) and compare ranking against Day 5's RF ranking as a sanity check — large disagreement is worth a note, not necessarily a problem.
- Record how many boosting rounds early stopping actually used — useful context for Day 7's tuning search space.

**Deliverable:** XGBoost metrics on `X_val` + gain-based feature importance table.

---

## Step 3 — LightGBM Baseline (Untuned)

**Achievable:**
- Fit `LGBMClassifier` with comparable defaults to Step 2 (similar depth/leaves, `learning_rate`, early stopping on `X_val`) — the goal here is a fair head-to-head with XGBoost, not two arbitrarily different configs.
- Evaluate on `X_val`.
- Extract feature importances (`importance_type='gain'`), compare ranking against XGBoost's from Step 2.
- Note LightGBM's training time vs. XGBoost's at this data scale — this project's whole premise leans on "large-scale," so a training-time note here is a legitimate, resume-relevant data point, not just trivia.

**Deliverable:** LightGBM metrics on `X_val` + feature importance table + training-time comparison note.

---

## Step 4 — CatBoost Baseline (Untuned)

CatBoost's core strength is native categorical handling — worth a real design decision here, not just running it on the same one-hot matrix as the other two out of convenience.

**Achievable:**
- Decide explicitly: run CatBoost on the same one-hot `FEATURES_DAY6` matrix for a clean three-way comparison, **or** reconstruct the pre-one-hot categorical columns (loan_type, loan_purpose, lien_status, etc. as native categorical features via `cat_features=[...]`) to let CatBoost use its actual strength. Document which was chosen and why — if resume/portfolio value matters, showing you understood the trade-off (and ideally running both) is worth more than defaulting to whichever was easier.
- Fit `CatBoostClassifier` with early stopping on `X_val` (`eval_set`, `early_stopping_rounds`), class weighting from Step 1.
- Evaluate on `X_val`.
- Extract feature importances, compare ranking against Steps 2–3.

**Deliverable:** CatBoost metrics on `X_val` + feature importance table + the one-hot-vs-native-categorical decision documented.

---

## Step 5 — Four-Way Model Comparison

**Achievable:**
- Consolidated metrics table: Logistic Regression, Decision Tree, Random Forest (from Day 5) alongside XGBoost, LightGBM, CatBoost (today) — one table, `X_val` throughout.
- Overlaid ROC and PR curves across all six models (or at minimum the three Day-6 boosted models plus RF as the baseline-to-beat) — reuse Day 5's plotting pattern for visual consistency across the project.
- Cross-model feature-importance comparison: build one combined ranking table (XGBoost gain / LightGBM gain / CatBoost gain / RF from Day 5) — this is a preview of Day 8's SHAP work, not a replacement for it, but useful now for sanity-checking which features consistently matter across every method tried so far.
- Explicitly identify the single best-performing untuned boosted model on `X_val` — this becomes the primary candidate for Day 7's Optuna tuning, though the others aren't discarded yet.

**Deliverable:** the six-model comparison table + ROC/PR overlay + combined feature-importance table.

---

## Step 6 — One Honest Look at `X_test`

`X_test` has been reserved since Day 4/5 specifically for this moment — a single, final, un-iterated-on check, not a second validation set to tune against.

**Achievable:**
- Take the single best model identified in Step 5 (by `X_val` performance) and evaluate it **once** on `X_test`.
- Report the same metric set (ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix) and compare directly against its `X_val` numbers — a large gap between val and test performance is the signal to worry about overfitting to `X_val` via repeated comparison, not proceed silently.
- **Do not** go back and re-tune anything based on this result — if the test number is concerning, that's a Day 7 conversation about regularization, not a reason to re-run Day 6 against test again. Document the number and move on; that discipline is the whole point of holding test out this long.

**Deliverable:** one clearly-labeled "final Day 6 model, test-set performance" result — the number that will anchor Day 7's tuning target and eventually the resume line.

---

## Step 7 — Wrap-up

Write a short Day 6 summary: the feature-ablation decision from Step 0, headline `X_val` metrics for all three boosted models, the winning model and its one-time `X_test` result from Step 6, and open questions for Day 7:

- Which hyperparameters most need Optuna's attention given today's untuned results — depth/leaves, learning rate, regularization (`reg_alpha`/`reg_lambda`, `min_child_weight`), or subsampling (`subsample`/`colsample_bytree`)?
- Given the stratified-subsample compute trade-off used for Day 5's Random Forest, what subsample strategy should Optuna's search use to stay within a reasonable tuning budget at this row count?
- Does the winning boosted model's feature-importance ranking meaningfully disagree with Day 5's RF ranking in a way that changes which features Day 8's SHAP analysis should focus on?

---

## End-of-Day-6 Checklist

- [ ] One-hot family integrity check completed for Day 5's 26 drop candidates (no family fully emptied, or exception explicitly justified)
- [ ] Feature-selection ablation run (full 103 vs. trimmed 77) on a quick boosted model; final `FEATURES_DAY6` set decided and documented
- [ ] Shared `eval_set`/`eval_metric`/early-stopping protocol defined once, reused across all three boosted models
- [ ] Per-library imbalance handling decided and documented (XGBoost `scale_pos_weight`, LightGBM `is_unbalance`/`scale_pos_weight`, CatBoost `auto_class_weights`/`class_weights`)
- [ ] XGBoost fit, evaluated on `X_val`, gain-based importances extracted
- [ ] LightGBM fit, evaluated on `X_val`, gain-based importances extracted, training time noted vs. XGBoost
- [ ] CatBoost fit, evaluated on `X_val`, importances extracted, one-hot-vs-native-categorical decision documented
- [ ] Six-model comparison table (LR/DT/RF/XGB/LGBM/CatBoost) + ROC/PR overlay produced
- [ ] Combined cross-model feature-importance table produced
- [ ] Best model on `X_val` identified and evaluated exactly once on `X_test`; val-vs-test gap reported
- [ ] Day 6 summary + open questions for Day 7 written

**Not for Day 6 (save for later days):** hyperparameter tuning / Optuna search (Day 7), SHAP explainability (Day 8), fairness metric computation (Day 9), deployment demo (Day 10). `X_test` is not to be touched again until the final tuned model in Day 7 needs its own one-time check.

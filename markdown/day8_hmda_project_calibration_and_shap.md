# HMDA 2024 Mortgage Lending Project — Day 8

**Inputs from Day 7 (`7_collab_hyperparameter_tuning.ipynb`):**
- Tuned XGBoost, 78-feature canonical schema (`day7_model_features.csv`), `is_race_proxy=1` on `tract_minority_population_percent`
- Tuning target: XGBoost; objective = PR-AUC on X_val; trials = 20; trial subsample = 1000000.
- scale_pos_weight fixed = 0.3396. Feature set = 78 (from day7_model_features.csv).
- Best PR-AUC (X_val) = 0.9496. Best params: {"max_depth": 13, "learning_rate": 0.008616500012778569, "n_estimators": 2500, "subsample": 0.9944859888550582, "colsample_bytree": 0.5952111558253492, "min_child_weight": 20, "reg_lambda": 0.418806703627406, "reg_alpha": 0.12419627835407367}
- Note: X_test was used only for this final evaluation (tuning used X_val only).
- Artifacts: day7_best_xgb_params.json, day7_tuned_metrics.csv, day7_optuna_progress.csv, figures/day7/*.png.
- Tuned XGBoost X_val: ROC-AUC=0.8930 PR-AUC=0.9531 P=0.9142 R=0.8565 F1=0.8844
- Tuned XGBoost X_test (single look): ROC-AUC=0.8933 PR-AUC=0.9532 P=0.9143 R=0.8571 F1=0.8848

**Day 8 focus:** README §7.4 (full evaluation — calibration + business-threshold confusion matrix, not just the AUC numbers already in hand) and §7.5 (SHAP explainability) together, since neither is complete without the other — a threshold choice needs to be explainable, and SHAP needs a model that's actually been checked for calibration first.

---

## Step 0 — Pre-Flight: Column-Order Alignment Check

Everything since Day 6 has run on positional numpy arrays with no column names attached. SHAP is the first step where that becomes a real risk — a silent off-by-one here mislabels every feature-attribution plot without erroring.

**Achievable:**
- Reload the exact array-building code from Day 7 (however `X_train`/`X_val`/`X_test` were filtered down to the 78 columns) and confirm, index by index, that the resulting array's column order matches `day7_model_features.csv`'s row order — not just the same 78 names, the same **order**.
- Add an explicit assertion for this (e.g. re-derive the filtered column list from the raw 103-column schema using the same filter logic, and compare it element-wise against the CSV's `feature` column) so this can't silently drift again in Day 9/10.
- Confirm the tuned model from Day 7 (`day7_best_xgb_params.json` + retrained weights) is reloadable and reproduces the same val/test metrics before building anything on top of it today.

**Deliverable:** one passing assertion cell confirming array-order ↔ CSV-order alignment, kept at the top of the notebook as a standing guard.

---

## Step 1 — Calibration Check

`scale_pos_weight` reweights the loss function, which typically distorts predicted probabilities away from true empirical frequencies — worth checking before treating XGBoost's output as "probability of approval" anywhere in Days 8–10.

**Achievable:**
- Plot a calibration curve (reliability diagram) on `X_val`: predicted probability (binned) vs. actual observed approval rate in each bin.
- If the curve shows meaningful distortion (a class-weighted model typically over- or under-predicts extreme probabilities), decide whether to apply post-hoc recalibration (Platt scaling or isotonic regression, fit on `X_val`, never `X_test`) or explicitly document that the model's output should be read as a **ranking score**, not a calibrated probability, for the rest of the project.
- If recalibration is applied, re-plot the calibration curve after correction to confirm it actually improved.
- **Important ordering note:** if recalibration is applied (Platt scaling or isotonic regression), keep the **raw, uncalibrated model** as the one SHAP explains in Steps 3–6 below. Both calibration methods are monotonic, so they don't change ranking, threshold choice, or which features matter — but isotonic regression in particular breaks the additive assumption SHAP's math relies on, making attributions harder to interpret cleanly. Use the calibrated probabilities only for the business-facing threshold/risk-tier output in Step 2 and `explain_application` in Step 5; use the raw model's score for the actual `TreeExplainer` calls.

**Deliverable:** calibration plot (before/after if recalibrated) + a one-line decision on how model output should be interpreted going forward + the explicit raw-vs-calibrated split documented for Steps 3–6.

---

## Step 2 — Business-Relevant Operating Threshold

All metrics so far have used the default 0.5 cutoff. README's denial-risk framing (0–20% low / 20–50% moderate / 50–75% high / 75–100% very high) implies a deliberate threshold choice matters more here than raw AUC.

**Achievable:**
- Use the PR curve on `X_val` to inspect the precision/recall trade-off across thresholds, not just at 0.5.
- Pick and document an operating threshold with an explicit rationale (e.g. optimizing F1, or holding recall on true approvals above some bar since a false denial is the fair-lending-sensitive error type per the README's FNR framing) — this doesn't need to be optimal, it needs to be **defensible and written down**.
- Recompute the confusion matrix at the chosen threshold (not 0.5) on `X_val`, and once on `X_test` alongside the existing default-threshold numbers for comparison.

**Deliverable:** chosen threshold + rationale, confusion matrix at that threshold for both `X_val` and `X_test`.

---

## Step 3 — SHAP Setup at Scale

**Achievable:**
- Use `shap.TreeExplainer` — XGBoost is natively supported and fast; no need for the model-agnostic (and far slower) `KernelExplainer`.
- Don't run SHAP on the full multi-million-row `X_test`/`X_train` — draw a **stratified explanation sample** (e.g. 50,000–100,000 rows, stratified on `approved` and, if feasible, cross-checked against the demographics lookup table so later steps have adequate per-group representation) rather than the full set.
- Pass feature names from `day7_model_features.csv` explicitly into every SHAP plot call, using the Step 0 alignment guard — this is where a silent misalignment would actually show up as wrong labels on a plot, not an error.
- Note SHAP compute time at this sample size — another legitimate scale data point for the project narrative.

**Deliverable:** a reusable `X_explain` sample + fitted `shap.TreeExplainer` + computed SHAP values, ready for the plots below.

---

## Step 4 — Global Interpretability

**Achievable:**
- SHAP summary (beeswarm) plot — the headline global-importance visual for the whole project.
- SHAP mean-|value| bar plot — cleaner ranking view for the write-up.
- Build one combined ranking table: SHAP mean-|value| alongside Day 5/6's LR coefficient rank, DT/RF impurity rank, and XGBoost/LightGBM/CatBoost gain rank — SHAP is generally the more trustworthy ranking (accounts for interactions, not biased toward high-cardinality splits the way impurity/gain can be), so where it disagrees with the earlier gain-based rankings, say so explicitly and treat SHAP as the tiebreaker for the project's final feature-importance narrative.

**Deliverable:** summary + bar plots, and the final cross-method importance comparison table for the project.

---

## Step 5 — Local (Per-Application) Interpretability

This is the README §7.5 deliverable directly — the "predicted probability of approval: 21.6%... top contributing factors" style output.

**Achievable:**
- Build a small reusable `explain_application(row_index)` function that returns: predicted approval probability, derived denial-risk (`1 − P(approved)`), risk tier (using the README's 0–20/20–50/50–75/75–100% bands), and the top contributing SHAP factors in plain language.
- Run it on a handful of illustrative cases from the explanation sample: a clear high-confidence approval, a clear high-confidence denial, and one or two borderline cases near the Step 2 operating threshold — borderline cases are usually the most interesting for an underwriter-facing explanation.
- Produce a SHAP waterfall or force plot for each example case.

**Deliverable:** the `explain_application` function + rendered output for 3–4 example applications.

---

## Step 6 — Race-Proxy SHAP Check

Direct, quantified follow-up to the `is_race_proxy` flag carried since Day 7 — turns a documented suspicion into actual evidence ahead of Day 9.

**Achievable:**
- Pull the SHAP values specifically for `tract_minority_population_percent` across the explanation sample.
- Report its rank in the Step 4 global importance table, and its distribution of SHAP contribution magnitude/direction.
- Using the demographics lookup table (row-aligned from Day 4/5), check whether this feature's SHAP value differs systematically by `derived_race` — if applicants in different race categories get pushed toward denial by this feature at different rates, that's a concrete, quantified proxy-risk finding to hand to Day 9, not just a flag on a CSV.

**Deliverable:** a short proxy-risk finding — SHAP-value distribution for `tract_minority_population_percent`, by race group, with a plain-language interpretation.

---

## Step 7 — Wrap-up

Write a short Day 8 summary: calibration finding and whether recalibration was applied, the chosen operating threshold and its confusion matrix on val/test, the final cross-method feature-importance ranking, the race-proxy SHAP finding, and open questions for Day 9:

- Does the race-proxy SHAP signal from Step 6 translate into an actual disparity in the model's error rates or calibration by group?
- Should Day 9's fairness evaluation use the default 0.5 threshold or the Step 2 business threshold — do subgroup disparities look different at each?
- Are there other geography-adjacent features (beyond the one already flagged) whose SHAP behavior suggests proxy risk that Day 4/7 didn't originally flag?

---

## End-of-Day-8 Checklist

- [ ] Column-order alignment between array filtering and `day7_model_features.csv` explicitly verified
- [ ] Tuned Day 7 model reloaded and reproduces reported val/test metrics
- [ ] Calibration curve plotted; recalibration applied or explicitly declined, with rationale
- [ ] Business-relevant operating threshold chosen and documented, distinct from default 0.5
- [ ] Confusion matrix recomputed at the chosen threshold for `X_val` and `X_test`
- [ ] Stratified SHAP explanation sample built; `TreeExplainer` fit
- [ ] SHAP summary + bar plots produced
- [ ] Combined cross-method feature-importance table (SHAP + all prior rankings) produced
- [ ] `explain_application` function built and run on 3–4 example cases with waterfall/force plots
- [ ] Race-proxy SHAP finding for `tract_minority_population_percent` computed by `derived_race`
- [ ] Day 8 summary + open questions for Day 9 written

**Not for Day 8 (save for later days):** full subgroup fairness metric computation (TPR/FPR/calibration by group — Day 9), any further hyperparameter re-tuning, deployment demo (Day 10).

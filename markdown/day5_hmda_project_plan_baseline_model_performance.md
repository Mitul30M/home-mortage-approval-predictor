# HMDA 2024 Mortgage Lending Project — Day 5

**Inputs from Day 4:**
- `X_train` (6,611,985 × 103), `y_train` (6,611,985 × 1, `approved` int8) — confirmed
- `X_test` / `y_test` — schema not yet independently shown to me; Step 1 below folds a parity check into the baseline-prep work so this isn't a silent gap going into modeling
- `test_demographics_lookup.parquet` (or equivalent) from Day 4, preserving `derived_race` / `derived_ethnicity` / `derived_sex` out-of-band for later fairness work

**Day 5 focus:** two things, in order —
1. **Step 0 — unsupervised segmentation** on a stratified 1M-row sample, purely descriptive, before any label-driven modeling starts. This is exploratory context, not a substitute for the deferred EDA, and cluster labels are *not* automatically fed into the baseline models below.
2. **Baseline supervised models** — Logistic Regression, Decision Tree, and Random Forest — evaluated consistently and used to produce a documented feature-selection recommendation, per the Day 5 conversation about where feature selection actually lives in this project.

---

## Step 0 — Stratified Clustering (Unsupervised Segmentation)

Goal: see if there are natural applicant/loan segments in the data *before* looking at approval as a label — done on a manageable stratified sample so it doesn't become a second full-scale EDA.

**Achievable:**
- Draw a **stratified sample of 1,000,000 rows** from the training population (join `X_train` + `y_train` on index, stratify the sample on `approved` so the sample's approval rate stays at ~74.6%, matching the full training population — verify this explicitly after sampling, don't just assume `train_test_split(..., stratify=...)` got it exactly right at this size).
- **Preprocessing for clustering (distinct from the modeling preprocessing already done):**
  - Standardize the continuous columns (`loan_amount`, `loan_to_value_ratio`, `property_value`, `income`, `debt_to_income_ratio`, `loan_term`, `tract_population`, `tract_minority_population_percent`, `ffiec_msa_md_median_family_income`, `tract_to_msa_income_percentage`, `tract_owner_occupied_units`, `tract_one_to_four_family_homes`, `tract_median_age_of_housing_units`, `loan_to_income_ratio`) — unscaled continuous features next to 0/1 one-hot columns will otherwise dominate the distance metric by magnitude alone.
  - Leave the one-hot categorical blocks as-is, but be aware that Euclidean distance treats each one-hot level as an independent dimension — with ~90 categorical columns vs. ~13 continuous ones, categorical structure will dominate the clustering unless you either (a) accept that and interpret clusters primarily as categorical-profile groups, or (b) down-weight the categorical block, or (c) reduce dimensionality first. Pick one and document why.
  - Given the column-count imbalance, a PCA step before clustering (e.g. to ~20–30 components capturing most variance) is worth trying and comparing against clustering on the raw 103-dim space — document which was used.
- **Algorithm choice:** use **MiniBatchKMeans**, not full KMeans/HDBSCAN/agglomerative — at 1M rows × 103 (or fewer, post-PCA) columns, MiniBatchKMeans is the only one of these that stays practically fast while still giving centroid-based, interpretable clusters. Document this choice and the trade-off (approximate vs. exact centroids) in one line.
- **Choosing k:** run the elbow method (inertia vs. k) and silhouette score on a smaller sub-sample (e.g. 50,000 rows) first — don't compute silhouette on the full 1M set, it's O(n²) and impractical at this scale. Use the sub-sample result to pick a candidate k, then fit MiniBatchKMeans on the full 1M-row sample with that k.
- **Profile each cluster:**
  - Mean/median of each continuous feature per cluster
  - Dominant one-hot categories per cluster (e.g. most common `loan_type_*`, `applicant_age_*` level)
  - **Approval rate within each cluster**, compared against the overall ~74.6% — clusters that deviate meaningfully are the most informative output of this step
  - Cross-tab cluster membership against the Day 4 demographics lookup table (`derived_race`, `derived_sex`) — descriptive only, explicitly labeled as such, not a fairness test (that's Day 9)
- **Decision point to document, not resolve today:** whether cluster ID becomes an engineered feature for the supervised models later, or stays a purely descriptive/reporting artifact. Default to *not* feeding it into Steps 2–4 below unless there's a clear reason — adding an unsupervised label as a supervised feature needs its own leakage reasoning (cluster fit on train-only, applied to test, same as any other train-fit transform).

**Deliverable:** a cluster-profile table (cluster ID → size, key feature means, dominant categories, in-cluster approval rate, demographic composition) plus a short write-up of what k was chosen, why, and what the segments seem to represent.

---

## Step 1 — Baseline-Model Prep

**Achievable:**
- Reload `X_train`/`X_test`/`y_train`/`y_test`; confirm **column-for-column schema parity** between train and test (same 103 columns, same order, same dtypes) — this is the outstanding Day 4 check, folded in here so it happens before any model touches the data.
- Confirm the demographics lookup table row-aligns to `X_test`'s index — needed later, not today, but worth failing loudly now rather than at Day 9.
- Build one shared evaluation function used by all three baselines below: ROC-AUC, PR-AUC, precision/recall/F1 at the default 0.5 threshold, and a confusion matrix — so Logistic Regression, Decision Tree, and Random Forest are compared on identical terms.
- Decide and document the imbalance-handling approach for baselines: `class_weight='balanced'` (or `'balanced_subsample'` for Random Forest) across all three models, consistent with the README's "class weighting, not oversampling" decision.

**Deliverable:** a reusable `evaluate_model(model, X_test, y_test)` function and a confirmed train/test schema-parity check.

---

## Step 2 — Logistic Regression Baseline

**Achievable:**
- Fit `LogisticRegression(class_weight='balanced')` on `X_train` — given the row count (6.6M), watch fit time; if `lbfgs` is impractically slow, document switching to `saga` or a `SGDClassifier(loss='log_loss')` equivalent and note the trade-off.
- Evaluate with the shared function from Step 1.
- Extract coefficients and rank features by absolute standardized coefficient magnitude.
- Compute VIF **only on the continuous, non-dummy features** — the one-hot groups will show infinite/undefined VIF by construction (each group sums to 1 across its rows), which is expected and not a real collinearity finding; don't let it get mis-flagged as a problem.
- Note any coefficients that are near-zero and not statistically distinguishable from zero (via standard errors/p-values if using `statsmodels` alongside sklearn for this step) as drop candidates.

**Deliverable:** coefficient table (feature, coefficient, direction, near-zero flag) + VIF table for continuous features only.

---

## Step 3 — Decision Tree Baseline

**Achievable:**
- Fit a shallow, interpretable `DecisionTreeClassifier(max_depth=5–8, class_weight='balanced')` — depth capped deliberately for readability, not for best performance (that's what Random Forest and Day 6's boosted models are for).
- Evaluate with the shared function.
- Extract `feature_importances_` and rank.
- Compare this ranking against the Logistic Regression ranking from Step 2 (e.g. side-by-side table, or Spearman correlation between the two rankings) — where they agree, that's a strong drop/keep signal; where they disagree, flag it as a likely non-linear or interaction effect the linear model can't see.
- Optionally visualize the top 2–3 levels of the tree for the write-up — useful for explaining an early split in plain language (e.g. "the top split is on `debt_to_income_ratio`").

**Deliverable:** feature-importance table + a ranking-comparison table against Logistic Regression.

---

## Step 4 — Random Forest Baseline

Included at "baseline" tier per today's scope, even though it's already an ensemble — treat it as the bridge model whose feature-importance signal you can trust more than a single tree's, before committing to Day 6's boosted models.

**Achievable:**
- Fit `RandomForestClassifier(n_estimators=100–200, class_weight='balanced_subsample', n_jobs=-1)`. At 6.6M rows, consider fitting on a large stratified subsample (e.g. 1–2M rows, same stratified-on-`approved` approach as Step 0) to keep Day 5 runtime reasonable — document this trade-off explicitly if you subsample; full-data Random Forest can be revisited in Day 6 if needed.
- Evaluate with the shared function.
- Extract `feature_importances_`, add to the same ranking-comparison table from Step 3.
- Note where Random Forest's ranking diverges from both Logistic Regression and the single Decision Tree — this is your first signal of which features carry real non-linear/interaction value worth keeping into Day 6, versus features only a single overfit tree liked.

**Deliverable:** feature-importance table added to the running comparison, plus a one-line note on any subsampling used and why.

---

## Step 5 — Cross-Baseline Comparison & Feature-Selection Recommendation

This is the actual "where does feature selection happen" answer from earlier — Day 5 produces the recommendation, it doesn't yet act on it.

**Achievable:**
- Build one combined ranking table: feature → LR coefficient rank, DT importance rank, RF importance rank, and an average/composite rank.
- Flag **consistent low-signal features** (bottom-ranked across all three methods) as drop candidates for Day 6.
- Flag features where rankings strongly disagree (e.g. low in LR, high in RF) as "keep — likely non-linear/interaction signal," explicitly not drop candidates despite the linear model's read.
- Re-state the one-hot VIF caveat from Step 2 here too, since this table is where someone might misread it a second time.

**Deliverable:** the actual feature-selection recommendation — a table of `feature | LR rank | DT rank | RF rank | recommendation (drop / keep / investigate)` — documented but not yet applied to the training data.

---

## Step 6 — Model Comparison & Evaluation Summary

**Achievable:**
- Consolidated metrics table across all three baselines: ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix.
- Overlaid ROC curves and PR curves for the three models on one plot each.
- One-paragraph note on which baseline currently looks strongest purely on discrimination ability — framed as context for Day 6, not a final model choice.

**Deliverable:** comparison table + ROC/PR overlay plots.

---

## Step 7 — Wrap-up

Write a short Day 5 summary: best-performing baseline and its headline metrics, the cluster-segmentation findings from Step 0 (including any clusters with notably different approval rates or demographic composition — flagged as descriptive, not causal), the feature-selection recommendation table, and open questions for Day 6:

- Does dropping the Step 5 "drop candidate" features actually hurt LightGBM/CatBoost performance, or confirm they were noise?
- Do any Step 0 clusters warrant separate modeling or just monitoring as a segment in later evaluation?
- Given Random Forest's importance ranking, are there interaction effects worth engineering explicitly before Day 6, or is that better left to the boosted models to discover on their own?

---

## End-of-Day-5 Checklist

- [ ] Stratified 1M-row clustering sample drawn, approval rate confirmed ≈74.6%
- [ ] Clustering preprocessing decided and documented (scaling approach, PCA or not, categorical weighting)
- [ ] k chosen via elbow/silhouette on a sub-sample; MiniBatchKMeans fit on the full 1M sample
- [ ] Cluster profile table produced (feature means, dominant categories, in-cluster approval rate, demographic composition)
- [ ] Train/test schema parity confirmed (outstanding Day 4 item, closed here)
- [ ] Shared `evaluate_model` function built and reused across all three baselines
- [ ] Logistic Regression fit, evaluated, coefficients ranked, VIF computed on continuous features only
- [ ] Decision Tree fit, evaluated, feature importances ranked and compared to Logistic Regression
- [ ] Random Forest fit, evaluated, feature importances ranked and added to the comparison
- [ ] Combined feature-selection recommendation table produced (drop / keep / investigate)
- [ ] Consolidated metrics table + ROC/PR overlay plots produced
- [ ] Day 5 summary + open questions for Day 6 written

**Not for Day 5 (save for later days):** applying the feature-selection recommendation (Day 6 decision), hyperparameter tuning/Optuna, SHAP/explainability, fairness metric computation (Day 9), deployment.

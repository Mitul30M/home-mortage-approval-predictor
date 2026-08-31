# Home Mortgage Application Approval Predictor — Project Summary

**Domain:** Finance / Credit Risk (Lending Analytics)
**Dataset:** 2024 HMDA National Loan-Level Dataset (~12.2M real records, ~100 fields each)
**Type:** Traditional ML — binary classification (approve vs. deny), with responsible-AI / fairness evaluation
**Scope:** decision-support system — **not** an autonomous approval/rejection system

---

## Table of Contents

1. [Problem Framing & Objective](#1-problem-framing--objective)
2. [Data Source & Target Definition](#2-data-source--target-definition)
3. [Step 1 — Data Ingestion & Foundation](#3-step-1--data-ingestion--foundation)
4. [Step 2 — Data Cleaning & Fair-Lending Descriptive Pass](#4-step-2--data-cleaning--fair-lending-descriptive-pass)
5. [Step 3 — Deferred EDA](#5-step-3--deferred-eda)
6. [Step 4 — Preprocessing & Feature Engineering](#6-step-4--preprocessing--feature-engineering)
7. [Step 5 — Baseline Models](#7-step-5--baseline-models)
8. [Step 6 — Stronger Models](#8-step-6--stronger-models)
9. [Step 7 — Hyperparameter Tuning (Optuna)](#9-step-7--hyperparameter-tuning-optuna)
10. [Step 8 — Calibration & SHAP Explainability](#10-step-8--calibration--shap-explainability)
11. [Step 9 — Fairness Analysis Across Demographic Groups](#11-step-9--fairness-analysis-across-demographic-groups)
12. [Step 10 — Final Demo Notebook](#12-step-10--final-demo-notebook)
13. [Key Methodological Decisions](#13-key-methodological-decisions)
14. [Limitations & Responsible-Use Statement](#14-limitations--responsible-use-statement)
15. [Deliverables & Artifacts](#15-deliverables--artifacts)

---

## 1. Problem Framing & Objective

The project asks two questions simultaneously:

1. **Predictive:** Given only pre-decision applicant, loan, and property information, how likely is a mortgage application to be approved?
2. **Responsible-AI:** Does a model that predicts this well also behave *consistently* across race, ethnicity, and sex subgroups?

The target variable is `approved`, derived from `action_taken`:

| `action_taken` | Meaning | `approved` |
|---|---|---|
| `1` | Loan originated | `1` |
| `3` | Application denied | `0` |
| all other codes | No clean approve/deny decision | **Excluded** |

**Why "approval" as the positive class (not "denial"):** the model's positive predictions correspond to the outcome an applicant wants — keeping precision/recall/SHAP outputs intuitive ("predicted probability of approval," not a double-negative).

The output is a **probability of approval**. For underwriting workflows this is read as a **denial-risk score** = `1 − P(approve)`, with illustrative tiers:

| Denial-risk band | Label |
|---|---|
| 0–20% | Low (high approval probability) |
| 20–50% | Moderate |
| 50–75% | High |
| 75–100% | Very high (low approval probability) |

---

## 2. Data Source & Target Definition

- **Source:** [2024 HMDA Snapshot National Loan-Level Dataset](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/) — FFIEC/CFPB, Home Mortgage Disclosure Act
- **Scale:** 12,229,298 records × 99 columns (raw, all strings)
- **Genuineness:** real regulatory reporting data — every row is an actual mortgage application
- **Decision subset** (originated `action_taken=1` + denied `action_taken=3`): **8,276,018 records** (67.7% of full dataset), approval rate **74.63%**

---

## 3. Step 1 — Data Ingestion & Foundation (`notebooks/1_data_load.ipynb`)

### What we did
- Loaded all 12.2M rows × 99 columns using an explicit `dtype` mapping (not default string inference) — raw all-string load used **8.62 GB RAM**.
- Kept **60 columns** split into four buckets:
  - **33 model-safe features** (used later for prediction)
  - **7 fair-lending audit-only columns** (used to *measure* disparities, but excluded from the model — race/sex/age directly would let the model learn a demographic signal, which is both ethically and legally wrong for a lending model)
  - **13 leakage-prone columns** excluded (e.g., `rate_spread`, `denial_reason_*`, `total_loan_costs`, `origination_charges` — populated *after* the decision)
  - **6 ID/geo columns** excluded from the model by default (`lei`, `census_tract`, etc.; used for grouping/joins, not raw features)
- Converted numeric fields to `float32`/`int32`, low-cardinality strings to `category` → memory down to **1.75 GB** (≈6× reduction).
- Saved typed artifacts:
  - `hmda_2024_typed.parquet` (12,229,298 × 65) — the typed full dataset
  - `hmda_2024_decisions.parquet` (8,276,018 × 66) — the originated+denied subset
  - `hmda_2024_model.parquet` (12,229,298 × 34) — leakage-free model inputs for the full population
  - `hmda_2024_model_decisions.parquet` (8,276,018 × 34) — leakage-free model inputs for the decision subset
  - `1_column_dictionary.csv` (60 rows) — column name × dtype × role × `use_in_model` flag

### Key decisions
- **Leakage-first column triage:** before any statistical filtering, every column was classified as pre-decision (usable) vs. post-decision/outcome-derived (excluded). This is the difference between a model that looks good in testing and one that would actually work in production.
- **Target = approval, not denial:** deliberate framing so that "predicted probability of approval" and SHAP attributions are intuitive to read for underwriters and applicants alike.
- **Duplicates:** 29,475 exact duplicate rows (0.2410%) flagged and logged; resolved in Step 2.

### Findings
- `action_taken` breakdown: Loan originated 50.50%, Denied 17.17%, Withdrawn 12.57%, Purchased loan 10.41%, Closed for incompleteness 4.76%, Approved-not-accepted 2.95%, Preapproval outcomes ~1.3%.
- 31 of 65 columns have any missing values.
- Demographic composition of full dataset: White 64.22%, Race Not Available 17.46%, Black 8.81%, Asian 6.06%.

---

## 4. Step 2 — Data Cleaning & Fair-Lending Descriptive Pass (`notebooks/2_data_cleaning_fairlending.ipynb`)

### What we did
- Investigated the outlier/invalid-value problems Step 1 surfaced:
  - `income` minimum = **−155,317** (thousands of $) — impossible; 7,182 rows (0.087%) had negative income → re-nulled and re-imputed at median.
  - `loan_to_value_ratio` mean 2,741 with std ~4.36M — impossible for a percentage (realistically 0–200); 2,150 rows had LTV > 500% → set to NaN.
  - `loan_amount`, `property_value`, `interest_rate` capped at the 99.5th percentile.
- **Cleaning rules table:**

| Column | Rule | Rows affected | % of decision subset |
|---|---|---|---|
| income | negative values → NaN | 7,182 | 0.087% |
| income | capped at 99.5th pct (1,191) | 38,758 | 0.468% |
| loan_to_value_ratio | <0 or >500 → NaN | 2,150 | 0.026% |
| loan_amount | capped at 99.5th pct (2,005,000) | 39,791 | 0.481% |
| property_value | capped at 99.5th pct (3,635,000) | 38,697 | 0.468% |

- **Duplicates:** 11,036 exact duplicates dropped (0.1333%) → final row count **8,264,982**.
- **Missingness strategy table** — grouped by *why* missing:
  - *Structurally missing* (`denial_reason_2/3/4`, 96–99.9%): left as-is (missing = "not applicable").
  - *Reporting-exemption driven* (`total_points_and_fees`, `prepayment_penalty_term`, etc.): treated as its own meaningful category.
  - *Real missing data* (`income`, `property_value`, `debt_to_income_ratio`, `loan_to_value_ratio`, `interest_rate`): planned for median imputation + `*_missing` flag in Step 4.
- **First fair-lending descriptive pass** (raw, unadjusted — explicitly flagged):
  - Approval rate by `derived_race`: **Asian 76.35%**, White 73.99%, AI/AN 61.23%, Black 60.95%.
  - Approval rate by `derived_sex`: **Male 73.31%**, Female 70.39%.
  - Denial reasons by race: DTI and credit history are the two most common denial reasons; their ranking differs across race groups.
  - Median income/loan_amount by demographic: Asian median income $133k vs. AI/AN $76k; Male $95k vs. Female $76k — context for interpreting raw gaps.
- **Exclusion rule for primary comparisons** (carried forward): `Joint`, `*_Not Available`, `Free Form Text Only` excluded from the primary race/ethnicity/sex comparison; sizes reported separately. Primary comparison subsets:
  - `derived_race`: 3,973,081 rows (48.07%); 6 groups after exclusion
  - `derived_ethnicity`: 539,278 rows; 2 groups
  - `derived_sex`: 3,897,740 rows; 2 groups

### Why these decisions matter
- Capping at the 99.5th percentile (not an arbitrary dollar figure) keeps genuinely large-but-real loans while removing data-entry noise — documented with row-impact counts so a reviewer can verify.
- The "exclude Joint/NA/Free Form Text Only" rule is applied identically in Step 2's descriptive pass, Step 9's fairness evaluation, and the final demo — methodological consistency across the project.

---

## 5. Step 3 — Deferred EDA (`notebooks/hmda_eda.ipynb`)

### What we did
- **Deferred deliberately** (not skipped): the README's Step 3 "EDA at scale" was pushed to *after* the modeling pipeline was stable, because EDA at this scale is expensive and the modeling notebooks needed the cleaned, leakage-free frame first. It was finally executed once the pipeline was reproducible.
- Run on `hmda_2024_clean.parquet` — **pre-split, pre-imputation** — so it has real `NaN`s in `income`, `property_value`, `debt_to_income_ratio`, `loan_to_value_ratio`, and `interest_rate`, making it the right file for genuine missingness EDA (modeling-stage files already have these imputed and flagged, which would hide the patterns EDA is supposed to surface).
- `loan_to_income_ratio` recomputed inline (it was engineered in Step 4 and doesn't exist in this file).
- **9 sections (A.1–A.9):** univariate continuous (histograms + box plots + summary stats), univariate categorical (bar charts + pie charts for low-cardinality fields only), target/demographic composition, missingness (bar chart + co-occurrence heatmap), feature-vs-target bivariate, feature-vs-feature scatter (including the `tract_minority_population_percent` × `tract_to_msa_income_percentage` proxy pair), correlation heatmaps (full + geography-focused), demographic-specific distributions, optional geography.
- **All plots rendered inline, none saved to `figures/`** — EDA is exploratory scratch work, not a versioned deliverable.
- Uses a **250k reproducible sample** for plots (full file for statistics).

### Findings (smoke-tested)
- Approved share 0.7465 (matches Step 1/2).
- 16 continuous + 17 categorical + 5 pie-low fields identified.

---

## 6. Step 4 — Preprocessing & Feature Engineering (`notebooks/4_preprocessing_feature_engineering.ipynb`)

### What we did
- **Sentinel/exempt code audit:** scanned `MODEL_COLUMNS` against the LAR data-fields reference. Found `applicant_credit_score_type` has 238,185 rows coded `'1111'` (= Exempt). No other sentinel contamination found. Categorical sentinels (`'1111'` → Exempt, `'8888'` → Not-Applicable, `'9999'` → No Co-Applicant) handled explicitly, not treated as ordinary missing values.
- **Categorical encoding** (17 features):
  - **One-hot** for low-cardinality fields: `loan_type` (4), `loan_purpose` (6), `lien_status` (2), `occupancy_type` (3), `construction_method` (2), `derived_dwelling_category` (4), `applicant_credit_score_type`, `co_applicant_credit_score_type`, `submission_of_application`, `reverse_mortgage`, `open_end_line_of_credit`, `business_or_commercial_purpose`, `preapproval`, `negative_amortization`, `interest_only_payment`, `balloon_payment`.
  - High-cardinality geo IDs confirmed **absent** from the feature set (no raw `census_tract`/`lei` leaked in).
- **Derived features:**
  - `loan_to_income_ratio = loan_amount / income` — re-verified no zero/negative `income` remain post-Step-2 cleaning (109,336 income≤0 re-nulled). Capped at 99.9th percentile (42.63) to avoid division-by-near-zero extremes.
  - `loan_to_value_ratio` already present in the cleaned base (README's loan-to-property-value requirement already covered).
  - `rate_spread_bucket` computed (below_typical / typical / high / elevated) — **leakage-excluded**, not added to the model frame (rate_spread is post-decision).
  - `applicant_age` promoted as a model categorical (8 levels + Unknown).
- **Train/Val/Test split** (stratified on `approved`):

| Partition | Rows | Features | Approved % |
|---|---|---|---|
| X_train | 5,950,786 | 103 | 0.7465 |
| X_val | 661,199 | 103 | 0.7465 |
| X_test | 1,652,997 | 103 | 0.7465 |

- All feature dtypes `float32`. Train/Val/Test schema-matched (same columns, same order, same dtypes).
- **Demographics lookup tables** built row-index-aligned: `val_demographics_lookup.parquet` (661,199 × 5), `test_demographics_lookup.parquet` (1,652,997 × 5) — columns: `derived_race`, `derived_ethnicity`, `derived_sex`, `applicant_age`, `row_id`. Race/sex preserved **out-of-band** (not in the model matrix) for Step 9's fairness work.
- Test demographic group sizes (derived_race): White 1,062,536; Race Not Available 287,999; Black 145,119; Asian 100,127; plus smaller groups.

### Why these decisions matter
- The train/test split is done here (not Step 5) because categorical encoding decisions — specifically any target/frequency encoding — are only leakage-safe when fit on a training partition. Encoding fit on the full dataset before splitting would leak test information into the training distribution.
- `tract_minority_population_percent` explicitly **flagged `is_race_proxy=1`** in the column dictionary and **kept out of the model** — a well-known redlining-adjacent proxy. It is retained only for the fairness-audit dimension.

---

## 7. Step 5 — Baseline Models (`notebooks/5_baseline_models.ipynb`)

### What we did
- **Step 0 — Unsupervised segmentation** (descriptive, not fed into models): stratified 1M-row sample from training data, PCA to 26 components (90% variance), MiniBatchKMeans with k=6 (silhouette 0.091). Cluster approval rates ranged 62.1%–86.2% vs. 74.65% overall. Cluster labels treated as a descriptive artifact, not an engineered feature.
- **Step 1 — Shared evaluation function:** ROC-AUC, PR-AUC, precision, recall, F1 at default 0.5, plus confusion matrix. `class_weight='balanced'` for LR and DT; `class_weight='balanced_subsample'` for RF (1.5M subsample, approval rate confirmed 0.7465).
- **Three baselines** evaluated identically:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.8012 | 0.9127 | 0.8869 | 0.7225 | 0.7963 |
| Decision Tree (max_depth 5–8) | 0.8051 | 0.9045 | 0.8873 | 0.7327 | 0.8026 |
| **Random Forest** (100–200 trees) | **0.8732** | **0.9417** | 0.8569 | **0.9652** | **0.9078** |

- **Step 5 — Combined feature-selection recommendation** (one table, three methods): 77 keep / 26 drop. Consistent low-signal features across all three methods flagged as drop candidates (e.g., `loan_type_4`, many rare `co_applicant_credit_score_type` levels, several `applicant_credit_score_type` Exempt levels, `negative_amortization_Exempt`, `balloon_payment_Exempt`). Features where rankings disagree (e.g., low in LR, high in RF) flagged as "keep — likely non-linear/interaction signal."
- RF feature importances (#1 income, #2 loan_to_income_ratio, #3 loan_to_value_ratio).

### Why these decisions matter
- The three baselines establish a **progressive comparison** (logistic → single tree → ensemble) rather than jumping straight to the most complex model.
- RF recall of 0.9652 looks impressive but is partly a `class_weight='balanced_subsample'` artifact — PR-AUC and precision matter just as much for a fair comparison.
- The feature-selection recommendation was **documented but not yet applied** — Step 6 actually ablates it.

---

## 8. Step 6 — Stronger Models (`notebooks/6_stronger_models.ipynb`)

### What we did
- **Step 0 — Ablate the Step 5 feature-selection recommendation:** trained one quick LightGBM on the full 103-feature set and one on the trimmed 77-feature set, same hyperparameters, same early stopping on X_val. Gap = **0.0004** → within noise → **FEATURES_Step6 = 78** (trimmed; force-kept one level of a fully-emptied categorical family). This is the actual resolution to the "domain knowledge vs. computational necessity" question.
- **Step 1 — Shared setup:** `scale_pos_weight = 0.3396` (= neg_count / pos_count on y_train) for XGBoost/LightGBM; CatBoost `auto_class_weights='Balanced'`. Early stopping: 50 rounds on X_val, eval_metric = AUC, consistent across all three.
- **Steps 2–4 — Three boosted baselines (untuned):**

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **XGBoost** | **0.8851** | **0.9495** | 0.9124 | 0.8418 | 0.8757 |
| LightGBM | 0.8823 | 0.9482 | 0.9118 | 0.8363 | 0.8724 |
| CatBoost | 0.8794 | 0.9470 | 0.9112 | 0.8311 | 0.8693 |

- CatBoost run on the one-hot FEATURES_Step6 matrix (native-categorical reconstruction deferred because derived features like `loan_to_income_ratio`, `applicant_age`, `*_missing` were engineered in Step 4 and not persisted in pre-encoded form).
- **Step 6 — Single honest look at X_test** (best model = XGB): ROC-AUC=0.8854, PR-AUC=0.9495, P=0.9125, R=0.8421, F1=0.8759. **Val-vs-test ROC-AUC gap = +0.0002** → essentially no overfitting signal, the model generalizes cleanly.

### Why these decisions matter
- The ablation test means we don't blindly trust Step 5's "77 keep / 26 drop" table — we actually tested it and found the trimmed set performs identically, so we proceed with the smaller, faster feature set.
- XGBoost chosen as the tuning candidate because it leads both ROC-AUC (0.8851) and PR-AUC (0.9495) on X_val.

---

## 9. Step 7 — Hyperparameter Tuning (`notebooks/7_collab_hyperparameter_tuning.ipynb`)

### What we did
- **Optuna**, 20 trials, objective = PR-AUC on X_val. Each trial trained on a stratified 1M-row subsample (compute trade-off: tuning on the full 5.95M train set would be prohibitively slow at this row count; the subsample preserves class balance and the full X_val is used for validation).
- `scale_pos_weight` **fixed at 0.3396** (not tuned — the README specifies class weighting, not naive oversampling, at this scale).
- Feature set = 78 (from `Step7_model_features.csv`).
- **Best params:**

```json
{
  "max_depth": 13,
  "learning_rate": 0.008616500012778569,
  "n_estimators": 2500,
  "subsample": 0.9944859888550582,
  "colsample_bytree": 0.5952111558253492,
  "min_child_weight": 20,
  "reg_lambda": 0.418806703627406,
  "reg_alpha": 0.12419627835407367
}
```

- **Artifacts:** `Step7_best_xgb_params.json`, `Step7_tuned_metrics.csv`, `Step7_optuna_progress.csv`, `figures/Step7/*.png`.
- **Reproduced metrics:**

| Split | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| X_val | 0.8930 | 0.9531 | 0.9142 | 0.8565 | 0.8844 |
| X_test (single look) | 0.8933 | 0.9532 | 0.9143 | 0.8571 | 0.8848 |

- X_test used **only once** for this final evaluation; tuning used X_val only.

### Why these decisions matter
- 20 trials is an explicit, documented compute trade-off: more trials could squeeze out marginal gains, but 20 trials on a 1M subsample reproduces the full-data-tuned metrics almost exactly (val/test ROC-AUC differ by 0.0003).
- `scale_pos_weight` fixed (not tuned) keeps the imbalance-handling strategy identical across all Steps — the tuned model's probabilities are directly comparable to the baseline models' probabilities.

---

## 10. Step 8 — Calibration & SHAP Explainability (`notebooks/8_calibration_shap.ipynb`)

### What we did
- **Step 0 — Column-order alignment guard:** re-derived the filtered column list from the raw 103-column schema and compared element-wise against `Step7_model_features.csv`. A standing guard cell at the top of the notebook — SHAP is the first step where a silent off-by-one would mislabel every feature-attribution plot without erroring.
- **Step 1 — Calibration check:** `scale_pos_weight` reweights the loss function and typically distorts predicted probabilities away from empirical frequencies. Applied **Platt (logistic) scaling** fit on X_val only. The raw, uncalibrated model is **retained for SHAP** (both Platt and isotonic regression are monotonic, so they don't change ranking, threshold choice, or which features matter — but isotonic regression breaks the additive assumption SHAP's math relies on). Calibrated probabilities used only for the business-facing threshold/risk-tier output.
- **Step 2 — Business-relevant operating threshold:** Fβ=0.3 (precision-weighted) on the PR curve → **THRESH = 0.858**. This threshold prioritizes precision over recall, reflecting that a false approval (over-approval, FPR) is a principal+interest-at-risk financial risk, while the fair-lending-sensitive error (FNR — a qualified applicant wrongly flagged) is reported by subgroup with the same weight as overall accuracy.

**Confusion matrix at THRESH = 0.858:**

| Split | TPR | FPR | FNR | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| X_val | 0.7347 | 0.1456 | 0.2653 | 0.9369 | 0.7347 | 0.8236 |
| X_test | 0.7357 | 0.1458 | 0.2643 | 0.9370 | 0.7357 | 0.8242 |

- **Step 3 — SHAP setup:** `TreeExplainer` on the raw model, stratified explanation sample of **50,000 rows from X_val** (stratified on `approved`, aligned positionally with `val_demographics_lookup.parquet`). Compute time ≈ 1164.6s. SHAP values shape (50000, 78), expected_value ≈ −0.00079.
- **Step 4 — Global interpretability:** beeswarm plot + bar plot + cross-method importance table (SHAP mean-|value| ranked alongside Step 5/6's LR coefficient rank, DT/RF impurity rank, and XGB/LGBM/CatBoost gain rank).

**Top 10 features by mean |SHAP|:**

| Rank | Feature | SHAP mean |abs| |
|---|---|---|
| 1 | `loan_purpose_1` | 0.5046 |
| 2 | `debt_to_income_ratio_missing` | 0.3776 |
| 3 | `loan_to_value_ratio` | 0.3425 |
| 4 | `income` | 0.2675 |
| 5 | `property_value` | 0.2623 |
| 6 | `loan_to_income_ratio` | 0.1671 |
| 7 | `tract_minority_population_percent` | 0.1469 |
| 8 | `lien_status_1` | 0.1171 |
| 9 | `preapproval_1` | 0.1089 |
| 10 | `co_applicant_credit_score_type_10` | 0.1084 |

**Cross-method comparison (SHAP rank vs. gain/coefficient rank):**
- `loan_purpose_1`: #1 SHAP vs. #18 LR coef — SHAP captures the non-linear/interaction effect the linear model misses.
- `tract_minority_population_percent`: #7 SHAP vs. #50 XGB gain — SHAP values this geography-adjacent proxy more highly than XGBoost's gain-based importance (which is biased toward high-cardinality splits).
- `income`: #4 SHAP vs. #1 RF importance — RF over-weights income relative to the SHAP interaction-aware ranking.
- **SHAP is treated as the tiebreaker** for the project's final feature-importance narrative.

- **Step 5 — Local interpretability:** `explain_application(row_index)` function returns calibrated P(approve), denial-risk tier, and top SHAP factors in plain language. Three example applications:
  - **Application 7184** (P=0.983, actual=1): top factors `preapproval_1 (+4.10)`, `preapproval_2 (+1.74)`, `loan_term (+0.93)`, `loan_purpose_1 (+0.77)`.
  - **Application 1207** and **Application 1826** — similar waterfall-style explanations.
- **Step 6 — Race-proxy SHAP check:** `tract_minority_population_percent` SHAP mean differs by `derived_race` — the feature's contribution to the log-odds is not uniform across race groups, with the largest mean SHAP for Asian and Native Hawaiian/PI applicants (small-n caveat applies). This is descriptive evidence of a proxy signal, **not** a causal fairness finding.

### Figures (`figures/Step8/`)
- `calibration_raw.png`, `calibration_calibrated.png` — reliability diagrams before/after Platt scaling.
- `confusion_matrix.png` — confusion matrix at THRESH=0.858.
- `shap_beeswarm.png`, `shap_bar.png` — global SHAP plots.
- `proxy_shap_by_race.png` — `tract_minority_population_percent` SHAP by `derived_race`.
- `waterfall_7184.png`, `waterfall_1207.png`, `waterfall_1826.png` — local per-application explanations.

### Why these decisions matter
- **Raw model retained for SHAP, calibrated for business output:** this is a deliberate architectural split. Using the calibrated model for SHAP would break SHAP's additive assumption (isotonic) and distort the feature attribution. Using the raw model for SHAP preserves the math while using calibrated probabilities for the risk-tier output preserves the business-facing interpretation.
- **THRESH=0.858 vs. 0.5:** the default 0.5 threshold produces a TPR of only ~0.35 at this operating point (TPR=0.3459, FNR=0.6541) — meaning 65% of actually-approved applicants would be wrongly flagged as high-risk. The precision-weighted 0.858 threshold lifts TPR to ~0.74, which is the right operating point for a decision-support system.

---

## 11. Step 9 — Fairness Analysis Across Demographic Groups (`notebooks/9_fairness_analysis.ipynb`)

### What we did
- **Step 0 — Setup & methodology:**
  - Confirmed/rebuilt `test_demographics_lookup.parquet` (row-index-aligned to X_test) for Step 9.
  - Reapplied Step 2's exclusion rule (`Joint`, `*_Not Available`, `Free Form Text Only` excluded from primary comparison; sizes reported separately).
  - **Evaluation protocol:** X_val is the working partition for Steps 1–5 (investigative); X_test gets exactly one confirmatory look in Step 6.
  - Evaluated at **both** thresholds: 0.5 (default) and 0.858 (business).
  - **Small-sample rule:** flag any group with n < 500 in the evaluation subset.
- **Step 1 — Subgroup confusion-matrix metrics** (TPR, FPR, FNR, precision, recall, F1, n) for `derived_race`, `derived_ethnicity`, `derived_sex` at both thresholds.
- **Step 2 — Quantify disparities:** gap between highest- and lowest-scoring group per metric per dimension, at both thresholds.
- **Step 3 — Calibration by group:** reliability diagrams for the calibrated model's output, split by `derived_race` and `derived_sex`.
- **Step 4 — Tie back the Step 8 race-proxy finding:** put the Step 8 SHAP-by-race finding for `tract_minority_population_percent` side-by-side with Step 1–2's actual error-rate disparities.
- **Step 5 — Profile the false-negative population:** examine the ~457k actually-approved applicants predicted as "deny" at THRESH=0.858 by demographic composition.
- **Step 6 — One honest confirmatory look at X_test:** repeat the core subgroup table on X_test; compare against X_val findings.

### Reproduced metrics (Step 7 tuned XGBoost, max_depth=13, Platt-calibrated)
- val ROC-AUC=0.8930, test ROC-AUC=0.8933.
- Business threshold (X_val Fβ=0.3) = **0.8582**.
- **FN population (val+test combined) at THRESH = 457,066** of actually-approved applicants wrongly predicted as "deny."

### Subgroup error-rate disparities (FNR gaps, highest vs. lowest group)

| Dimension | @0.5 FNR gap | @0.858 FNR gap | Verdict |
|---|---|---|---|
| `derived_race` | 0.0125 (NHPI vs 2+ minority) | 0.0064 (White vs 2+ minority) | **Helped** at business threshold |
| `derived_ethnicity` | 0.0012 (Hispanic vs Non-Hispanic) | 0.0057 (Hispanic vs Non-Hispanic) | **Widened** at business threshold |
| `derived_sex` | 0.0012 (Female vs Male) | 0.0028 (Female vs Male) | **Widened** at business threshold |

### False-negative population profile
- 457,066 actually-approved applicants (val+test) are predicted as deny at THRESH.
- Over-representation ratios by `derived_race`:
  - **Black or African American: 1.275×** (127.5% of their share of actually-approved)
  - **Native Hawaiian or Other Pacific Islander: 1.193×**
  - **American Indian or Alaska Native: 1.14×**
  - **Asian: 0.72×** (under-represented among FNs)
  - **White: ≈1.0×** (baseline)
- **Small-sample caveats:** several race categories are small-n (American Indian/Alaska Native n=511, Native Hawaiian/PI n=149, 2+ minority races n=162) — their point estimates carry wide uncertainty and should not be treated as headlines.

### Step 8 SHAP proxy tie-back
- `tract_minority_population_percent` ranks #7 by SHAP but only #50 by XGBoost gain.
- The group with the most adverse proxy SHAP (Asian, highest mean SHAP contribution) does **not** show the worst FNR — Asian is actually under-represented among false negatives (0.72×). The evidence is descriptive, not causal.

### 0.5 vs. 0.858 threshold verdict
- The precision-weighted business threshold **helped shrink race FNR disparities** relative to the default 0.5 threshold (gap halved from 0.0125 to 0.0064).
- But it **widened ethnicity and sex FNR disparities** (Hispanic gap grew 0.0012 → 0.0057; sex gap grew 0.0012 → 0.0028).
- This is an important finding: the choice of operating threshold is not neutral with respect to subgroup fairness — the same threshold that helps one dimension can hurt another.

### X_test confirmatory comparison
- Disparities present at X_test at similar magnitudes to X_val:
  - `derived_race` @0.858: val mean FNR 0.2625, test mean FNR 0.3011.
  - `derived_ethnicity` @0.858: val 0.2666, test 0.2812.
  - `derived_sex` @0.858: val 0.2652, test 0.3197.
- Agreement across an independent split strengthens the finding; no further iteration based on this result.

### Figures (`figures/Step9/`)
- Calibration-by-group diagrams (by `derived_race` and `derived_sex`).
- SHAP-by-race proxy tie-back diagram.
- Demographic-specific distribution plots.

### Limitations (stated explicitly)
1. This step **measures and reports** disparities — it does not implement a fairness-constrained optimizer or retrain the model to close any gaps found. This is a deliberate, stated limitation of the project, not an oversight.
2. The original Step 3 plan (regression-adjusted approval gaps, significance testing on the raw historical decisions) was **never executed** — Step 9 reports the *trained model's* subgroup error rates, which is a related but different question from whether the raw historical decisions were disparate after controls.
3. Several race/ethnicity categories are small-n (n < 500) — point estimates carry wide uncertainty and should not be treated as headlines.

---

## 12. Step 10 — Final Demo Notebook (`notebooks/home_mortage_application_approval_classifier.ipynb`)

### What we did
A self-contained, end-to-end demonstration notebook that loads the Optuna-tuned XGBoost and walks through the full pipeline on 3 representative test applications.

- **Step 0 — Setup & pre-flight:** imports, paths, the column-order alignment guard (asserting the numpy array column order exactly matches `Step7_model_features.csv`), loads the tuned model from `artifacts/hmda_tuned_xgboost.pkl`/`.json`, reproduces the Step 8 hold-out metrics (test ROC-AUC=0.8933, PR-AUC=0.9532) as proof the correct artifact was loaded.
- **Step 1 — Select 3 representative test applications** using **calibrated** probabilities:
  - **HIGH_APPROVAL** (index 384459): calibrated P(approve)=0.9832, actual=1
  - **DENIED_LOW** (index 409154): calibrated P(approve)=0.1014, actual=0
  - **BORDERLINE** (index 658691): calibrated P(approve)=0.8580, actual=1
- **Step 2 — Uncalibrated model metrics + per-application display:** test @0.5 ROC-AUC=0.8933, PR-AUC=0.9532; confusion matrix @0.858 (uncalibrated): TN=407,337, FP=11,668, FN=807,112, TP=426,880 (TPR=0.3459, FPR=0.0278, FNR=0.6541). Per-application summary with calibrated probabilities, denial-risk tiers, and actual labels.
- **Step 3 — Build, persist & reload the calibrated model:** Platt scaler fit on X_val only, persisted to `artifacts/calibrated_model.pkl`, reloaded and verified (coef match: True). Demonstrates that the calibrator survives a save/reload cycle (~16 min retraining saved).
- **Step 4 — Compare calibrated vs uncalibrated:** full test-set metrics under the loaded calibrated model, side-by-side raw vs calibrated for the 3 selected applications, calibration curve (reliability diagram).

**Calibrated model at THRESH=0.858 (loaded):** TPR=0.7360, FPR=0.1459, FNR=0.2640, P=0.9370, R=0.7360, F1=0.8242.

- **Step 5 — SHAP global interpretability:** TreeExplainer on raw model, 50k explanation sample from X_val. Beeswarm + bar plots. Top features consistent with Step 8: loan_purpose_1, debt_to_income_ratio_missing, loan_to_value_ratio, income, property_value.
- **Step 6 — SHAP local interpretability:** waterfall plots for each of the 3 selected applications. **Bug fixed:** the original code indexed the 50k-sample SHAP array (`sv[i]`) with test-set indices (e.g., 384459) — fixed by computing SHAP values directly for the 3 test points (`explainer.shap_values(X_test_3)`) and using `sv_3[idx]` (3 elements) instead of `sv[i]` (50k elements). Also fixed `data=X_explain[i]` → `X_test_3[idx]` in the `shap.Explanation` constructor.
- **Step 7 — Wrap-up summary:** key results written to `markdown/final_demo_summary.md`.

### Figures (`figures/final_demo/`)
- `calibration_comparison.png` — calibrated vs uncalibrated reliability diagrams.
- `shap_beeswarm.png`, `shap_bar.png` — global SHAP.
- `waterfall_HIGH_APPROVAL.png`, `waterfall_DENIED_LOW.png`, `waterfall_BORDERLINE.png` — local explanations.

---

## 13. Key Methodological Decisions

### Data & leakage
| Decision | Rationale |
|---|---|
| Target = approval (not denial) | Positive predictions correspond to the outcome an applicant wants; keeps precision/recall/SHAP intuitive |
| Keep only originated + denied (`action_taken` 1 or 3) | Other codes don't represent a lender approve/deny decision |
| Leakage-first column triage (before statistical filtering) | Difference between a model that looks good in testing and one that works in production |
| Exclude `rate_spread`, `denial_reason_*`, `total_loan_costs`, `origination_charges` | Post-decision fields — not known at application time |
| `tract_minority_population_percent` flagged `is_race_proxy=1`, kept out of model | Well-known redlining-adjacent proxy; retained for fairness audit only |
| Race/sex kept out of model inputs | Ethically and legally wrong to let a lending model learn a demographic signal directly |

### Modeling
| Decision | Rationale |
|---|---|
| `scale_pos_weight=0.3396` fixed (not oversampling) | Class weighting, not naive oversampling, at this scale (12M+ rows) |
| `scale_pos_weight` not tuned (Step 7) | Keeps imbalance-handling identical across all Steps; probabilities directly comparable |
| 20 Optuna trials on 1M subsample | Explicit compute trade-off; reproduces full-data-tuned metrics (val/test ROC-AUC differ by 0.0003) |
| Feature ablation before trusting Step 5's "77 keep / 26 drop" | Don't apply a feature-selection recommendation on faith — test it |
| FEATURES_Step6 = 78 (trimmed) | Ablation gap = 0.0004, within noise |
| Raw model retained for SHAP, calibrated for business output | Platt scaling is monotonic (preserves ranking) but breaks SHAP's additive assumption if used directly |

### Threshold & evaluation
| Decision | Rationale |
|---|---|
| THRESH = 0.858 (Fβ=0.3) | Precision-weighted; a false approval is a principal+interest-at-risk financial risk |
| Default 0.5 threshold retained for comparison | TPR at 0.5 is only ~0.35 — meaning 65% of actually-approved applicants wrongly flagged as high-risk |
| Business threshold **helped** race FNR, **widened** ethnicity/sex FNR | Operating threshold choice is not neutral with respect to subgroup fairness |
| X_test used exactly once (Step 6, Step 7, Step 9) | Standard test-discipline pattern — no re-tuning against test |

### Fairness
| Decision | Rationale |
|---|---|
| Measures and reports disparities; does not optimize | Honest scope boundary; a fairness-constrained optimizer is a different project |
| Evaluated at both 0.5 and 0.858 | Resolves the Step 8 open question; the threshold choice materially affects subgroup disparities |
| FNR reported with same weight as overall accuracy | Per README: FNR is the fair-lending-sensitive metric (a qualified applicant wrongly flagged) |
| Small-sample rule (n < 500 flagged) | Several race categories too small for reliable point estimates |
| Step 3 regression-adjusted analysis never executed | Acknowledged explicitly in limitations; Step 9 reports model error rates, not raw decision disparities |

---

## 14. Limitations & Responsible-Use Statement

1. **Decision-support only.** This system outputs a risk score plus contributing factors, intended to sit in front of a human underwriter — not replace one. It is explicitly not an autonomous approval/rejection system.
2. **Measures, does not correct.** The fairness component measures and reports subgroup disparities — it does not implement a fairness-constrained optimizer or retrain the model to close any gaps found. Any decision to act on these findings is a policy decision, not a modeling one.
3. **Missing Step 3 regression-adjusted analysis.** The original plan for regression-adjusted approval gaps and significance testing on the raw historical decisions was never executed. Step 9 reports the *trained model's* subgroup error rates — a related but different question from whether the raw historical decisions were disparate after controlling for income, DTI, LTV, and credit score.
4. **Small-sample uncertainty.** Several race/ethnicity categories (American Indian/Alaska Native n≈511, Native Hawaiian/PI n≈149, 2+ minority races n≈162) have point estimates with wide confidence intervals and should not be treated as headlines.
5. **Proxy variables.** `tract_minority_population_percent` is a documented geography-adjacent proxy for race. Its SHAP contribution varies by race group, but the SHAP signal does not translate cleanly into measured FNR disparities (Asian has the highest adverse proxy SHAP but is under-represented among false negatives). This is descriptive evidence, not a causal fairness finding.
6. **Calibration is global, not per-group.** Platt scaling is fit once on the full X_val; per-group calibration plots show it holds up reasonably well, but systematic per-group miscalibration could exist that a global fit doesn't capture.
7. **Calibration gap at X_test.** The X_test mean FNR (0.3011 at THRESH=0.858) is higher than the X_val mean FNR (0.2625) — the calibrated model is slightly less well-calibrated on the held-out test set. This is expected and monitored, not a signal to re-tune.
8. **Deployment pending.** The README specifies a lightweight FastAPI/Streamlit inference demo (submit application features → receive approval probability, denial-risk tier, top SHAP factors). This was not yet implemented at the time of the final demo notebook.

---

## 15. Deliverables & Artifacts

### Notebooks
| Notebook | Purpose |
|---|---|
| `notebooks/1_data_load.ipynb` | Ingest 12.2M records, dtype mapping, column triage, save typed parquets |
| `notebooks/2_data_cleaning_fairlending.ipynb` | Outlier cleaning, dedup, missingness strategy, raw fair-lending descriptive pass |
| `notebooks/hmda_eda.ipynb` | Deferred full EDA (A.1–A.9), inline-only plots, 250k sample |
| `notebooks/4_preprocessing_feature_engineering.ipynb` | Sentinel audit, categorical encoding, derived features, train/val/test split, demographics lookups |
| `notebooks/5_baseline_models.ipynb` | LR/DT/RF baselines, stratified clustering, feature-selection recommendation |
| `notebooks/6_stronger_models.ipynb` | XGB/LGBM/CatBoost ablation + untuned baselines, single X_test look |
| `notebooks/7_collab_hyperparameter_tuning.ipynb` | Optuna 20-trial tuning, best params & artifacts |
| `notebooks/8_calibration_shap.ipynb` | Platt calibration, THRESH=0.858, TreeExplainer SHAP (global + local), proxy check |
| `notebooks/9_fairness_analysis.ipynb` | Subgroup TPR/FPR/FNR/precision/recall/F1 by race/ethnicity/sex at both thresholds, disparity gaps, calibration-by-group, FN profiling, X_test confirmatory |
| `notebooks/home_mortage_application_approval_classifier.ipynb` | Final demo: load model, 3 representative test applications, calibrated vs uncalibrated comparison, SHAP global + local |

### Artifacts (`artifacts/`)
- `hmda_tuned_xgboost.pkl` / `.json` — Optuna-tuned XGBClassifier (max_depth=13)
- `calibrated_model.pkl` — Platt (logistic) scaler, survives save/reload
- `Step7_best_xgb_params.json`, `Step7_tuned_metrics.csv`, `Step7_optuna_progress.csv`

### Figures (`figures/`)
- `figures/Step5/roc_overlay.png`, `pr_overlay.png`
- `figures/Step6/roc_overlay.png`, `pr_overlay.png`
- `figures/Step7/roc_tuned.png`, `pr_tuned.png`
- `figures/Step8/calibration_raw.png`, `calibration_calibrated.png`, `confusion_matrix.png`, `shap_beeswarm.png`, `shap_bar.png`, `proxy_shap_by_race.png`, `waterfall_7184.png`, `waterfall_1207.png`, `waterfall_1826.png`
- `figures/Step9/` — calibration-by-group, proxy tie-back, demographic distribution plots
- `figures/final_demo/` — `calibration_comparison.png`, `shap_beeswarm.png`, `shap_bar.png`, `waterfall_HIGH_APPROVAL.png`, `waterfall_DENIED_LOW.png`, `waterfall_BORDERLINE.png`

### Data (`data/processed/`)
- `hmda_2024_typed.parquet`, `hmda_2024_decisions.parquet`, `hmda_2024_clean.parquet`, `hmda_2024_cleaned_modelling.parquet`
- `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test` parquets
- `val_demographics_lookup.parquet`, `test_demographics_lookup.parquet`
- `Step7_model_features.csv` (78 canonical features, with `is_race_proxy` flag)
- `1_column_dictionary.csv`

### Markdown summaries
- `markdown/Step1_hmda_project_plan.md` through `markdown/Step9_hmda_project_plan.md` — daily planning documents
- `markdown/Step5_baseline_summary.md`, `markdown/Step6_stronger_models_summary.md`, `markdown/Step8_calibration_shap_summary.md`, `markdown/Step9_fairness_summary.md` — daily result summaries
- `markdown/_tuning_summary.md` — Step 7 tuning summary
- `markdown/final_demo_summary.md` — final demo wrap-up
- **`markdown/project_summary.md`** — this document

### Reproduced headline metrics

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.8933 |
| Test PR-AUC | 0.9532 |
| Test Precision @0.5 | 0.9143 |
| Test Recall @0.5 | 0.8571 |
| Test F1 @0.5 | 0.8848 |
| THRESH (Fβ=0.3) | 0.8582 |
| Test TPR @0.858 | 0.7360 |
| Test FPR @0.858 | 0.1459 |
| Test FNR @0.858 | 0.2640 |
| FN population (val+test) @0.858 | 457,066 |
| `scale_pos_weight` | 0.3396 |
| Feature count | 78 (canonical) |
| Race-proxy feature | `tract_minority_population_percent` (`is_race_proxy=1`) |
| Calibrated model artifact | `artifacts/calibrated_model.pkl` |
| Calibrated model coef match on reload | True |

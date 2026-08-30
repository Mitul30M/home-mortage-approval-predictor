# Project Proposal: Large-Scale Mortgage Approval Prediction & Fair-Lending Analytics

**Domain:** Finance (Credit Risk / Lending Analytics)
**Type:** Traditional ML — large-scale structured data, binary classification
**Duration:** 10 days (4–5 hrs/day)
**Dataset:** 2024 HMDA National Loan-Level Dataset (~12M+ real records)

---

## 1. Problem statement

A lending institution receives thousands of mortgage applications daily and has accumulated millions of historical records. Reviewing every application manually at scale is inefficient, and human underwriting decisions can be inconsistent across reviewers.

This project asks two questions simultaneously:

1. **Predictive:** Based only on information available at the time of application, how likely is a given mortgage application to be approved?
2. **Responsible-AI:** Does a model that predicts this well also behave *consistently* across different applicant demographic groups, or does it introduce systematic disparities?

Answering only the first question produces a good classifier. Answering both produces a decision-support system a real financial institution could plausibly discuss with a risk, compliance, or fair-lending team.

---

## 2. Objective

> Build a machine-learning decision-support system that estimates the probability of mortgage approval from pre-decision applicant, loan, and property information — and evaluate whether that model's error rates and calibration are consistent across race, ethnicity, and sex subgroups.

This is explicitly **not** an autonomous approval/rejection system. The output is a risk score plus contributing factors, intended to sit in front of a human underwriter — not replace one.

---

## 3. Dataset

**Source:** [2024 HMDA Snapshot National Loan-Level Dataset](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/) — published by the FFIEC/CFPB under the Home Mortgage Disclosure Act.

- **Scale:** ~12 million+ real loan-level application records from thousands of reporting institutions across the U.S.
- **Genuineness:** This is real regulatory reporting data, not a synthetic or Kaggle-competition dataset — every row corresponds to an actual mortgage application filed with a real institution.
- **Richness:** ~100 fields per record, including applicant financial characteristics, loan terms, property details, underwriting-adjacent variables, the final action taken, and derived demographic fields (race, ethnicity, sex) — which is what makes the fairness component possible.
- **Documentation:** [LAR Data Fields reference](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields/) — required reading before feature engineering, since most fields are coded rather than plain text.

---

## 4. Target variable

Derived from the `action_taken` field. The modeling target is `approved`, with **approval as the positive class (1)**:

| `action_taken` code | Meaning | `approved` (target) |
|---|---|---|
| `1` | Loan originated | `1` |
| `3` | Application denied | `0` |
| all other codes (withdrawn, incomplete, purchased loan, preapproval-only, etc.) | No clean approve/deny decision was made | Excluded from the modeling set |

```python
df_model["approved"] = (df["action_taken"] == "1").astype("int8")
df_model_decisions["approved"] = df_decisions["approved"]
```

This filtering decision is documented explicitly in the notebook, since it directly shapes what the model can and cannot claim to predict. Framing the target as *approval* rather than *denial* is a deliberate choice, not just a sign flip — it means the model's positive predictions correspond to the outcome an applicant wants, which keeps precision/recall/SHAP outputs intuitive to read (e.g. "predicted probability of approval," not a double-negative "predicted probability of not-being-denied").

---

## 5. The ML problem, formally

Given applicant, loan, and property features **known at the time of application**:

```
X = Applicant + Loan + Property + Financial Features (pre-decision only)
```

Predict:

```
P(Y = Approved | X)
```

The output is a **probability of approval**, not a hard label — e.g. `0.784` means a 78.4% predicted chance of approval. For underwriting/risk-review workflows, this is more naturally read as a **denial-risk score**, which is just the complement:

```
denial_risk = 1 − P(Y = Approved | X)
```

So the institution can set its own risk-tier thresholds off `denial_risk` (illustrative, not prescriptive):

```
0–20%    Low predicted denial risk    (high predicted approval probability)
20–50%   Moderate
50–75%   High
75–100%  Very high                    (low predicted approval probability)
```

---

## 6. Why this matters (real stakeholders)

| Stakeholder | How they'd use it |
|---|---|
| **Underwriters** | A quantitative signal ("this application has elevated predicted denial risk") to prioritize deeper manual review |
| **Risk & analytics teams** | Portfolio-level insight into which applicant/loan characteristics associate with denial, and how that shifts over time |
| **Compliance / fair-lending teams** | A monitoring framework for subgroup error-rate and calibration disparities — the analytical capability such a team would rely on |
| **ML/FinTech hiring teams** | Evidence of large-scale tabular processing, leakage-aware feature engineering, imbalanced classification, model comparison, calibration, explainability, and fairness evaluation — the full traditional-ML skill set in one project |

---

## 7. Methodology

### 7.1 Leakage-aware feature engineering
Not every column in the historical dataset would have been known at decision time. Before any modeling, every field is checked against the codebook and classified as **pre-decision** (usable) or **post-decision / outcome-derived** (excluded). This decision is documented column-by-column — it's the difference between a model that looks good in testing and one that would actually work in production.

### 7.2 Data preparation at scale
- Convert raw CSV → Parquet for efficient repeated loading (Colab-friendly)
- Handle HMDA's sentinel/"exempt" codes explicitly — not treated as ordinary missing values
- Encode categoricals (target/frequency encoding for high-cardinality fields, one-hot for low-cardinality)
- Engineer derived features: loan-to-income ratio, loan-to-property-value ratio, rate-spread bucket, applicant age bucket

### 7.3 Model comparison
Progressive comparison rather than jumping straight to the most complex option:

```
Logistic Regression → Random Forest → XGBoost / LightGBM → CatBoost
```

- Class imbalance handled via class weighting (`scale_pos_weight`), not naive oversampling at this scale
- Hyperparameter tuning via Optuna, searched on a stratified subsample, validated on the full held-out test set (an explicit, documented compute trade-off)

### 7.4 Evaluation
- ROC-AUC, PR-AUC (more informative given imbalance)
- Precision / Recall / F1
- Calibration curve
- Confusion matrix at a business-relevant operating threshold

### 7.5 Explainability
SHAP-based feature attribution on a sampled explanation set, producing per-application output such as:

```
Predicted probability of approval: 21.6%
Predicted denial risk: 78.4% (High)

Top contributing factors (pushing toward denial):
• High loan-to-income ratio
• High loan amount relative to property value
• Loan characteristics associated with elevated historical denial rates
```

### 7.6 Fairness evaluation
Subgroup analysis across `derived_race`, `derived_ethnicity`, and `derived_sex`:

```
Overall model performance
        ↓
Group-level performance (TPR, FPR, precision, recall)
        ↓
Error-rate comparison across groups
        ↓
Calibration comparison across groups
        ↓
Investigate and report significant disparities
```

**Positive-class note:** since `approved` (not `denied`) is the positive class, standard metric names need care when discussing them with a fair-lending audience:
- **True Positive Rate (TPR)** = share of *actually-approved* applicants the model correctly predicts as approved.
- **False Negative Rate (FNR)** = share of *actually-approved* applicants the model incorrectly predicts as denied — this is the metric of greatest fair-lending concern (a qualified applicant wrongly flagged as high-risk), and it should be reported by subgroup with the same weight as overall accuracy, not as a secondary metric.
- **False Positive Rate (FPR)** = share of *actually-denied* applicants the model incorrectly predicts as approved — an over-approval error, not an equity harm in the traditional sense, but still worth monitoring for portfolio risk.

**Scope note:** this project *measures and reports* subgroup disparities — it does not implement a fairness-constrained optimizer. That's a deliberately honest scope boundary, and stating it explicitly is part of the deliverable.

### 7.7 Deployment
A lightweight FastAPI/Streamlit inference demo: submit application features → receive approval probability, derived denial-risk tier, and top SHAP-based contributing factors. Framed explicitly as a decision-support layer, not an autonomous approval system.

---

## 8. Deliverables

- Cleaned, leakage-audited feature pipeline (documented column-inclusion/exclusion rationale)
- Trained and tuned classification model with full evaluation report
- SHAP-based explainability outputs
- Subgroup fairness report (metrics + written discussion of limitations)
- Simple inference demo (API or Streamlit app)
- README covering: problem framing, data source, leakage handling, modeling decisions, fairness findings, and a "limitations and responsible use" statement

---

## 9. Timeline (10 days, 4–5 hrs/day)

| Day | Focus |
|---|---|
| 1 | Ingest data (CSV → Parquet), orient on schema, define target |
| 2 | Leakage audit — build pre-decision feature set |
| 3 | EDA at scale (missingness, class balance, distributions, group sizes) |
| 4 | Preprocessing & feature engineering |
| 5 | Baseline models (Logistic Regression, Decision Tree) |
| 6 | Stronger models (LightGBM / CatBoost) |
| 7 | Hyperparameter tuning (Optuna, subsample strategy) |
| 8 | Evaluation & SHAP interpretability |
| 9 | Fairness analysis across demographic groups |
| 10 | Deployment demo + documentation polish |

---

---

## Day 9 Progress (deferred EDA + fairness analysis)

**Status (as of Day 9 build):** both notebooks built, syntax-validated, and core pipeline smoke-tested.

### Deliverables produced
- `notebooks/hmda_eda.ipynb` — deferred full EDA on `data/processed/hmda_2024_clean.parquet` (pre-split, pre-imputation, 8.26M rows). Sections A.1–A.9. All plots inline, **none saved to `figures/`**. 39 cells (28 code / 11 md). Recomputes `loan_to_income_ratio` inline; uses a 250k reproducible sample for plots and full file for stats.
- `notebooks/9_fairness_analysis.ipynb` — Day 9 fairness milestone. Self-contained (reloads model, recomputes Platt calibration + the 0.858 business threshold from scratch). Steps 0–7 covering subgroup TPR/FPR/FNR/precision/recall/F1 by `derived_race`, `derived_ethnicity`, `derived_sex` at both 0.5 and 0.858; disparity-gap quantification; 0.5-vs-0.858 verdict; calibration-by-group diagrams; Day 8 SHAP proxy tie-back; false-negative population profiling; one confirmatory X_test look; wrap-up + limitations + open questions for Day 10. 20 cells (9 code / 11 md). Writes `markdown/day9_fairness_summary.md`.
- `figures/day9/` — created for fairness figures (written at runtime).
- Builder scripts: `/tmp/opencode/build_eda.py`, `/tmp/opencode/build_fairness.py` (reproducible notebook generation).

### Key methodological decisions captured
- **Choice A (recommended):** only `artifacts/hmda_tuned_xgboost.pkl`/`.json` persisted; `TreeExplainer` and the Platt `LogisticRegression` calibrator are reconstructed live each run — no separate SHAP or calibrator artifact, deterministic and shap-version-stable.
- **Day 2 exclusion rule** reapplied for the primary comparison subset: exclude `Joint`, `*_Not Available`, `Free Form Text Only` per demographic dimension (sizes reported separately).
- **Small-sample rule:** flag any group with n < 500 in the evaluation subset.
- **Protocol:** X_val for Steps 1–5 (investigative); X_test gets exactly one confirmatory look (Step 6).
- **Scope honored:** measures and reports disparities; does not implement a fairness-constrained optimizer or retrain the model (README boundary). The original Day 3 regression-adjusted analysis was never executed; Day 9 reports model error rates (a related but different question).

### Validation results (smoke-tested)
- Reproduced val ROC-AUC=0.8930 / PR-AUC=0.9531; test ROC-AUC=0.8933.
- Business threshold (X_val Fbeta=0.3) = 0.8582; FN population (val+test) at threshold = 457,066.
- Race FNR disparity *shrank* at the 0.858 threshold vs 0.5 (helped); ethnicity and sex FNR disparity *widened*.
- False-negative population over-representation: Black (1.28×), Native Hawaiian/PI (1.19×), American Indian/Alaska Native (1.14×); Asian under-represented (0.72×); White ≈ baseline (1.0×).
- EDA notebook: data loads, 250k sample, approved share 0.7465, 16 continuous + 17 categorical + 5 pie-low fields.

### Remaining
- Run the full notebooks end-to-end in Jupyter (heavy: 8M-row EDA plots and 50k-row SHAP) to generate all figures and the summary markdown.
- Day 10: deployment demo (FastAPI/Streamlit) + documentation polish + consider Day-3 regression-adjusted analysis.

### Day 9 notebook bug fixes (post-build validation)
Two issues found and fixed in `notebooks/hmda_eda.ipynb` (and the same root cause in `notebooks/9_fairness_analysis.ipynb`):

1. **Invisible plots** — both notebooks imported `matplotlib` with `matplotlib.use('Agg')`, a non-interactive backend, and lacked `%matplotlib inline`. With Agg active, `plt.show()` renders to an off-screen buffer the notebook never captures, so all plot cells showed `outputs: []`. Fixed by replacing the `matplotlib.use('Agg')` line with `%matplotlib inline` as the first executable line (before `import matplotlib.pyplot`).

2. **`sns.heatmap` crash** (`notebooks/hmda_eda.ipynb` only) — three heatmap calls passed an invalid `xrotation=45` kwarg that the installed seaborn version does not support, producing `AttributeError: QuadMesh.set() got an unexpected keyword argument 'xrotation'`. Fixed by removing `xrotation=45` and rotating the x-tick labels manually after each call: `ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')`.

Both notebooks re-validated after the fixes: 0 syntax errors, 0 leftover bad patterns; heatmap cells confirmed to render without error.

### Day 9 runtime bugs fixed (post-build validation, in `notebooks/9_fairness_analysis.ipynb`)
Two IndexError/KeyError bugs found when the notebook executes end-to-end:

1. **`pc_test_p` IndexError (calibration cell):** `pc_test_p = {dim: pc_test[primary_mask(val_demo, dim)]}` indexed the 1.65M-element `pc_test` array with a 661k-element `val_demo` mask → `IndexError: boolean index did not match`. Fixed: `primary_mask(test_demo, dim)` so the mask length matches `pc_test`.

2. **Step 6 `UndefinedVariableError`:** the X_test confirmatory comparison called `.query('metric=="FNR"')` on `RES` and `TEST` DataFrames — but those come from `subgroup_metrics`, which has columns `group, n, TPR, FPR, FNR, precision, recall, F1, small` (no `metric` column). Only the `compute_gaps`-produced `GAPS` DataFrames have a `metric` column. Fixed: replaced `.query('metric=="FNR"')['FNR']` with direct `['FNR']` column access.

Both notebooks now validated end-to-end: 0 syntax errors, 0 leftover bad patterns.

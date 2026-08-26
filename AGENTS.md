# Project Proposal: Large-Scale Mortgage Denial Risk Prediction & Fair-Lending Analytics

**Domain:** Finance (Credit Risk / Lending Analytics)
**Type:** Traditional ML — large-scale structured data, binary classification
**Duration:** 10 days (4–5 hrs/day)
**Dataset:** 2024 HMDA National Loan-Level Dataset (~12M+ real records)

---

## 1. Problem statement

A lending institution receives thousands of mortgage applications daily and has accumulated millions of historical records. Reviewing every application manually at scale is inefficient, and human underwriting decisions can be inconsistent across reviewers.

This project asks two questions simultaneously:

1. **Predictive:** Based only on information available at the time of application, how likely is a given mortgage application to be denied?
2. **Responsible-AI:** Does a model that predicts this well also behave *consistently* across different applicant demographic groups, or does it introduce systematic disparities?

Answering only the first question produces a good classifier. Answering both produces a decision-support system a real financial institution could plausibly discuss with a risk, compliance, or fair-lending team.

---

## 2. Objective

> Build a machine-learning decision-support system that estimates the probability of mortgage denial from pre-decision applicant, loan, and property information — and evaluate whether that model's error rates and calibration are consistent across race, ethnicity, and sex subgroups.

This is explicitly **not** an autonomous approval/rejection system. The output is a risk score plus contributing factors, intended to sit in front of a human underwriter — not replace one.

---

## 3. Dataset

**Source:** [2024 HMDA Snapshot National Loan-Level Dataset](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/) — published by the FFIEC/CFPB under the Home Mortgage Disclosure Act.

- **Scale:** ~12 million+ real loan-level application records from thousands of reporting institutions across the U.S.
- **Genuineness:** This is real regulatory reporting data, not a synthetic or Kaggle-competition dataset — every row corresponds to an actual mortgage application filed with a real institution.
- **Richness:** ~100 fields per record, including applicant financial characteristics, loan terms, property details, underwriting-adjacent variables, the final action taken, and derived demographic fields (race, ethnicity, sex) — which is what makes the fairness component possible.
- **Documentation:** [LAR Data Fields reference](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields/) — required reading before feature engineering, since most fields are coded rather than plain text.

**NOTE: REFER TO /resources/HMDA_LAR_Data_Fields_Official_Doc.pdf**
---

## 4. Target variable

Derived from the `action_taken` field:

| Value | Meaning |
|---|---|
| `1` | Denied → target = 1 |
| `0` | Originated / approved → target = 0 |
| other codes (withdrawn, incomplete, purchased loan, etc.) | Excluded from the modeling set — these don't represent a clean approve/deny outcome |

This filtering decision is documented explicitly in the notebook, since it directly shapes what the model can and cannot claim to predict.

---

## 5. The ML problem, formally

Given applicant, loan, and property features **known at the time of application**:

```
X = Applicant + Loan + Property + Financial Features (pre-decision only)
```

Predict:

```
P(Y = Denied | X)
```

The output is a **probability**, not a hard label — e.g. `0.784` — so the institution can set its own risk-tier thresholds (illustrative, not prescriptive):

```
0–20%    Low predicted denial risk
20–50%   Moderate
50–75%   High
75–100%  Very high
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
Predicted probability of denial: 78.4%
Risk level: High

Top contributing factors:
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

**Scope note:** this project *measures and reports* subgroup disparities — it does not implement a fairness-constrained optimizer. That's a deliberately honest scope boundary, and stating it explicitly is part of the deliverable.

### 7.7 Deployment
A lightweight FastAPI/Streamlit inference demo: submit application features → receive denial probability, risk tier, and top SHAP-based contributing factors. Framed explicitly as a decision-support layer, not an autonomous approval system.

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

## 10. One-line interview summary

> "I built a large-scale mortgage decision-support model using over 12 million real 2024 U.S. mortgage application records. The goal was to predict the probability of mortgage denial using only information available at decision time, while evaluating predictive performance, calibration, and subgroup fairness. I compared several traditional ML models and deployed the best-performing one as an inference service."

**Deeper conversation tree available on request:** Why HMDA? → Target definition → Leakage → Feature engineering → Class imbalance → Model selection → SHAP → Fairness → Deployment.

---

## 11. Limitations & responsible use

- This model is a **decision-support tool**, not an autonomous underwriting system. Final lending decisions remain with human underwriters.
- Subgroup performance differences, if found, are **reported and discussed**, not corrected via constrained optimization — that scope boundary is intentional given the project timeline.
- Historical lending data can reflect historical patterns and biases; a model trained on it may reproduce those patterns unless explicitly audited, which is precisely why the fairness-evaluation component exists.

# HMDA 2024 Mortgage Lending Project — Day 1

**Dataset:** HMDA LAR 2024, 12,229,298 records × 99 columns (loaded from Drive)
**Project goals (across all days):**
1. Fair lending / disparity analysis (race, ethnicity, sex, denial reasons)
2. Predictive modeling (approve vs. deny classification)
3. General EDA / dashboard

**Day 1 focus:** Get the data into a clean, memory-manageable, well-understood state so Days 2+ (EDA deep-dive, disparity analysis, modeling) can move fast. No modeling or deep disparity work today — this is foundation day.

---

## Step 1 — Environment & Efficient Loading

The raw file is huge (12.2M rows × 99 cols, all loaded as strings). Loading naively will blow up RAM in Colab.

**Achievable:**
- Load with `dtype` mapping instead of default (string) inference — most columns are actually categorical codes or numeric.
- Read only the columns you'll plausibly need across all 3 goals (see Step 2) rather than all 99, OR load in chunks and immediately downcast.
- Convert numeric-looking fields (`loan_amount`, `income`, `interest_rate`, `property_value`, `loan_to_value_ratio`, `tract_*` fields) to `float32`/`int32`.
- Convert low-cardinality string fields (`state_code`, `derived_race`, `derived_sex`, `action_taken`, `loan_type`, etc.) to `category` dtype.
- Save the reduced/typed dataframe to **Parquet** (not CSV) so Day 2+ notebooks load in seconds instead of minutes.

**Deliverable:** `hmda_2024_reduced.parquet` saved to Drive.

---

## Step 2 — Column Triage (map columns to project goals)

With 99 columns, decide upfront which you actually need. Roughly:

| Purpose | Key columns |
|---|---|
| Outcome / target | `action_taken` (1=originated, 3=denied, others=withdrawn/incomplete/purchased) |
| Fair lending | `derived_race`, `derived_ethnicity`, `derived_sex`, `applicant_age`, `denial_reason_1-4`, `income`, `debt_to_income_ratio` |
| Loan characteristics | `loan_amount`, `loan_purpose`, `loan_type`, `lien_status`, `interest_rate`, `rate_spread`, `property_value`, `loan_to_value_ratio`, `loan_term` |
| Geography | `state_code`, `county_code`, `census_tract`, `derived_msa_md` |
| Tract context (proxy variables) | `tract_minority_population_percent`, `tract_to_msa_income_percentage`, `ffiec_msa_md_median_family_income` |
| Institution | `lei` |

**Achievable:** Produce a short data dictionary (markdown or CSV) listing the ~25–30 columns you're keeping, their dtype, and which of the 3 project goals they serve. Drop or archive the rest for now (you can always re-pull from raw later).

---

## Step 3 — Understand the Target Variable (`action_taken`)

This drives both the disparity analysis and the future model.

**Achievable:**
- Print value counts for `action_taken` and map codes to labels (1=Originated, 2=Approved not accepted, 3=Denied, 4=Withdrawn, 5=Closed for incompleteness, 6=Purchased loan, 7/8=Preapproval outcomes).
- Decide the analysis population: for fair lending + modeling, the standard approach is to keep only **originated (1)** and **denied (3)** records, and set those aside as `action_taken_binary` (1=approved, 0=denied). Document *why* you're excluding withdrawn/incomplete/purchased (they don't represent a lender approve/deny decision).
- Report what % of rows fall into each bucket before you filter — this is a real EDA finding worth keeping.

**Deliverable:** A short note (in-notebook markdown cell) stating the filtering rule you'll apply from Day 2 onward, plus the resulting row count.

---

## Step 4 — Data Quality Audit

**Achievable:**
- Missingness: `% null` per kept column (HMDA often uses sentinel codes like `1111`, `9999`, `Exempt` instead of true NaN — check for these in `loan_term`, `income`, `debt_to_income_ratio`, `combined_loan_to_value_ratio`, etc., and note them explicitly rather than treating them as real values).
- Duplicates: check for exact duplicate rows.
- Sanity-check numeric ranges: `loan_amount`, `income`, `interest_rate`, `property_value` — flag obvious outliers/placeholder values (e.g., income of 0 or negative, interest rate of 0 on originated loans).
- Check cardinality of `derived_race`, `derived_ethnicity`, `derived_sex` — note "Not Available"/"Not Applicable" categories since these matter a lot for fair-lending work later.

**Deliverable:** A data quality summary table (columns × %missing × notes) kept in the notebook.

---

## Step 5 — Light Initial EDA (descriptive only, no disparity claims yet)

Keep this high-level — save the deep disparity comparisons for Day 2.

**Achievable, pick a handful:**
- Distribution of `action_taken` (bar chart).
- Distribution of `loan_amount` and `income` (histograms, probably log-scale given skew).
- Row counts by `state_code` (top 10 states).
- Row counts by `derived_race`, `derived_sex` (just counts, not yet crossed with outcome).
- Note dataset time scope (`activity_year` — likely all 2024, confirm).

**Deliverable:** 4–6 simple charts + one paragraph of observations in the notebook.

---

## Step 6 — Wrap-up

- Save the cleaned/reduced Parquet file + the column dictionary + the filtering-rule note to Drive so Day 2's notebook can load them directly without re-touching the raw 12M-row file.
- Write a short "Day 1 summary" markdown cell: final row count, final column count, key open questions for Day 2 (e.g., "how do denial rates vary by race once we control for income/loan amount?").

---

## End-of-Day-1 Checklist

- [ ] Raw data loaded with proper dtypes (not all-string)
- [ ] Reduced Parquet file saved to Drive
- [ ] Column dictionary documented (~25–30 kept columns)
- [ ] `action_taken` understood, binary target rule decided and documented
- [ ] Missingness / sentinel-value audit done
- [ ] Duplicate check done
- [ ] 4–6 basic EDA charts produced
- [ ] Day 1 summary + open questions for Day 2 written

**Not for Day 1 (save for later days):** approval-rate-by-race crosstabs, statistical disparity testing, feature engineering for modeling, train/test split, any model training.

Feature selection isn't one method, it's a few filters applied in sequence. Here's how I'd think about it for your specific case (predicting approve/deny + doing fair lending analysis):

**1. Start from purpose, not statistics**

You have two different feature sets, not one:
- **Modeling features** — used as model inputs to predict approval.
- **Fairness-audit features** — race, ethnicity, sex, age — used to *measure* disparities, but generally **excluded** as direct model inputs (including them lets the model learn a race/sex signal directly, which is both ethically and legally the wrong move for a lending model). You still keep them in the dataset — just not fed to the classifier.

**2. Kill leakage first (before anything statistical)**

This matters more than any selection algorithm. A few HMDA-specific traps:
- `rate_spread`, `denial_reason_*`, `total_loan_costs`, `origination_charges` are often only populated *after* a decision is made — using them to predict `action_taken` is leaking the answer.
- `purchaser_type` and `action_taken` itself are obviously off-limits.
- Ask "would this value exist yet, at the moment the lender is deciding?" If no, drop it.

**3. Domain/regulatory logic next**

HMDA has known "legitimate underwriting" variables that regulators and researchers typically use: `loan_amount`, `income`, `debt_to_income_ratio`, `loan_to_value_ratio` (or `combined-loan-to-value`), `property_value`, `loan_term`, `loan_purpose`, `lien_status`, occupancy type, `applicant_credit_score_type`. These map to real underwriting logic, so they belong regardless of what stats say.

**4. Then apply statistical/quantitative filters**

Once leakage and domain junk are removed, narrow further with:
- **Missingness threshold** — drop anything with, say, >40–50% missing unless it's important enough to impute carefully.
- **Variance/cardinality check** — near-constant columns or IDs with huge cardinality (`census_tract`, `lei`) aren't useful as raw model features; use them for grouping/joins instead, or engineer aggregates from them (e.g., tract-level minority % is fine, tract ID itself isn't).
- **Correlation / multicollinearity** — check correlation matrix or VIF among numeric features (e.g., `loan_amount` vs `property_value` vs `loan_to_value_ratio` are likely redundant — LTV is often derived from the other two).
- **Mutual information / chi-square** against the target for categorical features, to get a quick signal ranking before modeling.

**5. Model-based selection (once you're actually modeling, Day 3+)**

- Fit a baseline model (logistic regression or gradient boosting) with the domain-justified set, look at feature importances / SHAP values / coefficients.
- Use L1 (Lasso) regularization or recursive feature elimination to see what survives.
- Don't let the model choose features you can't justify to a human — for a fair-lending-adjacent project, interpretability matters more than squeezing out 0.5% accuracy.

**6. Sanity-check against proxy discrimination**

Even after excluding race/sex directly, watch for close **proxies**: `tract_minority_population_percent`, zip/county-level geography, or `census_tract` can act as a proxy for race. If you include them, flag it explicitly and test whether they're driving disparate impact — this is a well-known issue in fair lending modeling (redlining-adjacent).

If it's useful, I can turn this into a concrete feature list (keep/drop/engineer) against your actual 99 columns for Day 1's column dictionary — just say the word.
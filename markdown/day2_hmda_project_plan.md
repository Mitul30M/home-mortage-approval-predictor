# HMDA 2024 Mortgage Lending Project — Day 2

**Inputs from Day 1:** `hmda_2024_typed.parquet` (12,229,298 × 63), `hmda_2024_decisions.parquet` (8,276,018 × 64, 74.63% approval rate), `1_column_dictionary.csv`

**Day 2 focus:** Day 1's own EDA surfaced real data-quality problems that have to be fixed *before* any disparity or modeling work is trustworthy. So Day 2 is split in two halves: (A) clean up what Day 1 found broken, and (B) do the first real fair-lending descriptive pass (approval rates & denial reasons by race/ethnicity/sex) — still no regression/statistical testing, that's Day 3.

---

## What Day 1's output actually told us (starting point for today)

- `income` has a **minimum of -155,317** (in thousands of $) and `loan_to_value_ratio` has a **mean of 2,741 with std ~4.36M** — both are impossible for real values (income shouldn't be negative; LTV is a percentage, realistically 0–200). The loan_amount and income histograms from Day 1 are unreadable — a few extreme outliers compressed the whole chart into one bar.
- 29,384 exact duplicate rows (0.24%) — not yet investigated or resolved.
- Missingness is very uneven: `denial_reason_2/3/4` are 96–99.9% missing (expected — most loans only have one denial reason), but `income` (14.5%), `property_value` (22.4%), `debt_to_income_ratio` (33.2%), `loan_to_value_ratio` (35.3%), and `interest_rate` (36.9%) have real, non-trivial missingness that Day 3 modeling will need a plan for.
- Demographic composition of the decision subset: `derived_race` is 64.2% White, 17.5% Race Not Available, 8.8% Black, 6.1% Asian, plus smaller groups; `derived_sex` is 34.9% Joint, 33.7% Male, 22.5% Female, 8.9% Not Available. These "Not Available" / "Joint" buckets are large enough that we need an explicit rule for how to treat them in comparisons, not just ignore them.

---

## Step 1 — Investigate & Fix the Outlier / Invalid-Value Problems

Don't jump straight to capping — first find out *why* the values are broken.

**Achievable:**
- For `income`, `loan_amount`, `loan_to_value_ratio`, `property_value`, `interest_rate`, `rate_spread`: print `.describe()` at the 1st/5th/95th/99th/99.9th percentiles (not just mean/std) to see where the distribution actually breaks down.
- Check the raw string values behind the worst outliers (`df.loc[df['income'] < 0]`, `df.loc[df['loan_to_value_ratio'] > 500]`, etc.) — is it a handful of extreme rows, or a systemic issue? Are these plausibly real (e.g. commercial-scale loans) or clearly bad data entry?
- Decide and **document** a concrete rule per column, e.g.:
  - `income`: drop or null out negative values; consider capping at a high percentile (e.g. 99.5th) rather than an arbitrary dollar figure.
  - `loan_to_value_ratio`: values should realistically sit in roughly 0–200%; anything wildly outside that is almost certainly a data error, not a real jumbo/exotic loan.
  - `loan_amount` / `property_value`: cap or flag rather than silently drop, since some legitimately large loans exist — the goal is separating "large but real" from "impossible."
- Report how many rows each rule affects (e.g., "capping income at the 99.5th percentile affects X rows, 0.0Y% of the decision subset") — this is the kind of number a reviewer will ask for.
- Re-plot the loan_amount and income histograms after cleaning (log scale or percentile-clipped) to confirm they're now actually readable.

**Deliverable:** a short "data cleaning rules" table (column, rule, rows affected) plus corrected histograms.

---

## Step 2 — Resolve Duplicate Rows

**Achievable:**
- Pull a sample of the 29,384 duplicate rows and inspect them — same LEI + same everything, or could they be legitimate repeated applications?
- Decide: drop all but the first occurrence (standard default) unless something in the sample suggests otherwise.
- Document the decision and the resulting row count after dropping.

---

## Step 3 — Finalize a Missingness Strategy (decisions, not just percentages)

Group the missing columns by *why* they're missing, since the strategy differs:

| Missingness type | Example columns | Approach |
|---|---|---|
| Structurally missing (not an error) | `denial_reason_2/3/4` | Leave as-is — these are only populated when a loan has multiple denial reasons. Missing = "not applicable," not "unknown." |
| Reporting-exemption driven | `total_points_and_fees`, `prepayment_penalty_term`, `intro_rate_period`, `lender_credits`, `discount_points` | Many institutions are exempt from reporting these under HOEPA/Reg Z thresholds — treat "missing" as its own meaningful category rather than something to impute away. |
| Real missing data relevant to modeling | `income`, `property_value`, `debt_to_income_ratio`, `loan_to_value_ratio`, `interest_rate` | These matter for both fair-lending and modeling. For Day 2, just decide and note the plan (e.g., median imputation with a missingness-flag column) — full implementation can wait for Day 3's feature engineering. |

**Achievable:** produce this table filled in with your own findings (which bucket each of the ~30 missing columns falls into), and a one-line plan for each of the "real missing data" columns.

---

## Step 4 — First Fair-Lending Descriptive Pass

Still descriptive only — no regression adjustment or significance testing yet (Day 3).

**Achievable:**
- Decide how to handle `Joint`, `*_Not Available`, and `Free Form Text Only` categories: recommended default is to **exclude them from the primary race/ethnicity/sex comparison table** (report their size separately) so the comparison focuses on single, known categories.
- Compute and chart **approval rate by `derived_race`** and **by `derived_sex`** on the cleaned decision subset (a simple `groupby('derived_race')['approved'].mean()` bar chart).
- Break down **`denial_reason_1`** by `derived_race` — which denial reasons are most common within each group, and do the rankings differ?
- Compare **median `income` and median `loan_amount`** by demographic group — useful context for interpreting the approval-rate gaps, since some of the gap may trace to differences in loan/income profile rather than the decision itself.
- Explicitly label all of this as **raw/unadjusted** in your notebook markdown — the whole point of Day 3 is controlling for income, DTI, LTV, credit score before drawing any conclusion about disparate treatment.

**Deliverable:** approval-rate-by-group chart, denial-reason-by-group table, and a markdown note clearly flagging these as uncontrolled comparisons.

---

## Step 5 — Produce the Modeling-Ready Base File

**Achievable:**
- Apply the Step 1 cleaning rules and Step 2 duplicate resolution to `df_decisions`.
- Save the result as `hmda_2024_clean.parquet` — this becomes the starting point for Day 3's feature engineering and modeling, so it should already be duplicate-free and outlier-handled.
- Keep the raw `hmda_2024_decisions.parquet` from Day 1 untouched as a reference/audit trail.

---

## Step 6 — Wrap-up

Write a short Day 2 summary: final cleaned row count, % of rows affected by cleaning rules, and the raw approval-rate gaps found in Step 4 (numbers only, no causal claims yet). List open questions for Day 3, e.g.:
- Once we control for income, loan amount, DTI, and LTV, do the race/sex approval-rate gaps shrink, disappear, or persist?
- Is `tract_minority_population_percent` correlated with `derived_race` strongly enough to act as a proxy in a model that excludes race directly?
- What imputation approach for `income`/`property_value`/`DTI`/`LTV` best preserves the fair-lending signal without introducing its own bias?

---

## End-of-Day-2 Checklist

- [ ] Outlier/invalid-value investigation done for income, loan_amount, LTV, property_value, interest_rate, rate_spread
- [ ] Cleaning rules decided, documented, and applied (with rows-affected counts)
- [ ] Corrected loan_amount / income histograms produced (readable, not compressed by outliers)
- [ ] Duplicate rows investigated and resolved
- [ ] Missingness strategy table completed (structural vs. exemption-driven vs. real missing)
- [ ] Approval rate by `derived_race` and `derived_sex` computed and charted (raw, unadjusted)
- [ ] Denial reasons broken down by `derived_race`
- [ ] Median income/loan_amount compared by demographic group
- [ ] `hmda_2024_clean.parquet` saved as the Day 3 modeling base
- [ ] Day 2 summary + open questions for Day 3 written

**Not for Day 2 (save for later days):** regression-adjusted disparity testing, statistical significance tests, proxy-discrimination analysis, feature engineering for the model, train/test split, model training.

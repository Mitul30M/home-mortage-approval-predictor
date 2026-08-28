# HMDA 2024 Mortgage Lending Project — Day 4

**Inputs from `2_data_cleaning_fairlending.ipynb`:**
- `hmda_2024_clean.parquet` (8,264,982 × 64) — deduped, outlier-cleaned per Day 2's rules
- `hmda_2024_cleaned_modelling.parquet` — leakage-free feature subset (via `1_column_dictionary.csv`'s `use_in_model == 'YES'`), numeric fields median-imputed with `*_missing` flags added where missingness ≥30%, categoricals filled with `'Missing'`, `conforming_loan_limit` dropped (100% empty)
- Your imputation-strategy doc, explaining the column-by-column decisions behind the above

**Resequencing note:** the README's Day 3 ("EDA at scale") is being deliberately pushed to **after modeling, before Day 9** — not skipped. This plan flags the couple of narrow, unavoidable checks Day 4 still needs (cardinality counts for encoding decisions, a sentinel-code sanity check) so nothing gets silently lost in the reshuffle. Those are scoped tightly to "what feature engineering needs to proceed correctly," not a substitute for the full EDA deliverable.

**Day 4 focus:** turn the cleaned, imputed, leakage-free frame into an actual model-ready matrix — sentinel-code audit, categorical encoding, derived features, and (moved up from Day 5, see Step 4) the train/test split, since categorical encoding needs to be fit train-only to stay leakage-safe.

---

## Step 1 — Sentinel / Exempt Code Audit

HMDA's LAR schema uses sentinel codes (e.g. `1111` = Exempt, `8888` = Not Applicable, `9999` = No Co-Applicant) for several fields. These are *not* `NaN` in the raw data, so Day 2's median imputation would not have touched them — if any slipped through as literal numbers, the median already computed in `hmda_2024_cleaned_modelling.parquet` may be contaminated.

**Achievable:**
- Cross-check the LAR data-fields reference for known sentinel values on your current `MODEL_COLUMNS`, particularly `prepayment_penalty_term`, `intro_rate_period`, `total_points_and_fees`, `applicant_credit_score_type`, `co_applicant_credit_score_type`, `construction_method`, and `total_units`.
- For each, check `value_counts()` on the *pre-imputation* `hmda_2024_clean.parquet` version of the column for suspicious spikes at round sentinel-like numbers (1111, 8888, 9999, etc.) sitting alongside otherwise-continuous values.
- If contamination is found: re-null the sentinel values in that column, re-run median imputation, and re-apply the same ≥30%-missing flag rule Day 2 used — keep the rule identical so the imputation doc stays internally consistent.
- If no contamination is found, document that explicitly (a clean "checked, none found" note is still a deliverable — it's what a reviewer will ask about).

**Deliverable:** a short sentinel-code audit table — `column | sentinel value(s) checked | contamination found? | action taken`.

---

## Step 2 — Encode Categorical Features

**Achievable:**
- Split your current `model_feats` into low-cardinality vs. high-cardinality categoricals and list them explicitly — most of your `MODEL_COLUMNS` (`loan_type`, `loan_purpose`, `lien_status`, `occupancy_type`, `construction_method`, `derived_dwelling_category`, `applicant_credit_score_type`, `co_applicant_credit_score_type`, `submission_of_application`, `reverse_mortgage`, `open_end_line_of_credit`, `business_or_commercial_purpose`, `preapproval`, `negative_amortization`, `interest_only_payment`, `balloon_payment`) are low-cardinality — one-hot encode these.
- Confirm no raw high-cardinality geo ID slipped through the leakage audit (your `MODEL_COLUMNS` note says "no raw geo IDs" — this is the point to actually verify that against the column dictionary, not just assume it).
- If any genuinely high-cardinality field remains, use frequency encoding by default; only reach for target encoding if there's a clear predictive reason, since target encoding is the one method here that **must** be fit train-only (see Step 4) — flag that dependency explicitly wherever it's used.
- Keep the `*_missing` flag columns as-is (already 0/1, no encoding needed).

**Deliverable:** an encoding map table — `column | method | cardinality | notes` — plus a one-line confirmation that no raw geo ID leaked into the final feature set.

---

## Step 3 — Engineer Derived Features

Per the README's feature-engineering scope: loan-to-income ratio, loan-to-property-value ratio, rate-spread bucket, applicant age bucket.

**Achievable:**
- `loan_to_income_ratio = loan_amount / income` — re-verify no zero/negative `income` values remain post-Day-2 cleaning before dividing (should be none, since negatives were nulled in Step 1 of Day 2, but confirm rather than assume).
- `loan_to_value_ratio` already satisfies the loan-to-property-value ratio requirement post-cleaning — no need to re-derive; just note in the feature dictionary that this README item is already covered.
- `rate_spread_bucket` — bin the cleaned `rate_spread` into a small number of ordinal bands (e.g. below-typical / typical / elevated / high); pick cutoffs from the cleaned distribution's own quantiles rather than arbitrary round numbers.
- `applicant_age_bucket` — check which age signal actually exists in your schema (`applicant_age` if present as a real field, vs. only `applicant_age_above_62` as a binary flag) before designing bucket edges — don't assume a continuous age field is available.
- Add each new engineered column as a row in `1_column_dictionary.csv`, matching the pattern already used for the `*_missing` flags (`role`, `use_in_model`).

**Deliverable:** the new engineered columns added to the modeling frame, and their entries in the column dictionary.

---

## Step 4 — Train/Test Split (moved up from Day 5)

Encoding decisions in Step 2 (specifically, any target encoding) are only leakage-safe if fit on a training partition — so the split needs to happen now, not at the start of Day 5.

**Achievable:**
- Stratify the split on `approved` to preserve class balance across train/test.
- Report demographic group sizes (`derived_race`, `derived_ethnicity`, `derived_sex`) in the resulting test split as a sanity check — not a fairness analysis, just confirming no group got split down to an unusably small test count before Day 9 needs it.
- Fit any target/frequency encoders from Step 2 on the training split only, then transform both splits with the fitted encoder.
- Keep `derived_race`, `derived_ethnicity`, `derived_sex` **out of the model feature matrix** but preserved in a separate row-index-aligned lookup table for the test split, so Day 9 can join fairness metrics back without needing to touch the model inputs.

**Deliverable:** `X_train` / `X_test` / `y_train` / `y_test` (or equivalent parquet files) plus a `test_demographics_lookup.parquet` keyed on the same row index.

---

## Step 5 — Final Preprocessing Sanity Checks

**Achievable:**
- Confirm `X_train` and `X_test` have identical columns, dtypes, and encoding — a common silent bug when any encoding step is applied separately per split.
- Re-confirm no outcome-derived or post-decision columns are present in the final feature matrix (final re-check against the Day 1 leakage audit, not new analysis).
- Report final feature count, dtype breakdown, and train/test row counts.

**Deliverable:** a short `preprocessing_summary` note (markdown cell is fine given the time crunch) — feature count, encoding choices, train/test sizes, and a one-line confirmation that the leakage audit and the new split are consistent with each other.

---

## Step 6 — Wrap-up

Write a short Day 4 summary: final engineered feature count, encoding choices made, train/test split sizes, and explicit confirmation that the fairness-relevant demographic columns are preserved out-of-band for Day 9.

Also carry forward — don't lose — the deferred EDA scope, so it doesn't quietly disappear from the plan:

- Class-balance check on the *actual* train/test split (not just the full dataset, which Day 1/2 already covered)
- Distribution checks on the newly engineered features (`loan_to_income_ratio`, `rate_spread_bucket`, `applicant_age_bucket`)
- Demographic group-size confirmation ahead of Day 9's fairness evaluation

---

## End-of-Day-4 Checklist

- [ ] Sentinel/exempt-code audit completed for `MODEL_COLUMNS`, contamination confirmed or ruled out
- [ ] Categorical features encoded (one-hot for low-cardinality; frequency/target for any high-cardinality remainder)
- [ ] Confirmed no raw high-cardinality geo ID present in the final feature set
- [ ] `loan_to_income_ratio`, `rate_spread_bucket`, `applicant_age_bucket` engineered and added to the column dictionary
- [ ] Train/test split created, stratified on `approved`, with demographic group sizes checked in the test split
- [ ] Any target/frequency encoders fit train-only and applied to both splits
- [ ] `test_demographics_lookup.parquet` saved, row-index-aligned to `X_test`
- [ ] `X_train`/`X_test` schema-matched (same columns, dtypes, encoding)
- [ ] Day 4 summary + deferred-EDA scope note written

**Not for Day 4 (save for later days):** model training, hyperparameter tuning, SHAP/explainability, fairness metric computation (Day 9), and the full formal EDA report (deliberately deferred, not skipped — see resequencing note above).

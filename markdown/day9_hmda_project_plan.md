# HMDA 2024 Mortgage Lending Project — Day 9

This file covers two separate pieces of work: **(A)** the deferred full EDA, finally being done now that the model pipeline is stable, and **(B)** the actual Day 9 milestone — fairness analysis across demographic groups. Do A first; it's independent of the model and doesn't block B, but closes out a debt that's been carried since the README's Day 3.

---

# Part A — Deferred EDA Notebook (`hmda_eda.ipynb`)

**Data source:** `hmda_2024_clean.parquet` — the Day 2 output, **before** train/test split, **before** one-hot encoding, and (importantly) **before** imputation. This is deliberate: this file still has real `NaN`s in `income`, `property_value`, `debt_to_income_ratio`, `loan_to_value_ratio`, and `interest_rate`, which makes it the right file for genuine missingness EDA — the modeling-stage files already have these imputed and flagged, which would hide the exact patterns EDA is supposed to surface.

**One column caveat:** `loan_to_income_ratio` was engineered later, in Day 4, so it won't exist in this file. If you want it in the EDA, just recompute it inline (`loan_amount / income`) — cheap, and keeps this notebook self-contained rather than depending on a later one-hot-encoded file.

**Important instruction, different from every other notebook so far:** **do not save these plots to a `figures/` directory.** Every other day's plots (Day 5–8) are versioned deliverables — model performance, SHAP outputs, calibration curves — meant to be referenced later. EDA plots are exploratory scratch work for your own understanding; rendering them inline (`plt.show()`, no `savefig`) is enough, and skipping the save step keeps the repo from filling up with throwaway images that don't correspond to a specific reported finding.

## A.1 — Univariate: Continuous Features

**Achievable — for each of:** `income`, `loan_amount`, `property_value`, `loan_to_value_ratio`, `debt_to_income_ratio`, `interest_rate`, `rate_spread`, `loan_term`, `tract_population`, `tract_minority_population_percent`, `ffiec_msa_md_median_family_income`, `tract_to_msa_income_percentage`, `tract_owner_occupied_units`, `tract_one_to_four_family_homes`, `tract_median_age_of_housing_units`:
- Histogram (use log-scale x-axis or percentile-clipping where a column is still right-skewed post-cleaning — several of these were exactly what Day 2's cleaning fixed, so confirm they're actually readable now)
- Box plot, to visually re-confirm Day 2's outlier cleaning worked (should look clean — if any still show extreme whiskers, that's worth flagging as a Day 2 follow-up, not something to silently ignore)
- One consolidated summary-statistics table: mean, median, std, skew, kurtosis, min/max for all of the above in one place

## A.2 — Univariate: Categorical / Discrete Features

**Achievable — for each of:** `loan_type`, `loan_purpose`, `lien_status`, `occupancy_type`, `construction_method`, `derived_dwelling_category`, `applicant_credit_score_type`, `co_applicant_credit_score_type`, `submission_of_application`, `reverse_mortgage`, `open_end_line_of_credit`, `business_or_commercial_purpose`, `preapproval`, `negative_amortization`, `interest_only_payment`, `balloon_payment`, `total_units`:
- Bar chart of value counts / proportions
- **Pie charts only for genuinely low-cardinality fields** (≤5–6 categories) where proportions are the interesting story — good candidates: `loan_purpose`, `occupancy_type`, `preapproval`, `lien_status`, `derived_dwelling_category`. Don't pie-chart anything with more categories than that (e.g. `applicant_credit_score_type`) — it becomes unreadable.

## A.3 — Target and Demographic Composition

**Achievable:**
- Pie or bar chart of `approved` class balance — re-confirm the ~74.6% figure on this deduped file (Day 1's original 74.63% was pre-dedup; worth checking whether removing 11,036 duplicate rows shifted it at all).
- Bar/pie chart of `derived_race`, `derived_ethnicity`, `derived_sex` composition — same check, re-confirm Day 1's original composition numbers on the deduped subset.

## A.4 — Missingness

**Achievable:**
- Missingness bar chart: % missing per column, sorted descending, on this file's actual (pre-imputation) numbers — this supersedes Day 2/3's missingness table with corrected, dedup-adjusted denominators.
- A missingness co-occurrence view (e.g. `missingno.matrix` or a boolean correlation heatmap) — does missing `debt_to_income_ratio` tend to co-occur with missing `loan_to_value_ratio`? This is useful context for why `debt_to_income_ratio_missing` turned out to be Day 8's #2 SHAP feature.

## A.5 — Bivariate: Feature vs. Target

**Achievable:**
- Box plots of `income`, `loan_amount`, `debt_to_income_ratio`, `loan_to_value_ratio` grouped by `approved` (0/1) — a visual precursor to what the model later learned; compare shapes against Day 8's SHAP findings for the same features.
- Bar chart of approval rate by each key categorical feature (`loan_purpose`, `loan_type`, `occupancy_type`, `preapproval`, etc.) — same `groupby().mean()` pattern Day 2 used for demographics, now applied to loan/property characteristics.
- Approval rate by `derived_race` / `derived_sex`, recomputed on this deduped file — a direct consistency check against Day 2's original raw numbers (Asian 76.3%, White 74.0%, ... Male 73.3%, Female 70.4%) to confirm dedup didn't meaningfully shift them.

## A.6 — Bivariate: Feature vs. Feature

**Achievable:**
- Scatterplots: `income` vs. `loan_amount`, `property_value` vs. `loan_amount`, `loan_to_value_ratio` vs. `debt_to_income_ratio` — color or facet by `approved` where it adds a clear visual story.
- **Scatterplot specifically for the race-proxy question:** `tract_minority_population_percent` vs. `tract_to_msa_income_percentage` — directly relevant groundwork for Day 9 Part B's proxy discussion.

## A.7 — Correlation / Multivariate

**Achievable:**
- Correlation heatmap (Pearson) across all continuous features — cross-check against Day 5's VIF findings (max 6.44, no severe multicollinearity); this is the same question asked a different way.
- A second, focused correlation heatmap restricted to the `tract_*` geography fields plus `ffiec_msa_md_median_family_income` — the geography cluster most relevant to the proxy-discrimination question.

## A.8 — Demographic-Specific Distributions (sets up Day 9 Part B directly)

**Achievable:**
- Box or violin plots of `income`, `loan_amount`, `debt_to_income_ratio`, `loan_to_value_ratio` by `derived_race` and by `derived_sex` — extends Day 2's median-only comparison into a full distributional view.
- **The most directly relevant plot for what comes next:** histogram or box plot of `tract_minority_population_percent` by `derived_race` — this is the visual, model-free counterpart to Day 8's SHAP finding that this feature's contribution differs by race group. If the visual pattern here doesn't match the SHAP pattern, that's worth understanding before Part B proceeds.

## A.9 — Optional: Geography (Lower Priority)

**Achievable, time-permitting:** since `hmda_2024_clean.parquet` (unlike the model-feature files) still has `state_code`, a simple top-10-states-by-volume bar chart and a state-level approval-rate comparison is easy descriptive context — not required, skip if time is tight.

## Deliverable for Part A

`hmda_eda.ipynb` — every plot above, rendered inline, not saved to disk. No modeling changes result from this notebook; it's a standalone, closing-the-loop deliverable for the README's Day 3 requirement.

---

# Part B — Day 9: Fairness Analysis Across Demographic Groups

**Inputs from Day 8:**
- Tuned XGBoost (Run 2: depth 13), Platt-calibrated on `X_val`
- Operating threshold = 0.858 (precision-weighted, Fbeta=0.3)
- Val confusion matrix @0.858: TN 143,192 / FP 24,410 / FN 130,934 / TP 362,663 → FNR 26.5%
- Test confusion matrix @0.858: TN 357,918 / FP 61,087 / FN 326,132 / TP 907,860 → FNR 26.4%
- **~457,000 actually-approved applicants (val+test combined) predicted as "deny" at this threshold** — the population this entire step needs to characterize by demographic group
- Day 8 finding: `tract_minority_population_percent` ranks #7 by SHAP (vs. #50 by XGBoost gain), with mean SHAP contribution varying by `derived_race` (lowest for American Indian/Alaska Native, highest for 2+ minority races and Joint — though several of these groups are small-n)

**Honest note on scope:** the earlier-drafted "Day 3" plan (regression-adjusted approval gaps, significance testing on the raw decisions) was never executed — the project moved straight to preprocessing after Day 2's raw descriptive pass. Day 9, as scoped by the README, is about the **trained model's** subgroup error rates and calibration, which is a related but different question from "were the raw historical decisions adjusted-for-controls disparate." Worth naming this gap explicitly in your final write-up rather than letting Day 9 quietly stand in for both.

**README's explicit scope boundary, worth restating before starting:** this step *measures and reports* disparities. It does not implement a fairness-constrained optimizer or retrain the model to close any gaps found. That's a deliberate, stated limitation of the project — not an oversight to fix today.

## Step 0 — Setup and Methodology Decisions

**Achievable:**
- Check whether `test_demographics_lookup.parquet` (planned back in Day 4) actually exists. If it doesn't, build it now the same way `val_demographics_lookup.parquet` was built — row-index-aligned to `X_test`.
- Reapply Day 2's exclusion rule (`Joint`, `*_Not Available`, `Free Form Text Only` excluded from the primary comparison, sizes reported separately) — same methodology, now applied to model performance metrics instead of raw approval rates, for consistency across the whole project.
- Decide and fix the evaluation protocol: **`X_val` is the working partition for Steps 1–5** (iterate freely, this is investigative), **`X_test` gets exactly one confirmatory look in Step 6** — same test-discipline pattern established since Day 6.
- Evaluate at **both** the default 0.5 threshold and Day 8's 0.858 business threshold — this resolves the open question Day 8's own summary left for today.
- Set an explicit small-sample caution rule (e.g., flag any group with n < 500 in the evaluation subset) — Day 8's proxy-SHAP table already showed several race categories this small (American Indian/Alaska Native n=511, Native Hawaiian/PI n=149, 2+ minority races n=162); their point estimates need a caveat, not a headline.

**Deliverable:** confirmed/built `test_demographics_lookup.parquet`, the primary (exclusion-applied) evaluation subset, and the two-threshold evaluation plan documented.

## Step 1 — Subgroup Confusion-Matrix Metrics

**Achievable:**
- For `derived_race`, `derived_ethnicity`, and `derived_sex` separately: compute TPR, FPR, FNR, precision, recall, F1, and group size `n`, at **both** thresholds (0.5 and 0.858), on the primary `X_val` subset.
- Report FNR in the main table with the same visual weight as any other metric — per the README's explicit instruction, this is the fair-lending-sensitive metric and shouldn't be relegated to a footnote.
- Also report FPR clearly labeled as the over-approval / portfolio-risk metric (not an equity harm in the traditional sense, per the README, but still worth tracking).

**Deliverable:** one consolidated table per demographic dimension (race / ethnicity / sex), each row = group, columns = `n, TPR, FPR, FNR, precision, recall, F1`, at both thresholds.

## Step 2 — Quantify the Disparities

**Achievable:**
- For each metric in Step 1's tables, compute the gap between the highest- and lowest-scoring group (or against the largest group as an explicit reference category) — turn the table into an actual disparity number, not just a list to eyeball.
- Cross-reference against Step 0's small-sample flag: are the largest gaps coming from well-populated groups (a real finding) or from the smallest groups (likely noise, needs a caveat, not a headline)?
- **Directly compare the 0.5-threshold gaps against the 0.858-threshold gaps** — does moving to the precision-weighted business threshold shrink, widen, or leave subgroup disparities unchanged? This is arguably the single most important empirical question Day 9 can answer, and it follows directly from Day 8's threshold decision.

**Deliverable:** a disparity-gap table (metric, group with highest value, group with lowest value, gap, small-sample caveat if applicable) for both thresholds, plus a one-paragraph verdict on whether the business threshold helped or hurt subgroup fairness relative to 0.5.

## Step 3 — Calibration by Group

**Achievable:**
- Plot separate reliability diagrams (same method as Day 8's global one) for the **calibrated** model's output, split by `derived_race` and by `derived_sex` — does Day 8's Platt scaling, fit globally, hold up equally well within each subgroup, or does it systematically over/under-predict for specific groups?
- If Day 8's isotonic-vs-Platt comparison wasn't finished, note here whether the choice of calibration method changes the by-group picture — a global calibration curve that looks fine can still hide subgroup-level miscalibration, which is exactly the kind of thing a fairness review needs to catch.
- Document, don't necessarily fix: if per-group miscalibration is found, that's a limitation to report (per the README's "measure and report" scope), not something to patch today.

**Deliverable:** by-group calibration plots + a written note on whether calibration quality is consistent across groups.

## Step 4 — Tie Back the Day 8 Race-Proxy Finding

**Achievable:**
- Put Day 8's SHAP-by-race finding for `tract_minority_population_percent` side by side with Step 1–2's actual error-rate disparities. Does the group with the largest/most-adverse SHAP contribution from this feature also show the worst FNR or worst calibration? Or does the SHAP signal *not* line up with an actual outcome disparity?
- State plainly whether the evidence supports, partially supports, or doesn't support the proxy-discrimination concern that's been flagged and carried forward since Day 4/7 — this is the direct payoff of having tracked that flag through six days of work.

**Deliverable:** one clear paragraph connecting (or explicitly not connecting) the SHAP proxy signal to a measured outcome disparity.

## Step 5 — Profile the False-Negative Population

**Achievable:**
- Directly examine the ~457,000-person false-negative population (actually approved, model predicts deny at the 0.858 threshold) by `derived_race` and `derived_sex` composition.
- Compare that composition against each group's overall share of the *actually-approved* population — is any group over-represented among people the model would wrongly flag as high-risk, relative to their base rate?
- Cross-check against Day 2's original raw denial-reason breakdown by race for context — does the false-negative population's profile resemble the historically-denied population's profile in any of the loan/financial characteristics, which would help explain (without excusing) why the model is making these specific errors?

**Deliverable:** a demographic composition breakdown of the false-negative population, with a plain-language interpretation.

## Step 6 — One Honest Final Look at `X_test`

**Achievable:**
- Repeat Step 1's core table (TPR/FPR/FNR by group, both thresholds) exactly once on `X_test` + `test_demographics_lookup.parquet`.
- Compare directly against the `X_val` findings from Steps 1–2: do the same disparities show up, at similar magnitude? Agreement across an independent split is meaningfully stronger evidence than a val-only finding; disagreement is itself worth reporting honestly, not explaining away.
- No further iteration based on this result — same discipline as every prior "final test look" in this project.

**Deliverable:** the val-vs-test disparity comparison, clearly labeled as the project's one confirmatory check.

## Step 7 — Wrap-up

Write a plain-language Day 9 summary covering: which groups show meaningfully worse error rates and/or calibration (with small-sample caveats attached where relevant), whether the 0.858 threshold helped or hurt subgroup fairness relative to 0.5, whether the Day 8 SHAP proxy signal actually predicted the measured disparities, the false-negative population's demographic profile, and an explicit **limitations paragraph** — including the never-executed regression-adjusted analysis from the original Day 3 plan, the small-sample caveats on several race categories, and the README's stated scope boundary (measured and reported, not corrected).

Open questions for Day 10:
- Should the deployment demo surface a subgroup monitoring view (e.g. periodic FNR-by-group tracking) as an ongoing fairness check, given the README's framing of this as a compliance-relevant capability?
- Given the disparities found (if any), is the 0.858 threshold still the right documented choice, or does today's finding warrant revisiting it — and if revisited, who would that decision belong to (a modeling choice vs. a policy choice)?
- What would the regression-adjusted analysis from the original Day 3 plan have added on top of today's model-error-based findings, if there's time to circle back before wrapping the project?

---

## End-of-Day-9 Checklist

**Part A:**
- [ ] `hmda_eda.ipynb` created on `hmda_2024_clean.parquet` (pre-split, pre-imputation)
- [ ] Univariate continuous (histograms, box plots, summary stats) completed for all listed columns
- [ ] Univariate categorical (bar charts, pie charts for low-cardinality only) completed
- [ ] Target/demographic composition re-confirmed on deduped file
- [ ] Missingness bar chart + co-occurrence view completed
- [ ] Feature-vs-target bivariate plots completed
- [ ] Feature-vs-feature scatterplots completed, including the tract-geography proxy pair
- [ ] Correlation heatmaps (full + geography-focused) completed
- [ ] Demographic-specific distribution plots completed, including `tract_minority_population_percent` by race
- [ ] No plots saved to a `figures/` directory (exploratory notebook only)

**Part B:**
- [ ] `test_demographics_lookup.parquet` confirmed or built
- [ ] Day 2 exclusion rule reapplied for the primary comparison subset
- [ ] Subgroup TPR/FPR/FNR/precision/recall/F1 computed for race, ethnicity, sex at both 0.5 and 0.858 thresholds
- [ ] Disparity gaps quantified, small-sample caveats applied
- [ ] 0.5-vs-0.858 threshold comparison completed with an explicit verdict
- [ ] Calibration-by-group plots completed
- [ ] Day 8 SHAP proxy finding explicitly connected to (or dissociated from) measured disparities
- [ ] False-negative population profiled by demographic composition
- [ ] One confirmatory look completed on `X_test`, compared against `X_val` findings
- [ ] Day 9 summary + limitations paragraph + open questions for Day 10 written

**Not for Day 9 (per README's explicit scope boundary):** implementing a fairness-constrained optimizer, retraining the model to close any disparity found, deployment (Day 10).

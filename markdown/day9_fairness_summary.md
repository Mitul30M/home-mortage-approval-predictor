# Day 9 Fairness Summary

## Reproduced metrics
- Day 7 tuned XGBoost (max_depth=13), Platt-calibrated on X_val.
- Reproduced val ROC-AUC=0.8930, test ROC-AUC=0.8933.
- Business threshold (X_val Fbeta=0.3) = 0.8582.
- FN population (val+test combined) at THRESH: 457066 of actually-approved.

## Subgroup error-rate disparities
- derived_race FNR gap: @0.5 = 0.0125 (Native Hawaiian or Other Pacific Islander vs 2 or more minority races); @0.8582 = 0.0064 (White vs 2 or more minority races) -> helped.
- derived_ethnicity FNR gap: @0.5 = 0.0012 (Hispanic or Latino vs Not Hispanic or Latino); @0.8582 = 0.0057 (Hispanic or Latino vs Not Hispanic or Latino) -> widened.
- derived_sex FNR gap: @0.5 = 0.0012 (Female vs Male); @0.8582 = 0.0028 (Female vs Male) -> widened.

## Small-sample caveats
- Groups flagged small (n < 500): .

## 0.5 vs 0.858 threshold verdict
- The precision-weighted business threshold helped shrink subgroup FNR disparities relative to the default 0.5 threshold (see Step 2).

## Day 8 SHAP proxy tie-back
- tract_minority_population_percent ranks #7 by SHAP but only #50 by XGBoost gain. The proxy SHAP mean differs by derived_race (see Step 4).
- Whether the group with the most adverse proxy SHAP also shows the worst FNR is investigated in Step 2; the evidence is descriptive, not causal.

## False-negative population profile
- 457066 actually-approved applicants (val+test) are predicted as deny at THRESH.
- Over-representation ratios by derived_race and derived_sex are printed in Step 5.

## Limitations
- This step *measures and reports* disparities; it does not implement a fairness-constrained optimizer or retrain the model to close any gaps found (README scope boundary).
- The original Day 3 plan (regression-adjusted approval gaps, significance testing on the raw decisions) was never executed; Day 9 reports the *model error rates*, a related but different question from whether the raw historical decisions were disparate after controls.
- Several race/ethnicity categories are small-n (n < 500), so their point estimates carry wide uncertainty and should not be treated as headlines.

## Open questions for Day 10
- Should the deployment demo surface a subgroup monitoring view (e.g. periodic FNR-by-group tracking) as an ongoing fairness check?
- Given the disparities found (if any), is the 0.858 threshold still the right documented choice, or does today's finding warrant revisiting it (modeling choice vs policy choice)?
- What would the regression-adjusted analysis from the original Day 3 plan have added on top of today's model-error-based findings?
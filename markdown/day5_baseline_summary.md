# Day 5 Baseline Models - Summary

- Best baseline on X_val by ROC-AUC: RF (0.8732)
- Metrics (X_val):
    roc_auc  pr_auc  precision  recall      f1
LR   0.8012  0.9127     0.8869  0.7225  0.7963
DT   0.8051  0.9045     0.8873  0.7327  0.8026
RF   0.8732  0.9417     0.8569  0.9652  0.9078

- Clustering: k=6 via silhouette (15k subsample); preprocessing = StandardScaler(continuous)+PCA(26 comps, 90% var)+MiniBatchKMeans. Demographic composition on VAL (leakage-safe).
- Feature-selection recommendation counts: {'keep': 77, 'drop (low signal across methods)': 26}
- Drop candidates (low signal all 3 methods): ['loan_type_4', 'co_applicant_credit_score_type_8', 'co_applicant_credit_score_type_6', 'applicant_credit_score_type_14', 'negative_amortization_2', 'co_applicant_credit_score_type_12', 'co_applicant_credit_score_type_Exempt', 'co_applicant_credit_score_type_15', 'prepayment_penalty_term_missing', 'balloon_payment_Exempt', 'negative_amortization_Exempt', 'interest_only_payment_Exempt', 'co_applicant_credit_score_type_11', 'negative_amortization_1', 'derived_dwelling_category_Multifamily:Manufactured', 'co_applicant_credit_score_type_4', 'applicant_credit_score_type_6', 'applicant_credit_score_type_15', 'co_applicant_credit_score_type_14', 'co_applicant_credit_score_type_5', 'loan_purpose_5', 'applicant_credit_score_type_Exempt', 'applicant_credit_score_type_5', 'applicant_credit_score_type_13', 'applicant_credit_score_type_4', 'co_applicant_credit_score_type_13']
- Open questions for Day 6:
  1. Does dropping 'drop candidate' features hurt LightGBM/CatBoost or confirm noise?
  2. Do any Step-0 clusters warrant separate modeling vs just monitoring?
  3. Given RF importances, engineer explicit interactions or let boosted models discover them?
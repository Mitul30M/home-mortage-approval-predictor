"""Persist the final Day-6-validated modeling feature set to a CSV.

Recomputes FEATURES_DAY6 deterministically (independent of the Day 6 notebook
namespace) and writes data/processed/modelling/day7_model_features.csv with:
  feature        - the one-hot column name (78 rows)
  family         - prefix before first '_', for grouping/audit
  is_race_proxy  - 1 only for tract_minority_population_percent (kept, flagged)
  in_model       - 1
This CSV is the single source of truth for Day 7 tuning, Day 8 SHAP,
Day 9 fairness, and Day 10 serving.
"""
import os as _os, pathlib as _pl
import numpy as np, pandas as pd

_ROOT = None
for _cand in [_pl.Path(_os.getcwd()), _pl.Path(_os.getcwd()).parent,
              _pl.Path('/Volumes/Mitul/Projects/home-mortage-approval-predictor')]:
    if (_cand / 'data' / 'processed' / 'modelling').exists():
        _ROOT = _cand; break
if _ROOT is None:
    _ROOT = _pl.Path(_os.getcwd())
M = str(_ROOT / 'data' / 'processed' / 'modelling') + '/'

# All 103 model columns (matches Day 6 ALL_FEATURES)
ALL_FEATURES = list(pd.read_parquet(M + 'X_train.parquet').columns)

# Day 5 recommendation table
fs = pd.read_csv(M + 'day5_feature_selection.csv')
drops = set(fs[fs['recommendation'].str.startswith('drop')]['feature'])
print('raw Day5 drop count:', len(drops))

# Group by one-hot family (prefix before first '_')
fam = {}
for c in ALL_FEATURES:
    fam.setdefault(c.split('_', 1)[0], []).append(c)
fully_emptied = [f for f, members in fam.items() if set(members).issubset(drops)]
print('families fully emptied by raw drops:', fully_emptied)

adjusted_drops = set(drops)
for f in fully_emptied:
    keep = fam[f][0]            # keep first level as implicit reference
    adjusted_drops.discard(keep)
    print(f"  force-keep {keep} (family '{f}' would otherwise be fully dropped)")
print('adjusted drop count:', len(adjusted_drops))

FEATURES_FULL = ALL_FEATURES
FEATURES_TRIM = [c for c in ALL_FEATURES if c not in adjusted_drops]
print('FEATURES_FULL:', len(FEATURES_FULL), '| FEATURES_TRIM:', len(FEATURES_TRIM))

# Build the canonical feature table
df = pd.DataFrame({'feature': FEATURES_TRIM})
df['family'] = df['feature'].apply(lambda s: s.split('_', 1)[0])
df['is_race_proxy'] = (df['feature'] == 'tract_minority_population_percent').astype(int)
df['in_model'] = 1

out = M + 'day7_model_features.csv'
df.to_csv(out, index=False)
print('wrote', out, '| rows:', len(df), '| race-proxy flagged:',
      int(df['is_race_proxy'].sum()))
assert len(df) == 78, f"expected 78 features, got {len(df)}"
print('OK: final model feature set = 78 columns (canonical).')

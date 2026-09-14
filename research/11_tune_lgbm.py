"""Random-search LightGBM hyperparameters on DEV (2012-2020 scored, walk-forward yearly).
Selection happens on DEV only; the chosen config is then frozen for VALID and the locked HOLDOUT."""
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.evaluate import walk_forward, scores, fit_predict_sym
from ufc.model import feature_sets, lgbm_factory, recency_weight

d = pd.read_parquet("data/model_table.parquet")
fs = feature_sets(d)
keys = fs["ufc"] + fs["skills"] + fs["sherdog"] + fs["geo"] + fs["wiki"] + fs["fm"]
mask = (d.date >= "2012-01-01") & (d.date < "2021-01-01")
rng = np.random.default_rng(3)
space = dict(num_leaves=[4, 6, 8, 12, 16, 24], min_child_samples=[40, 80, 120, 200, 300], colsample_bytree=[0.15, 0.25, 0.35, 0.5, 0.7],
             learning_rate=[0.006, 0.01, 0.015, 0.025], n_estimators=[300, 500, 800, 1200], reg_lambda=[1, 5, 15, 40, 80],
             subsample=[0.6, 0.8, 1.0], halflife=[None, 4, 8])
rows = []
for i in range(24):
    cfg = {k: v[rng.integers(len(v))] for k, v in space.items()}
    hl = cfg.pop("halflife")
    fn = lambda tr, te: fit_predict_sym(lgbm_factory(**cfg, subsample_freq=1), tr, te, keys, raw=["age", "n_fights"], context=fs["context"], weight=recency_weight(hl))[0]
    p = walk_forward(d, fn, start="2012-01-01", end="2021-01-01")
    s = scores(d.y[mask], p.reindex(d.index)[mask])
    rows.append({**cfg, "halflife": hl, **s}); print(i, rows[-1], flush=True)
R = pd.DataFrame(rows).sort_values("logloss")
R.to_csv("reports/11_lgbm_tuning_dev.csv", index=False)
print(R.head(8).to_string(index=False))
best = R.iloc[0].to_dict()
json.dump({k: (None if (isinstance(v, float) and np.isnan(v)) else (int(v) if k in ("num_leaves", "min_child_samples", "n_estimators") else v))
           for k, v in best.items() if k in space}, open("data/lgbm_best.json", "w"), indent=1)

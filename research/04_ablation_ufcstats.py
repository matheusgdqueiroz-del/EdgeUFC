"""Which information categories carry signal? LightGBM walk-forward on DEV (2010-2020), yearly retrain.
  only_<g>  : that group (+context)      drop_<g> : everything except that group"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from ufc.evaluate import make_pairs, side_cols, fit_predict_sym, walk_forward, scores, CONTEXT
from ufc.groups import group_of

f = pd.read_parquet("data/fights.parquet"); X = pd.read_parquet("data/features_ufcstats.parquet"); prof = pd.read_parquet("data/profiles.parquet")
d = make_pairs(f, X, prof)
dev = d[(d.date >= "2010-01-01") & (d.date < "2021-01-01")]
keys = side_cols(d)
groups = {}
for k in keys: groups.setdefault(group_of(k), []).append(k)
print({g: len(v) for g, v in groups.items()})

def gbm_factory(): return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.015, num_leaves=12, min_child_samples=100,
                                          subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=10, verbose=-1)
def run(ks, raw=()):
    return walk_forward(d, lambda tr, te: fit_predict_sym(gbm_factory, tr, te, ks, raw=raw, context=CONTEXT)[0])

rows = []
def log(name, ks, raw=()):
    s = scores(dev.y, run(ks, raw).reindex(dev.index)); s.update(model=name, n_feat=len(ks)); rows.append(s); print(s, flush=True)

log("context_only", [])
log("all", keys, raw=["age", "n_fights"])
for g, ks in groups.items():
    log(f"only_{g}", ks)
    log(f"drop_{g}", [k for k in keys if k not in ks], raw=["age", "n_fights"] if g not in ("physical_age", "record_style") else ())
out = pd.DataFrame(rows).set_index("model")
out.to_csv("reports/04_ablation_ufcstats_dev.csv"); print(out.sort_values("logloss").to_string())

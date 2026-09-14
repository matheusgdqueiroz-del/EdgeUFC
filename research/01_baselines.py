"""Baselines vs first models on the DEV period (walk-forward, yearly retrain)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from ufc.evaluate import make_pairs, side_cols, fit_predict_sym, walk_forward, scores, CONTEXT

f = pd.read_parquet("data/fights.parquet"); X = pd.read_parquet("data/features_ufcstats.parquet"); prof = pd.read_parquet("data/profiles.parquet")
d = make_pairs(f, X, prof)
dev = d[(d.date >= "2010-01-01") & (d.date < "2021-01-01")]
keys = side_cols(d)
print("side features:", len(keys))

def rule(col, sign=1):
    diff = sign * (d[f"a_{col}"] - d[f"b_{col}"])
    return pd.Series(np.where(diff > 0, 0.6, np.where(diff < 0, 0.4, 0.5)), index=d.index)

res = {}
res["coin"] = pd.Series(0.5, index=d.index)
res["more_ufc_fights"] = rule("n_fights")
res["younger"] = rule("age", -1)
res["longer_reach"] = rule("reach")
res["better_winrate"] = rule("win_rate_s")
res["glicko_raw"] = d.glicko_p

def lr(cols, C=1.0, raw=(), ctx=()):
    fac = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return lambda tr, te: fit_predict_sym(fac, tr, te, cols, raw, list(ctx))[0]

for e in ["elo", "elo_mov", "elo_fast", "elo_new", "glicko"]:
    res[f"lr_{e}"] = walk_forward(d, lr([e]))
res["lr_elo+glicko+age+reach"] = walk_forward(d, lr(["elo", "glicko", "age", "reach", "height"]))
res["lr_all_diffs"] = walk_forward(d, lr(keys, C=0.05))

def gbm(tr, te):
    fac = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=80,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1)
    return fit_predict_sym(fac, tr, te, keys, raw=["age", "n_fights", "elo"], context=CONTEXT)[0]
res["lgbm_all"] = walk_forward(d, gbm)

rows = []
for k, p in res.items():
    s = scores(dev.y, p.reindex(dev.index)); s["model"] = k; rows.append(s)
out = pd.DataFrame(rows).set_index("model")
print(out.to_string())
out.to_csv("reports/01_baselines_dev.csv")

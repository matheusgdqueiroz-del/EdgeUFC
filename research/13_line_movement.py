"""Can we predict where the line moves between open and close? (the professional's definition of edge)

target  m = logit(p_close) - logit(p_open)       (positive = money came in on side a)
inputs  known when lines open: logit(p_open), model probability (out-of-sample), feature differences.
        weigh-in misses are excluded (they happen after most lines open).
Walk-forward yearly; reports out-of-sample R^2 and whether predicted movement direction is right, then
whether "bet at open on the side the model expects to shorten" earned closing-line value and profit.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from ufc.model import feature_sets, logit

period = sys.argv[1] if len(sys.argv) > 1 else "dev"
hi = 2020 if period == "dev" else 2024
P = pd.read_parquet(f"data/oos_{period}.parquet")
d = pd.read_parquet("data/model_table.parquet").loc[P.index]
fs = feature_sets(d)
keys = [k for k in fs["ufc"] + fs["sherdog"] + fs["geo"] if "missed_weight" not in k]
ok = d.p_open.notna() & d.p_close.notna() & (d.open_overround.between(1.0, 1.2))
d, P = d[ok], P[ok]
m = logit(d.p_close.values) - logit(d.p_open.values)
X = pd.DataFrame({f"d_{k}": d[f"a_{k}"].values - d[f"b_{k}"].values for k in keys}, index=d.index)
X["lp_open"] = logit(d.p_open.values)
X["lp_model_minus_open"] = logit(P["ens"].values) - logit(d.p_open.values)
yr = d.date.dt.year.values
pred = pd.Series(np.nan, index=d.index)
pred_lin = pd.Series(np.nan, index=d.index)
for Y in range(2012, hi + 1):
    tr, te = (yr < Y) & (yr >= 2008) & np.isfinite(X.lp_model_minus_open.values), yr == Y
    if te.sum() == 0:
        continue
    # symmetric augmentation: swapping sides negates every input and the target
    Xtr = pd.concat([X[tr], -X[tr]]); ytr = np.concatenate([m[tr], -m[tr]])
    g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=8, min_child_samples=150, colsample_bytree=0.4,
                          subsample=0.8, subsample_freq=1, reg_lambda=20, verbose=-1).fit(Xtr, ytr)
    pred[te] = (g.predict(X[te]) - g.predict(-X[te])) / 2
    r = Ridge(alpha=50).fit(Xtr[["lp_open", "lp_model_minus_open"]].fillna(0), ytr)
    pred_lin[te] = (r.predict(X[te][["lp_open", "lp_model_minus_open"]].fillna(0)) - r.predict(-X[te][["lp_open", "lp_model_minus_open"]].fillna(0))) / 2
sc = pred.notna()
for name, pr in (("lgbm", pred), ("ridge(open, model-open)", pred_lin)):
    e = m[sc.values] - pr[sc].values
    r2 = 1 - np.mean(e ** 2) / np.mean(m[sc.values] ** 2)
    big = np.abs(pr[sc].values) > 0.1
    print(f"{name}: OOS R^2 vs no-move baseline {r2:.3f}; direction hit-rate when |pred|>0.1: {np.mean(np.sign(pr[sc].values[big]) == np.sign(m[sc.values][big])):.3f} (n={big.sum()})")
print("mean |movement| (logit):", round(np.mean(np.abs(m)), 3), " corr(model-open, move):", round(np.corrcoef(X.lp_model_minus_open[sc], m[sc.values])[0, 1], 3))

# bet at the opening price on sides predicted to shorten by > thr
for thr in (0.05, 0.1, 0.2):
    rows = []
    for side in ("a", "b"):
        sgn = 1 if side == "a" else -1
        sel = sc & (sgn * pred > thr)
        price_open = d.loc[sel, f"{side}_open_dec"]; price_close = d.loc[sel, f"{side}_close_mean_dec"]
        won = (d.loc[sel, "y"] == 1) if side == "a" else (d.loc[sel, "y"] == 0)
        nc = d.loc[sel, "y"].isna() & (d.loc[sel, "outcome"] != "D/D")
        prof = np.where(nc, 0, np.where(won, price_open - 1, -1))
        rows.append(pd.DataFrame({"date": d.date[sel], "clv": price_open / price_close - 1, "profit": prof}))
    b = pd.concat(rows)
    if len(b):
        print(f"thr {thr}: bets {len(b)}, mean CLV {b.clv.mean():+.3%}, ROI at open {b.profit.mean():+.3%}, by year ROI:",
              b.groupby(b.date.dt.year).profit.mean().round(3).to_dict())

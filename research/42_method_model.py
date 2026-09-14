"""v2 / method model on DEV (walk-forward yearly, scored 2010-2020, no odds): does it beat base rates?

Baseline = frequencies by (women, weight class, scheduled rounds) from earlier years.
Scores: log loss of ITD, of method|winner (3 classes), of under_k; plus a calibration check for ITD.
Saves out-of-sample predictions 2010-2020 to data/v2/method_dev.parquet.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.fetch import ROOT
from ufc.evaluate import walk_forward
from ufc.model import feature_sets
from ufc.engine import stats_keys
from ufc.method_model import targets, predict_all, METHODS, FIN

d = pd.read_parquet(ROOT / "data" / "model_table.parquet")
for p in sorted((ROOT / "data" / "v2").glob("features_*.parquet")):
    d = d.merge(pd.read_parquet(p), on="fight_id", how="left")
d = d.join(targets(d))
fs = feature_sets(d)
keys = stats_keys(fs) + [k for k in fs["ufc"] if k.startswith("pr_")]
raw = [k for k in FIN if f"a_{k}" in d]

parts = []
for Y in range(2010, 2021):
    tr = d[(d.date < f"{Y}-01-01") & (d.date >= "2001-01-01")]
    te = d[d.date.dt.year == Y]
    parts.append(predict_all(tr, te, keys, raw))
    print(Y, flush=True)
P = pd.concat(parts)
P["fight_id"] = d.loc[P.index, "fight_id"]
P.to_parquet(ROOT / "data" / "v2" / "method_dev.parquet")


def base_rates(train, test, col, classes=None):
    grp = ["women", "weightclass", "sched_rounds"]
    tr = train[train[col].notna()]
    if classes is None:
        g = tr.groupby(grp)[col].agg(["sum", "size"])
        prior = tr[col].mean()
        r = (g["sum"] + 20 * prior) / (g["size"] + 20)
        return test[grp].apply(tuple, axis=1).map(r.to_dict()).fillna(prior).values
    out = []
    prior = tr[col].value_counts(normalize=True).reindex(classes).fillna(0.01)
    for c in classes:
        g = tr.assign(hit=(tr[col] == c).astype(float)).groupby(grp).hit.agg(["sum", "size"])
        r = (g["sum"] + 20 * prior[c]) / (g["size"] + 20)
        out.append(test[grp].apply(tuple, axis=1).map(r.to_dict()).fillna(prior[c]).values)
    return np.column_stack(out)


def bll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


rows = []
te_all = d.loc[P.index]
B = {c: [] for c in ["itd"] + [f"under_{k}" for k in (1, 2, 3, 4)]}
BM = []
for Y in range(2010, 2021):
    tr = d[(d.date < f"{Y}-01-01") & (d.date >= "2001-01-01")]
    te = te_all[te_all.date.dt.year == Y]
    for c in B:
        B[c].append(pd.Series(base_rates(tr, te, c), index=te.index))
    # method given winner, winner-oriented base rates are symmetric
    BM.append(pd.DataFrame(base_rates(tr, te, "meth3", METHODS), index=te.index, columns=METHODS))
for c in B:
    b = pd.concat(B[c])
    if c not in P:
        continue
    ok = te_all[c].notna() & P[c].notna()
    rows.append(dict(target=c, n=int(ok.sum()), base_rate_ll=round(bll(b[ok].values, te_all.loc[ok, c].values), 4),
                     model_ll=round(bll(P.loc[ok, c].values, te_all.loc[ok, c].values), 4)))
bm = pd.concat(BM)
ok = te_all.meth3.notna()
x = te_all[ok]
side = np.where(x.y.values == 1, "a", "b")
pm = np.array([[P.loc[i, f"{s}_{m}"] for m in METHODS] for i, s in zip(x.index, side)])
pm = pm / pm.sum(1, keepdims=True)
yi = np.array([METHODS.index(m) for m in x.meth3])
ll_model = -np.mean(np.log(np.clip(pm[np.arange(len(yi)), yi], 1e-6, 1)))
ll_base = -np.mean(np.log(np.clip(bm.loc[x.index].values[np.arange(len(yi)), yi], 1e-6, 1)))
rows.append(dict(target="method|winner", n=len(yi), base_rate_ll=round(ll_base, 4), model_ll=round(ll_model, 4)))
res = pd.DataFrame(rows)
res["gain"] = res.base_rate_ll - res.model_ll
print(res.to_string(index=False))
res.to_csv(ROOT / "reports" / "42_method_model_dev.csv", index=False)
q = pd.qcut(P.itd, 10)
print(pd.DataFrame({"pred": P.itd, "actual": te_all.itd}).groupby(q).mean().round(3).to_string())

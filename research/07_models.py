"""Model comparison + data-source increments (walk-forward, no hindsight).

  python research/07_models.py dev     -> scored on 2010-2020 (selection allowed)
  python research/07_models.py valid   -> scored on 2021-2024 (confirmation only)

Writes out-of-sample predictions for every bout from 2008 on to data/oos_<period>.parquet so later
scripts (stacking, betting, error analysis) never touch in-sample predictions.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.evaluate import walk_forward, scores, fit_predict_sym, VALID_START, HOLDOUT_START
from ufc.model import feature_sets, lgbm_factory, lr_factory, recency_weight, Stacker, residual_lgbm, logit

period = sys.argv[1] if len(sys.argv) > 1 else "dev"
lo, hi = ("2010-01-01", "2021-01-01") if period == "dev" else ("2021-01-01", "2025-01-01")
d = pd.read_parquet("data/model_table.parquet")
fs = feature_sets(d)
print({k: len(v) for k, v in fs.items()})
score_mask = (d.date >= lo) & (d.date < hi)
has_mkt = d.p_close.notna() if "p_close" in d else pd.Series(False, index=d.index)

RAW = ["age", "n_fights"]
from ufc.engine import lgbm_params
TUNED, TUNED_HL = lgbm_params()
def gbm(keys, hl=TUNED_HL, raw=RAW, **kw):
    cfg = {**TUNED, **kw, "subsample_freq": 1}
    return lambda tr, te: fit_predict_sym(lgbm_factory(**cfg), tr, te, keys, raw=raw, context=fs["context"], weight=recency_weight(hl))[0]
def lr(keys, C=0.05):
    return lambda tr, te: fit_predict_sym(lr_factory(C), tr, te, keys, raw=(), context=[c for c in fs["context"] if c != "year"])[0]

ALL = fs["ufc"] + fs["skills"] + fs["sherdog"] + fs["geo"] + fs["wiki"] + fs["fm"] + fs["pv"]
runs = {
    "lgbm_ufc": gbm(fs["ufc"]),
    "lgbm_ufc+skills": gbm(fs["ufc"] + fs["skills"]),
    "lgbm_ufc+skills+sherdog": gbm(fs["ufc"] + fs["skills"] + fs["sherdog"]),
    "lgbm_ufc+skills+sherdog+geo": gbm(fs["ufc"] + fs["skills"] + fs["sherdog"] + fs["geo"]),
    "lgbm_ufc+skills+sherdog+geo+fm": gbm(fs["ufc"] + fs["skills"] + fs["sherdog"] + fs["geo"] + fs["fm"]),
    "lgbm_all": gbm(ALL),
    "lgbm_skills_only": gbm(fs["skills"] + ["age", "reach", "height"], raw=["age"]),
    "lgbm_sherdog+fm+age_only": gbm(fs["sherdog"] + fs["fm"] + ["age", "reach", "height"], raw=["age"]),
    "lr_all": lr(ALL),
}
runs = {k: v for k, v in runs.items() if not (("sherdog" in k and not fs["sherdog"]))}
start = "2008-01-01"
preds = {}
for name, fn in runs.items():
    preds[name] = walk_forward(d, fn, start=start, end=hi)
    s = scores(d.y[score_mask], preds[name].reindex(d.index)[score_mask]); s["model"] = name
    sm = scores(d.y[score_mask & has_mkt], preds[name].reindex(d.index)[score_mask & has_mkt]) if has_mkt.any() else {}
    print(name, s, "| on market-covered:", {k: sm.get(k) for k in ("n", "logloss", "acc")}, flush=True)
P = pd.DataFrame(preds)
P["ens"] = (P["lgbm_all"] + P["lr_all"]) / 2
P = P.join(d[["date", "y", "fight_id"] + [c for c in ("p_close", "p_open") if c in d]])

rows = []
for col in [c for c in P.columns if c.startswith(("lgbm", "lr", "ens"))] + [c for c in ("p_open", "p_close") if c in P]:
    m = score_mask.reindex(P.index, fill_value=False) & P[col].notna() & (has_mkt.reindex(P.index) if has_mkt.any() else True)
    s = scores(P.y[m], P[col][m]); s["model"] = col; rows.append(s)

# ---- market stacking (walk-forward: stacker for year Y fitted on years < Y) ----
if has_mkt.any():
    for base in ("ens", "lgbm_all"):
        for mcol in ("p_close", "p_open"):
            out, coefs = pd.Series(np.nan, index=P.index), []
            for Y in range(2011, int(hi[:4])):
                trm = (P.date.dt.year < Y) & P[mcol].notna() & P[base].notna() & P.y.notna()
                tem = (P.date.dt.year == Y) & P[mcol].notna() & P[base].notna()
                if trm.sum() < 500 or tem.sum() == 0:
                    continue
                st = Stacker().fit(P[mcol][trm].values, P[base][trm].values, P.y[trm].values)
                out[tem] = st.predict(P[mcol][tem].values, P[base][tem].values)
                coefs.append((Y, *np.round(st.coef, 3)))
            P[f"stack_{base}_{mcol}"] = out
            m = score_mask.reindex(P.index, fill_value=False) & out.notna()
            s = scores(P.y[m], out[m]); s["model"] = f"stack_{base}_{mcol}"; rows.append(s)
            print(f"stack {base}+{mcol} coefs (year, market, model):", coefs[-4:])
    # residual model on top of the closing line
    allk = ALL + fs["mkt_hist"]
    P["resid_close"] = walk_forward(d[has_mkt], lambda tr, te: residual_lgbm(tr, te, allk, fs["context"]), start="2011-01-01", end=hi).reindex(P.index)
    m = score_mask.reindex(P.index, fill_value=False) & P.resid_close.notna()
    s = scores(P.y[m], P.resid_close[m]); s["model"] = "resid_lgbm_on_close"; rows.append(s)

res = pd.DataFrame(rows).set_index("model")
print(f"\n=== {period} ({lo} .. {hi}) — rows restricted to bouts with market odds where available ===")
print(res.sort_values("logloss").to_string())
res.to_csv(f"reports/07_models_{period}.csv")
P.to_parquet(f"data/oos_{period}.parquet")

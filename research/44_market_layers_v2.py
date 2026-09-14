"""v2 / market layers on DEV (walk-forward yearly, scored 2013-2020) at every line stage.

Compares, per stage s = 0..9 (0 = open, 9 = close):
  v1 : per-stage 2-parameter stack with the v1 stats model (research/40 design; v1 used stack_open at 0, ens3 at 9)
  v2 : MarketLayers2 (pooled stages) with a chosen stats model (v1 or v2 stats), components and ensemble
Metrics: log loss vs the line at that stage (event-clustered bootstrap for ens vs v1 stack), steady-rule bets at the
stage's average price: bets per event, ROI, closing-line value.

  python research/44_market_layers_v2.py <stats_variant>
Ran with a third residual-LightGBM layer (since removed from ufc/market2.py); results are cached in data/v2.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.fetch import ROOT
from ufc.sources.ufcstats import load_fights
from ufc.sources import odds
from ufc.model import Stacker, feature_sets, logit
from ufc.engine import stats_keys
from ufc.betting import simulate, summary
from ufc.market2 import MarketLayers2, stage_rows

STEADY = dict(min_edge=0.03, frac=0.125, max_units=1.5)
variant = sys.argv[1] if len(sys.argv) > 1 else "C0.003+promo+soft70"
V2 = ROOT / "data" / "v2"

d = pd.read_parquet(ROOT / "data" / "model_table.parquet")
for p in sorted(V2.glob("features_*.parquet")):
    d = d.merge(pd.read_parquet(p), on="fight_id", how="left")
o = odds.build(load_fights())
d = d.drop(columns=[c for c in o.columns if c != "fight_id" and c in d.columns]).merge(o, on="fight_id", how="left")
s1 = pd.read_parquet(V2 / "stats_oos_base.parquet").rename(columns={"stats": "stats_v1"})
s2 = pd.read_parquet(V2 / f"stats_oos_{variant}.parquet").rename(columns={"stats": "stats_v2"})
d = d.merge(s1, on="fight_id", how="left").merge(s2, on="fight_id", how="left")
d = d[(d.date >= "2008-01-01") & (d.date < "2021-01-01") & d.p_s0.notna() & d.p_s9.notna() & d.stats_v1.notna()].reset_index(drop=True)
fs = feature_sets(d)
keys = stats_keys(fs) + [k for k in fs["ufc"] if k.startswith("pr_")] + fs["mkt_hist"]
yr = d.date.dt.year.values

preds = []
cache = V2 / f"market_dev_{variant}.parquet"
for Y in ([] if cache.exists() else range(2013, 2021)):
    tr, te = d[(yr < Y)], d[yr == Y]
    for tag, col in (("v1stats", "stats_v1"), ("v2stats", "stats_v2")):
        trx, tex = tr.assign(stats=tr[col]), te.assign(stats=te[col])
        ml = MarketLayers2(keys, fs["context"]).fit(trx)
        rows = stage_rows(tex, range(10))
        P = ml.predict(rows)
        P.columns = [f"{tag}_{c}" for c in P.columns]
        P["fight_id"], P["stage"] = rows.fight_id.values, rows.stage.values
        preds.append(P.reset_index(drop=True))
    # v1-style per-stage stack
    for k in range(10):
        ok = tr.y.notna()
        st = Stacker().fit(tr.loc[ok, f"p_s{k}"].values, tr.loc[ok, "stats_v1"].values, tr.loc[ok, "y"].values)
        preds.append(pd.DataFrame({"fight_id": te.fight_id.values, "stage": k, "v1_stack": st.predict(te[f"p_s{k}"].values, te.stats_v1.values)}))
    print(Y, flush=True)

if preds:
    pd.concat(preds).groupby(["fight_id", "stage"], as_index=False).first().to_parquet(cache)
P = pd.read_parquet(cache)
rows = stage_rows(d[yr >= 2013].assign(stats=d.stats_v1), range(10)).reset_index(drop=True)
R = rows.merge(P, on=["fight_id", "stage"])
n_events = R.event_id.nunique()


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_gain(x, a, b, n=1000, seed=0):
    ok = x.y.notna().values
    g = pd.DataFrame({"ev": x.event_id.values[ok], "diff": ll(x[a].values[ok], x.y.values[ok]) - ll(x[b].values[ok], x.y.values[ok])}).groupby("ev")["diff"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    s, c = g["sum"].values, g["size"].values
    bs = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    return s.sum() / c.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


out = []
models = ["v1_stack", "v1stats_ens", "v2stats_stage_stack", "v2stats_edge_glm", "v2stats_resid", "v2stats_ens"]
for k in range(10):
    x = R[R.stage == k]
    ok = x.y.notna()
    row = dict(stage=k, line_ll=round(ll(x.p_line[ok].values, x.y[ok].values).mean(), 4))
    for m in models:
        row[f"{m}_ll"] = round(ll(x[m][ok].values, x.y[ok].values).mean(), 4)
    g, lo, hi = boot_gain(x, "v1_stack", "v2stats_ens")
    row.update(gain_v2ens_vs_v1=round(g, 4), ci=f"[{lo:+.4f}, {hi:+.4f}]")
    for m in ("v1_stack", "v2stats_ens"):
        b = simulate(x.assign(date=x.date), m, "a_line_dec", "b_line_dec", **STEADY)
        sm = summary(b)
        j = x.loc[b.index]
        close = np.where(b.bet_side == "a", j.a_s9_dec, j.b_s9_dec)
        row[f"{m}_bets_ev"] = round(sm["bets"] / n_events, 2)
        row[f"{m}_roi"] = sm["roi"]
        row[f"{m}_clv"] = round(float(np.mean(b.price.values / close - 1)), 4) if len(b) else np.nan
        row[f"{m}_yrs_up"] = int((b.groupby(b.date.dt.year).profit.sum() > 0).sum())
    out.append(row)
res = pd.DataFrame(out)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(res.to_string(index=False))
res.to_csv(ROOT / "reports" / f"44_market_layers_dev.csv", index=False)

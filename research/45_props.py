"""v2 / prop markets (never examined before this script): does the method model beat closing prop prices?

Pre-registered in PREREGISTRATION.md Addendum B (4):
  * method model = ufc/method_model.py as developed on DEV outcomes (research/42), walk-forward yearly for 2021+
  * winner probability = v1 engine ens3 out of sample (market-aware at the close), else the closing line
  * prop layer per market family: logistic on [logit(fair prop price), logit(model)] + intercept, refit yearly on
    earlier prop years only; scored from the second prop year
  * bets: steady rule (edge > 3%, 1/8 Kelly, max 1.5u) at the MEAN closing price across books quoting the prop
  * edge claimed only if ROI 90% bootstrap CI (clustered by event) excludes 0 AND the latest full year is positive
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from ufc.fetch import ROOT
from ufc.sources.ufcstats import load_fights
from ufc.model import feature_sets, logit
from ufc.engine import stats_keys
from ufc.method_model import targets, predict_all, FIN
from ufc import props
from ufc.betting import kelly_units

V2 = ROOT / "data" / "v2"
STEADY = dict(min_edge=0.03, frac=0.125, max_units=1.5)
# NOTE: the first run grouped "fight goes to decision" and "fighter wins by decision" into one family by mistake
# (m.split("_")[1] == "dec"); that output is kept in reports/45_props_as_run_family_bug.csv. Corrected below.
FAMILY = lambda m: "dec" if m == "dec" else ("under" if m.startswith("under") else {"dec": "fdec"}.get(m.split("_")[1], m.split("_")[1]))

# ---------------------------------------------------------------- 1. method model out of sample 2021+
d = pd.read_parquet(ROOT / "data" / "model_table.parquet")
for p in sorted(V2.glob("features_*.parquet")):
    d = d.merge(pd.read_parquet(p), on="fight_id", how="left")
d = d.join(targets(d))
mpath = V2 / "method_oos_2021.parquet"
if not mpath.exists():
    fs = feature_sets(d)
    keys = stats_keys(fs) + [k for k in fs["ufc"] if k.startswith("pr_")]
    raw = [k for k in FIN if f"a_{k}" in d]
    parts = []
    for Y in range(2021, 2027):
        tr = d[(d.date < f"{Y}-01-01") & (d.date >= "2001-01-01")]
        te = d[d.date.dt.year == Y]
        parts.append(predict_all(tr, te, keys, raw))
        print("method", Y, flush=True)
    P = pd.concat(parts)
    P["fight_id"] = d.loc[P.index, "fight_id"]
    P.to_parquet(mpath)
M = pd.read_parquet(mpath)

# ---------------------------------------------------------------- 2. prop markets matched to bouts
lines = pd.read_csv(ROOT / "data" / "bfo" / "event_lines.csv")
C = props.consensus(props.market_table(lines))
fights = load_fights()
C = props.link_bouts(C, fights)
E = pd.read_parquet(ROOT / "data" / "engine_oos.parquet")[["fight_id", "ens3", "p_close"]]
x = C.merge(d[["fight_id", "date", "event_id", "y", "outcome", "method", "fight_secs", "sched_rounds", "weightclass"]], on="fight_id") \
     .merge(E, on="fight_id", how="left").merge(M, on="fight_id", how="left")
x = x[x.date >= "2021-01-01"]
x["p_a"] = x.ens3.fillna(x.p_close)
x = x[x.p_a.notna() & x.itd.notna() & x.fair.notna() & x.fair.between(0.01, 0.99)]
x["model"] = props.model_prob(x)
x["won"] = props.settle(x)
x["family"] = x.market.map(FAMILY)
x = x[x.model.notna()].reset_index(drop=True)
print("prop rows", len(x), "bouts", x.fight_id.nunique(), "by year", x.groupby(x.date.dt.year).fight_id.nunique().to_dict())
print(x.groupby("family").agg(n=("won", "size"), books=("n_books", "mean"), overround=("overround", "median")).round(3).to_string())

# ---------------------------------------------------------------- 3. walk-forward prop layer + bets
yr = x.date.dt.year.values
first = int(yr.min())
x["layer"] = np.nan
for Y in range(first + 1, 2027):
    for fam, g in x.groupby("family"):
        tr = g[(g.date.dt.year < Y) & g.won.notna()]
        te = g[g.date.dt.year == Y]
        if len(te) == 0 or len(tr) < 200:
            continue
        X = np.column_stack([logit(tr.fair.values), logit(np.clip(tr.model.values, 1e-4, 1 - 1e-4))])
        lr = LogisticRegression(C=1.0).fit(X, tr.won.values)
        Xt = np.column_stack([logit(te.fair.values), logit(np.clip(te.model.values, 1e-4, 1 - 1e-4))])
        x.loc[te.index, "layer"] = lr.predict_proba(Xt)[:, 1]
        if Y == 2026:
            print(f"layer coef {fam}: fair {lr.coef_[0][0]:.2f} model {lr.coef_[0][1]:.2f} intercept {lr.intercept_[0]:.2f} (n={len(tr)})")


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


s = x[x.layer.notna() & x.won.notna()]
print("\nLog loss (scored years):")
print(s.groupby("family").apply(lambda g: pd.Series(dict(n=len(g), market=ll(g.fair, g.won).mean(), model=ll(g.model, g.won).mean(),
                                                         layer=ll(g.layer, g.won).mean()))).round(4).to_string())


def bets(frame, prob_col, price="mean"):
    f = frame[frame[prob_col].notna()].copy()
    p = f[prob_col].values
    dy, dn = f[f"yes_{price}"].values, f[f"no_{price}"].values
    ev_y, ev_n = p * dy - 1, (1 - p) * dn - 1
    yes = np.where(np.isnan(ev_n), True, np.where(np.isnan(ev_y), False, ev_y >= ev_n))
    pb, dec, ev = np.where(yes, p, 1 - p), np.where(yes, dy, dn), np.where(yes, ev_y, ev_n)
    st = kelly_units(pb, dec, STEADY["frac"], STEADY["max_units"])
    ok = (ev > STEADY["min_edge"]) & (st >= 0.25) & np.isfinite(dec)
    won = np.where(yes, f.won.values, 1 - f.won.values)
    prof = np.where(np.isnan(f.won.values), 0, np.where(won == 1, st * (dec - 1), -st))
    f["stake"], f["profit"], f["ev"], f["side"] = np.where(ok, st, 0), np.where(ok, prof, 0), ev, np.where(yes, "yes", "no")
    return f[ok]


def boot_roi(b, n=2000, seed=1):
    g = b.groupby("event_id").agg(p=("profit", "sum"), s=("stake", "sum"))
    rng = np.random.default_rng(seed)
    r = [g.p.values[i].sum() / g.s.values[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    return np.percentile(r, 5), np.percentile(r, 95)


rows = []
events = s.event_id.nunique()
for prob in ("layer", "model"):
    for fam in ["all"] + sorted(s.family.unique()):
        g = s if fam == "all" else s[s.family == fam]
        b = bets(g, prob)
        if b.empty:
            continue
        lo, hi = boot_roi(b)
        by = b.groupby(b.date.dt.year).apply(lambda z: z.profit.sum() / z.stake.sum())
        rows.append(dict(prob=prob, family=fam, bets=len(b), bets_per_event=round(len(b) / events, 2), staked=round(b.stake.sum(), 1),
                         profit=round(b.profit.sum(), 1), roi=round(b.profit.sum() / b.stake.sum(), 4), roi_90ci=f"[{lo:+.3f}, {hi:+.3f}]",
                         yes_share=round((b.side == "yes").mean(), 2), roi_by_year=by.round(3).to_dict()))
R = pd.DataFrame(rows)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 120)
print("\nBets at mean closing prop price, steady rule:")
print(R.to_string(index=False))
R.to_csv(ROOT / "reports" / "45_props.csv", index=False)
x.to_parquet(V2 / "props_scored.parquet")

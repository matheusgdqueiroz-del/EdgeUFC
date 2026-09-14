"""v2 / line stages (DEV only, scored 2013-2020): how much edge is left at each point of a line's life?

BestFightOdds sparklines give the average price at 10 evenly spaced points from the opening line (stage 0)
to the close (stage 9). For each stage: market log loss, a walk-forward stack (stage line + out-of-sample
stats model), and the fixed steady staking rule bet at that stage's average price, with closing-line value.
Also fits one pooled stage-aware stack whose market/model weights vary linearly with the stage.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from ufc.fetch import ROOT
from ufc.sources.ufcstats import load_fights
from ufc.sources import odds
from ufc.model import Stacker, logit
from ufc.betting import simulate, summary

STEADY = dict(min_edge=0.03, frac=0.125, max_units=1.5)
HI = 2020
E = pd.read_parquet(ROOT / "data" / "engine_oos.parquet")[["fight_id", "event_id", "date", "y", "outcome", "stats"]]
o = odds.build(load_fights()).drop(columns=["p_close", "p_open"], errors="ignore")
d = E.merge(o, on="fight_id")
d = d[(d.date >= "2008-01-01") & (d.date < f"{HI + 1}-01-01") & d.stats.notna() & d.p_s0.notna() & d.p_s9.notna()].reset_index(drop=True)
yr = d.date.dt.year.values
S = 10


def ll(p, y):
    m = ~np.isnan(y)
    p = np.clip(p[m], 1e-6, 1 - 1e-6)
    return -np.mean(y[m] * np.log(p) + (1 - y[m]) * np.log(1 - p))


class StageStack:
    """logit p = (a0 + a1 s) logit(p_stage) + (b0 + b1 s) logit(stats), s in [0, 1]; symmetric, no intercept."""

    def _X(self, pm, pmod, s):
        lm, lmod = logit(pm), logit(pmod)
        return np.column_stack([lm, lm * s, lmod, lmod * s])

    def fit(self, pm, pmod, s, y):
        X = self._X(pm, pmod, s)
        self.m = LogisticRegression(C=10.0, fit_intercept=False).fit(np.vstack([X, -X]), np.concatenate([y, 1 - y]))
        return self

    def predict(self, pm, pmod, s):
        return self.m.predict_proba(self._X(pm, pmod, s))[:, 1]


rows, pooled_preds = [], {k: np.full(len(d), np.nan) for k in range(S)}
stage_preds = {k: np.full(len(d), np.nan) for k in range(S)}
for Y in range(2013, HI + 1):
    tr, te = (yr < Y) & ~np.isnan(d.y.values), yr == Y
    # pooled stage-aware stack on all stages of training bouts
    P = np.concatenate([d.loc[tr, f"p_s{k}"].values for k in range(S)])
    M = np.concatenate([d.loc[tr, "stats"].values for k in range(S)])
    Sg = np.concatenate([np.full(tr.sum(), k / (S - 1)) for k in range(S)])
    Yv = np.concatenate([d.loc[tr, "y"].values for k in range(S)])
    ss = StageStack().fit(P, M, Sg, Yv)
    for k in range(S):
        st = Stacker().fit(d.loc[tr, f"p_s{k}"].values, d.loc[tr, "stats"].values, d.loc[tr, "y"].values)
        stage_preds[k][te] = st.predict(d.loc[te, f"p_s{k}"].values, d.loc[te, "stats"].values)
        pooled_preds[k][te] = ss.predict(d.loc[te, f"p_s{k}"].values, d.loc[te, "stats"].values, np.full(te.sum(), k / (S - 1)))
        if Y == HI:
            rows.append(dict(stage=k, coef_market=st.coef[0], coef_model=st.coef[1]))
coefs = pd.DataFrame(rows)

sc = yr >= 2013
y = d.y.values
out = []
n_events = d.loc[sc, "event_id"].nunique()
for k in range(S):
    d[f"stack_s{k}"], d[f"pooled_s{k}"] = stage_preds[k], pooled_preds[k]
    x = d[sc]
    b = simulate(x, f"stack_s{k}", f"a_s{k}_dec", f"b_s{k}_dec", **STEADY)
    bp = simulate(x, f"pooled_s{k}", f"a_s{k}_dec", f"b_s{k}_dec", **STEADY)
    clv = np.where(b.bet_side == "a", b[f"a_s{k}_dec"] if f"a_s{k}_dec" in b else np.nan, np.nan)
    j = x.loc[b.index]
    close_price = np.where(b.bet_side == "a", j.a_close_mean_dec, j.b_close_mean_dec)
    sm, smp = summary(b), summary(bp)
    out.append(dict(stage=k, market_ll=round(ll(x[f"p_s{k}"].values, x.y.values), 4), stack_ll=round(ll(x[f"stack_s{k}"].values, x.y.values), 4),
                    pooled_ll=round(ll(x[f"pooled_s{k}"].values, x.y.values), 4),
                    bets=sm["bets"], bets_per_event=round(sm["bets"] / n_events, 2), roi=sm["roi"], profit=sm["profit"], max_dd=sm["max_dd"],
                    clv=round(float(np.mean(b.price.values / close_price - 1)), 4) if len(b) else np.nan,
                    pooled_bets=smp["bets"], pooled_roi=smp["roi"],
                    years_up=f"{int((b.groupby(b.date.dt.year).profit.sum() > 0).sum())}/{HI - 2012}"))
res = pd.DataFrame(out).merge(coefs, on="stage")
pd.set_option("display.width", 250)
print(f"DEV scored 2013-{HI}: {sc.sum()} bouts, {n_events} events")
print(res.to_string(index=False))
res.to_csv(ROOT / "reports" / "40_line_stages_dev.csv", index=False)

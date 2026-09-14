"""The prediction engine used BOTH for the final historical evaluation and for live picks.

v2 (live default, reports/PREREGISTRATION.md Addendum C):
  stats  : LightGBM + logistic (C 0.003, soft labels for split decisions), promotion-level features included
  market : ufc/market2.py stage-aware layers, usable at any point between the opening line and the close
v1 layers (kept to reproduce the v1 audit)
  1. stats    : market-free ensemble (LightGBM + logistic) on every non-market feature
  2. stack    : closing/opening line + stats (walk-forward logistic stack)
  3. edge_glm : offset ridge logistic around the line with pre-registered mispricing candidates
  4. resid    : LightGBM correction around the closing line
  5. ens3     : mean of 2-4 (primary probability for bets placed near the close)
  stack_open  : line-open version (for bets placed when lines open)
Staking: fractional Kelly with (min edge, fraction, cap) chosen from earlier out-of-sample years only.
"""
import json
import numpy as np
import pandas as pd

from ufc.fetch import ROOT
from ufc.evaluate import walk_forward, fit_predict_sym, fit_predict_soft
from ufc.model import (feature_sets, lgbm_factory, lr_factory, Stacker, OffsetRidgeLogit, residual_lgbm, logit)
from ufc.betting import walk_forward_policy, simulate, kelly_units, GRID

DATA = ROOT / "data"
RAW = ["age", "n_fights"]


def lgbm_params():
    p = DATA / "lgbm_best.json"
    if not p.exists():
        return {}, None
    cfg = json.load(open(p))
    hl = cfg.pop("halflife", None)
    return cfg, hl


def stats_keys(fs, version="v2"):
    keys = fs["ufc"] + fs["skills"] + fs["sherdog"] + fs["geo"] + fs["wiki"] + fs["fm"]
    return keys if version == "v2" else [k for k in keys if not k.startswith("pr_")]


def stats_predict(train, test, fs, version="v2"):
    cfg, _ = lgbm_params()
    keys = stats_keys(fs, version)
    ctx_lr = [c for c in fs["context"] if c != "year"]
    if version == "v1":
        g = fit_predict_sym(lgbm_factory(**cfg, subsample_freq=1), train, test, keys, raw=RAW, context=fs["context"])[0]
        l = fit_predict_sym(lr_factory(0.05), train, test, keys, raw=(), context=ctx_lr)[0]
        return (g + l) / 2
    g = fit_predict_soft(lgbm_factory(**cfg, subsample_freq=1), train, test, keys, RAW, fs["context"], soft=0.7)
    l = fit_predict_soft(lr_factory(0.003), train, test, keys, (), ctx_lr, soft=0.7, pipeline=True)
    return (g + l) / 2


# ---------------------------------------------------------------- edge GLM design (pre-registered list)
def edge_design(d, p_line_col="p_close"):
    Lm = logit(d[p_line_col].values)
    Lo = logit(d.p_open.fillna(d[p_line_col]).values)
    Lmod = logit(d.stats.values)
    anti = lambda c: (d[f"a_{c}"] - d[f"b_{c}"]).astype(float).values if f"a_{c}" in d else np.zeros(len(d))
    main = (d.bout_order <= 4).values
    pa, pb = d[p_line_col].values, 1 - d[p_line_col].values
    Z = pd.DataFrame(index=d.index)
    Z["def_skill"] = anti("sk_d_str"); Z["adj_sapm"] = anti("d_adj_sapm"); Z["sig_def"] = anti("d_sig_def")
    Z["tda15"] = anti("c_tda15"); Z["age"] = anti("age"); Z["replacement"] = anti("replacement")
    Z["missed_weight"] = anti("missed_weight"); Z["debut"] = anti("debut")
    Z["main_dog_band"] = (main & (pa >= .375) & (pa <= .45)).astype(float) - (main & (pb >= .375) & (pb <= .45)).astype(float)
    Z["longshot"] = (pa <= .2).astype(float) - (pb <= .2).astype(float)
    Z["steam"] = Lm - Lo
    Z["disagree_x_debut"] = (Lmod - Lm) * ((d.a_debut + d.b_debut) > 0).values
    X = np.column_stack([Lm, Lmod - Lm, Z.fillna(0).values])
    return X, Lm


class EdgeGLM:
    GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)

    def fit(self, X, y, off, years):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.mu[:2] = 0; self.sd[:2] = 1  # market log-odds and model-market gap stay on their natural scale
        self.sd[self.sd == 0] = 1
        Y = years.max() + 1
        inner_tr, inner_va = years < Y - 3, years >= Y - 3
        self.C = 0.003
        if inner_tr.sum() > 800 and inner_va.sum() > 300:
            best = np.inf
            for C in self.GRID:
                m = OffsetRidgeLogit(1 / C).fit(self._z(X[inner_tr]), y[inner_tr], off[inner_tr])
                p = np.clip(m.predict(self._z(X[inner_va]), off[inner_va]), 1e-6, 1 - 1e-6)
                l = -np.mean(y[inner_va] * np.log(p) + (1 - y[inner_va]) * np.log(1 - p))
                if l < best:
                    best, self.C = l, C
        self.m = OffsetRidgeLogit(1 / self.C).fit(self._z(X), y, off)
        return self

    def _z(self, X):
        return (X - self.mu) / self.sd

    def predict(self, X, off):
        return self.m.predict(self._z(X), off)


class MarketLayers:
    """stack + edge GLM + residual LightGBM around the closing line, and a stack on the opening line."""

    def fit(self, tr):
        tr = tr[tr.p_close.notna() & tr.stats.notna() & tr.y.notna()]
        self.fs = feature_sets(tr)
        self.stack = Stacker().fit(tr.p_close.values, tr.stats.values, tr.y.values)
        X, off = edge_design(tr)
        self.edge = EdgeGLM().fit(X, tr.y.values, off, tr.date.dt.year.values)
        self.train = tr  # residual LightGBM is refit inside predict (cheap) to reuse residual_lgbm
        o = tr[tr.p_open.notna()]
        self.stack_open = Stacker().fit(o.p_open.values, o.stats.values, o.y.values) if len(o) > 500 else None
        return self

    def predict(self, te):
        out = pd.DataFrame(index=te.index)
        c = te.p_close.notna() & te.stats.notna()
        if c.any():
            x = te[c]
            out.loc[x.index, "stack"] = self.stack.predict(x.p_close.values, x.stats.values)
            X, off = edge_design(x)
            out.loc[x.index, "edge_glm"] = self.edge.predict(X, off)
            keys = stats_keys(self.fs) + self.fs["mkt_hist"]
            out.loc[x.index, "resid"] = residual_lgbm(self.train, x, keys, self.fs["context"])
            out["ens3"] = out[["stack", "edge_glm", "resid"]].mean(axis=1)
        o = te.p_open.notna() & te.stats.notna()
        if self.stack_open is not None and o.any():
            out.loc[te.index[o], "stack_open"] = self.stack_open.predict(te.p_open[o].values, te.stats[o].values)
        return out


def market_layers(d, years_fit, years_pred):
    yrs = d.date.dt.year
    tr, te = d[yrs.isin(years_fit)], d[yrs.isin(years_pred)]
    if (tr.p_close.notna() & tr.stats.notna() & tr.y.notna()).sum() < 500 or te.empty:
        return pd.DataFrame(index=te.index)
    return MarketLayers().fit(tr).predict(te)


def full_walk_forward(d, start_year=2008, end=None, final=False, version="v2"):
    """Every bout from start_year: stats OOS (yearly retrain) then market layers from start_year + 3 (yearly).
    v2 adds p2_s0..p2_s9: the stage-aware engine probability with the line at each sparkline stage."""
    fs = feature_sets(d)
    end = end or ("2027-01-01" if final else "2025-01-01")
    d = d.copy()
    d["stats"] = walk_forward(d, lambda tr, te: stats_predict(tr, te, fs, version), start=f"{start_year}-01-01", end=end, final=final).reindex(d.index)
    if version == "v2":
        from ufc.market2 import MarketLayers2, stage_rows
        yrs = d.date.dt.year
        parts = []
        for Y in range(start_year + 3, pd.Timestamp(end).year):
            hist, cur = d[(yrs < Y) & d.stats.notna()], d[(yrs == Y) & d.stats.notna()]
            if cur.empty:
                continue
            rows = stage_rows(cur, range(10))
            P = MarketLayers2().fit(hist).predict(rows)
            parts.append(pd.DataFrame({"idx": rows.index, "stage": rows.stage.values, "p": P.ens.values}))
        W = pd.concat(parts).pivot_table(index="idx", columns="stage", values="p")
        W.columns = [f"p2_s{k}" for k in W.columns]
        return d.join(W)
    last = pd.Timestamp(end).year - 1
    layers = [market_layers(d, list(range(start_year, Y)), [Y]) for Y in range(start_year + 3, last + 1)]
    L = pd.concat([x for x in layers if len(x)])
    return d.join(L)


def policy_history(E, last_year, price="vig5", first_year=2013):
    """Walk-forward chosen staking policies; returns placed bets and per-year policy table."""
    E = add_prices(E)
    return walk_forward_policy(E, "ens3", f"a_{price}", f"b_{price}", first_year, last_year)


def add_prices(E, vig=5):
    E = E.copy()
    E[f"a_vig{vig}"] = 1 / (E.p_close * (1 + vig / 100)); E[f"b_vig{vig}"] = 1 / ((1 - E.p_close) * (1 + vig / 100))
    return E


def choose_policy(E, price_a, price_b, prob="ens3", grid=GRID, max_drawdown=None):
    """Best policy over ALL past out-of-sample bets (used for live recommendations).
    max_drawdown (units) rejects policies whose historical peak-to-trough loss exceeded it."""
    import itertools
    from ufc.betting import summary
    best, bv = None, -np.inf
    for v in itertools.product(*grid.values()):
        c = dict(zip(grid, v))
        b = simulate(E, prob, price_a, price_b, **c)
        if len(b) < 150:
            continue
        if max_drawdown is not None and summary(b)["max_dd"] > max_drawdown:
            continue
        val = np.sum(np.log1p(b.profit.values / 100))
        if val > bv:
            best, bv = c, val
    return best

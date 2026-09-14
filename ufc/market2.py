"""v2 market-aware layers: one engine for any point of a line's life (stage s: 0 = opening line, 1 = close).

Trained on several sparkline stages of every past bout at once, so the weight on the line versus the stats model
varies smoothly with the line's maturity and with how much UFC history the fighters have.
  stage_stack : symmetric logistic on logit(line), logit(stats) and their interactions with s / experience
  edge_glm    : offset ridge logistic around the line (corrections shrink to "trust the line")
  ens         : mean of the two (a residual LightGBM layer was tried on DEV and dropped: reports/v2_attempts.csv)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ufc.model import OffsetRidgeLogit, logit

TRAIN_STAGES = (0, 3, 6, 9)


def stage_rows(d, stages):
    """Stack bouts once per stage with the line at that stage: p_line, a_line_dec, b_line_dec, s."""
    out = []
    for k in stages:
        x = d.copy()
        x["p_line"], x["s"], x["stage"] = x[f"p_s{k}"], k / 9, k
        x["a_line_dec"], x["b_line_dec"] = x[f"a_s{k}_dec"], x[f"b_s{k}_dec"]
        out.append(x)
    x = pd.concat(out)
    return x[x.p_line.notna() & x.stats.notna()]


def _exp(x):
    n = np.minimum(x.a_n_fights.fillna(0).values, x.b_n_fights.fillna(0).values)
    debut = ((x.a_n_fights.fillna(0) == 0) | (x.b_n_fights.fillna(0) == 0)).values.astype(float)
    return np.log1p(n), debut


def stack_X(x):
    lm, lmod, s = logit(x.p_line.values), logit(x.stats.values), x.s.values
    ln, debut = _exp(x)
    return np.column_stack([lm, lm * s, lmod, lmod * s, lmod * debut, lmod * ln])


def edge_X(x):
    lm, lmod, s = logit(x.p_line.values), logit(x.stats.values), x.s.values
    lo = logit(x.p_open.fillna(x.p_line).values)
    ln, debut = _exp(x)
    anti = lambda c: (x[f"a_{c}"] - x[f"b_{c}"]).astype(float).fillna(0).values if f"a_{c}" in x else np.zeros(len(x))
    main = (x.bout_order <= 4).values
    pa, pb = x.p_line.values, 1 - x.p_line.values
    gap = lmod - lm
    cols = [lm, gap, gap * s, gap * debut, gap * ln, lm - lo, (lm - lo) * s,
            anti("sk_d_str"), anti("d_adj_sapm"), anti("d_sig_def"), anti("c_tda15"), anti("age"), anti("replacement"), anti("missed_weight"),
            anti("debut"), (main & (pa >= .375) & (pa <= .45)).astype(float) - (main & (pb >= .375) & (pb <= .45)).astype(float),
            (pa <= .2).astype(float) - (pb <= .2).astype(float)]
    return np.column_stack(cols)


class EdgeGLM2:
    GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)

    def fit(self, X, y, off, years):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.mu[:3] = 0; self.sd[:3] = 1
        self.sd[self.sd == 0] = 1
        Y = years.max() + 1
        itr, iva = years < Y - 3, years >= Y - 3
        self.C = 0.003
        if itr.sum() > 3000 and iva.sum() > 1000:
            best = np.inf
            for C in self.GRID:
                m = OffsetRidgeLogit(1 / C).fit(self._z(X[itr]), y[itr], off[itr])
                p = np.clip(m.predict(self._z(X[iva]), off[iva]), 1e-6, 1 - 1e-6)
                l = -np.mean(y[iva] * np.log(p) + (1 - y[iva]) * np.log(1 - p))
                if l < best:
                    best, self.C = l, C
        self.m = OffsetRidgeLogit(1 / self.C).fit(self._z(X), y, off)
        return self

    def _z(self, X):
        return (X - self.mu) / self.sd

    def predict(self, X, off):
        return self.m.predict(self._z(X), off)


class MarketLayers2:
    def __init__(self, stages=TRAIN_STAGES):
        self.stages = stages

    def fit(self, hist):
        tr = stage_rows(hist[hist.y.notna()], self.stages)
        y = tr.y.values
        X = stack_X(tr)
        self.stack = LogisticRegression(C=10.0, fit_intercept=False).fit(np.vstack([X, -X]), np.concatenate([y, 1 - y]))
        self.edge = EdgeGLM2().fit(edge_X(tr), y, logit(tr.p_line.values), tr.date.dt.year.values)
        return self

    def predict(self, x):
        """x: rows with p_line, s, stats, p_open and features (one row per bout-stage)."""
        out = pd.DataFrame(index=x.index)
        out["stage_stack"] = self.stack.predict_proba(stack_X(x))[:, 1]
        out["edge_glm"] = self.edge.predict(edge_X(x), logit(x.p_line.values))
        out["ens"] = out[["stage_stack", "edge_glm"]].mean(axis=1)
        return out

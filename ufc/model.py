"""Model definitions shared by research backtests and the live predictor."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from ufc.evaluate import fit_predict_sym, design, CONTEXT

EVENT_CONTEXT = ["apex", "no_crowd", "elevation", "high_altitude", "opp_stance", "h2h_n", "common_n"]
SHERDOG = ("pro_", "nonufc_", "gelo", "wins_over_ufc_level")
GEO = ("home_country", "travel_km", "tz_shift", "alt_gap")
WIKI = ("missed_weight", "replacement")
SKILLS = ("sk_",)
FM = ("fm_",)
PV = ("pv_",)
MKT_HIST = ("mr", "line_gap")  # derived from past betting lines: only for market-aware models
MARKET = ("p_open", "p_close")


def side_keys(d):
    return sorted(c[2:] for c in d.columns if c.startswith("a_") and "b_" + c[2:] in d.columns
                  and pd.api.types.is_numeric_dtype(d["a_" + c[2:]]))


def feature_sets(d):
    keys = [k for k in side_keys(d) if not k.startswith(("open_dec", "close_", "sd_linked", "glicko_p", "fm_src_bout"))
            and not k.endswith(("_dec",))]
    ufc = [k for k in keys if not k.startswith(SHERDOG + GEO + WIKI + SKILLS + FM + PV + MKT_HIST)]
    pick = lambda pref: [k for k in keys if k.startswith(pref)]
    ctx = CONTEXT + [c for c in EVENT_CONTEXT if c in d.columns]
    return dict(ufc=ufc, sherdog=pick(SHERDOG), geo=pick(GEO), wiki=pick(WIKI), skills=pick(SKILLS), fm=pick(FM), pv=pick(PV), mkt_hist=pick(MKT_HIST), context=ctx)


def lgbm_factory(**kw):
    params = dict(n_estimators=600, learning_rate=0.012, num_leaves=12, min_child_samples=120, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.35, reg_lambda=15, min_split_gain=0.0, verbose=-1)
    params.update(kw)
    return lambda: lgb.LGBMClassifier(**params)


def lr_factory(C=0.05):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def recency_weight(halflife_years):
    if not halflife_years:
        return None
    return lambda tr: 0.5 ** ((tr.date.max() - tr.date).dt.days / 365.25 / halflife_years)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Stacker:
    """Symmetric logistic stack: logit(p) = a*logit(p_market) + b*logit(p_model) (no intercept)."""

    def fit(self, pm, pmod, y):
        X = np.column_stack([logit(pm), logit(pmod)])
        X = np.vstack([X, -X]); yy = np.concatenate([y, 1 - y])
        self.m = LogisticRegression(C=10.0, fit_intercept=False).fit(X, yy)
        return self

    def predict(self, pm, pmod):
        return self.m.predict_proba(np.column_stack([logit(pm), logit(pmod)]))[:, 1]

    @property
    def coef(self):
        return self.m.coef_[0]


def residual_lgbm(train, test, keys, context, market_col="p_close", **kw):
    """LightGBM that starts from the market's log-odds and learns only corrections (symmetric)."""
    tr = train[train.y.notna() & train[market_col].notna()]
    Xtr = pd.concat([design(tr, keys, (), context), design(tr, keys, (), context, swap=True)])
    ytr = np.concatenate([tr.y.values, 1 - tr.y.values])
    base = np.concatenate([logit(tr[market_col].values), -logit(tr[market_col].values)])
    params = dict(n_estimators=150, learning_rate=0.01, num_leaves=6, min_child_samples=250, subsample=0.7,
                  subsample_freq=1, colsample_bytree=0.3, reg_lambda=30, verbose=-1)
    params.update(kw)
    m = lgb.LGBMClassifier(**params).fit(Xtr, ytr, init_score=base)
    te = test
    b1 = logit(te[market_col].values)
    r1 = m.predict(design(te, keys, (), context), raw_score=True)
    r2 = m.predict(design(te, keys, (), context, swap=True), raw_score=True)
    # raw_score excludes init_score; combine symmetric corrections around the market
    corr = (r1 - r2) / 2
    return 1 / (1 + np.exp(-(b1 + corr)))


class OffsetRidgeLogit:
    """Logistic regression with a fixed offset (the market's log-odds) and an L2 penalty that shrinks every
    correction toward ZERO, i.e. toward trusting the market. Symmetric data augmentation, no intercept."""

    def __init__(self, lam=1.0, iters=50):
        self.lam, self.iters = lam, iters

    def fit(self, X, y, offset):
        X = np.vstack([X, -X]); off = np.concatenate([offset, -offset]); yy = np.concatenate([y, 1 - y])
        b = np.zeros(X.shape[1])
        for _ in range(self.iters):
            eta = off + X @ b
            p = 1 / (1 + np.exp(-eta))
            g = X.T @ (p - yy) + self.lam * b
            H = (X * (p * (1 - p))[:, None]).T @ X + self.lam * np.eye(len(b))
            step = np.linalg.solve(H, g)
            b -= step
            if np.abs(step).max() < 1e-8:
                break
        self.coef_ = b
        return self

    def predict(self, X, offset):
        return 1 / (1 + np.exp(-(offset + X @ self.coef_)))

"""Pairing, walk-forward backtesting and scoring.

Protocol (fixed before any modelling):
  * DEV period       2010-01-01 .. 2020-12-31  -> free to explore / choose features and models
  * VALIDATION       2021-01-01 .. 2024-12-31  -> used sparingly to confirm choices
  * LOCKED HOLDOUT   2025-01-01 ..             -> scored once at the end (final=True)
Every prediction for date D comes from a model trained only on bouts dated < D's retraining cutoff.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HOLDOUT_START = pd.Timestamp("2025-01-01")
VALID_START = pd.Timestamp("2021-01-01")

CONTEXT = ["sched_rounds", "title", "women", "weight_lbs", "bout_order", "main_event", "n_bouts", "year"]


def make_pairs(fights, feats, profiles, extra=None):
    d = fights.merge(feats, on="fight_id")
    if extra is not None:
        d = d.merge(extra, on="fight_id", how="left")
    prof = profiles.set_index("fid")
    for side, col in (("a", "f1_id"), ("b", "f2_id")):
        p = prof.reindex(d[col])
        d[f"{side}_height"] = p.height_in.values
        d[f"{side}_reach"] = p.reach_in.values
        d[f"{side}_age"] = ((d.date.values - p.dob.values) / np.timedelta64(1, "D") / 365.25)
        d[f"{side}_ape"] = d[f"{side}_reach"] - d[f"{side}_height"]
        st = p.stance.fillna("Unknown").values
        d[f"{side}_southpaw"] = (st == "Southpaw").astype(float)
        d[f"{side}_switch"] = (st == "Switch").astype(float)
        d[f"{side}_age_debut"] = d[f"{side}_age"] - d[f"{side}_tenure_days"] / 365.25
    d["main_event"] = (d.bout_order == 0).astype(float)
    d["year"] = d.date.dt.year + d.date.dt.dayofyear / 366
    d["title"] = d.title.astype(float); d["women"] = d.women.astype(float)
    d["opp_stance"] = ((d.a_southpaw != d.b_southpaw) | (d.a_switch != d.b_switch)).astype(float)
    # antisymmetric pair-level features -> a_/b_ halves so that swapping sides negates them
    for c in ("h2h_score", "common_score"):
        if c in d:
            d[f"a_{c}"], d[f"b_{c}"] = d[c] / 2, -d[c] / 2
    if "glicko_p" in d:
        d["a_glicko_p"], d["b_glicko_p"] = d.glicko_p, 1 - d.glicko_p
    d["y"] = d.winner
    return d


def side_cols(d):
    return sorted(c[2:] for c in d.columns if c.startswith("a_") and "b_" + c[2:] in d.columns)


def design(d, keys, raw=(), context=CONTEXT, swap=False):
    """Diff features a-b (or b-a if swap) + chosen raw per-side values + symmetric context."""
    A, B = ("b", "a") if swap else ("a", "b")
    X = {f"d_{k}": d[f"{A}_{k}"].values - d[f"{B}_{k}"].values for k in keys}
    for k in raw:
        X[f"s1_{k}"] = d[f"{A}_{k}"].values
        X[f"s2_{k}"] = d[f"{B}_{k}"].values
    for c in context:
        X[c] = d[c].values
    return pd.DataFrame(X, index=d.index)


def fit_predict_sym(model_factory, train, test, keys, raw=(), context=CONTEXT, weight=None):
    """Train on both orientations; predict symmetric probability for side a."""
    tr = train[train.y.notna()]
    Xtr = pd.concat([design(tr, keys, raw, context), design(tr, keys, raw, context, swap=True)])
    ytr = np.concatenate([tr.y.values, 1 - tr.y.values])
    w = None if weight is None else np.concatenate([weight(tr), weight(tr)])
    m = model_factory()
    if w is None:
        m.fit(Xtr, ytr)
    else:
        m.fit(Xtr, ytr, sample_weight=w)
    p1 = m.predict_proba(design(test, keys, raw, context))[:, 1]
    p2 = m.predict_proba(design(test, keys, raw, context, swap=True))[:, 1]
    return (p1 + 1 - p2) / 2, m


def soft_target(d, q=0.7):
    """Winner label softened for near coin-flip results: split decisions and DQs count as q for the winner,
    majority decisions as (1 + q) / 2 (as selected on DEV in research/41)."""
    t = d.y.astype(float).copy()
    for methods, qq in ((["S-DEC", "DQ"], q), (["M-DEC"], (1 + q) / 2)):
        m = d.method.isin(methods) & d.y.notna()
        t[m] = np.where(d.y[m] == 1, qq, 1 - qq)
    return t


def fit_predict_soft(model_factory, train, test, keys, raw=(), context=CONTEXT, soft=None, pipeline=False):
    """Like fit_predict_sym, but each bout contributes both labels weighted by its soft target (soft=None: hard labels)."""
    tr = train[train.y.notna()]
    X1, X2 = design(tr, keys, raw, context), design(tr, keys, raw, context, swap=True)
    q = soft_target(tr, soft).values if soft else tr.y.values.astype(float)
    X = pd.concat([X1, X1, X2, X2])
    y = np.concatenate([np.ones(len(tr)), np.zeros(len(tr)), np.zeros(len(tr)), np.ones(len(tr))])
    w = np.concatenate([q, 1 - q, q, 1 - q])
    keep = w > 0
    m = model_factory()
    fit_kw = {("logisticregression__sample_weight" if pipeline else "sample_weight"): w[keep]}
    m.fit(X[keep], y[keep], **fit_kw)
    p1 = m.predict_proba(design(test, keys, raw, context))[:, 1]
    p2 = m.predict_proba(design(test, keys, raw, context, swap=True))[:, 1]
    return (p1 + 1 - p2) / 2


def walk_forward(d, predict_fn, start="2010-01-01", end=None, freq="YS", final=False, min_train_date="2001-01-01"):
    """predict_fn(train_df, test_df) -> probabilities for test rows. Retrains at each period start."""
    end = pd.Timestamp(end) if end else (d.date.max() + pd.Timedelta(days=1) if final else HOLDOUT_START)
    if not final and end > HOLDOUT_START:
        raise ValueError("holdout is locked; pass final=True only for the final evaluation")
    cuts = list(pd.date_range(start, end, freq=freq)) + [end]
    cuts = sorted(set(c for c in cuts if c <= end))
    preds = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        test = d[(d.date >= lo) & (d.date < hi)]
        if test.empty:
            continue
        train = d[(d.date < lo) & (d.date >= min_train_date)]
        p = predict_fn(train, test)
        preds.append(pd.Series(p, index=test.index))
    return pd.concat(preds)


def scores(y, p):
    ok = y.notna() & p.notna()
    y, p = y[ok].values.astype(float), np.clip(p[ok].values, 1e-6, 1 - 1e-6)
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    acc = np.mean((p > 0.5) == (y == 1)) if len(y) else np.nan
    ece = calib_error(y, p)
    return dict(n=len(y), logloss=round(ll, 4), brier=round(np.mean((p - y) ** 2), 4), acc=round(acc, 4),
                auc=round(roc_auc_score(y, p), 4) if len(set(y)) > 1 else np.nan, ece=round(ece, 4))


def calib_error(y, p, bins=10):
    """Expected calibration error on the favourite's probability (orientation free)."""
    q = np.where(p >= 0.5, p, 1 - p)
    yy = np.where(p >= 0.5, y, 1 - y)
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(q, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(yy[m].mean() - q[m].mean())
    return e


def calibration_table(y, p, bins=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0)):
    ok = y.notna() & p.notna()
    y, p = y[ok].values, p[ok].values
    q = np.where(p >= 0.5, p, 1 - p); yy = np.where(p >= 0.5, y, 1 - y)
    t = pd.DataFrame({"q": q, "won": yy})
    t["bin"] = pd.cut(t.q, bins, include_lowest=True)
    return t.groupby("bin", observed=True).agg(n=("won", "size"), predicted=("q", "mean"), actual=("won", "mean")).round(3)


if __name__ == "__main__":
    y = pd.Series([1, 0, 1, 1.0]); p = pd.Series([0.9, 0.2, 0.6, 0.4])
    s = scores(y, p)
    assert s["acc"] == 0.75 and 0 < s["logloss"] < 1
    print("evaluate ok", s)

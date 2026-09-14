"""v2 / market-free stats model variants on DEV (walk-forward yearly, scored 2010-2020).

  python research/41_stats_variants.py base soft_s65 ...

Each variant's out-of-sample predictions are cached in data/v2/stats_<name>.parquet and compared with `base`
(the v1 stats model) by an event-clustered paired bootstrap. Every run is appended to reports/v2_attempts.csv.
"""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.fetch import ROOT
from ufc.evaluate import walk_forward, design
from ufc.model import feature_sets, lgbm_factory, lr_factory
from ufc.engine import lgbm_params, stats_keys, RAW

DATA, V2 = ROOT / "data", ROOT / "data" / "v2"
V2.mkdir(exist_ok=True)
LOG = ROOT / "reports" / "v2_attempts.csv"
START, END = "2010-01-01", "2021-01-01"


def fit_predict_soft(factory, train, test, keys, raw, context, q_col=None, weight=None, lr=False):
    """Symmetric fit; with q_col, each bout contributes both labels weighted by its soft label q (prob side a won)."""
    tr = train[train.y.notna()]
    X1, X2 = design(tr, keys, raw, context), design(tr, keys, raw, context, swap=True)
    q = tr[q_col].values if q_col else tr.y.values
    w0 = np.ones(len(tr)) if weight is None else weight(tr)
    X = pd.concat([X1, X1, X2, X2])
    y = np.concatenate([np.ones(len(tr)), np.zeros(len(tr)), np.zeros(len(tr)), np.ones(len(tr))])
    w = np.concatenate([q, 1 - q, q, 1 - q]) * np.tile(w0, 4)
    keep = w > 0
    m = factory()
    if lr:
        m.fit(X[keep], y[keep], logisticregression__sample_weight=w[keep])
    else:
        m.fit(X[keep], y[keep], sample_weight=w[keep])
    p1 = m.predict_proba(design(test, keys, raw, context))[:, 1]
    p2 = m.predict_proba(design(test, keys, raw, context, swap=True))[:, 1]
    return (p1 + 1 - p2) / 2


def stats_model(keys_fn=None, q_col=None, lr_C=0.05, w_lgbm=0.5, lgbm_over=None, seeds=(None,)):
    def predict(train, test, d, fs):
        cfg, hl = lgbm_params()
        cfg = {**cfg, **(lgbm_over or {})}
        keys = keys_fn(fs, d) if keys_fn else stats_keys(fs)
        gs = [fit_predict_soft(lgbm_factory(**cfg, subsample_freq=1, **({} if s is None else {"random_state": s})), train, test, keys, RAW, fs["context"], q_col)
              for s in seeds]
        g = np.mean(gs, axis=0)
        if w_lgbm >= 1:
            return g
        l = fit_predict_soft(lr_factory(lr_C), train, test, keys, (), [c for c in fs["context"] if c != "year"], q_col, lr=True)
        return w_lgbm * g + (1 - w_lgbm) * l
    return predict


def soft_labels(d, q_split, q_major=None, q_dq=None):
    q = d.y.astype(float).copy()
    for meth, qq in (("S-DEC", q_split), ("M-DEC", q_major if q_major is not None else (1 + q_split) / 2), ("DQ", q_dq if q_dq is not None else q_split)):
        m = (d.method == meth) & d.y.notna()
        q[m] = np.where(d.y[m] == 1, qq, 1 - qq)
    return q


def load_table():
    d = pd.read_parquet(DATA / "model_table.parquet")
    extra = sorted(V2.glob("features_*.parquet"))  # new feature blocks merged by fight_id (a_/b_ columns)
    for p in extra:
        d = d.merge(pd.read_parquet(p), on="fight_id", how="left")
    return d


def parse_variant(name, d):
    """Composable variant names, e.g. 'C0.003+soft60+promo+w0.6'. 'base' = v1 stats model."""
    kw, extra = {}, []
    for tok in name.split("+"):
        if tok == "base":
            continue
        elif tok.startswith("C"):
            kw["lr_C"] = float(tok[1:])
        elif tok.startswith("w"):
            kw["w_lgbm"] = float(tok[1:])
        elif tok.startswith("soft"):
            kw["q_col"] = "q" + tok[4:]
        elif tok.startswith("bag"):
            kw["seeds"] = tuple(range(1, int(tok[3:]) + 1))
        elif tok == "lgbm_only":
            kw["w_lgbm"] = 1.0
        elif tok in BLOCKS:
            extra.append(BLOCKS[tok])
        else:
            raise ValueError(tok)
    if extra:
        kw["keys_fn"] = lambda fs, d: stats_keys(fs) + [k for k in feature_sets(d)["ufc"] if k.startswith(tuple(extra))]
    return stats_model(**kw)


BLOCKS = {"promo": "pr_"}  # feature blocks built into data/v2/features_*.parquet, excluded from 'base'
LEGACY = {"lr_C001": "C0.01", "lr_C0003": "C0.003", "lr_C005_w70": "w0.7", "soft_s60": "soft60", "soft_s70": "soft70", "soft_s80": "soft80"}


def run_variant(name, d):
    path = V2 / f"stats_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)["p"]
    fs = feature_sets(d)
    fs["ufc"] = [k for k in fs["ufc"] if not k.startswith(tuple(BLOCKS.values()))]  # new blocks only enter through their variant
    fn = parse_variant(LEGACY.get(name, name), d)
    t = time.time()
    p = walk_forward(d, lambda tr, te: fn(tr, te, d, fs), start=START, end=END)
    pd.DataFrame({"p": p, "fight_id": d.loc[p.index, "fight_id"]}).to_parquet(path)
    print(f"{name}: {time.time() - t:.0f}s", flush=True)
    return p


def ll_vec(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare(d, base, var, n_boot=2000, seed=0):
    idx = base.index.intersection(var.index)
    x = d.loc[idx]
    ok = x.y.notna().values
    lb, lv = ll_vec(base[idx].values[ok], x.y.values[ok]), ll_vec(var[idx].values[ok], x.y.values[ok])
    ev = x.event_id.values[ok]
    g = pd.DataFrame({"ev": ev, "diff": lb - lv}).groupby("ev")["diff"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    k = len(g)
    boots = []
    s, n = g["sum"].values, g["size"].values
    for _ in range(n_boot):
        i = rng.integers(0, k, k)
        boots.append(s[i].sum() / n[i].sum())
    return dict(n=int(ok.sum()), base=lb.mean(), var=lv.mean(), gain=lb.mean() - lv.mean(), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5))


def log_attempt(layer, change, r, accepted, notes=""):
    ids = pd.read_csv(LOG)
    row = dict(id=len(ids) + 1, date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), layer=layer, change=change, dev_metric="logloss 2010-2020",
               v1=round(r["base"], 5), v2=round(r["var"], 5), gain=round(r["gain"], 5), ci_lo=round(r["lo"], 5), ci_hi=round(r["hi"], 5),
               accepted=accepted, notes=notes)
    pd.concat([ids, pd.DataFrame([row])]).to_csv(LOG, index=False)


if __name__ == "__main__":
    d = load_table()
    for qq in (60, 70, 80):
        d[f"q{qq}"] = soft_labels(d, qq / 100)
    args = sys.argv[1:]
    ref = "base"
    if args and args[0].startswith("--ref="):
        ref = args.pop(0).split("=", 1)[1]
    base = run_variant(ref, d)
    for name in args:
        if name == ref:
            continue
        v = run_variant(name, d)
        r = compare(d, base, v)
        acc = "yes" if r["lo"] > 0 else "no"
        print(f"{name} vs {ref}: n={r['n']} ref {r['base']:.5f} var {r['var']:.5f} gain {r['gain']:+.5f} CI [{r['lo']:+.5f}, {r['hi']:+.5f}] accepted={acc}", flush=True)
        log_attempt("stats", name if ref == "base" else f"{name} vs {ref}", r, acc)

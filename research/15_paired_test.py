"""Paired per-bout log-loss comparison (bootstrap CI, per-year consistency) between a model and a baseline."""
import sys
import numpy as np, pandas as pd
from ufc.fetch import ROOT

def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def compare(P, a, b, lo, hi):
    m = P[a].notna() & P[b].notna() & P.y.notna() & (P.date >= lo) & (P.date < hi)
    dif = ll(P.y[m], P[b][m]) - ll(P.y[m], P[a][m])  # positive = a better
    rng = np.random.default_rng(0)
    boots = [dif.values[rng.integers(0, len(dif), len(dif))].mean() for _ in range(4000)]
    by = pd.Series(dif.values).groupby(P.date[m].dt.year.values).mean()
    return dict(model=a, baseline=b, n=int(m.sum()), base_ll=round(ll(P.y[m], P[b][m]).mean(), 4), model_ll=round(ll(P.y[m], P[a][m]).mean(), 4),
                gain=round(dif.mean(), 4), ci95=f"[{np.percentile(boots, 2.5):.4f}, {np.percentile(boots, 97.5):.4f}]",
                t=round(dif.mean() / dif.std() * np.sqrt(len(dif)), 2), years_better=f"{(by > 0).sum()}/{len(by)}")

if __name__ == "__main__":
    f = sys.argv[1]; lo, hi = sys.argv[2], sys.argv[3]
    P = pd.read_parquet(ROOT / "data" / f)
    pairs = [x.split(":") for x in sys.argv[4:]]
    print(pd.DataFrame([compare(P, a, b, lo, hi) for a, b in pairs]).to_string(index=False))

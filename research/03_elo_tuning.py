"""Tune win-Elo hyperparameters on 2005-2014 only (frozen afterwards). Score = log loss of the raw Elo
expectation after a 1-parameter logistic rescale fitted on 1994-2004 + the tuning window itself is NOT used:
we rescale on the preceding years to stay honest."""
import itertools, math
import numpy as np, pandas as pd
f = pd.read_parquet("data/fights.parquet")
f = f.sort_values(["date", "event_id", "bout_order"], ascending=[True, True, False])

def run(k=40, init_debut=1500, mov=1.0, layoff_regress=0.0, new_k_boost=0.0, title_mult=1.0):
    r, n, last = {}, {}, {}
    out = []
    for date, day in f.groupby("date", sort=True):
        pre = []
        for x in day.itertuples():
            ra = r.get(x.f1_id, init_debut); rb = r.get(x.f2_id, init_debut)
            # regress toward 1500 in proportion to layoff years
            if layoff_regress and x.f1_id in last: ra = 1500 + (ra - 1500) * (1 - layoff_regress) ** ((date - last[x.f1_id]).days / 365)
            if layoff_regress and x.f2_id in last: rb = 1500 + (rb - 1500) * (1 - layoff_regress) ** ((date - last[x.f2_id]).days / 365)
            pre.append((x, ra, rb))
            out.append((x.date, x.winner, ra - rb))
        for x, ra, rb in pre:
            s = 0.5 if x.outcome == "D/D" else x.winner
            last[x.f1_id] = last[x.f2_id] = date
            if s != s:
                r[x.f1_id], r[x.f2_id] = ra, rb; continue
            e = 1 / (1 + 10 ** ((rb - ra) / 400))
            m = (mov if x.method in ("KO", "DOC", "SUB") else 1.0) * (title_mult if x.title else 1.0)
            ka = k * m * (1 + new_k_boost / (1 + n.get(x.f1_id, 0))); kb = k * m * (1 + new_k_boost / (1 + n.get(x.f2_id, 0)))
            r[x.f1_id] = ra + ka * (s - e); r[x.f2_id] = rb - kb * (s - e)
            n[x.f1_id] = n.get(x.f1_id, 0) + 1; n[x.f2_id] = n.get(x.f2_id, 0) + 1
    return pd.DataFrame(out, columns=["date", "y", "diff"])

def score(df, lo, hi):
    tr = df[(df.date < lo) & (df.date >= "1998") & df.y.notna()]
    te = df[(df.date >= lo) & (df.date < hi) & df.y.notna()]
    # 1-parameter slope (symmetric) by grid on the preceding data
    best = min(np.linspace(0.0005, 0.01, 40), key=lambda b: -np.mean(np.log(np.where(tr.y == 1, 1 / (1 + np.exp(-b * tr["diff"])), 1 - 1 / (1 + np.exp(-b * tr["diff"]))))))
    p = 1 / (1 + np.exp(-best * te["diff"]))
    return -np.mean(np.log(np.where(te.y == 1, p, 1 - p))), best

rows = []
grid = dict(k=[20, 40, 60, 90], init_debut=[1450, 1500], mov=[1.0, 1.5], layoff_regress=[0.0, 0.1], new_k_boost=[0.0, 2.0])
for vals in itertools.product(*grid.values()):
    cfg = dict(zip(grid.keys(), vals))
    df = run(**cfg)
    ll, b = score(df, pd.Timestamp("2005-01-01"), pd.Timestamp("2015-01-01"))
    ll2, _ = score(df, pd.Timestamp("2015-01-01"), pd.Timestamp("2021-01-01"))
    rows.append({**cfg, "ll_tune_05_14": round(ll, 4), "ll_dev_15_20": round(ll2, 4), "slope": round(b, 5)})
out = pd.DataFrame(rows).sort_values("ll_tune_05_14")
out.to_csv("reports/03_elo_tuning.csv", index=False)
print(out.head(12).to_string(index=False)); print(out.tail(5).to_string(index=False))

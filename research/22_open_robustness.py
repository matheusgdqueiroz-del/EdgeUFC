"""Addendum A robustness checks for opening-line betting (O4)."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.fetch import ROOT
from ufc.evaluate import walk_forward
from ufc.model import feature_sets, Stacker
from ufc.engine import stats_predict
from ufc.betting import walk_forward_policy, summary

mode = sys.argv[1] if len(sys.argv) > 1 else "valid"
final = mode == "final"
end = "2027-01-01" if final else "2025-01-01"
E = pd.read_parquet(ROOT / "data" / f"engine_oos_{mode}.parquet")
d = pd.read_parquet(ROOT / "data" / "model_table.parquet")
d["a_missed_weight"] = 0.0; d["b_missed_weight"] = 0.0
fs = feature_sets(d)
d["stats"] = walk_forward(d, lambda tr, te: stats_predict(tr, te, fs), start="2008-01-01", end=end, final=final).reindex(d.index)
yrs = d.date.dt.year
d["stack_open_noweigh"] = np.nan
for Y in range(2011, pd.Timestamp(end).year):
    tr = (yrs < Y) & d.p_open.notna() & d.stats.notna() & d.y.notna()
    te = (yrs == Y) & d.p_open.notna() & d.stats.notna()
    if tr.sum() > 500 and te.sum():
        d.loc[te, "stack_open_noweigh"] = Stacker().fit(d.p_open[tr].values, d.stats[tr].values, d.y[tr].values).predict(d.p_open[te].values, d.stats[te].values)
E["stack_open_noweigh"] = d.stack_open_noweigh.reindex(E.index)
last = int(E.date.max().year) if final else 2024
periods = [("VALIDATION 2021-2024", "2021-01-01", "2025-01-01")] + ([("LOCKED HOLDOUT 2025-01-01..", "2025-01-01", "2027-01-01")] if final else [])
rows = []
clean = E.open_overround.between(1.02, 1.10) & ((E.p_open - E.p_close).abs() < 0.15)
for label, prob, sub in [("O4 stack_open @open", "stack_open", E), ("O4-noweigh", "stack_open_noweigh", E), ("O4-clean", "stack_open", E[clean]),
                         ("O4-noweigh + clean", "stack_open_noweigh", E[clean])]:
    bets, _ = walk_forward_policy(sub, prob, "a_open_dec", "b_open_dec", 2013, last)
    for name, lo, hi in periods:
        bb = bets[(bets.date >= lo) & (bets.date < hi)]
        s = summary(bb); s.update(variant=label, period=name)
        if len(bb) > 30:
            rng = np.random.default_rng(2); pr, st = bb.profit.values, bb.stake.values
            boots = [pr[i].sum() / st[i].sum() for i in (rng.integers(0, len(pr), len(pr)) for _ in range(3000))]
            s["roi_ci90"] = f"[{np.percentile(boots, 5):.3f}, {np.percentile(boots, 95):.3f}]"
            close = np.where(bb.bet_side == "a", E.loc[bb.index, "a_close_mean_dec"], E.loc[bb.index, "b_close_mean_dec"])
            s["mean_clv"] = round(float(np.nanmean(bb.price / close - 1)), 4)
        rows.append(s)
T = pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv(ROOT / "reports" / f"22_open_robustness_{mode}.csv", index=False)

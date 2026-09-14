"""POST-HOC (after holdout): staking with a drawdown cap, walk-forward, vs the pre-registered growth-optimal rule."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.fetch import ROOT
from ufc.engine import add_prices
from ufc.betting import walk_forward_policy, summary, GRID

E = add_prices(pd.read_parquet(ROOT / "data" / "engine_oos_final.parquet"))
last = int(E.date.max().year)
grid = dict(min_edge=[0.02, 0.04, 0.06, 0.08, 0.12], frac=[0.1, 0.15, 0.25, 0.5], max_units=[1.0, 2.0, 3.0, 5.0])
rows, curves = [], {}
for label, prob, a, b in [("near close, 5% margin", "ens3", "a_vig5", "b_vig5"), ("at opening price", "stack_open", "a_open_dec", "b_open_dec")]:
    for cap in (None, 25.0, 15.0):
        bets, ch = walk_forward_policy(E[E[prob].notna()], prob, a, b, 2013, last, grid=grid, max_drawdown=cap)
        curves[(label, cap)] = bets
        for name, lo, hi in [("2013-2020", "2013", "2021"), ("2021-2024", "2021", "2025"), ("2025-2026", "2025", "2027"), ("2021-2026 (untouched)", "2021", "2027")]:
            bb = bets[(bets.date >= lo) & (bets.date < hi)]
            s = summary(bb); s.update(setting=label, dd_cap=cap or "none", period=name); rows.append(s)
        print(label, cap, ch[["year", "min_edge", "frac", "max_units"]].tail(3).to_dict("records"))
T = pd.DataFrame(rows)
print(T.to_string(index=False)); T.to_csv(ROOT / "reports" / "23_risk_capped_staking.csv", index=False)
pd.concat({f"{k[0]}|{k[1]}": v[["date", "stake", "price", "profit", "p_bet", "bet_side"]] for k, v in curves.items()}).to_csv(ROOT / "reports" / "23_bets_by_setting.csv")

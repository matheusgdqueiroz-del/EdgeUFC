"""Bankroll plans for betting the moment lines open (user's plan), 5-year horizon, bet limit R$5,000.

Same resampling as research/51 (whole months of 2021-Sep 2026 out-of-sample bets, 4,000 paths).
Price cases: as tested (opening line, estimated Pinnacle margin 4.5%) and 3% worse prices (margin 7.5%), in case
Pinnacle's own opener is sharper than the first book's opener that the backtest used.
"""
import importlib.util, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.fetch import ROOT

spec = importlib.util.spec_from_file_location("plans51", ROOT / "research" / "51_bankroll_plans.py")
plans = importlib.util.module_from_spec(spec); spec.loader.exec_module(plans)

CAP = 5000.0
rows = []
for case, margin in (("as tested", 1.045), ("prices 3% worse", 1.075)):
    U, X, blocks = plans.month_blocks(0, margin)
    for start in (300, 500):
        for k in (1, 2, 3, 4, 5, 6, 8, 10):
            B, low, snaps, _ = plans.simulate(U, X, blocks, k, start, months=60, max_stake=CAP, years=(1, 2, 3, 5))
            rows.append(dict(prices=case, start=start, unit_per_100=k,
                             y1_typical=round(np.median(snaps[1])), y2_typical=round(np.median(snaps[2])), y3_typical=round(np.median(snaps[3])),
                             y5_typical=round(np.median(B)), y5_bad_luck=round(np.percentile(B, 10)), y5_good_luck=round(np.percentile(B, 90)),
                             down_after_1y=round((snaps[1] < start).mean() * 100, 1), ever_lost_half=round((low <= start * 0.5).mean() * 100, 1),
                             ever_lost_80=round((low <= start * 0.2).mean() * 100, 1), reach_10k_3y=round((snaps[3] >= 10_000).mean() * 100, 1),
                             reach_100k_5y=round((B >= 100_000).mean() * 100, 1)))
    print(case, "done", flush=True)
R = pd.DataFrame(rows)
R.to_csv(ROOT / "reports" / "52_bankroll_plans_early.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(R.to_string(index=False))

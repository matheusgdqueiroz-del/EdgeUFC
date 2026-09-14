"""Would the predictor have made money? Walk-forward staking with honest prices.

For each scenario (probability source x price):
  * each year's staking policy (min edge, Kelly fraction, max units) is chosen ONLY from simulated bets in
    earlier years (out-of-sample predictions throughout)
  * bets are then settled on that year's results
Scenarios
  close_mean : bet just before the fight at the average closing price (realistic single-book bettor)
  close_best : best closing price across books (line shopping; optimistic)
  open       : bet when lines open at the opening price, using the opening market as the stack input
Also reports closing-line value (CLV) for opening bets and bootstrap confidence intervals.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.betting import simulate, summary, walk_forward_policy

period = sys.argv[1] if len(sys.argv) > 1 else "dev"
P = pd.read_parquet(f"data/oos_{period}.parquet")
d = pd.read_parquet("data/model_table.parquet").loc[P.index]
E = pd.read_parquet(f"data/edge_{period}.parquet")
for c in ("stack", "edge_glm", "resid_lgbm", "ens3"):
    P[f"edge_{c}"] = E[c].reindex(P.index)
for c in ["a_close_mean_dec", "b_close_mean_dec", "a_close_best_dec", "b_close_best_dec", "a_open_dec", "b_open_dec", "outcome"]:
    P[c] = d[c]
first, last = (2013, 2020) if period == "dev" else (2021, 2024)
P = P[P.date.dt.year <= last]

scen = [
    ("model_only@close_mean", "ens", "a_close_mean_dec", "b_close_mean_dec"),
    ("stack_close@close_mean", "stack_ens_p_close", "a_close_mean_dec", "b_close_mean_dec"),
    ("stack_close@close_best", "stack_ens_p_close", "a_close_best_dec", "b_close_best_dec"),
    ("resid_close@close_mean", "resid_close", "a_close_mean_dec", "b_close_mean_dec"),
    ("edge_glm@close_mean", "edge_edge_glm", "a_close_mean_dec", "b_close_mean_dec"),
    ("edge_ens3@close_mean", "edge_ens3", "a_close_mean_dec", "b_close_mean_dec"),
    ("edge_ens3@close_best", "edge_ens3", "a_close_best_dec", "b_close_best_dec"),
    ("stack_open@open", "stack_ens_p_open", "a_open_dec", "b_open_dec"),
    ("market_close@close_mean (control: should lose ~vig)", "p_close", "a_close_mean_dec", "b_close_mean_dec"),
]
rows, all_bets = [], {}
for name, pcol, ad, bd in scen:
    if pcol not in P:
        continue
    bets, chosen = walk_forward_policy(P, pcol, ad, bd, first, last)
    all_bets[name] = bets
    s = summary(bets); s["scenario"] = name
    if len(bets) > 30:  # bootstrap ROI CI over bets
        rng = np.random.default_rng(0); pr, st = bets.profit.values, bets.stake.values
        boots = [pr[i].sum() / st[i].sum() for i in (rng.integers(0, len(pr), len(pr)) for _ in range(2000))]
        s["roi_ci90"] = f"[{np.percentile(boots, 5):.3f}, {np.percentile(boots, 95):.3f}]"
    rows.append(s)
    print(f"\n### {name}\n", chosen.to_string(index=False), flush=True)
res = pd.DataFrame(rows).set_index("scenario")
print(f"\n=== walk-forward betting {first}-{last} (units; 1u = 1% bankroll) ===")
print(res.to_string())
res.to_csv(f"reports/09_betting_{period}.csv")

# CLV: did opening-line bets beat the closing line?
b = all_bets.get("stack_open@open")
if b is not None and len(b):
    close = np.where(b.bet_side == "a", P.loc[b.index, "a_close_mean_dec"], P.loc[b.index, "b_close_mean_dec"])
    clv = b.price / close - 1
    print(f"\nopening bets: mean CLV {np.nanmean(clv):+.3%}, share beating close {np.nanmean(clv > 0):.2f}  (positive CLV = the market moved toward our side)")

# hindsight sensitivity grid (NOT a valid performance estimate; shows how fragile the policy is)
if "stack_ens_p_close" in P:
    grid = []
    for me in (0.0, 0.03, 0.06, 0.1):
        for fr in (0.1, 0.25, 0.5):
            s = summary(simulate(P[P.date.dt.year >= first], "stack_ens_p_close", "a_close_mean_dec", "b_close_mean_dec", min_edge=me, frac=fr, max_units=5))
            grid.append(dict(min_edge=me, frac=fr, **s))
    print("\nhindsight grid (for fragility only):"); print(pd.DataFrame(grid).to_string(index=False))

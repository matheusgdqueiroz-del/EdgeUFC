"""Pre-registered evaluation (see reports/PREREGISTRATION.md). Runs the PRODUCTION engine (ufc/engine.py).

  python research/20_final_evaluation.py valid   -> walk-forward through 2024, scores VALIDATION 2021-2024
  python research/20_final_evaluation.py final   -> walk-forward through today, scores the LOCKED HOLDOUT too,
                                                    and saves data/engine_oos.parquet for live predictions
"""
import sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.fetch import ROOT
from ufc.evaluate import walk_forward, scores, calibration_table, fit_predict_sym
from ufc.model import lr_factory
from ufc.engine import full_walk_forward, add_prices
from ufc.betting import walk_forward_policy, summary
sys.path.insert(0, str(ROOT / "research"))
from importlib import import_module
paired = import_module("15_paired_test").compare

mode = sys.argv[1] if len(sys.argv) > 1 else "valid"
final = mode == "final"
D = ROOT / "data"; R = ROOT / "reports"
d = pd.read_parquet(D / "model_table.parquet")
end = "2027-01-01" if final else "2025-01-01"
E = full_walk_forward(d, final=final, end=end)
E["baseline_elo_age"] = walk_forward(d, lambda tr, te: fit_predict_sym(lr_factory(1.0), tr, te, ["elo_tuned", "age"], (), [])[0],
                                     start="2008-01-01", end=end, final=final).reindex(E.index)
E.to_parquet(D / f"engine_oos_{mode}.parquet")
if final:
    E.to_parquet(D / "engine_oos.parquet")

periods = [("DEV 2011-2020 (selection-biased)", "2011-01-01", "2021-01-01"), ("VALIDATION 2021-2024", "2021-01-01", "2025-01-01")]
if final:
    periods.append(("LOCKED HOLDOUT 2025-01-01..", "2025-01-01", "2027-01-01"))
out = {}
lines = []
for name, lo, hi in periods:
    m = (E.date >= lo) & (E.date < hi) & E.p_close.notna() & E.ens3.notna() & E.y.notna()
    sc = pd.DataFrame([dict(model=c, **scores(E.y[m], E[c][m])) for c in ["p_close", "p_open", "ens3", "stack", "edge_glm", "resid", "stack_open", "stats", "baseline_elo_age"]]).set_index("model")
    pt = pd.DataFrame([paired(E[m], a, b, lo, hi) for a, b in [("ens3", "p_close"), ("stack", "p_close"), ("stack_open", "p_open"), ("stats", "baseline_elo_age"), ("stats", "p_close")]])
    lines += [f"\n##### {name}", sc.to_string(), pt.to_string(index=False), "calibration ens3:", calibration_table(E.y[m], E.ens3[m]).to_string()]
    out[name] = dict(scores=sc.reset_index().to_dict("records"), paired=pt.to_dict("records"))

# betting: walk-forward policies, settled per period
B = add_prices(E[E.ens3.notna()])
last_year = int(E.date.max().year) if final else 2024
bet_rows = []
for label, prob, a, b in [("ens3 @ 5% margin book (O3)", "ens3", "a_vig5", "b_vig5"), ("ens3 @ avg closing price", "ens3", "a_close_mean_dec", "b_close_mean_dec"),
                          ("ens3 @ best closing price", "ens3", "a_close_best_dec", "b_close_best_dec"), ("stack_open @ opening price (O4)", "stack_open", "a_open_dec", "b_open_dec")]:
    bets, chosen = walk_forward_policy(B, prob, a, b, 2013, last_year)
    chosen.to_csv(R / f"20_policy_{mode}_{prob}_{a}.csv", index=False)
    for name, lo, hi in periods:
        bb = bets[(bets.date >= lo) & (bets.date < hi)] if len(bets) else bets
        s = summary(bb); s.update(scenario=label, period=name)
        if len(bb) > 30:
            rng = np.random.default_rng(1); pr, st = bb.profit.values, bb.stake.values
            boots = [pr[i].sum() / st[i].sum() for i in (rng.integers(0, len(pr), len(pr)) for _ in range(3000))]
            s["roi_ci90"] = f"[{np.percentile(boots, 5):.3f}, {np.percentile(boots, 95):.3f}]"
        if "open" in label and len(bb):
            close = np.where(bb.bet_side == "a", B.loc[bb.index, "a_close_mean_dec"], B.loc[bb.index, "b_close_mean_dec"])
            s["mean_clv"] = round(float(np.nanmean(bb.price / close - 1)), 4)
        bet_rows.append(s)
    bets.to_csv(R / f"20_bets_{mode}_{prob}_{a}.csv")
BT = pd.DataFrame(bet_rows)
lines += ["\n##### BETTING (walk-forward policies; units = % of bankroll)", BT.to_string(index=False)]
txt = "\n".join(lines)
print(txt)
open(R / f"20_final_evaluation_{mode}.txt", "w", encoding="utf8").write(txt)
BT.to_csv(R / f"20_betting_{mode}.csv", index=False)

"""Where does the predictor succeed / fail? Stability over time, experience levels, divisions, confidence,
agreement with the market. Uses only out-of-sample walk-forward predictions (data/oos_<period>.parquet)."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.evaluate import scores, calibration_table

period = sys.argv[1] if len(sys.argv) > 1 else "dev"
col = sys.argv[2] if len(sys.argv) > 2 else "ens"
P = pd.read_parquet(f"data/oos_{period}.parquet")
d = pd.read_parquet("data/model_table.parquet").set_index(P.index.name) if P.index.name else pd.read_parquet("data/model_table.parquet")
d = d.loc[P.index]
lo, hi = ("2010", "2021") if period == "dev" else ("2021", "2025")
m = (P.date >= lo) & (P.date < hi) & P[col].notna() & P.y.notna()
P, d = P[m], d[m]
mk = "p_close" if "p_close" in P else None


def table(groups, name):
    rows = []
    for g, idx in groups:
        if idx.sum() < 40:
            continue
        s = scores(P.y[idx], P[col][idx]); s["group"] = g
        if mk:
            k = idx & P[mk].notna()
            sm = scores(P.y[k], P[mk][k]) if k.sum() > 40 else {}
            s["market_logloss"] = sm.get("logloss"); s["market_acc"] = sm.get("acc")
        rows.append(s)
    t = pd.DataFrame(rows).set_index("group")
    print(f"\n--- by {name} ---"); print(t.to_string())
    return t


out = {}
out["year"] = table([(y, P.date.dt.year == y) for y in sorted(P.date.dt.year.unique())], "year")
exp = np.minimum(d.a_n_fights, d.b_n_fights)
out["experience"] = table([("both debut", (d.a_n_fights == 0) & (d.b_n_fights == 0)),
                           ("one debutant", ((d.a_n_fights == 0) ^ (d.b_n_fights == 0))),
                           ("min 1-2 UFC fights", exp.between(1, 2)), ("min 3-5", exp.between(3, 5)), ("min 6+", exp >= 6)], "UFC experience")
out["division"] = table([(w, d.weightclass == w) for w in d.weightclass.value_counts().index[:12]] +
                        [("women", d.women == 1), ("men", d.women == 0)], "division")
q = np.maximum(P[col], 1 - P[col])
out["confidence"] = table([(f"{a:.2f}-{b:.2f}", (q >= a) & (q < b)) for a, b in [(.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, .8), (.8, 1.01)]], "model confidence")
out["card"] = table([("main event", d.bout_order == 0), ("5 rounds", d.sched_rounds == 5), ("title", d.title == 1),
                     ("other", (d.bout_order > 0) & (d.sched_rounds != 5))], "card position")
out["method"] = table([(k, d.method.isin(v)) for k, v in {"KO/TKO": ["KO", "DOC"], "SUB": ["SUB"], "DEC": ["U-DEC", "S-DEC", "M-DEC"], "split dec": ["S-DEC"]}.items()], "how the fight ended (post-fight, diagnostic only)")
if mk:
    agree = (P[col] > .5) == (P[mk] > .5)
    out["agreement"] = table([("model & market agree", agree & P[mk].notna()), ("disagree", ~agree & P[mk].notna())], "agreement with market")
print("\ncalibration:"); print(calibration_table(P.y, P[col]).to_string())

# confident misses (for semantic review)
miss = P.assign(q=q, f1=d.f1_name, f2=d.f2_name, method=d.method, wc=d.weightclass)
miss = miss[((miss[col] > .5) != (miss.y == 1))].sort_values("q", ascending=False)
print("\nmost confident misses:")
print(miss[["date", "f1", "f2", col] + ([mk] if mk else []) + ["y", "method", "wc"]].head(25).round(3).to_string())
pd.concat(out, names=["breakdown"]).to_csv(f"reports/08_breakdowns_{period}_{col}.csv")

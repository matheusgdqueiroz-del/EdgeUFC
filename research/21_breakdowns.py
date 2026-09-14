"""Where the engine succeeds and fails (production engine outputs, out-of-sample only).

  python research/21_breakdowns.py valid|final
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.fetch import ROOT
from ufc.evaluate import scores

mode = sys.argv[1] if len(sys.argv) > 1 else "valid"
E = pd.read_parquet(ROOT / "data" / f"engine_oos_{mode}.parquet")
lo = "2021-01-01"
E = E[(E.date >= lo) & E.y.notna() & E.p_close.notna() & E.ens3.notna()]


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def table(groups, name):
    rows = []
    for g, m in groups:
        m = m & E.index.isin(E.index)
        if m.sum() < 40:
            continue
        x = E[m]
        rows.append(dict(group=g, n=int(m.sum()), market_ll=round(ll(x.y, x.p_close).mean(), 4), engine_ll=round(ll(x.y, x.ens3).mean(), 4),
                         stats_ll=round(ll(x.y, x.stats).mean(), 4), gain_vs_market=round((ll(x.y, x.p_close) - ll(x.y, x.ens3)).mean(), 4),
                         market_acc=round(((x.p_close > .5) == (x.y == 1)).mean(), 3), engine_acc=round(((x.ens3 > .5) == (x.y == 1)).mean(), 3),
                         stats_acc=round(((x.stats > .5) == (x.y == 1)).mean(), 3)))
    t = pd.DataFrame(rows)
    t.insert(0, "breakdown", name)
    return t


exp = np.minimum(E.a_n_fights, E.b_n_fights)
q = np.maximum(E.ens3, 1 - E.ens3)
fav = np.maximum(E.p_close, 1 - E.p_close)
dis = (E.ens3 > .5) != (E.p_close > .5)
T = pd.concat([
    table([(str(y), E.date.dt.year == y) for y in sorted(E.date.dt.year.unique())], "year"),
    table([("both debut", (E.a_n_fights == 0) & (E.b_n_fights == 0)), ("one debutant", (E.a_n_fights == 0) ^ (E.b_n_fights == 0)),
           ("min 1-2 UFC fights", exp.between(1, 2)), ("min 3-5", exp.between(3, 5)), ("min 6+", exp >= 6)], "UFC experience"),
    table([(w, E.weightclass == w) for w in E.weightclass.value_counts().index[:11]] + [("women", E.women == 1), ("men", E.women == 0)], "division"),
    table([(f"{a:.2f}-{b:.2f}", (q >= a) & (q < b)) for a, b in [(.5, .55), (.55, .6), (.6, .7), (.7, .8), (.8, 1.01)]], "engine confidence"),
    table([(f"{a:.2f}-{b:.2f}", (fav >= a) & (fav < b)) for a, b in [(.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]], "market favourite price"),
    table([("main event", E.bout_order == 0), ("main card (1-4)", E.bout_order.between(1, 4)), ("prelims", E.bout_order >= 5), ("title fight", E.title == 1)], "card position"),
    table([("engine & market agree", ~dis), ("engine disagrees with market", dis)], "agreement"),
    table([(k, E.method.isin(v)) for k, v in {"KO/TKO": ["KO", "DOC"], "submission": ["SUB"], "decision": ["U-DEC", "S-DEC", "M-DEC"], "split decision": ["S-DEC"]}.items()], "how it ended (diagnostic)"),
])
pd.set_option("display.width", 220)
print(T.to_string(index=False))
T.to_csv(ROOT / "reports" / f"21_breakdowns_{mode}.csv", index=False)
miss = E.assign(q=q)[((E.ens3 > .5) != (E.y == 1))].sort_values("q", ascending=False)
print("\nmost confident misses:")
print(miss[["date", "f1_name", "f2_name", "ens3", "p_close", "y", "method", "weightclass"]].head(15).round(3).to_string(index=False))

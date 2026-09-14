"""v2 / prop results robustness (declared AFTER seeing research/45; reported as robustness, not as new evidence).

Uses data/v2/props_scored.parquet (walk-forward layer probabilities from research/45).
  R1 prices 2% / 4% worse than the mean closing price
  R2 only markets quoted by 3+ books
  R3 edge threshold 6% instead of 3%
  R4 at most one bet per bout (largest edge)
  R5 flat 1-unit stakes
  R6 worst price across books (the "yes" side's lowest quote is not stored, so use mean price minus the spread to best)
"""
import importlib.util, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.fetch import ROOT
from ufc.betting import kelly_units

x = pd.read_parquet(ROOT / "data" / "v2" / "props_scored.parquet")
s = x[x.layer.notna() & x.won.notna()].copy()
FAMS = ["fdec", "itd"]


def bets(f, prob="layer", haircut=0.0, min_edge=0.03, flat=False, one_per_bout=False, price="mean"):
    f = f[f[prob].notna()].copy()
    p = f[prob].values
    dy = f[f"yes_{price}"].values * (1 - haircut) if price != "worst" else 2 * f.yes_mean.values - f.yes_best.values
    dn = f[f"no_{'mean' if price == 'worst' else price}"].values * (1 - haircut)
    if price == "worst":
        dn = 2 * f.no_mean.values - f.no_best.values
    ev_y, ev_n = p * dy - 1, (1 - p) * dn - 1
    yes = np.where(np.isnan(ev_n), True, np.where(np.isnan(ev_y), False, ev_y >= ev_n))
    pb, dec, ev = np.where(yes, p, 1 - p), np.where(yes, dy, dn), np.where(yes, ev_y, ev_n)
    st = np.where(flat, 1.0, kelly_units(pb, dec, 0.125, 1.5))
    ok = (ev > min_edge) & (st >= 0.25) & np.isfinite(dec) & (dec > 1)
    won = np.where(yes, f.won.values, 1 - f.won.values)
    f["stake"] = st
    f["profit"] = np.where(won == 1, st * (dec - 1), -st)
    f["ev"] = ev
    b = f[ok]
    if one_per_bout:
        b = b.sort_values("ev", ascending=False).drop_duplicates("fight_id")
    return b


def row(name, b, events):
    g = b.groupby("event_id").agg(p=("profit", "sum"), s=("stake", "sum"))
    rng = np.random.default_rng(1)
    r = [g.p.values[i].sum() / g.s.values[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(2000))]
    by = b.groupby(b.date.dt.year).apply(lambda z: z.profit.sum() / z.stake.sum()).round(3).to_dict()
    return dict(check=name, bets=len(b), per_event=round(len(b) / events, 2), roi=round(b.profit.sum() / b.stake.sum(), 4),
                ci90=f"[{np.percentile(r, 5):+.3f}, {np.percentile(r, 95):+.3f}]", years_up=f"{sum(v > 0 for v in by.values())}/{len(by)}", by_year=by)


ev_n = s.event_id.nunique()
out = []
for fams, label in ((FAMS, "fdec+itd"), (["fdec"], "fdec"), (["itd"], "itd")):
    g = s[s.family.isin(fams)]
    out.append(row(f"{label}: as run (mean price)", bets(g), ev_n))
    out.append(row(f"{label}: R1 price -2%", bets(g, haircut=0.02), ev_n))
    out.append(row(f"{label}: R1 price -4%", bets(g, haircut=0.04), ev_n))
    out.append(row(f"{label}: R2 3+ books", bets(g[g.n_books >= 3]), ev_n))
    out.append(row(f"{label}: R3 edge > 6%", bets(g, min_edge=0.06), ev_n))
    out.append(row(f"{label}: R4 one bet per bout", bets(g, one_per_bout=True), ev_n))
    out.append(row(f"{label}: R5 flat 1u", bets(g, flat=True), ev_n))
    out.append(row(f"{label}: R6 worst-ish price", bets(g, price="worst"), ev_n))
    out.append(row(f"{label}: best price", bets(g, price="best"), ev_n))
R = pd.DataFrame(out)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print(R.to_string(index=False))
R.to_csv(ROOT / "reports" / "46_props_robustness.csv", index=False)

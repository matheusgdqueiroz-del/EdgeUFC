"""v2 / Pinnacle realism on DEV (2008-2020 archive of Pinnacle closing moneylines, UFC_Final dataset).

1. Pinnacle closing margin (overround) by year and by price band.
2. Is Pinnacle's closing line sharper than the BestFightOdds consensus close? (log loss on the same bouts)
3. Synthetic Pinnacle price = consensus fair probability with Pinnacle's margin split proportionally.
   Compare steady-rule bets (v2 DEV layers at the close) settled at REAL Pinnacle prices vs the synthetic ones,
   so the synthetic model can be trusted for 2021+ where no Pinnacle archive exists.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from ufc.fetch import ROOT
from ufc.sources.ufcstats import load_fights, slug
from ufc.sources import odds
from ufc.betting import simulate, summary
from ufc.evaluate import scores

STEADY = dict(min_edge=0.03, frac=0.125, max_units=1.5)
f = load_fights()
o = odds.build(f)
d = f.merge(o, on="fight_id", how="left")
d["y"] = d.winner
uf = pd.read_csv(ROOT / "data_raw/UFC_Final/data/datasets_for_analysis/final_datasets/odds_w_outcomes_one_row_per_fight.csv")
uf = uf[uf.Pinnacle_win.notna() & uf.Pinnacle_lose.notna()].copy()
uf["date"] = pd.to_datetime(uf.Card_Date)
uf["s1"], uf["s2"] = uf.Winner_Cleaned.map(slug), uf.Loser_Cleaned.map(slug)
by = {k: g for k, g in uf.groupby("date")}
rows = []
for r in d[d.date < "2021-01-01"].itertuples():
    s1, s2 = slug(r.f1_name), slug(r.f2_name)
    best, bs = None, 0
    for off in (0, -1, 1):
        g = by.get(r.date + pd.Timedelta(days=off))
        if g is None:
            continue
        for e in g.itertuples():
            a = (fuzz.ratio(s1, e.s1) + fuzz.ratio(s2, e.s2)) / 2
            b = (fuzz.ratio(s1, e.s2) + fuzz.ratio(s2, e.s1)) / 2
            if max(a, b) > bs:
                bs, best = max(a, b), ((e.Pinnacle_win, e.Pinnacle_lose) if a >= b else (e.Pinnacle_lose, e.Pinnacle_win))
    if bs >= 85:
        rows.append((r.fight_id, best[0], best[1]))
pin = pd.DataFrame(rows, columns=["fight_id", "a_pin_dec", "b_pin_dec"])
x = d.merge(pin, on="fight_id")
x = x[(x.a_pin_dec > 1) & (x.b_pin_dec > 1)]
x["pin_over"] = 1 / x.a_pin_dec + 1 / x.b_pin_dec
x = x[x.pin_over.between(1.0, 1.15)]
x["p_pin"] = (1 / x.a_pin_dec) / x.pin_over
print("matched bouts", len(x))
fav = np.maximum(x.p_pin, 1 - x.p_pin)
print("\nPinnacle closing overround by year:", x.groupby(x.date.dt.year).pin_over.median().round(4).to_dict())
print("by favourite band:", x.groupby(pd.cut(fav, [.5, .6, .7, .8, .9, 1])).pin_over.median().round(4).to_dict())
print("consensus (BFO) close overround same bouts:", round(x.close_overround.median(), 4))

s = x[x.y.notna() & x.p_close.notna()]
print("\nSharpness on the same bouts:")
print("  Pinnacle fair close:", scores(s.y, s.p_pin))
print("  BFO consensus close:", scores(s.y, s.p_close))

# synthetic Pinnacle price from consensus fair probability
M = x[x.date.dt.year >= 2016].pin_over.median()
print(f"\nmargin used for synthetic Pinnacle prices (median 2016-2020): {M:.4f}")
x["a_syn_dec"], x["b_syn_dec"] = 1 / (x.p_close * M), 1 / ((1 - x.p_close) * M)
print("median |synthetic - real| price (decimal):", round(np.nanmedian(np.abs(x.a_syn_dec - x.a_pin_dec)), 4),
      " mean real/synthetic - 1:", round(np.nanmean(x.a_pin_dec / x.a_syn_dec - 1), 4))

# bets with v2 DEV layers at the close (stage 9) and v1 ens3 (out of sample)
P = pd.read_parquet(ROOT / "data" / "v2" / "market_dev_C0.003+promo+soft70.parquet")
P = P[P.stage == 9].assign(v2=lambda z: (z.v2stats_stage_stack + z.v2stats_edge_glm) / 2)[["fight_id", "v2", "v1_stack"]]
E = pd.read_parquet(ROOT / "data" / "engine_oos.parquet")[["fight_id", "ens3"]]
z = x.merge(P, on="fight_id").merge(E, on="fight_id", how="left")
z = z[z.date >= "2013-01-01"]
out = []
for prob in ("v1_stack", "ens3", "v2"):
    for price in ("pin", "syn", "close_mean"):
        b = simulate(z, prob, f"a_{price}_dec", f"b_{price}_dec", **STEADY)
        sm = summary(b)
        out.append(dict(prob=prob, price=price, bets=sm["bets"], per_event=round(sm["bets"] / z.event_id.nunique(), 2), roi=sm["roi"], max_dd=sm["max_dd"],
                        years_up=int((b.groupby(b.date.dt.year).profit.sum() > 0).sum()), n_years=b.date.dt.year.nunique()))
R = pd.DataFrame(out)
print("\nSteady rule at the close, DEV 2013-2020 (bouts with a Pinnacle archive price):")
print(R.to_string(index=False))
R.to_csv(ROOT / "reports" / "47_pinnacle_dev.csv", index=False)
pd.DataFrame({"margin_2016_2020": [M]}).to_csv(ROOT / "data" / "v2" / "pinnacle_margin.csv", index=False)

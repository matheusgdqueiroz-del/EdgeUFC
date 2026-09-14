"""Hypotheses fixed on 2008-2020 (research/10), re-tested as flat 1u bets at average closing prices on 2021-2026."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from ufc.fetch import ROOT
d = pd.read_parquet(ROOT / "data" / "model_table.parquet")
d = d[d.p_close.notna() & (d.date >= "2021-01-01")]

def flat(name, ma, mb):
    rows = []
    for side, m in (("a", ma), ("b", mb)):
        m = m & (d.y.notna() | (d.outcome == "D/D"))
        dec = d[f"{side}_close_mean_dec"]; won = (d.y == 1) if side == "a" else (d.y == 0)
        rows.append(pd.DataFrame({"date": d.date[m], "profit": np.where(won, dec - 1, -1.0)[m.values], "won": won[m],
                                  "p": (d.p_close if side == "a" else 1 - d.p_close)[m]}))
    b = pd.concat(rows)
    yr = b.groupby(b.date.dt.year).profit.mean()
    return dict(hypothesis=name, bets=len(b), roi=round(b.profit.mean(), 4), win_rate=round(b.won.mean(), 3), implied=round(b.p.mean(), 3),
                p_value=round(stats.ttest_1samp(b.profit, 0).pvalue, 3) if len(b) > 5 else np.nan, years_positive=f"{(yr > 0).sum()}/{len(yr)}")
H = [flat("younger by 5+ years", (d.a_age - d.b_age) <= -5, (d.a_age - d.b_age) >= 5),
     flat("main card slight underdog (37.5-45%)", (d.bout_order <= 4) & d.p_close.between(.375, .45), (d.bout_order <= 4) & (1 - d.p_close).between(.375, .45)),
     flat("big underdog (<=20%)", d.p_close <= .2, d.p_close >= .8),
     flat("big favourite (>=80%)", d.p_close >= .8, d.p_close <= .2),
     flat("bet AGAINST UFC debutant facing 3+ fight veteran", (d.b_n_fights == 0) & (d.a_n_fights >= 3), (d.a_n_fights == 0) & (d.b_n_fights >= 3)),
     flat("bet AGAINST fighter who missed weight", (d.b_missed_weight == 1) & (d.a_missed_weight == 0), (d.a_missed_weight == 1) & (d.b_missed_weight == 0)),
     flat("bet AGAINST short-notice replacement", (d.b_replacement == 1) & (d.a_replacement == 0), (d.a_replacement == 1) & (d.b_replacement == 0)),
     flat("home-country fighter vs foreigner", (d.a_home_country == 1) & (d.b_home_country == 0), (d.b_home_country == 1) & (d.a_home_country == 0)),
     flat("travelled 5000+ km less", (d.a_travel_km - d.b_travel_km) <= -5000, (d.a_travel_km - d.b_travel_km) >= 5000)]
T = pd.DataFrame(H); print(T.to_string(index=False)); T.to_csv(ROOT / "reports" / "25_hypotheses_2021_2026.csv", index=False)

"""Market as a predictor + cross-source validation of the odds we scraped.

1. Coverage and quality (log loss / accuracy / calibration) of BFO opening and closing probabilities by year.
2. Discrepancy check against two independent archives:
     UFC_Final (per-book closing odds 2008-2020, includes Pinnacle) and ultimate_ufc_dataset (2010-2024).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from ufc.sources.ufcstats import slug
from ufc.evaluate import scores, calibration_table, HOLDOUT_START

f = pd.read_parquet("data/fights.parquet"); o = pd.read_parquet("data/odds.parquet")
d = f.merge(o, on="fight_id", how="left")
d["y"] = d.winner
pre = d[d.date < HOLDOUT_START]
print("coverage by year (closing prob present):")
print(pre.groupby(pre.date.dt.year).p_close.apply(lambda s: f"{s.notna().mean():.2f}").loc[2007:].to_string())

rows = []
for lo, hi in [(2008, 2012), (2013, 2016), (2017, 2020), (2021, 2024)]:
    x = pre[(pre.date.dt.year >= lo) & (pre.date.dt.year <= hi) & pre.p_close.notna() & pre.p_open.notna()]
    for col in ("p_open", "p_close"):
        s = scores(x.y, x[col]); s.update(period=f"{lo}-{hi}", market=col); rows.append(s)
print(pd.DataFrame(rows).set_index(["period", "market"]).to_string())
x = pre[(pre.date >= "2010") & pre.p_close.notna()]
print("\nclosing-line calibration 2010-2024:"); print(calibration_table(x.y, x.p_close).to_string())

# ---------- cross-source validation ----------
def attach(ext, date_col, n1, n2, p1):
    ext = ext.copy(); ext["s1"], ext["s2"] = ext[n1].map(slug), ext[n2].map(slug)
    ext["date"] = pd.to_datetime(ext[date_col])
    by = {dd: g for dd, g in ext.groupby("date")}
    out = []
    for r in d[d.p_close.notna()].itertuples():
        s1, s2 = slug(r.f1_name), slug(r.f2_name)
        best, bs = None, 0
        for off in (0, -1, 1):
            g = by.get(r.date + pd.Timedelta(days=off))
            if g is None: continue
            for e in g.itertuples():
                a = (fuzz.ratio(s1, e.s1) + fuzz.ratio(s2, e.s2)) / 2; b = (fuzz.ratio(s1, e.s2) + fuzz.ratio(s2, e.s1)) / 2
                if max(a, b) > bs:
                    bs, best = max(a, b), (getattr(e, p1) if a >= b else 1 - getattr(e, p1))
        if bs >= 85:
            out.append((r.Index, best))
    return pd.Series(dict(out))

uf = pd.read_csv("data_raw/UFC_Final/data/datasets_for_analysis/final_datasets/odds_w_outcomes_one_row_per_fight.csv")
w, l = uf.meanodds_win, uf.meanodds_lose
uf["p_win_mean"] = (1 / w) / (1 / w + 1 / l)
pw, pl = uf.Pinnacle_win, uf.Pinnacle_lose
uf["p_win_pin"] = (1 / pw) / (1 / pw + 1 / pl)
d["uf_mean"] = attach(uf.rename(columns={"Winner_Cleaned": "W", "Loser_Cleaned": "L"}), "Card_Date", "W", "L", "p_win_mean")
d["uf_pin"] = attach(uf.rename(columns={"Winner_Cleaned": "W", "Loser_Cleaned": "L"}), "Card_Date", "W", "L", "p_win_pin")
um = pd.read_csv("data_raw/ultimate_ufc_dataset/data/ultimate_ufc_dataset/ufc-master.csv")
dr, db = np.where(um.RedOdds > 0, 1 + um.RedOdds / 100, 1 + 100 / um.RedOdds.abs()), np.where(um.BlueOdds > 0, 1 + um.BlueOdds / 100, 1 + 100 / um.BlueOdds.abs())
um["p_red"] = (1 / dr) / (1 / dr + 1 / db)
d["um"] = attach(um, "Date", "RedFighter", "BlueFighter", "p_red")

for col in ("uf_mean", "uf_pin", "um"):
    x = d[d[col].notna() & d.p_close.notna() & d.y.notna()]
    diff = (x.p_close - x[col]).abs()
    print(f"\n{col}: n={len(x)}  mean|diff|={diff.mean():.3f}  p90|diff|={diff.quantile(.9):.3f}  share>0.10: {(diff > .10).mean():.3f}")
    print("  BFO   ", scores(x.y, x.p_close)); print("  other ", scores(x.y, x[col]))
    big = x[diff > 0.15][["date", "f1_name", "f2_name", "p_close", col, "p_open", "match_score"]].head(8)
    print(big.round(3).to_string())
d[["fight_id", "uf_mean", "uf_pin", "um"]].to_parquet("data/odds_external.parquet")

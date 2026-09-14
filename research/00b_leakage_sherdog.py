"""Canary for Sherdog-derived features only (fast)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.sources.ufcstats import load_fights
from ufc.sources import sherdog_features as sf
T = pd.Timestamp("2016-06-01")
fights = load_fights()
link = sf.link_fighters(fights)
recs = sf.load_records(); bios = pd.read_csv(sf.SD / "fighters.csv")
early = fights[(fights.date < T) & (fights.date >= "2012-01-01")]
a = sf.record_features(early, link, recs, bios).set_index("fight_id")
b = sf.record_features(early, link, recs[recs.date < T].sample(frac=1, random_state=3), bios).set_index("fight_id")  # truncated AND shuffled
n = a.select_dtypes("number").columns
dd = ~np.isclose(a[n].values.astype(float), b.loc[a.index, n].values.astype(float), equal_nan=True)
print(f"sherdog: compared {len(a)} bouts x {len(n)} features; mismatching cells: {int(dd.sum())}")
if dd.sum():
    print(pd.Series(dd.sum(0), index=n)[lambda s: s > 0]); raise SystemExit("LEAK DETECTED")
print("NO LEAKAGE in Sherdog features")

# skill model + market ratings: truncate bouts at T
import json
from ufc import skills, market_ratings
from ufc.fetch import ROOT
perf = pd.read_parquet(ROOT / "data" / "perf.parquet"); odds = pd.read_parquet(ROOT / "data" / "odds.parquet")
cfg = json.load(open(ROOT / "data" / "skills_cfg.json"))
cut = fights[fights.date < T]
for name, full, trunc in [("skills", skills.run(fights, perf, cfg)[0], skills.run(cut, perf[perf.fight_id.isin(set(cut.fight_id))], cfg)[0]),
                          ("market_ratings", market_ratings.run(fights, odds)[0], market_ratings.run(cut, odds[odds.fight_id.isin(set(cut.fight_id))])[0])]:
    A = full.set_index("fight_id").loc[trunc.fight_id]; Bt = trunc.set_index("fight_id")
    n = Bt.select_dtypes("number").columns
    mm = int((~np.isclose(A[n].values.astype(float), Bt[n].values.astype(float), equal_nan=True)).sum())
    print(f"{name}: compared {len(Bt)} bouts x {len(n)} features; mismatching cells: {mm}")
    if mm: raise SystemExit("LEAK DETECTED")
print("NO LEAKAGE in skills / market ratings")

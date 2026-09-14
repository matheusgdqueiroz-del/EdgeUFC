"""Leakage canary for v2 promotion features: deleting every Sherdog record dated on/after 2016-07-01 must not change
features of bouts before that date (records are also shuffled)."""
import numpy as np, pandas as pd
from ufc.fetch import ROOT
from ufc.sources.ufcstats import load_fights
from ufc.sources import promotion_features as pf, sherdog_features as sf

CUT = pd.Timestamp("2016-07-01")
f = load_fights()
f = f[(f.date >= "2015-01-01") & (f.date < CUT)]
link = dict(pd.read_csv(ROOT / "data" / "sherdog_link.csv").values)
full = pf.promotion_features(f, link).set_index("fight_id").sort_index()

all_r, load_r = pf.all_records, sf.load_records
pf.all_records = lambda: all_r()[lambda r: r.date < CUT].sample(frac=1, random_state=3)
pf.load_records = lambda: load_r()[lambda r: r.date < CUT].sample(frac=1, random_state=4).sort_values(["date", "sd_url", "opp_url"], kind="mergesort")
trunc = pf.promotion_features(f, link).set_index("fight_id").sort_index()
diff = (full - trunc).abs().max().max()
print("bouts", len(full), "max abs difference", diff)
dd = (full - trunc).abs()
print(dd.max().round(4).to_string())
bad = dd.max(axis=1).sort_values(ascending=False).head(3)
print(pd.concat([full.loc[bad.index].T, trunc.loc[bad.index].T], axis=1, keys=["full", "trunc"]).dropna(how="all").to_string())
assert np.isclose(np.nan_to_num(full.values, nan=-9), np.nan_to_num(trunc.values, nan=-9)).all(), "LEAK: promotion features changed"
print("promotion features canary passed")

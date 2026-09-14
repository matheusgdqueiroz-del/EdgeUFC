"""Hindsight canary: features for bouts before T must be IDENTICAL whether or not the data after T exists.

We rebuild the engine features from a copy of the data truncated at T (all bouts, round stats and Sherdog
records on/after T deleted) and compare against the full build. Any difference = future information leaking
into the past.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.sources.ufcstats import load_fights, load_rounds
from ufc import features

T = pd.Timestamp("2016-06-01")
fights = load_fights(); rounds = load_rounds(fights)
full = features.run(fights, features.build_perf(fights, rounds), verbose=False).set_index("fight_id")

cut_f = fights[fights.date < T].reset_index(drop=True)
cut_r = rounds[rounds.fight_id.isin(set(cut_f.fight_id))]
trunc = features.run(cut_f, features.build_perf(cut_f, cut_r), verbose=False).set_index("fight_id")

a = full.loc[trunc.index, trunc.columns]
num = a.select_dtypes("number").columns
diff = ~np.isclose(a[num].values.astype(float), trunc[num].values.astype(float), equal_nan=True, rtol=1e-9, atol=1e-9)
bad = pd.Series(diff.sum(0), index=num)
print(f"engine: compared {len(trunc)} bouts x {len(num)} features; mismatching cells: {int(diff.sum())}")
if diff.sum():
    print(bad[bad > 0].sort_values(ascending=False).head(20))
    raise SystemExit("LEAK DETECTED")

# Sherdog record features: truncate the scraped records at T as well
try:
    from ufc.sources import sherdog_features as sf
    link = sf.link_fighters(fights)
    recs = sf.load_records(); bios = pd.read_csv(sf.SD / "fighters.csv")
    early = fights[(fights.date < T) & (fights.date >= "2012-01-01")]
    s_full = sf.record_features(early, link, recs, bios).set_index("fight_id")
    s_cut = sf.record_features(early, link, recs[recs.date < T], bios).set_index("fight_id")
    n2 = s_full.select_dtypes("number").columns
    dd = ~np.isclose(s_full[n2].values.astype(float), s_cut.loc[s_full.index, n2].values.astype(float), equal_nan=True)
    print(f"sherdog: compared {len(s_full)} bouts x {len(n2)} features; mismatching cells: {int(dd.sum())}")
    if dd.sum():
        b2 = pd.Series(dd.sum(0), index=n2); print(b2[b2 > 0]); raise SystemExit("LEAK DETECTED")
except FileNotFoundError:
    print("sherdog data not available yet; skipped")
print("NO LEAKAGE: pre-T features are unchanged by post-T data")

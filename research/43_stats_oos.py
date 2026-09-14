"""v2 / out-of-sample stats-model predictions from 2008 for a variant (feeds market-layer research).

  python research/43_stats_oos.py <variant> [end_year_exclusive=2021]
Writes data/v2/stats_oos_<variant>.parquet (walk-forward yearly). DEV only unless end > 2021.
"""
import sys, time, importlib.util, warnings; warnings.filterwarnings("ignore")
import pandas as pd
from ufc.fetch import ROOT
from ufc.evaluate import walk_forward
from ufc.model import feature_sets

spec = importlib.util.spec_from_file_location("sv", ROOT / "research" / "41_stats_variants.py")
sv = importlib.util.module_from_spec(spec); spec.loader.exec_module(sv)
name = sys.argv[1]
end = int(sys.argv[2]) if len(sys.argv) > 2 else 2021
d = sv.load_table()
for qq in (60, 70, 80):
    d[f"q{qq}"] = sv.soft_labels(d, qq / 100)
fs = feature_sets(d)
fs["ufc"] = [k for k in fs["ufc"] if not k.startswith(tuple(sv.BLOCKS.values()))]
fn = sv.parse_variant(name, d)
t = time.time()
p = walk_forward(d, lambda tr, te: fn(tr, te, d, fs), start="2008-01-01", end=f"{end}-01-01", final=end > 2025)
pd.DataFrame({"stats": p, "fight_id": d.loc[p.index, "fight_id"]}).to_parquet(ROOT / "data" / "v2" / f"stats_oos_{name}.parquet")
print(name, len(p), f"{time.time() - t:.0f}s")

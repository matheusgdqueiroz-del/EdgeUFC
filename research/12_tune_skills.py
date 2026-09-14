"""Tune the skill model's prior variance (p0) and drift (q) per stat on the one-step-ahead predictive
likelihood of the fight STATS themselves (2005-2014). Outcomes (who won) are never used here."""
import itertools, json, warnings; warnings.filterwarnings("ignore")
import pandas as pd
from ufc import skills

f = pd.read_parquet("data/fights.parquet"); p = pd.read_parquet("data/perf.parquet")
lo, hi = pd.Timestamp("2005-01-01"), pd.Timestamp("2015-01-01")
rows = []
for pm, qm, ph in itertools.product([0.06, 0.12, 0.25], [0.01, 0.03, 0.1], [1.0, 3.0]):
    cfg = dict(p0={k: v * pm for k, v in skills.DEFAULT["p0"].items()}, q={k: v * qm for k, v in skills.DEFAULT["q"].items()},
               phi={k: ph for k in skills.STATS})
    _, ll = skills.run(f, p, cfg, lo, hi)
    rows.append(dict(p0_mult=pm, q_mult=qm, phi=ph, **{f"ll_{k}": round(v, 5) for k, v in ll.items()})); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv("reports/12_skill_tuning.csv", index=False)
best = {}
for k in skills.STATS:
    r = R.loc[R[f"ll_{k}"].idxmax()]
    best[k] = dict(p0=skills.DEFAULT["p0"][k] * r.p0_mult, q=skills.DEFAULT["q"][k] * r.q_mult, phi=r.phi)
    print(k, "best p0 x", r.p0_mult, "q x", r.q_mult, "phi", r.phi)
json.dump({part: {k: v[part] for k, v in best.items()} for part in ("p0", "q", "phi")}, open("data/skills_cfg.json", "w"), indent=1)

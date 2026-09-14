"""v2 final moneyline evaluation (PREREGISTRATION.md Addendum C). Re-used period 2021-01-01 -> 2026-09-12, computed once.

  python research/48_v2_final.py
Writes data/engine_oos_v2.parquet (live engine history) and reports/48_v2_final.csv / .txt
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.fetch import ROOT
from ufc.engine import full_walk_forward
from ufc.betting import simulate, summary

DATA = ROOT / "data"
STEADY = dict(min_edge=0.03, frac=0.125, max_units=1.5)
path = DATA / "engine_oos_v2.parquet"
if not path.exists():
    d = pd.read_parquet(DATA / "model_table.parquet")
    full_walk_forward(d, final=True, version="v2").to_parquet(path)
E2 = pd.read_parquet(path)
E1 = pd.read_parquet(DATA / "engine_oos.parquet")[["fight_id", "stats", "ens3", "stack_open"]].rename(columns={"stats": "stats_v1"})
x = E2.merge(E1, on="fight_id", how="left")
lines = []
say = lambda s="": (print(s), lines.append(s))


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(frame, a, b, n=2000, seed=0):
    ok = frame[a].notna() & frame[b].notna() & frame.y.notna()
    f = frame[ok]
    g = pd.DataFrame({"ev": f.event_id.values, "d": ll(f[a].values, f.y.values) - ll(f[b].values, f.y.values)}).groupby("ev").d.agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    s, c = g["sum"].values, g["size"].values
    bs = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    return len(f), s.sum() / c.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def pin_prices(f, k, margin):
    p = f[f"p_s{k}"]
    return f.assign(a_px=1 / (p * margin), b_px=1 / ((1 - p) * margin))


def roi_ci(b, n=2000, seed=1):
    g = b.groupby("event_id").agg(p=("profit", "sum"), s=("stake", "sum"))
    rng = np.random.default_rng(seed)
    r = [g.p.values[i].sum() / g.s.values[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    return np.percentile(r, 5), np.percentile(r, 95)


for label, lo, hi in (("DEV context 2013-2020 (walk-forward, already used for design)", "2013-01-01", "2021-01-01"),
                      ("RE-USED PERIOD 2021-01-01 -> 2026-09-12", "2021-01-01", "2026-09-13")):
    f = x[(x.date >= lo) & (x.date < hi) & x.p_s0.notna() & x.p_s9.notna() & x.p2_s0.notna()].copy()
    events = f.event_id.nunique()
    say(f"\n=== {label}: {len(f)} bouts, {events} events ===")
    say("C1 log loss (gain = v1 - v2, positive favours v2; 95% CI clustered by event)")
    for k, v1, name in ((0, "stack_open", "opening line"), (9, "ens3", "closing line")):
        n, g, a, b = boot(f, v1, f"p2_s{k}")
        ok = f.y.notna() & f[v1].notna()
        say(f"  {name}: n={n} line {ll(f.loc[ok, f'p_s{k}'], f.loc[ok, 'y']).mean():.4f}  v1 {ll(f.loc[ok, v1], f.loc[ok, 'y']).mean():.4f}  "
            f"v2 {ll(f.loc[ok, f'p2_s{k}'], f.loc[ok, 'y']).mean():.4f}  gain {g:+.4f} [{a:+.4f}, {b:+.4f}]")
        n, g, a, b = boot(f, f"p_s{k}", f"p2_s{k}")
        say(f"     v2 vs the line itself: gain {g:+.4f} [{a:+.4f}, {b:+.4f}]")
    n, g, a, b = boot(f, "stats_v1", "stats")
    say(f"  stats model (market-free): v1 {ll(f.stats_v1[f.y.notna()], f.y[f.y.notna()]).mean():.4f} -> v2 gain {g:+.4f} [{a:+.4f}, {b:+.4f}]")

    say("C2 steady rule at synthetic Pinnacle prices (stage fair probability x margin)")
    rows = []
    for k, stage in ((0, "open"), (3, "early"), (6, "mid"), (9, "close")):
        for margin_label, m in (("primary", 1.045 if k <= 3 else 1.035), ("low 2.5%", 1.025), ("high 5.5%", 1.055)):
            g = pin_prices(f, k, m)
            models = [("v2", f"p2_s{k}")] + ([("v1", "stack_open")] if k == 0 else []) + ([("v1", "ens3")] if k == 9 else [])
            for mname, col in models:
                b = simulate(g, col, "a_px", "b_px", **STEADY)
                if b.empty:
                    continue
                b["event_id"] = g.loc[b.index, "event_id"]
                s = summary(b)
                c_lo, c_hi = roi_ci(b)
                j = g.loc[b.index]
                close = np.where(b.bet_side == "a", 1 / (j.p_s9 * 1.035), 1 / ((1 - j.p_s9) * 1.035))
                yrs = b.groupby(b.date.dt.year).profit.sum()
                rows.append(dict(stage=stage, margin=margin_label, model=mname, bets=s["bets"], per_event=round(s["bets"] / events, 2), roi=s["roi"],
                                 roi_90ci=f"[{c_lo:+.3f}, {c_hi:+.3f}]", profit_u=s["profit"], max_dd=s["max_dd"], clv=round(float(np.mean(b.price / close - 1)), 4),
                                 years_up=f"{int((yrs > 0).sum())}/{len(yrs)}"))
        # consensus mean price, as in v1 reports (close only has mean prices; other stages use the stage mean price)
        g = f.assign(a_px=f[f"a_s{k}_dec"], b_px=f[f"b_s{k}_dec"])
        for mname, col in [("v2", f"p2_s{k}")] + ([("v1", "stack_open")] if k == 0 else []) + ([("v1", "ens3")] if k == 9 else []):
            b = simulate(g, col, "a_px", "b_px", **STEADY)
            s = summary(b)
            yrs = b.groupby(b.date.dt.year).profit.sum()
            rows.append(dict(stage=stage, margin="consensus mean price", model=mname, bets=s["bets"], per_event=round(s["bets"] / events, 2), roi=s["roi"],
                             roi_90ci="", profit_u=s["profit"], max_dd=s["max_dd"], clv=np.nan, years_up=f"{int((yrs > 0).sum())}/{len(yrs)}"))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    say(R.to_string(index=False))
    if lo.startswith("2021"):
        R.to_csv(ROOT / "reports" / "48_v2_final.csv", index=False)
        g = pin_prices(f, 9, 1.035)
        b = simulate(g, "p2_s9", "a_px", "b_px", **STEADY)
        say("\nv2 close, primary Pinnacle margin, by year: " + str(b.groupby(b.date.dt.year).apply(lambda z: f"{len(z)} bets ROI {z.profit.sum() / z.stake.sum():+.3f}").to_dict()))
        g = pin_prices(f, 0, 1.045)
        b = simulate(g, "p2_s0", "a_px", "b_px", **STEADY)
        say("v2 open, primary Pinnacle margin, by year: " + str(b.groupby(b.date.dt.year).apply(lambda z: f"{len(z)} bets ROI {z.profit.sum() / z.stake.sum():+.3f}").to_dict()))
(ROOT / "reports" / "48_v2_final.txt").write_text("\n".join(lines), encoding="utf8")

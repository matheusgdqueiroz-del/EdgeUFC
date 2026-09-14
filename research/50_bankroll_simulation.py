"""Bankroll simulation of the v2 engine exactly as the user plans to bet (reais, step units, steady stake rule).

Every probability is out of sample (data/engine_oos_v2.parquet, walk-forward). Periods: 2013-2020 was used to design
the engine, 2021-Sep 2026 re-used; neither is a clean test. Stakes per bet: steady rule (edge > 3%, 1/8 Kelly, max 1.5
units, rounded to 0.5u) in UNITS, converted to reais with the unit in force when the card's bets are placed.

Unit systems
  user        start R$100; unit = R$1 per full R$100 of bankroll (R$1 at 100-199, R$2 at 200-299, ... R$100 at 10,000);
              below R$100 the unit is 1% of the bankroll
  aggressive  start R$200; unit R$10 at 200-299, R$20 at 300-399, R$30 at 400-499, ... (+R$10 per R$100, so a unit tends to
              10% of the bankroll); below R$200 the unit is 5% of the bankroll
  flat        start R$100; unit fixed at R$1 (no compounding, reference)
Prices: estimated Pinnacle (stage fair probability x 4.5% margin early / 3.5% late), book average at that stage, and REAL
Pinnacle closing prices where the 2008-2020 archive has them. Timing: bets placed 14 d (open), 3 d (mid-week) or 0 d
(fight day) before the card; the bankroll known at that moment sets the unit; stakes still riding on earlier cards are
not available cash. Ruin = bankroll below R$5.
"""
import warnings; warnings.filterwarnings("ignore")
import math
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from ufc.fetch import ROOT
from ufc.sources.ufcstats import slug
from ufc.betting import kelly_units

R, D = ROOT / "reports", ROOT / "data"
E = pd.read_parquet(D / "engine_oos_v2.parquet")
E = E[(E.date >= "2013-01-01") & E.p2_s0.notna() & E.p_s0.notna() & E.p_s9.notna()].copy()
STEADY = dict(min_edge=0.03, frac=0.125, max_units=1.5)
END = E.date.max()


# ------------------------------------------------------------------ real Pinnacle closing prices (archive 2008-2020)
def pinnacle_archive():
    uf = pd.read_csv(ROOT / "data_raw/UFC_Final/data/datasets_for_analysis/final_datasets/odds_w_outcomes_one_row_per_fight.csv")
    uf = uf[uf.Pinnacle_win.notna() & uf.Pinnacle_lose.notna()].copy()
    uf["date"] = pd.to_datetime(uf.Card_Date)
    uf["s1"], uf["s2"] = uf.Winner_Cleaned.map(slug), uf.Loser_Cleaned.map(slug)
    by = {k: g for k, g in uf.groupby("date")}
    rows = []
    for r in E[E.date < "2021-01-01"].itertuples():
        s1, s2 = slug(r.f1_name), slug(r.f2_name)
        best, bs = None, 0
        for off in (0, -1, 1):
            for e in (by.get(r.date + pd.Timedelta(days=off)) if by.get(r.date + pd.Timedelta(days=off)) is not None else pd.DataFrame()).itertuples():
                a = (fuzz.ratio(s1, e.s1) + fuzz.ratio(s2, e.s2)) / 2
                b = (fuzz.ratio(s1, e.s2) + fuzz.ratio(s2, e.s1)) / 2
                if max(a, b) > bs:
                    bs, best = max(a, b), ((e.Pinnacle_win, e.Pinnacle_lose) if a >= b else (e.Pinnacle_lose, e.Pinnacle_win))
        if bs >= 85:
            rows.append((r.Index, best[0], best[1]))
    p = pd.DataFrame(rows, columns=["idx", "a", "b"]).set_index("idx")
    ok = (p.a > 1) & (p.b > 1) & (1 / p.a + 1 / p.b).between(1.0, 1.15)
    return p[ok]




# ------------------------------------------------------------------ bets in units for one scenario
def scenario_bets(stage, price):
    p = E[f"p2_s{stage}"].values
    if price == "pin_est":
        m = 1.045 if stage <= 3 else 1.035
        da, db = 1 / (E[f"p_s{stage}"].values * m), 1 / ((1 - E[f"p_s{stage}"].values) * m)
    elif price == "avg":
        da, db = E[f"a_s{stage}_dec"].values, E[f"b_s{stage}_dec"].values
    else:
        if "a_pinreal" not in E:
            PIN = pinnacle_archive()
            E["a_pinreal"], E["b_pinreal"] = PIN.a, PIN.b
        da, db = E.a_pinreal.values, E.b_pinreal.values
    ev_a, ev_b = p * da - 1, (1 - p) * db - 1
    side_a = ev_a >= ev_b
    pb, dec, ev = np.where(side_a, p, 1 - p), np.where(side_a, da, db), np.where(side_a, ev_a, ev_b)
    u = kelly_units(pb, dec, STEADY["frac"], STEADY["max_units"])
    u = np.where(u >= 0.25, np.round(u * 2) / 2, 0.0)
    ok = (ev > STEADY["min_edge"]) & (u > 0) & np.isfinite(dec)
    won = np.where(side_a, E.y.values == 1, E.y.values == 0)
    nc = np.isnan(E.y.values) & (E.outcome.values != "D/D")
    mult = np.where(nc, 0.0, np.where(won, dec - 1, -1.0))  # profit per unit staked
    b = pd.DataFrame({"event_id": E.event_id.values, "date": E.date.values, "units": u, "mult": mult, "dec": dec})[ok]
    return b


def events_arrays(b):
    g = b.sort_values("date").groupby("event_id", sort=False)
    ev = [(grp.date.iloc[0], grp.units.values, grp.mult.values) for _, grp in g]
    ev.sort(key=lambda x: x[0])
    return ev


# ------------------------------------------------------------------ unit systems
def unit_user(B):
    return math.floor(B / 100) if B >= 100 else B / 100


def unit_aggressive(B):
    return 10.0 * (math.floor(B / 100) - 1) if B >= 200 else 0.05 * B


SYSTEMS = {"user": (100.0, unit_user), "aggressive": (200.0, unit_aggressive), "flat": (100.0, lambda B: 1.0)}
LEAD = {0: 14, 6: 3, 9: 0}


def run(events, system, lead_days, t0=None, t1=None, min_stake=0.0, ruin=5.0, max_stake=None):
    """Returns timeline DataFrame (date, bankroll) after each card settles, plus stats."""
    start, unit_fn = SYSTEMS[system]
    lead = pd.Timedelta(days=lead_days)
    evs = [e for e in events if (t0 is None or e[0] >= t0) and (t1 is None or e[0] < t1)]
    B, pending, timeline, n_bets, staked, ruined = start, [], [], 0, 0.0, False
    for date, units, mult in evs:
        place = date - lead
        still = []
        for (sd, stake, pnl) in pending:  # settle cards that finished before this card's bets are placed
            if sd < place:
                B += pnl
                timeline.append((sd, B))
            else:
                still.append((sd, stake, pnl))
        pending = still
        if B < ruin:
            ruined = True
            break
        outstanding = sum(s for _, s, _ in pending)
        cash = B - outstanding
        stakes = units * unit_fn(B)
        if max_stake:  # bookmaker limit per bet
            stakes = np.minimum(stakes, max_stake)
        if min_stake:
            keep = stakes >= min_stake
            stakes, m = stakes[keep], mult[keep]
        else:
            m = mult
        tot = stakes.sum()
        if tot <= 0 or cash <= 0:
            continue
        if tot > cash:
            stakes = stakes * cash / tot
            tot = cash
        pending.append((date, tot, float((stakes * m).sum())))
        n_bets += len(stakes)
        staked += tot
    for sd, stake, pnl in sorted(pending, key=lambda x: x[0]):
        B += pnl
        timeline.append((sd, B))
    tl = pd.DataFrame(timeline, columns=["date", "bankroll"])
    if ruined:
        tl = pd.concat([tl, pd.DataFrame([{"date": tl.date.max() if len(tl) else t0, "bankroll": B}])])
    return tl, dict(start=start, final=B, bets=n_bets, staked=staked, ruined=ruined or B < ruin)


def path_stats(tl, start, t0, t1):
    """Drawdown and calendar-period stats from a bankroll timeline."""
    if tl.empty:
        return {}
    s = pd.concat([pd.Series([start], index=[pd.Timestamp(t0) - pd.Timedelta(days=1)]), tl.groupby("date").bankroll.last()])
    peak = s.cummax()
    out = dict(min_bankroll=round(s.min(), 2), max_dd_pct=round(float(((peak - s) / peak).max() * 100), 1))
    idx = pd.date_range(pd.Timestamp(t0).normalize(), pd.Timestamp(t1), freq="D")
    daily = s.reindex(s.index.union(idx)).sort_index().ffill().reindex(idx)
    for name, rule in (("week", "W"), ("month", "ME"), ("quarter", "QE"), ("year", "YE")):
        ends = daily.resample(rule).last()
        starts = ends.shift(1).fillna(start)
        r = (ends / starts - 1).iloc[:-1] if rule in ("YE",) and ends.index[-1] > pd.Timestamp(t1) else (ends / starts - 1)
        active = r[r != 0]
        out[f"{name}s_up_pct"] = round(float((r > 0).mean() * 100), 1)
        out[f"{name}s_down_pct"] = round(float((r < 0).mean() * 100), 1)
        out[f"worst_{name}_pct"] = round(float(r.min() * 100), 1)
        out[f"median_{name}_pct"] = round(float(r.median() * 100), 1)
    return out


if __name__ == "__main__":  # importable: research/51 reuses the bet builder and unit rules
    SCEN = [("Est. Pinnacle, mid-week (3 days before)", 6, "pin_est"), ("Est. Pinnacle, lines just opened (14 days before)", 0, "pin_est"),
            ("Est. Pinnacle, fight day", 9, "pin_est"), ("Book average, mid-week", 6, "avg"), ("REAL Pinnacle closing odds (2013-2020 only)", 9, "pin_real")]
    ERAS = [("2013-01-01 to Sep 2026", pd.Timestamp("2013-01-01"), END + pd.Timedelta(days=1)),
            ("2021-01-01 to Sep 2026", pd.Timestamp("2021-01-01"), END + pd.Timedelta(days=1))]

    rows, curves = [], {}
    for label, stage, price in SCEN:
        ev = events_arrays(scenario_bets(stage, price))
        for era, t0, t1 in ERAS:
            if price == "pin_real" and t0.year >= 2021:
                continue
            tt1 = min(t1, pd.Timestamp("2021-01-01")) if price == "pin_real" else t1
            for system in SYSTEMS:
                variants = [(0.0, None)] + ([(1.0, None), (0.0, 500.0), (0.0, 2000.0)] if system == "user" else [(0.0, 500.0), (0.0, 2000.0)] if system == "aggressive" else [])
                for min_stake, cap in variants:
                    tl, st = run(ev, system, LEAD[stage], t0, tt1, min_stake=min_stake, max_stake=cap)
                    ps = path_stats(tl, st["start"], t0, tt1)
                    tag = " (skip stakes < R$1)" if min_stake else (f" (max R${cap:,.0f} per bet)" if cap else "")
                    rows.append(dict(scenario=label, era=era if price != "pin_real" else "2013-01-01 to 2020-12-31", system=system + tag, start=st["start"],
                                     final=round(st["final"], 2), multiple=round(st["final"] / st["start"], 2), ruined=st["ruined"], bets=st["bets"],
                                     staked=round(st["staked"], 2), **ps))
                    if not min_stake and not cap:
                        curves[(label, era, system)] = tl

    res = pd.DataFrame(rows)
    res.to_csv(R / "50_bankroll_simulation.csv", index=False)


    # ------------------------------------------------------------------ "start any week" windows
    def windows(ev, system, lead, era_t0, last=END, horizons=(("1 week", 7), ("1 month", 30), ("3 months", 91), ("1 year", 365))):
        out = []
        starts = pd.date_range(era_t0, last, freq="W-MON")
        for hname, days in horizons:
            finals = []
            for s in starts:
                e = s + pd.Timedelta(days=days)
                if e > last + pd.Timedelta(days=1):
                    break
                tl, st = run(ev, system, lead, s, e)
                finals.append((st["final"] / st["start"] - 1, st["ruined"], st["bets"]))
            f = np.array([x[0] for x in finals])
            nb = np.array([x[2] for x in finals])
            out.append(dict(horizon=hname, windows=len(f), profit_pct_of_windows=round((f > 0).mean() * 100, 1), loss_pct=round((f < 0).mean() * 100, 1),
                            no_bets_pct=round((nb == 0).mean() * 100, 1), median_return_pct=round(np.median(f) * 100, 1),
                            p10_return_pct=round(np.percentile(f, 10) * 100, 1), worst_return_pct=round(f.min() * 100, 1),
                            best_return_pct=round(f.max() * 100, 1), ruined_windows=int(sum(x[1] for x in finals))))
        return out


    wrows = []
    for label, stage, price in SCEN[:3] + SCEN[4:]:
        ev = events_arrays(scenario_bets(stage, price))
        for era, t0, _ in ERAS:
            if price == "pin_real" and t0.year >= 2021:
                continue
            for system in ("user", "aggressive"):
                evx = [e for e in ev if e[0] < pd.Timestamp("2021-01-01")] if price == "pin_real" else ev
                for w in windows(evx, system, LEAD[stage], t0, last=pd.Timestamp("2020-12-31") if price == "pin_real" else END):
                    wrows.append(dict(scenario=label, starts_from=era, system=system, **w))
    W = pd.DataFrame(wrows)
    W.to_csv(R / "50_start_any_week.csv", index=False)

    # ------------------------------------------------------------------ charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    plt.rcParams["text.usetex"] = False
    plt.rcParams["axes.formatter.use_mathtext"] = False
    esc = lambda t: t.replace("$", r"\$")  # "R$1 ... R$100" would otherwise be read as math

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = {"Est. Pinnacle, lines just opened (14 days before)": "#1b5aa6", "Est. Pinnacle, mid-week (3 days before)": "#2a9d8f",
              "Est. Pinnacle, fight day": "#b0372b", "REAL Pinnacle closing odds (2013-2020 only)": "#6a4c93"}
    for ax, (era, t0, t1) in zip(axes, ERAS):
        for (label, e, system), tl in curves.items():
            if e != era or system != "user" or label not in colors or tl.empty:
                continue
            s = pd.concat([pd.Series([100.0], index=[t0]), tl.groupby("date").bankroll.last()])
            ax.plot(s.index, s.values, color=colors[label], lw=1.6, label=esc(label.replace("Est. ", "est. ")))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"R${v:,.0f}"))
        ax.axhline(100, color="#888", lw=0.8, ls=":")
        ax.set_title(esc(f"Your system (R$100, unit = R$1 per R$100), starting {t0.date()}"))
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(R / "50_bankroll_curves.png", dpi=130)

    fig, ax = plt.subplots(figsize=(14, 5.2))
    t0 = pd.Timestamp("2021-01-01")
    for system, col, start in (("user", "#2a9d8f", 100), ("aggressive", "#e76f51", 200), ("flat", "#888", 100)):
        tl = curves[("Est. Pinnacle, mid-week (3 days before)", "2021-01-01 to Sep 2026", system)]
        s = pd.concat([pd.Series([float(start)], index=[t0]), tl.groupby("date").bankroll.last()])
        ax.plot(s.index, s.values, color=col, lw=1.6, label=esc({"user": "your system (R$100, +R$1 unit per R$100)", "aggressive": "aggressive (R$200, +R$10 unit per R$100)",
                                                              "flat": "flat R$1 units (R$100)"}[system]))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"R${v:,.0f}"))
    ax.set_title("Unit systems compared: est. Pinnacle mid-week prices, starting 2021-01-01")
    ax.grid(alpha=.25, which="both")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(R / "50_unit_systems.png", dpi=130)

    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
    cols = ["scenario", "era", "system", "start", "final", "multiple", "ruined", "bets", "min_bankroll", "max_dd_pct", "months_up_pct", "months_down_pct",
            "worst_month_pct", "quarters_up_pct", "worst_quarter_pct", "years_up_pct", "worst_year_pct"]
    print(res[cols].to_string(index=False))
    print()
    print(W.to_string(index=False))

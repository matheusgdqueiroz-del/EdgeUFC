"""Predict upcoming UFC cards and recommend moneyline stakes for Pinnacle.

  python -m ufc.predict              # use cached data
  python -m ufc.predict --update     # refresh all sources first (a few minutes)
  python -m ufc.predict --days 21    # how far ahead to look (default 30)

Writes predictions/latest.json (read by the desktop app), predictions/predictions_<date>.csv and appends every
recommendation to predictions/ledger.csv (the forward record: saved before the fights happen).
Requires data/engine_oos_v2.parquet (out-of-sample history the market layers and staking policy learn from),
produced by `python research/48_v2_final.py` and refreshed with `--retrain`.
"""
import argparse, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from ufc.fetch import ROOT
from ufc.sources.ufcstats import load_fights, load_rounds
from ufc import features
from ufc.build import assemble
from ufc.upcoming import upcoming_cards
from ufc.model import feature_sets
from ufc.engine import stats_predict, choose_policy
from ufc.market2 import MarketLayers2
from ufc.betting import kelly_units
from ufc.sources import pinnacle

DATA = ROOT / "data"
OUT = ROOT / "predictions"
# Default staking: fixed conservative rule (reports/24_fixed_policies.csv, reports/48_v2_final.csv)
STEADY = dict(min_edge=0.03, frac=0.125, max_units=1.5)
# Estimated Pinnacle moneyline margin: archived closes 2.4-3.4% (research/47); early lines assumed wider
PIN_MARGIN_EARLY, PIN_MARGIN_CLOSE = 1.045, 1.035


def step(msg):
    print(f"[step] {msg}", flush=True)


def american(dec):
    if dec is None or not np.isfinite(dec):
        return ""
    return f"+{(dec - 1) * 100:.0f}" if dec >= 2 else f"-{100 / (dec - 1):.0f}"


def _num(v, nd=None):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, nd) if nd is not None else v


def line_stage(days_to_fight, p_now, p_open):
    """Where the current line sits between its opening (0) and the close (1).
    ponytail: days-to-fight heuristic (no open timestamps on BFO pages); the engine's probabilities move little
    across stages 0.1-1 (see the note printed by run), stage 0 is kept for lines that have not moved since opening."""
    if p_open is not None and p_now is not None and abs(p_now - p_open) < 0.003:
        return 0.0
    return float(np.clip(1 - days_to_fight / 28, 1 / 9, 1))


def run(update=False, retrain=False, days=30, staking="steady"):
    if update:
        from ufc import update as upd
        upd.main()

    step("Loading fight history")
    fights = load_fights()
    perf = features.build_perf(fights, load_rounds(fights))
    step("Reading upcoming UFC cards")
    up = upcoming_cards(days_ahead=days, refresh=update)
    up = up[~up.fight_id.isin(fights.fight_id)] if len(up) else up
    OUT.mkdir(exist_ok=True)
    if up.empty:
        payload = dict(generated_at=pd.Timestamp.now().isoformat(timespec="seconds"), staking=staking, policy=STEADY,
                       policies={"steady": STEADY, "growth": STEADY}, bouts=[], days=days, history_through=str(fights.date.max().date()))
        (OUT / "latest.json").write_text(json.dumps(payload, indent=1), encoding="utf8")
        return payload
    women = set(fights.loc[fights.women, "f1_id"]) | set(fights.loc[fights.women, "f2_id"])
    up["women"] = up.f1_id.isin(women) | up.f2_id.isin(women) | up.women
    allf = pd.concat([fights, up[fights.columns]], ignore_index=True).sort_values(["date", "event_id", "bout_order"], ascending=[True, True, False]).reset_index(drop=True)
    step("Building fighter profiles (records, ratings, style, travel, odds)")
    d = assemble(allf, perf, out_name="live_table.parquet", verbose=False)
    hist, live = d[d.fight_id.isin(set(fights.fight_id))], d[d.fight_id.isin(set(up.fight_id))].copy()
    fs = feature_sets(d)

    path = DATA / "engine_oos_v2.parquet"
    if retrain or not path.exists():
        from ufc.engine import full_walk_forward
        step("Re-running the walk-forward history (this takes a while)")
        full_walk_forward(hist, final=True, version="v2").to_parquet(path)
    E = pd.read_parquet(path)

    step("Training the prediction model on every completed fight")
    live["stats"] = stats_predict(hist[hist.date >= "2001-01-01"], live, fs)
    step("Comparing with the betting market")
    ml = MarketLayers2().fit(E[E.stats.notna() & E.p_s0.notna()])
    today = pd.Timestamp.today().normalize()
    live["days_to_fight"] = (live.date - today).dt.days.clip(lower=0)
    # the market the engine reads is Pinnacle's own line when a fresh one exists, else the BestFightOdds consensus
    pin = pinnacle.match(live[["fight_id", "date", "f1_name", "f2_name"]], pinnacle.latest())
    live = live.drop(columns=[c for c in ("a_pin_dec", "b_pin_dec", "pin_snapshot") if c in live]).merge(pin, on="fight_id", how="left").set_index(live.index)
    live["pin_live"] = live.a_pin_dec.notna() & live.b_pin_dec.notna()
    p_pin = (1 / live.a_pin_dec) / (1 / live.a_pin_dec + 1 / live.b_pin_dec)
    live["p_line"] = np.where(live.pin_live, p_pin, live.p_close)
    has = live.p_line.notna()
    step(f"Pinnacle lines found for {int(live.pin_live.sum())} of {len(live)} bouts" if len(pin) else "No fresh Pinnacle lines: using estimated Pinnacle prices")
    live["s"] = [line_stage(dd, pn, po if po == po else None) if h else np.nan for dd, pn, po, h in zip(live.days_to_fight, live.p_line, live.p_open, has)]
    live["prob"] = live.stats
    if has.any():
        rows = live[has]
        live.loc[has, "prob"] = ml.predict(rows).ens.values
        spread = np.abs(ml.predict(rows.assign(s=1 / 9)).ens.values - ml.predict(rows.assign(s=1.0)).ens.values)
        print(f"[note] stage sensitivity: engine probability differs by {np.median(spread):.3f} (median) / {spread.max():.3f} (max) between early and closing weights", flush=True)
    margin = np.where(live.s.fillna(1) <= 1 / 3, PIN_MARGIN_EARLY, PIN_MARGIN_CLOSE)
    live["a_pin_est"] = np.where(live.pin_live, live.a_pin_dec, 1 / (live.p_line * margin))
    live["b_pin_est"] = np.where(live.pin_live, live.b_pin_dec, 1 / ((1 - live.p_line) * margin))
    H = E[E.p2_s9.notna() & (E.date >= "2013-01-01")]
    H = H.assign(a_pin=1 / (H.p_s9 * PIN_MARGIN_CLOSE), b_pin=1 / ((1 - H.p_s9) * PIN_MARGIN_CLOSE))
    growth = choose_policy(H, "a_pin", "b_pin", prob="p2_s9") or STEADY
    policies = {"steady": STEADY, "growth": {k: float(v) for k, v in growth.items()}}
    pol = policies[staking]

    step("Sizing stakes")
    bouts = []
    for r in live.sort_values(["date", "event", "bout_order"]).itertuples():
        has_mkt = _num(r.p_line) is not None
        p = float(r.prob)
        b = dict(fight_id=r.fight_id, event=r.event, date=str(r.date.date()), location=r.location, bout_order=int(r.bout_order),
                 weightclass=r.weightclass, women=bool(r.women), title=bool(r.title), rounds=int(r.sched_rounds) if r.sched_rounds == r.sched_rounds else 3,
                 prob_a=_num(p, 4), stats_a=_num(r.stats, 4), market_a=_num(r.p_line, 4), consensus_a=_num(r.p_close, 4), open_a=_num(r.p_open, 4), stage=_num(r.s, 2),
                 pin_live=bool(r.pin_live), pin_time=r.pin_snapshot if r.pin_live else None,
                 days_to_fight=int(r.days_to_fight), bet_side=None, units=0.0, edge=None)
        for side, name in (("a", r.f1_name), ("b", r.f2_name)):
            g = lambda k, nd=None: _num(getattr(r, f"{side}_{k}", None), nd)
            b[side] = dict(name=name, age=g("age", 1), height=g("height", 0), reach=g("reach", 0), ufc_wins=g("wins", 0), ufc_losses=g("losses", 0),
                           ufc_fights=g("n_fights", 0), pro_wins=g("pro_wins", 0), pro_losses=g("pro_losses", 0), streak=g("streak", 0),
                           layoff_days=g("days_since_last", 0), fm_rank=g("fm_rank", 0),
                           stance="Southpaw" if g("southpaw") == 1 else ("Switch" if g("switch") == 1 else "Orthodox"),
                           dec_avg=g("close_mean_dec", 4), dec_best=g("close_best_dec", 4), dec_open=g("open_dec", 4), pin_est=g("pin_est", 4),
                           price_avg=american(g("close_mean_dec")), price_best=american(g("close_best_dec")), price_open=american(g("open_dec")),
                           debut=g("debut") == 1, replacement=g("replacement") == 1, missed_weight=g("missed_weight") == 1)
        if has_mkt:
            ev_a, ev_b = p * b["a"]["pin_est"] - 1, (1 - p) * b["b"]["pin_est"] - 1
            bet_a = ev_a >= ev_b
            pb, dec_bet, ev = (p, b["a"]["pin_est"], ev_a) if bet_a else (1 - p, b["b"]["pin_est"], ev_b)
            units = float(kelly_units(pb, dec_bet, pol["frac"], pol["max_units"])) if ev > pol["min_edge"] else 0.0
            units = round(units * 2) / 2 if units >= 0.25 else 0.0
            b.update(edge_a=_num(ev_a, 4), edge_b=_num(ev_b, 4))
            if units > 0:
                b.update(bet_side="a" if bet_a else "b", units=units, edge=_num(ev, 4))
        bouts.append(b)

    now = pd.Timestamp.now().isoformat(timespec="seconds")
    payload = dict(generated_at=now, staking=staking, policy=pol, policies=policies, days=days, engine="v2",
                   history_through=str(fights.date.max().date()), bouts=bouts)
    (OUT / "latest.json").write_text(json.dumps(payload, indent=1), encoding="utf8")
    flat = [dict(generated_at=now, date=x["date"], event=x["event"], fight_id=x["fight_id"], fighter_a=x["a"]["name"], fighter_b=x["b"]["name"],
                 prob_a=x["prob_a"], market_a=x["market_a"], stage=x["stage"], pin_live=x["pin_live"], pin_est_a=x["a"]["pin_est"], pin_est_b=x["b"]["pin_est"],
                 bet_on=x[x["bet_side"]]["name"] if x["bet_side"] else "", units=x["units"], edge=x["edge"], staking=staking) for x in bouts]
    flat = pd.DataFrame(flat)
    flat.to_csv(OUT / f"predictions_{pd.Timestamp.today():%Y-%m-%d}.csv", index=False)
    ledger = OUT / "ledger.csv"
    flat.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)
    step("Done")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--retrain", action="store_true", help="recompute the out-of-sample history (slow)")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--staking", choices=["steady", "growth"], default="steady",
                    help="steady: 1/8 Kelly, edge>3%%, max 1.5u (low drawdown). growth: policy maximising historical growth (bigger swings)")
    a = ap.parse_args(argv)
    payload = run(a.update, a.retrain, a.days, a.staking)
    for ev in dict.fromkeys(b["event"] for b in payload["bouts"]):
        print(f"\n=== {ev} ===")
        for b in (x for x in payload["bouts"] if x["event"] == ev):
            pa = b["prob_a"]
            pick = b["a"] if pa >= .5 else b["b"]
            src = "Pinnacle" if b["pin_live"] else "est. Pinnacle"
            bet = f"BET {b['units']}u {b[b['bet_side']]['name']} ({src} {b[b['bet_side']]['pin_est']:.2f})" if b["bet_side"] else "no bet"
            print(f"  {b['a']['name']} vs {b['b']['name']}: {pick['name']} {max(pa, 1 - pa):.0%}  | {bet}")
    pol = payload["policy"]
    print(f"\nStaking ({a.staking}): bet when edge > {pol['min_edge']:.0%}, {pol['frac']:g} Kelly, max {pol['max_units']:g} units. 1 unit = 1% of bankroll.")


if __name__ == "__main__":
    main()

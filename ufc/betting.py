"""Betting simulation and staking policy selection.

Conventions
  * 1 unit = 1% of the starting bankroll (flat-unit accounting) unless `compound=True`.
  * No contest -> stake refunded. Draw -> moneyline bet loses (standard at most books).
  * A policy is (min_edge, kelly_fraction, max_units, price column). Choosing a policy for year Y may only
    look at simulated bets from years < Y (see `walk_forward_policy`).
"""
import itertools
import numpy as np
import pandas as pd


def kelly_units(p, dec, frac, max_units, unit_pct=1.0):
    """Fractional Kelly stake expressed in units (1 unit = unit_pct% of bankroll)."""
    f = (p * dec - 1) / (dec - 1)
    return np.clip(frac * f * 100 / unit_pct, 0, max_units)


def simulate(df, p_col, a_dec, b_dec, min_edge=0.03, frac=0.25, max_units=3.0, flat=False, min_units=0.25,
             p_lo=0.0, p_hi=1.0):
    """df needs: y (1 a wins, 0 b wins, NaN NC), outcome ('D/D' draws), p_col, odds columns. One bet max per fight."""
    d = df[[p_col, a_dec, b_dec, "y", "outcome", "date"]].dropna(subset=[p_col, a_dec, b_dec]).copy()
    pa, da, db = d[p_col].values, d[a_dec].values, d[b_dec].values
    ev_a, ev_b = pa * da - 1, (1 - pa) * db - 1
    side_a = ev_a >= ev_b
    p = np.where(side_a, pa, 1 - pa)
    price = np.where(side_a, da, db)
    ev = np.where(side_a, ev_a, ev_b)
    stake = np.where(flat, 1.0, kelly_units(p, price, frac, max_units))
    ok = (ev > min_edge) & (p >= p_lo) & (p <= p_hi) & (stake >= (min_units if not flat else 0))
    stake = np.where(ok, stake, 0.0)
    won = np.where(side_a, d.y.values == 1, d.y.values == 0)
    nc = d.y.isna().values & (d.outcome.values != "D/D")
    profit = np.where(nc, 0.0, np.where(won, stake * (price - 1), -stake))
    d["bet_side"] = np.where(ok, np.where(side_a, "a", "b"), "")
    d["stake"], d["price"], d["p_bet"], d["ev"], d["profit"] = stake, price, p, ev, profit
    return d[ok]


def summary(bets):
    if bets.empty:
        return dict(bets=0, staked=0.0, profit=0.0, roi=np.nan, hit=np.nan, max_dd=0.0, avg_price=np.nan)
    cum = bets.sort_values("date").profit.cumsum()
    dd = (cum.cummax().clip(lower=0) - cum).max()
    return dict(bets=len(bets), staked=round(bets.stake.sum(), 2), profit=round(bets.profit.sum(), 2),
                roi=round(bets.profit.sum() / bets.stake.sum(), 4), hit=round((bets.profit > 0).mean(), 3),
                max_dd=round(dd, 2), avg_price=round(bets.price.mean(), 3), avg_units=round(bets.stake.mean(), 2))


GRID = dict(min_edge=[0.0, 0.02, 0.04, 0.06, 0.08, 0.12], frac=[0.1, 0.25, 0.5], max_units=[2.0, 5.0])


def walk_forward_policy(df, p_col, a_dec, b_dec, first_year, last_year, grid=GRID, criterion="log_growth", min_hist_bets=150, max_drawdown=None):
    """For each year Y choose the policy that did best on years < Y (out-of-sample predictions only),
    then apply it to Y. Returns the concatenated bets actually 'placed' and the chosen policy per year."""
    placed, chosen = [], []
    yrs = df.date.dt.year
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    for Y in range(first_year, last_year + 1):
        hist = df[yrs < Y]
        best, best_val = None, -np.inf
        for c in combos:
            b = simulate(hist, p_col, a_dec, b_dec, **c)
            if len(b) < min_hist_bets:
                continue
            if max_drawdown is not None and summary(b)["max_dd"] > max_drawdown * max(1, (Y - first_year + 5) / 5):
                continue  # drawdown cap scaled with history length (a longer history contains deeper dips)
            if criterion == "log_growth":  # growth of a bankroll risking stake% per bet (units are % of bankroll)
                val = np.sum(np.log1p(b.profit.values / 100.0))
            else:
                val = b.profit.sum() / max(b.stake.sum(), 1)
            if val > best_val:
                best, best_val = c, val
        if best is None:
            continue
        cur = simulate(df[yrs == Y], p_col, a_dec, b_dec, **best)
        placed.append(cur)
        chosen.append(dict(year=Y, **best, hist_value=round(best_val, 3), **summary(cur)))
    return (pd.concat(placed) if placed else pd.DataFrame()), pd.DataFrame(chosen)


if __name__ == "__main__":
    # a sure edge must make money; a zero edge must place no bets
    t = pd.DataFrame(dict(date=pd.to_datetime(["2020-01-01"] * 4), y=[1, 1, 0, 1.0], outcome=["W/L"] * 4,
                          p=[0.6, 0.6, 0.6, 0.6], a=[2.2] * 4, b=[1.7] * 4))
    b = simulate(t, "p", "a", "b", min_edge=0.05, frac=0.5, max_units=5)
    assert len(b) == 4 and (b.bet_side == "a").all() and b.profit.sum() > 0
    assert simulate(t.assign(a=1.6), "p", "a", "b", min_edge=0.0).empty
    assert abs(kelly_units(0.6, 2.2, 1.0, 100) - (0.6 * 2.2 - 1) / 1.2 * 100) < 1e-9
    print("betting ok", summary(b))

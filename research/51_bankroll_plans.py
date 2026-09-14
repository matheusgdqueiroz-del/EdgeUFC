"""Which starting bankroll, unit size and step-up rule give the best shot at big growth without losing the bankroll?

Future paths are built by resampling whole MONTHS of the engine's out-of-sample bets from 2021-Sep 2026 (keeps the
real mix of good and bad cards together), 3 years each, 4,000 paths per plan. Three edge assumptions, from the bets'
own history: bet when lines open (strong), 3 days before (medium), fight day (weak edge ~ what a sharper future market
could leave). Stake per bet: steady rule in units (edge > 3%, 1/8 Kelly, max 1.5u, rounded to 0.5u), unit =
k reais per full R$100 of bankroll (below R$100: k% of the bankroll). Bets on a card never exceed the cash in hand.
Stakes under R$1 are skipped (bookmaker minimums). Optional monthly deposit.
"""
import importlib.util, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.fetch import ROOT

spec = importlib.util.spec_from_file_location("sim50", ROOT / "research" / "50_bankroll_simulation.py")
sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)

N_PATHS, MONTHS, SEED = 4000, 36, 7
CASES = {"lines open": 0, "3 days before": 6, "fight day": 9}
K = (1, 1.5, 2, 3, 4, 5)
STARTS = (100, 300, 500)


def month_blocks(stage, margin=None):
    b = sim.scenario_bets(stage, "pin_est", margin)
    b = b[b.date >= "2021-01-01"]
    cards = []
    for ev, g in b.groupby("event_id"):
        cards.append((g.date.iloc[0], g.units.values, g.mult.values))
    cards.sort(key=lambda c: c[0])
    M = max(len(c[1]) for c in cards)
    U = np.zeros((len(cards) + 1, M)); X = np.zeros((len(cards) + 1, M))  # last row = empty card (padding)
    for i, (_, u, m) in enumerate(cards):
        U[i, :len(u)], X[i, :len(m)] = u, m
    months = pd.Series([c[0].strftime("%Y-%m") for c in cards])
    all_months = pd.period_range("2021-01", pd.Timestamp(cards[-1][0]).strftime("%Y-%m"), freq="M").astype(str)
    blocks = [np.flatnonzero(months.values == mo) for mo in all_months]  # months with no card stay empty
    return U, X, blocks


def simulate(U, X, blocks, k, start, deposit=0.0, seed=SEED, months=MONTHS, max_stake=None, years=(1, 2, 3)):
    """Returns final bankroll, lowest bankroll, bankroll after years 1-3 and total deposited, one value per path."""
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(blocks), (N_PATHS, months))
    empty = len(U) - 1
    lens = np.array([[len(blocks[j]) for j in row] for row in pick])
    ends = np.cumsum(lens, 1)  # cards played by the end of each month
    T = int(ends[:, -1].max())
    C = np.full((N_PATHS, T), empty)
    for i, row in enumerate(pick):
        s = np.concatenate([blocks[j] for j in row])
        C[i, :len(s)] = s
    dep_at = np.zeros((N_PATHS, T + 1))
    if deposit:
        for i in range(N_PATHS):
            np.add.at(dep_at[i], ends[i, :-1], deposit)  # top-up at the start of every month after the first
    B = np.full(N_PATHS, float(start))
    low, hist, deposited = B.copy(), np.zeros((N_PATHS, T + 1)), np.full(N_PATHS, float(start))
    hist[:, 0] = B
    for t in range(T):
        B = B + dep_at[:, t]; deposited += dep_at[:, t]
        unit = np.where(B >= 100, k * np.floor(B / 100), k * B / 100)
        st = U[C[:, t]] * unit[:, None]
        st = np.where(st >= 1.0, st, 0.0)
        if max_stake:
            st = np.minimum(st, max_stake)
        tot = st.sum(1)
        st *= np.where(tot > B, np.maximum(B, 0) / np.maximum(tot, 1e-9), 1.0)[:, None]
        B = B + (st * X[C[:, t]]).sum(1)
        low = np.minimum(low, B)
        hist[:, t + 1] = B
    B = B + dep_at[:, T]; deposited += dep_at[:, T]
    idx = np.arange(N_PATHS)
    snaps = {y: hist[idx, ends[:, 12 * y - 1]] for y in years}
    return B, low, snaps, deposited


if __name__ == "__main__":
    rows = []
    for case, stage in CASES.items():
        U, X, blocks = month_blocks(stage)
        for start in STARTS:
            for k in K:
                B, low, snaps, dep = simulate(U, X, blocks, k, start)
                y1 = snaps[1]
                rows.append(dict(edge_case=case, start=start, unit_per_100=k, unit_pct=k,
                                 median_1y=round(np.median(y1)), median_2y=round(np.median(snaps[2])), median_3y=round(np.median(B)),
                                 bad_luck_3y_p10=round(np.percentile(B, 10)), good_luck_3y_p90=round(np.percentile(B, 90)),
                                 chance_down_after_1y=round((y1 < start).mean() * 100, 1), chance_down_after_3y=round((B < start).mean() * 100, 1),
                                 chance_ever_half=round((low <= start * 0.5).mean() * 100, 1), chance_ever_80pct_lost=round((low <= start * 0.2).mean() * 100, 1),
                                 chance_10x_3y=round((B >= 10 * start).mean() * 100, 1)))
        print(case, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "reports" / "51_bankroll_plans.csv", index=False)

    # the same plans with a R$50 monthly top-up (start R$300)
    dep_rows = []
    for case, stage in CASES.items():
        U, X, blocks = month_blocks(stage)
        for k in (1, 2, 3):
            B, low, snaps, dep = simulate(U, X, blocks, k, 300, deposit=50.0)
            dep_rows.append(dict(edge_case=case, start=300, monthly_deposit=50, unit_per_100=k, total_deposited=round(np.median(dep)),
                                 median_3y=round(np.median(B)), bad_luck_3y_p10=round(np.percentile(B, 10)), good_luck_3y_p90=round(np.percentile(B, 90)),
                                 chance_below_deposits_3y=round((B < dep).mean() * 100, 1)))
    D = pd.DataFrame(dep_rows)
    D.to_csv(ROOT / "reports" / "51_bankroll_plans_deposits.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(R.to_string(index=False)); print(); print(D.to_string(index=False))

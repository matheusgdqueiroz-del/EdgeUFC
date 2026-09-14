# UFC Edge Engine

Predicts UFC fights as calibrated win probabilities, compares them with the betting market, and recommends
a moneyline stake (in units) for Pinnacle only when the price is attractive. Every number in the research was produced
walk-forward: a prediction for a fight uses only information that existed before that fight.

## Use it

Double-click **UFC Edge** on the Desktop (or `UFC Edge.lnk` in this folder). The app opens in its own window:

- **Fight cards**: every upcoming bout with the engine's win probability, the market's, and a stake in units
  and money at the estimated Pinnacle price. Type the odds Pinnacle shows under each fighter for an exact stake;
  fights without a bet show the minimum odds that would make them one. Click the probability bar for the tale of
  the tape and the reasoning.
- **Update & predict**: downloads new results, odds and news, then re-prices every card (8–12 minutes).
- **Settings**: bankroll, currency, odds format (decimal / American), staking style (Steady / Growth), days ahead,
  and the Pinnacle odds feed key (The Odds API, free plan: 500 credits a month, 1 per update).

Prices are Pinnacle's own moneylines whenever Pinnacle lists the fight (`ufc/sources/pinnacle.py`); otherwise an estimate
(consensus fair price with a Pinnacle-sized margin) labelled "est. Pinnacle". Every Pinnacle fetch is saved to
`data/pinnacle/snapshots.csv`, so a real Pinnacle price history builds up from 14 Sep 2026. Past Pinnacle odds from
The Odds API need a paid plan, so backtests use archived Pinnacle closes (2008–2020) and the calibrated estimate after that.
- **Track record**: the full walk-forward audit.

The app is a small local server (`ufc/app.py`, standard library only) shown in an Edge app window; it stops by itself
two minutes after the window closes.

### First-time setup on a new computer

```bash
git clone https://github.com/matheusgdqueiroz-del/UFCEdge.git
```

```bash
pip install -r requirements.txt
```

```bash
python setup_data.py
```

`setup_data.py` downloads the public datasets that are not stored in the repository (the UFCStats mirror is required).
Then start the app with `pythonw -m ufc.app` from the project folder (or make a desktop shortcut to it), open
Settings and paste the Odds API key again: it is stored only in `predictions/settings.json`, which git ignores.
Page caches are not in the repository either; the first "Update & predict" re-downloads what it needs.

### Bankroll simulation

`python research/50_bankroll_simulation.py` replays the engine with real money rules (start R$100, unit R$1 per R$100;
or an aggressive R$200 / +R$10 per R$100 system) week by week. Results: `reports/50_*.csv` and `reports/50_*.png`.

Command-line equivalent:

```bash
python -m ufc.predict --update
```

That refreshes every source (a few minutes), builds features for the next 30 days of UFC cards, and prints
one table per event. Predictions are also saved to `predictions/predictions_<date>.csv`.

| column | meaning |
|---|---|
| `pick`, `win_prob` | who the engine favours and how likely (market-aware engine when odds exist) |
| `stats_model` | the market-free model's view of the same fighter (useful as a second opinion) |
| `market_prob` | vig-free probability implied by the current average line |
| `bet_on`, `price_avg`, `price_best` | the side with positive expected value, average and best current prices |
| `edge` | expected return per unit at the average price |
| `units` | recommended stake, 1 unit = 1% of your bankroll (0 = no bet) |
| `notes` | UFC debut, short-notice replacement, missed weight |

Every run appends its recommendations to `predictions/ledger.csv` before the fights: the forward track record.

Rules that come from the research, not taste:
- The edge is largest **when lines open** (see the report). Bet early at the price shown, never at a worse one.
- Recheck close to the fight: an edge that disappears at a worse price is no longer a bet.
- Default stakes: 1/8 Kelly, only when edge > 3%, max 1.5 units. `--staking growth` uses the policy that grew
  fastest historically (about 3x bigger stakes and drawdowns). Do not scale either up.

Other commands:

```bash
python -m ufc.update
```

```bash
python -m ufc.predict --retrain
```

`--retrain` recomputes the out-of-sample history the market layers and staking policy learn from
(run monthly or after a big data refresh; takes about 20–40 minutes).

## Version 2 (September 2026)

Chosen on 2010–2020 fights only (every attempt logged in `reports/v2_attempts.csv`), then re-checked once on
2021–Sep 2026 (already used for v1, so a re-check rather than an untouched test; `reports/48_v2_final.txt`).
Steady rule at estimated Pinnacle prices (consensus fair price with a 3.5% margin, 4.5% early):

| bet placed | v1 bets per card | v2 bets per card | v2 return (90% CI) | v2 years up |
|---|---|---|---|---|
| when lines open | 5.8 | 6.1 | +14.3% (+10.5 to +18.3) | 6 of 6 |
| mid-way to the fight | – | 5.7 | +7.6% (+3.8 to +11.5) | 4 of 6 |
| near fight time | 2.1 | 4.8 | +4.7% (+0.6 to +8.9) | 5 of 6 |

What changed: logistic part of the stats model regularised harder, split decisions as soft labels, level of the
promotions a fighter came from (`ufc/sources/promotion_features.py`), and a market layer trained on 10 points of
every line's life (BestFightOdds sparklines) so it knows how far to trust an early versus a late line (`ufc/market2.py`).
Pinnacle realism (`research/47_pinnacle.py`): archived Pinnacle closes 2008–2020 had a 2.4–3.4% margin and were
no sharper than the consensus; settling bets at real Pinnacle prices gave more bets and similar returns.
Prop markets (method of victory) looked profitable in research (`research/45_props.py`) but are not in the app:
Pinnacle does not offer them.

## What the evidence says for v1 (details: `reports/ufc_edge_audit.html`)

Scored on fights the design never saw (validation 2021–2024, locked holdout Jan 2025–Sep 2026):

| | closing line | engine | opening line | engine (open) |
|---|---|---|---|---|
| log loss 2021–24 | 0.6018 | 0.5992 | 0.6246 | 0.6147 |
| log loss 2025–26 | 0.5779 | 0.5716 | 0.5973 | 0.5820 |

- Beats the closing line in 6 of 6 untouched years (small gain, t = 3.4); beats opening lines clearly (t = 5.0).
- Default stake rule (1/8 Kelly, edge > 3%, max 1.5u), 2021–Sep 2026: +12.1% return over 1,418 opening-price
  bets (worst dip −18u, 6/6 years up); +9.6% over 324 near-close bets at a 5%-margin book (not significant).
- No edge on UFC debutants or light heavyweight; most pre-2021 betting "angles" faded afterwards.

## How it works

```
data sources ─► chronological feature engine ─► market-free model ─► market-aware layers ─► Kelly staking
```

1. **Data** (all re-collectable for future fights): UFCStats round-by-round stats (Greco1899 daily mirror),
   Sherdog full pro records and bios, BestFightOdds opening and closing lines, FightMatrix pre-fight ratings,
   Wikipedia weigh-in misses and late replacements, Open-Meteo geography for venues and home towns.
2. **Feature engine** (`ufc/features.py`, `ufc/skills.py`, `ufc/sources/*_features.py`): processes fight dates in
   order; each fighter is snapshotted before the date, then updated. ~300 features: opponent-adjusted
   striking/grappling, a Bayesian 14-skill Poisson state-space model, Elo/Glicko variants, strength of
   schedule re-rated with current ratings, pre-UFC record and a global Elo over all pro fights, FightMatrix
   ratings, travel/time-zone/altitude, style matchups, head-to-head and common opponents.
3. **Market-free model** (`ufc/engine.py: stats_predict`): LightGBM + logistic regression, symmetric in corners.
4. **Market-aware layers** (v2, `ufc/market2.py`): a stack of the line and the model whose weights depend on how
   old the line is and how experienced the fighters are, plus an offset ridge regression that corrects the line only
   where evidence exists. Live, the line's age is estimated from days to the fight.
5. **Staking**: fractional Kelly. The pre-registered evaluation picked (edge threshold, fraction, cap) each year
   from earlier out-of-sample bets only; the live default is a fixed conservative rule chosen afterwards for its
   low drawdown (reported separately in `reports/24_fixed_policies.csv`, so treat its history as less pristine).

## Honesty measures

- `reports/PREREGISTRATION.md`: the design was frozen before scoring 2021–2024 and the locked 2025–2026 holdout.
- `research/00_leakage_test.py`, `research/00b_leakage_sherdog.py`: features for fights before a cut-off are
  bit-identical whether or not later data exists (and with shuffled inputs).
- Odds cross-checked against two independent archives; profit re-settled at their prices.
- Controls: a recalibrated market alone and a market-plus-noise model both fail to make money.

## Reproduce the research

```bash
python -m ufc.build
```

Then run the scripts in `research/` in numeric order; results land in `reports/`.
Crawlers (`python -m ufc.sources.sherdog`, `.bfo`, `.fightmatrix`, `.wiki`) cache every page under
`data_raw/cache`, so re-runs are incremental.

## Known limits

- Opening lines often carry low betting limits; edges there are real in history but not scalable.
- UFCStats itself blocks scripts, so stats arrive via the daily GitHub mirror (a day of lag).
- A few old results were later overturned; features treat them as the final record.
- FightMatrix can backfill forgotten regional fights, slightly revising old ratings.

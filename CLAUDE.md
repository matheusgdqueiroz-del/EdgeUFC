# EdgeUFC ("UFC Edge")

## What it is
A personal sports-betting tool for one user in Brazil. It predicts upcoming events, compares its probabilities with the
betting market and recommends stakes. Today it covers UFC moneylines, bet at Pinnacle Brazil in reais. More sports may be added.

## Goal and philosophy
- Long-term bankroll growth from a real edge over the market, validated honestly.
- Predictions use only information available before the event; backtests are walk-forward (train on the past, predict what comes next).
- Data sources must keep being collectable for future events, otherwise they are useless live.
- Improvements should hold up out of sample, not just fit history.
- Simple, polished user experience on top; complexity under the hood.

## The user
- Bets as soon as lines are posted (early prices are where the edge has been largest).
- Small bankroll he can afford to lose; wants compounding growth and steadier months.
- Prefers plain-language explanations, concise answers and tables, and candid rather than overprotective talk about risk.

## How the repo is organized
- `ufc/` application package
  - `sources/` data collectors: fight stats, fighter records, historical odds, live Pinnacle odds, ratings, news, geography
  - `features.py`, `skills.py`, `build.py`: build a per-bout feature table from pre-fight snapshots
  - `engine.py`, `market2.py`: market-free stats model plus market-aware layers that blend it with the current line
  - `predict.py`: prices upcoming cards, writes `predictions/latest.json`, appends every recommendation to `predictions/ledger.csv`
  - `app.py`, `app_ui/`: local desktop app (fight cards, stakes, settings, track record)
- `research/`: numbered experiments and backtests; outputs in `reports/`
- `reports/PREREGISTRATION.md`, `reports/v2_attempts.csv`: record of design decisions and what was tried
- `data/`: processed datasets (in git); `data_raw/`: third-party repos and page caches (not in git, `setup_data.py` restores the repos)

## Current state (September 2026)
- Engine "v2": stats model (LightGBM + logistic regression) plus a market layer that accounts for how early or late the line is.
- Live prices come straight from Pinnacle; when Pinnacle has no line, an estimate is derived from other sportsbooks.
- Backtests: small edge near fight time, largest at opening lines (roughly +14% ROI at opening prices vs +5% near the close,
  2021–2026). That period has already been used for evaluation, so treat it as optimistic; the ledger started 2026-09-14 is the clean test.
- Staking: fractional Kelly in units, with the unit tied to bankroll size.
- Prop-market research exists but is not in the product (the user only bets moneylines).

## Running
- First time: `pip install -r requirements.txt`, then `python setup_data.py`
- App: `pythonw -m ufc.app` · command line: `python -m ufc.predict --update`
- Development machine is Windows with Miniconda Python.

## Open directions
- Add other sports.
- Steadier monthly results; track closing-line value from the ledger.
- Notify the user when new Pinnacle lines appear.

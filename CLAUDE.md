# EdgeUFC (app name "UFC Edge") — guide for AI assistants

Read this before touching the code. It exists so new threads don't re-derive (or misunderstand) the project.

## What this is
- A personal tool for one user (Matheus, Brazil) who bets **UFC moneylines only, at Pinnacle Brazil (pinnacle.bet.br), in reais**.
- It predicts win probabilities for upcoming UFC bouts, compares them with the market, and recommends a stake in units.
  Desktop app: double-click "UFC Edge" (`pythonw -m ufc.app`).
- Goal: long-run bankroll growth from a genuine, honestly validated edge, compounding over years.

## What it is NOT (common misconceptions)
- Not built with hindsight: every feature is a snapshot taken before the fight; models retrain yearly walk-forward.
- Not a big edge everywhere: the edge is small near the close and **concentrated in early (opening) lines**.
- Not a props tool: method/decision prop research exists (research/45–46) but is **not in the product** (Pinnacle BR has no such markets).
- Not using The Odds API anymore: replaced by direct scraping of Pinnacle's public JSON (commit 8471b17). An old key may
  still sit in `predictions/settings.json` (git-ignored, unused).
- Not a clean holdout on 2021–2026: that period was used to score v1 and re-check v2. See the protocol below.
- Not guaranteed profit, and not profitable every month (about 1 month in 4 loses even in the best backtest).

## The user and how to work with them
- Bets the moment Pinnacle posts a line; wants a line-open notification (not built yet).
- Bankroll plan recommended by research/52: start R$500, unit = R$5 per full R$100 of bankroll (steps down after losses too).
  The user can afford to lose R$500; don't be overprotective, but stay candid about uncertainty.
- Wants plain-language reports (no stats jargon), concise, tables, one clear next step.
- Wants thorough research and real improvements, never tuning that only fits the past.

## Run it
- App: `pythonw -m ufc.app` (local server on 127.0.0.1:8765 in an Edge window; stops 2 min after the window closes).
- Picks from the command line: `python -m ufc.predict --update` (refresh sources, predict next 30 days).
- Rebuild the feature table: `python -m ufc.build` (~7 min) → `data/model_table.parquet`.
- Rebuild the out-of-sample history used live: `python research/48_v2_final.py` or `python -m ufc.predict --retrain`
  → `data/engine_oos_v2.parquet`. Required after any feature/model change.
- New computer: `pip install -r requirements.txt`, `python setup_data.py` (clones data_raw repos; `data_raw/greco` is required).
- Windows + Git Bash quirks: set `PYTHONPATH` to the project root and `PYTHONIOENCODING=utf-8`; inline heredoc Python with
  relative paths can fail with "Bad file descriptor" (write the script to a file, use absolute paths, add `< /dev/null`).

## Pipeline
sources → `ufc/build.py` (assemble) → `data/model_table.parquet` (one row per bout; side features `a_*`/`b_*`) →
stats model → market layers → staking → `predictions/latest.json` → app (`ufc/app.py`, `ufc/app_ui/index.html`).

Sources (all obtainable for future fights):
- UFCStats via the Greco1899 daily CSV mirror (`data_raw/greco`; UFCStats blocks scripts) — `ufc/sources/ufcstats.py`.
- Sherdog pro records/bios — `sherdog.py`, `sherdog_features.py` (global Elo, pre-UFC record), `promotion_features.py` (level of competition, v2).
- BestFightOdds fighter pages: opening line, closing range, 10-point average-price sparkline = line stages 0 (open) … 9 (close) — `bfo.py`, `odds.py`.
- FightMatrix ratings, Wikipedia weigh-in misses/replacements, Open-Meteo geography (altitude, travel, time zones).
- Pinnacle live moneylines — `ufc/sources/pinnacle.py` (public JSON at sports2.pinnacle.bet.br, sportId 22, no auth). Every scrape is
  appended to `data/pinnacle/snapshots.csv`: the only real Pinnacle price history after 2020.
- Research-only archives: UFC_Final (real Pinnacle closing odds 2008–2020), ultimate_ufc_dataset.

Feature engine: `ufc/features.py` (chronological, opponent-adjusted stats, Elo/Glicko variants), `ufc/skills.py` (Bayesian Poisson skills),
`ufc/market_ratings.py` (market-implied ratings from past lines).

Models (`ufc/engine.py`, `version="v2"` is the default):
- stats (market-free): LightGBM (`data/lgbm_best.json`) + logistic regression C=0.003, symmetric in corners, soft labels 0.7 for split/majority decisions and DQs.
- market: `ufc/market2.py` `MarketLayers2` = mean of a stage-aware stack and a stage-aware offset-ridge "edge GLM", trained on line stages 0/3/6/9.
  Live: market input = Pinnacle's line when scraped, else the BFO consensus; stage estimated from days to the fight (`line_stage` in `ufc/predict.py`).
- v1 (`data/engine_oos.parquet`, `ens3` / `stack_open`) is kept only to reproduce the v1 audit.

Staking ("steady"): bet when expected edge > 3%, stake = 1/8 Kelly in units, max 1.5u, rounded to 0.5u. Price = real Pinnacle odds when scraped,
else estimate = consensus fair probability × margin (4.5% early, 3.5% late). The app lets the user type Pinnacle odds and shows the minimum odds that still make a bet.

Outputs: `predictions/latest.json` (app), `predictions/ledger.csv` (every recommendation saved before the fight = forward test),
`reports/ufc_edge_audit.html` (Track record tab, built by `research/30_report.py`).

## Research protocol (non-negotiable)
- `reports/PREREGISTRATION.md` holds the frozen designs: v1, Addenda A–C and corrections. Read it before evaluating anything.
- Periods: develop and tune on **DEV 2010–2020 only**. 2021-01-01 → 2026-09-12 is a **re-used period** (label it so). From 2026-09-14 the
  forward ledger (plus Pinnacle snapshots) is the only clean test.
- Log every attempted change, accepted or not, in `reports/v2_attempts.csv` (event-clustered paired bootstrap on DEV). No silent retries.
- Leakage canaries must pass after feature changes: `research/00_leakage_test.py`, `00b_leakage_sherdog.py`, `00c_leakage_promo.py`
  (truncate data at a date and shuffle inputs; earlier features must stay bit-identical). Never use constants computed over all years.
- Only data obtainable before a future fight. Beware "current snapshot" fields (e.g. a fighter's gym today) that leak later information.
- Sources ruled out: mmadecisions.com (robots disallow), betmma.tips (disallows AI crawlers), BFO line-chart API (obfuscated), OddsPortal (age-gated in Brazil).

## Key results (don't re-derive)
- Log loss: stats model ≈ 0.63 on DEV; closing line ≈ 0.61. Engine vs closing line +0.004 (2021–26); vs opening line +0.014.
- v2 steady rule at estimated Pinnacle prices, 2021 → Sep 2026 (re-used): bet at open 6.1 bets/card, ROI +14.3%; mid-way +7.6%; fight day +4.7%.
- Real Pinnacle closing odds 2013–2020: profitable (research/47, 50).
- Bankroll simulations (research/50–52): betting at open, R$100 with R$1 per R$100 → R$603 (2021 → 2026), ~75% of months up, every 1-year window up.
  5-year resampling: R$5 per R$100 is the fine line (~3% chance of ever losing half); R$8–10 per R$100 → 13–25%.
- Weak spots: no edge on UFC debutants; edge shrinks as the line matures.

## Known pitfalls (already fixed; don't reintroduce)
- Incremental BFO updates wrote mixed date formats → `odds.load_lines` parses ISO8601 leniently and `update.py` normalises dates.
- Name matching: reversed order (Wang Cong / Cong Wang) and Pinnacle nicknames (Patricio Pitbull, Renato Moicano) — `odds._sim`, `pinnacle.match`.
- `data/engine_oos_v2.parquet` goes stale after feature changes; rebuild before live predictions.
- Page caches (`data_raw/cache`) are not in git; `python -m ufc.update` re-fetches what it needs.

## Open work
- Engine improvements aimed at steadier months: more genuine early-line bets, closing-line-value tracking from the ledger and Pinnacle snapshots.
- Line-open notification when Pinnacle posts UFC lines (poll `pinnacle.current()`).
- Compare real Pinnacle opening prices and limits with the backtest assumptions once snapshots accumulate.

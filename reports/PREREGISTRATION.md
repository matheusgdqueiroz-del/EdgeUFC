# Pre-registration (frozen before scoring VALIDATION 2021–2024 and the LOCKED HOLDOUT 2025-01-01 → 2026-09-12)

Written after exploratory work on the DEV period (2010–2020) and BEFORE any model, feature set,
threshold or staking rule was scored on 2021+ data. Anything changed after this point must be reported
as a deviation.

## Why this exists
All feature discovery, hypothesis tests and model comparisons so far used 2008–2020. Numbers reported
on that period are selection-biased (I chose what worked there). Only the periods below give honest
estimates of how the engine would have done in real time.

## Data (all re-collectable for future fights)
UFCStats (Greco1899 daily CSV mirror), Sherdog (full pro records, bios), BestFightOdds (open + closing
lines), FightMatrix (pre-fight ratings), Wikipedia (weigh-in misses, replacements), Open-Meteo geocoding.
Wikimedia pageviews (hype hypothesis) are EXCLUDED from the primary models: the API throttles anonymous
clients so hard that collection could not complete reliably; any pageview analysis is exploratory only.

## Models (walk-forward, retrained every Jan 1 on all bouts dated before it)
1. `stats` — market-free ensemble: mean of LightGBM (hyperparameters from research/11 on DEV) and
   L2 logistic regression, all non-market features, symmetric (both corner orientations).
2. `stack` — logit p = a·logit(closing line) + b·logit(stats), fitted on out-of-sample years only.
3. `edge_glm` — offset ridge logistic around the closing line with the fixed candidate list in
   research/14 (defence, strikes absorbed, TD volume, age, replacement, missed weight, debut,
   main-card 37.5–45% band, ≤20% longshot band, open→close steam, model–market disagreement × debut);
   penalty chosen by nested time split inside each training window.
4. `resid_lgbm` — LightGBM correction on top of logit(closing line).
5. `ens3` — mean of 2, 3, 4. **Primary market-aware model.**
6. `stack_open` — stack on the OPENING line (for bets placed when lines open).

## Staking (walk-forward)
Fractional Kelly in units (1u = 1% bankroll); each year's (min edge, Kelly fraction, max units) chosen
only from simulated out-of-sample bets of earlier years, maximising log bankroll growth
(grid in ufc/betting.py). One bet max per bout. NC = refund, draw = loss.

## Primary outcomes
- O1 prediction: log-loss gain of `ens3` over the closing line (paired bootstrap), VALID and HOLDOUT.
- O2 prediction: `stats` vs closing line and vs an Elo+age logistic baseline.
- O3 betting (conservative): `ens3` at a synthetic single book = fair closing probability with a 5% margin.
- O4 betting (realistic early bettor): `stack_open` at opening prices, plus closing-line value.
Secondary: average closing price, best closing price, 7% margin, calibration, subgroup breakdowns.

## Success criteria
- Prediction edge claimed only if O1 gain > 0 with 95% CI excluding 0 on VALID, and gain > 0 on HOLDOUT.
- Betting edge claimed only if O3 ROI > 0 with bootstrap 90% CI excluding 0 on VALID AND ROI > 0 on HOLDOUT.
- Otherwise the report states that no reliable edge over the market was demonstrated.

## Frozen details (recorded 2026-09-14, before the validation run)
- LightGBM: num_leaves 8, min_child_samples 40, colsample 0.25, learning_rate 0.015, 800 trees, lambda 5,
  subsample 0.6, no recency weighting (research/11, scored 2012–2020 only).
- Skill model: per-stat p0/q/phi from research/12 (stat likelihood 2005–2014). Market-rating filter:
  p0 0.3, q 0.05, R 0.2 (line likelihood 2009–2014).
- DEV evidence that motivated the design (selection-biased): stats-model log loss 0.6342 vs closing line
  0.6125; stack with closing line 0.6067; stack with opening line 0.6159 vs opening 0.6281.
- Leakage canaries (research/00, 00b) pass for UFCStats engine, Sherdog, skills and market ratings.

## Addendum A (recorded after VALIDATION results, BEFORE the holdout run finished or was viewed)
VALIDATION outcome: O1 borderline (CI touched 0), O3 not significant, O4 strongly positive (ROI +9.6%, CLV +9%).
Because O4 bets are placed when lines open, weeks before weigh-ins, two robustness variants are declared now:
- O4-noweigh: stats model retrained with the weigh-in-miss feature neutralised (set to 0 everywhere).
- O4-clean: O4 bets restricted to "normal" markets: opening overround 1.02–1.10 and |p_open − p_close| < 0.15
  (guards against stale or erroneous opening lines inflating results).
Both are reported for VALIDATION and HOLDOUT regardless of outcome.

## Addendum B — v2 protocol (recorded 2026-09-14, before any v2 experiment was run)
Status: VALIDATION and HOLDOUT (2021-01-01 → 2026-09-12) have been scored for v1, so for moneyline
markets they are no longer untouched. v2 therefore follows these rules:
1. Every moneyline engine change is developed and accepted on DEV only (walk-forward, scored 2012–2020).
   Acceptance: paired-bootstrap log-loss gain with 95% CI excluding 0 on DEV for the layer it changes
   (stats model vs v1 stats, or market layer vs v1 ens3/stack_open), plus a plausible mechanism.
   Every attempt, accepted or not, is logged in `reports/v2_attempts.csv` (no silent retries).
2. v1 vs v2 on 2021–2026 is computed ONCE at the end and labelled "re-used period" in every report.
3. Intermediate line stages (BestFightOdds sparkline points between open and close) are new inputs, but
   their outcomes overlap already-seen fights; results on 2021–2026 are labelled "partially seen".
4. Prop markets (method of victory, inside the distance, goes to decision, rounds) have never been examined:
   - the method model is developed on outcomes up to 2020 only (no odds used);
   - prop odds exist from ~2021 (books currently listed by BestFightOdds); prop-market layers are refit
     yearly on earlier prop years only, scored from the second prop year onwards;
   - edge claimed only if ROI at the average price has a bootstrap 90% CI excluding 0 over the scored years
     AND is positive in the most recent full year.
5. From 2026-09-14 every live recommendation is saved before the fight (predictions/ledger.csv);
   this forward record is the only fully clean test from now on.

## Addendum C — v2 final moneyline evaluation (recorded 2026-09-14, before scoring v2 on 2021–2026)
User decision: moneyline only, placed at Pinnacle. Prop research (reports/45, 46) is kept as research, not product.
Frozen v2 engine (all choices made on DEV 2010–2020, see reports/v2_attempts.csv):
- stats: v1 LightGBM unchanged + logistic C 0.003 (was 0.05), soft labels 0.70 for split/majority decisions and DQs,
  plus promotion-level features (ufc/sources/promotion_features.py).
- market layer: mean of stage-aware stack and stage-aware edge GLM (ufc/market2.py, resid LightGBM dropped),
  trained on line stages 0, 3, 6, 9 of every earlier bout.
Comparisons on 2021-01-01 → 2026-09-12 ("re-used period", computed once):
- C1 log loss v2 vs v1 at the opening line (stage 0: v1 = stack_open) and at the close (stage 9: v1 = ens3), clustered bootstrap.
- C2 steady rule (edge > 3%, 1/8 Kelly, max 1.5u) at synthetic Pinnacle prices: stage fair probability with margin
  3.5% at the close and 4.5% at stages 0–3 (Pinnacle's archived closing margin was 2.4–3.4% in 2008–2020;
  early lines assumed wider). Reported per stage group: bets per event, ROI with 90% CI, years profitable.
- Sensitivity: margins 2.5% / 5.5%; consensus mean price as in v1.

### Addendum C outcome (recorded 2026-09-14 after the single run of research/48_v2_final.py)
Re-used period 2021-01-01 → 2026-09-12, 243 events, synthetic Pinnacle prices (primary margins):
- Log loss v2 vs v1: opening line +0.0013 [−0.0006, +0.0033]; close −0.0000 [−0.0017, +0.0016] (no significant change).
  v2 vs the line itself: open +0.0139 [+0.0094, +0.0185]; close +0.0040 [+0.0005, +0.0074].
- Steady rule, bets per event / ROI (90% CI) / years profitable:
  open v2 6.08 / +14.0% [+10.2, +17.9] / 6 of 6 (v1 5.84 / +12.4% / 5 of 6);
  stage 3 v2 5.47 / +8.4% [+4.5, +12.5] / 5 of 6; stage 6 v2 5.79 / +8.0% [+4.1, +11.9] / 5 of 6;
  close v2 4.85 / +4.3% [+0.2, +8.6] / 5 of 6 (v1 2.13 / +7.6% [+1.5, +14.2]).
Reading: v2 roughly doubles near-close volume at a lower ROI and slightly raises early-line volume and ROI.
The edge is still concentrated in early lines. Full table: reports/48_v2_final.csv / .txt.

### Correction after Addendum C (recorded 2026-09-14)
A truncation canary written afterwards (research/00c_leakage_promo.py) found that promotion strength was shrunk towards the
mean rating of ALL fights, including future ones (a single global constant). Fixed (anchor 1500; "last 5 fights" order made
input-independent), canary passes, DEV selection re-run (+0.0019 [+0.0002, +0.0036], still accepted), research/48 re-run.
As-run outputs kept as reports/48_v2_final_as_run_prior_leak.*. Corrected re-used-period results: open v2 6.09 bets/event,
ROI +14.3% [+10.5, +18.3], 6 of 6 years; stage 3 5.45, +8.5%; stage 6 5.74, +7.6%; close 4.81, +4.7% [+0.6, +8.9].
Live pricing from now on: real Pinnacle moneylines via The Odds API (user's key, saved in the app), estimates only when
Pinnacle has no line. Every fetch is appended to data/pinnacle/snapshots.csv (timestamped forward Pinnacle history).

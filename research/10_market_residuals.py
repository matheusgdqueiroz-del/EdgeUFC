"""What does the betting market fail to price? (the only kind of signal that can make money)

For every pre-fight feature difference: y ~ b0*logit(p_close) + b1*z(diff)   (symmetric, no intercept)
  discovery 2008-2015, replication 2016-2020, BH-FDR on discovery.
Plus classic hypotheses from the literature / folklore, each tested as flat 1-unit bets at closing prices
with year-by-year consistency (not just a pooled number):
  youth edge, travel edge, home country, women's favourites, slight underdogs on the main card,
  weight-miss fighters and their opponents, short-notice replacements, debutants, big favourites.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

d = pd.read_parquet("data/model_table.parquet")
d = d[d.p_close.notna() & (d.date < "2021-01-01")].copy()
rng = np.random.default_rng(11)
flip = rng.random(len(d)) < 0.5
L = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
lp = np.where(flip, -L(d.p_close), L(d.p_close))
y = np.where(flip, 1 - d.y, d.y)
okY = ~np.isnan(y)
disc = ((d.date < "2016-01-01") & okY).values
rep = ((d.date >= "2016-01-01") & okY).values
sides = sorted(c[2:] for c in d.columns if c.startswith("a_") and "b_" + c[2:] in d.columns and pd.api.types.is_numeric_dtype(d["a_" + c[2:]])
               and not c[2:].endswith("_dec"))


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


rows = []
for k in sides:
    diff = np.where(flip, d["b_" + k] - d["a_" + k], d["a_" + k] - d["b_" + k]).astype(float)
    out = {"feature": k}
    for tag, m in (("disc", disc), ("rep", rep)):
        ok = m & np.isfinite(diff)
        if ok.sum() < 500 or np.nanstd(diff[ok]) == 0:
            break
        x = (diff[ok] - np.nanmean(diff[ok])) / np.nanstd(diff[ok])
        try:
            r = sm.Logit(y[ok], np.column_stack([lp[ok], x])).fit(disp=0)
            out.update({f"{tag}_coef": r.params[1], f"{tag}_p": r.pvalues[1], f"{tag}_mkt_coef": r.params[0], f"{tag}_n": int(ok.sum())})
        except Exception:
            break
    if "rep_p" in out:
        rows.append(out)
R = pd.DataFrame(rows)
R["disc_q"] = bh(R.disc_p)
R["replicated"] = (R.disc_q < 0.2) & (R.rep_p < 0.05) & (np.sign(R.disc_coef) == np.sign(R.rep_coef))
R.sort_values("disc_p").to_csv("reports/10_market_residual_scan.csv", index=False)
print(f"features tested: {len(R)}; discovery q<0.20: {(R.disc_q < .2).sum()}; replicated: {R.replicated.sum()}")
print(R.sort_values("disc_p").head(25).round(4).to_string(index=False))
print("\nmarket coefficient (1.0 = perfectly calibrated closing line):", round(R.disc_mkt_coef.median(), 3), round(R.rep_mkt_coef.median(), 3))

# ---------------------------------------------------------------- folklore / literature hypotheses as bets
def flat_bets(name, side_a_mask, side_b_mask):
    """1u on side a where side_a_mask, on side b where side_b_mask, at mean closing price."""
    rows = []
    for side, m in (("a", side_a_mask), ("b", side_b_mask)):
        m = m & d.y.notna() | (m & (d.outcome == "D/D"))
        dec = d[f"{side}_close_mean_dec"]
        won = (d.y == 1) if side == "a" else (d.y == 0)
        prof = np.where(won, dec - 1, -1.0)
        rows.append(pd.DataFrame({"date": d.date[m], "profit": prof[m.values], "p": (d.p_close if side == "a" else 1 - d.p_close)[m], "won": won[m]}))
    b = pd.concat(rows)
    if len(b) < 30:
        return None
    yr = b.groupby(b.date.dt.year).profit.agg(["size", "mean"])
    t = stats.ttest_1samp(b.profit, 0)
    return dict(hypothesis=name, bets=len(b), roi=round(b.profit.mean(), 4), win_rate=round(b.won.mean(), 3), implied=round(b.p.mean(), 3),
                t_pvalue=round(t.pvalue, 4), years_positive=f"{(yr['mean'] > 0).sum()}/{len(yr)}",
                roi_2008_15=round(b[b.date < "2016"].profit.mean(), 4), roi_2016_20=round(b[b.date >= "2016"].profit.mean(), 4))


H = []
age_gap = d.a_age - d.b_age
H.append(flat_bets("younger by 5+ years", age_gap <= -5, age_gap >= 5))
if "a_travel_km" in d:
    tg = d.a_travel_km - d.b_travel_km
    H.append(flat_bets("travelled 5000+ km less", tg <= -5000, tg >= 5000))
    tz = d.a_tz_shift - d.b_tz_shift
    H.append(flat_bets("4+ fewer time zones crossed", tz <= -4, tz >= 4))
if "a_home_country" in d:
    H.append(flat_bets("home-country fighter vs foreigner", (d.a_home_country == 1) & (d.b_home_country == 0), (d.b_home_country == 1) & (d.a_home_country == 0)))
H.append(flat_bets("women's bouts: favourite", (d.women == 1) & (d.p_close > .5), (d.women == 1) & (d.p_close < .5)))
H.append(flat_bets("main card slight underdog (37.5-45%)", (d.bout_order <= 4) & d.p_close.between(.375, .45), (d.bout_order <= 4) & (1 - d.p_close).between(.375, .45)))
H.append(flat_bets("any slight underdog (35-45%)", d.p_close.between(.35, .45), (1 - d.p_close).between(.35, .45)))
H.append(flat_bets("big favourite (>=80%)", d.p_close >= .8, d.p_close <= .2))
H.append(flat_bets("big underdog (<=20%)", d.p_close <= .2, d.p_close >= .8))
H.append(flat_bets("UFC debutant vs veteran (bet debutant)", (d.a_n_fights == 0) & (d.b_n_fights >= 3), (d.b_n_fights == 0) & (d.a_n_fights >= 3)))
if "a_missed_weight" in d:
    H.append(flat_bets("fighter who missed weight", (d.a_missed_weight == 1) & (d.b_missed_weight == 0), (d.b_missed_weight == 1) & (d.a_missed_weight == 0)))
    H.append(flat_bets("short-notice replacement", (d.a_replacement == 1) & (d.b_replacement == 0), (d.b_replacement == 1) & (d.a_replacement == 0)))
H.append(flat_bets("coming off KO loss (bet them)", d.a_last_ko_loss == 1, d.b_last_ko_loss == 1))
H.append(flat_bets("layoff 2y+ (bet them)", d.a_days_since_last > 730, d.b_days_since_last > 730))
H.append(flat_bets("southpaw vs orthodox (bet southpaw)", (d.a_southpaw == 1) & (d.b_southpaw == 0) & (d.b_switch == 0), (d.b_southpaw == 1) & (d.a_southpaw == 0) & (d.a_switch == 0)))
H.append(flat_bets("4+ inch reach advantage", (d.a_reach - d.b_reach) >= 4, (d.b_reach - d.a_reach) >= 4))
H.append(flat_bets("high altitude: bet the fighter from altitude-closer home", (d.get("high_altitude", 0) == 1) & (d.get("a_alt_gap", 0) < d.get("b_alt_gap", 0) - 500), (d.get("high_altitude", 0) == 1) & (d.get("b_alt_gap", 0) < d.get("a_alt_gap", 0) - 500)))
Hd = pd.DataFrame([h for h in H if h])
Hd.to_csv("reports/10_hypothesis_bets.csv", index=False)
print("\nhypotheses as flat 1u bets at mean closing price (2008-2020; ROI per unit staked):")
print(Hd.to_string(index=False))

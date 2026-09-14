"""Exhaustive-but-honest signal scan.

DISCOVERY window 2005-2014, REPLICATION window 2015-2020 (VALID/HOLDOUT untouched).
Each bout is randomly oriented once (fixed seed) so no pattern can ride on UFCStats' listing order.

A. context-modulated effects: does the effect of a difference (a-b) change with context?
   model: y ~ diff + diff:context   (no intercept, antisymmetric)  -> LR test on discovery, sign+p on replication
B. level-modulated effects: does diff_i matter more when both fighters are high/low on j?
   model: y ~ diff_i + diff_i:mean_j
C. binned matchup cells: e.g. height bucket vs height bucket, age bucket vs age bucket, stance vs stance.
   cell win rate vs 50% (binomial), replicated.
All p-values are Benjamini-Hochberg corrected within each family on the discovery window.
"""
import warnings; warnings.filterwarnings("ignore")
import itertools
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

d = pd.read_parquet("data/model_table.parquet")
d = d[d.y.notna()].copy()
rng = np.random.default_rng(7)
flip = rng.random(len(d)) < 0.5
sides = sorted(c[2:] for c in d.columns if c.startswith("a_") and "b_" + c[2:] in d.columns)
A = {k: np.where(flip, d["b_" + k], d["a_" + k]) for k in sides}
B = {k: np.where(flip, d["a_" + k], d["b_" + k]) for k in sides}
y = np.where(flip, 1 - d.y, d.y)
disc = ((d.date >= "2005-01-01") & (d.date < "2015-01-01")).values
rep = ((d.date >= "2015-01-01") & (d.date < "2021-01-01")).values


def z(v, m):
    v = np.asarray(v, float); mu, sd = np.nanmean(v[m]), np.nanstd(v[m])
    return (v - mu) / (sd if sd > 0 else 1)


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def lr_test(X0, X1, yy):
    try:
        m0 = sm.Logit(yy, X0).fit(disp=0); m1 = sm.Logit(yy, X1).fit(disp=0)
        return 2 * (m1.llf - m0.llf), m1.params[-1], stats.chi2.sf(2 * (m1.llf - m0.llf), 1)
    except Exception:
        return np.nan, np.nan, np.nan


# ---------------------------------------------------------------- A. context modulation
wc = d.weight_lbs.values
contexts = {
    "women": d.women.values, "heavy(>=205)": (wc >= 205).astype(float), "light(<=135)": (wc <= 135).astype(float),
    "five_rounds": (d.sched_rounds == 5).astype(float).values, "main_event": d.main_event.values,
    "title": d.title.values, "late_era(year>=2012)": (d.date.dt.year >= 2012).astype(float).values,
}
for c in ("high_altitude", "apex", "no_crowd"):
    if c in d:
        contexts[c] = d[c].fillna(0).values
key_diffs = [k for k in sides if not k.startswith(("mu_",)) and np.nanstd(A[k] - B[k]) > 0]
rows = []
for k in key_diffs:
    diff = A[k] - B[k]
    for cname, cv in contexts.items():
        out = {"feature": k, "context": cname}
        for tag, m in (("disc", disc), ("rep", rep)):
            ok = m & np.isfinite(diff) & np.isfinite(cv)
            if ok.sum() < 400 or cv[ok].sum() < 60:
                break
            x = z(diff, ok)[ok]
            X0 = x[:, None]; X1 = np.column_stack([x, x * cv[ok]])
            lr, coef, p = lr_test(X0, X1, y[ok])
            out.update({f"{tag}_coef": coef, f"{tag}_p": p, f"{tag}_n_ctx": int(cv[ok].sum())})
        if "rep_p" in out:
            rows.append(out)
A_res = pd.DataFrame(rows)
A_res["disc_q"] = bh(A_res.disc_p.fillna(1))
A_res["replicated"] = (A_res.disc_q < 0.10) & (A_res.rep_p < 0.05) & (np.sign(A_res.disc_coef) == np.sign(A_res.rep_coef))
A_res.sort_values("disc_p").to_csv("reports/06a_context_modulation.csv", index=False)
print(f"A. context x diff tests: {len(A_res)}; discovery q<0.10: {(A_res.disc_q < .1).sum()}; replicated: {A_res.replicated.sum()}")
print(A_res.sort_values("disc_p").head(15).round(4).to_string(index=False))

# ---------------------------------------------------------------- B. level modulation among strongest features
top = ["age", "selo_strike", "d_sig_share", "c_slpm_minus_sapm", "reach", "height", "n_fights", "elo_tuned", "glicko",
       "d_td15", "d_td_def", "d_ctrl_share", "c_kd15", "c_kd_abs15", "days_since_last", "streak", "win_rate_s",
       "d_adj_slpm", "d_adj_sapm", "d_adj_td", "ko_loss_rate", "sub_win_rate", "late_early_ratio", "d_sig_def", "dec_margin_avg"]
top = [t for t in top if t in A]
rows = []
for i, j in itertools.permutations(top, 2):
    diff = A[i] - B[i]; lvl = (A[j] + B[j]) / 2
    out = {"diff": i, "level_of": j}
    for tag, m in (("disc", disc), ("rep", rep)):
        ok = m & np.isfinite(diff) & np.isfinite(lvl)
        if ok.sum() < 400:
            break
        x = z(diff, ok)[ok]; l = z(lvl, ok)[ok]
        lr, coef, p = lr_test(x[:, None], np.column_stack([x, x * l]), y[ok])
        out.update({f"{tag}_coef": coef, f"{tag}_p": p})
    if "rep_p" in out:
        rows.append(out)
B_res = pd.DataFrame(rows)
B_res["disc_q"] = bh(B_res.disc_p.fillna(1))
B_res["replicated"] = (B_res.disc_q < 0.10) & (B_res.rep_p < 0.05) & (np.sign(B_res.disc_coef) == np.sign(B_res.rep_coef))
B_res.sort_values("disc_p").to_csv("reports/06b_level_modulation.csv", index=False)
print(f"\nB. diff x level tests: {len(B_res)}; discovery q<0.10: {(B_res.disc_q < .1).sum()}; replicated: {B_res.replicated.sum()}")
print(B_res.sort_values("disc_p").head(15).round(4).to_string(index=False))

# ---------------------------------------------------------------- C. binned matchup cells
def cells(name, va, vb, labels):
    rows = []
    for tag, m in (("disc", disc), ("rep", rep)):
        ok = m & pd.notna(va) & pd.notna(vb) & (va != vb)
        t = pd.DataFrame({"a": va[ok], "b": vb[ok], "y": y[ok]})
        # orientation-free: express every bout as (lower label, higher label, did lower win)
        lo = np.where(t.a < t.b, t.a, t.b); hi = np.where(t.a < t.b, t.b, t.a)
        low_won = np.where(t.a < t.b, t.y, 1 - t.y)
        g = pd.DataFrame({"lo": lo, "hi": hi, "w": low_won}).groupby(["lo", "hi"]).w.agg(["size", "mean"])
        g.columns = [f"{tag}_n", f"{tag}_low_winrate"]
        rows.append(g)
    g = rows[0].join(rows[1], how="inner").reset_index()
    g = g[(g.disc_n >= 40) & (g.rep_n >= 40)]
    g["disc_p"] = [stats.binomtest(int(round(r.disc_low_winrate * r.disc_n)), int(r.disc_n)).pvalue for r in g.itertuples()]
    g["rep_p"] = [stats.binomtest(int(round(r.rep_low_winrate * r.rep_n)), int(r.rep_n)).pvalue for r in g.itertuples()]
    g["family"] = name
    g["lo"] = g.lo.map(lambda v: labels(v)); g["hi"] = g.hi.map(lambda v: labels(v))
    return g

H = lambda v: f"{int(v // 12)}'{int(v % 12)}\""
C_parts = [
    cells("height_in", np.round(A["height"]), np.round(B["height"]), H),
    cells("age_bucket(3y)", np.floor(A["age"] / 3) * 3, np.floor(B["age"] / 3) * 3, lambda v: f"{int(v)}-{int(v) + 2}"),
    cells("reach_bucket(2in)", np.floor(A["reach"] / 2) * 2, np.floor(B["reach"] / 2) * 2, lambda v: f"{int(v)}-{int(v) + 1}in"),
    cells("stance(0=orth,1=south,2=switch)", A["southpaw"] + 2 * A["switch"], B["southpaw"] + 2 * B["switch"], lambda v: ["orthodox", "southpaw", "switch"][int(v)]),
    cells("ufc_fights_bucket", np.minimum(A["n_fights"], 12) // 3, np.minimum(B["n_fights"], 12) // 3, lambda v: f"{int(v) * 3}-{int(v) * 3 + 2} fights"),
]
C_res = pd.concat(C_parts)
C_res["disc_q"] = bh(C_res.disc_p)
C_res["replicated"] = (C_res.disc_q < 0.10) & (C_res.rep_p < 0.05) & (np.sign(C_res.disc_low_winrate - .5) == np.sign(C_res.rep_low_winrate - .5))
C_res.sort_values("disc_p").to_csv("reports/06c_matchup_cells.csv", index=False)
print(f"\nC. matchup cells: {len(C_res)}; discovery q<0.10: {(C_res.disc_q < .1).sum()}; replicated: {C_res.replicated.sum()}")
print(C_res.sort_values("disc_p").head(20).round(3).to_string(index=False))

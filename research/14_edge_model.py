"""Market-aware "edge" models: start from the closing line and add only corrections with a reason to exist.

  A  stack      logit p = a*logit(market) + b*logit(stats model)
  C  edge GLM   logit p = a*logit(market) + b*(logit(model) - logit(market)) + sum_j c_j * z_j
                z_j = pre-registered mispricing candidates (from research/10 discovery 2008-2015):
                      striking defence, strikes absorbed (opp-adjusted), takedown volume, age gap, short-notice
                      replacement, missed weight, UFC debutant, main-card slight-underdog band, longshot band,
                      open->close steam, model-vs-market disagreement in debut bouts
  B  residual LightGBM on top of logit(market) with all features (challenger, more flexible, more overfit risk)
  A+B+C ensemble
All fitted walk-forward (yearly, trained on years < Y only, symmetric augmentation, no intercept).
Writes data/edge_<period>.parquet with out-of-sample probabilities for the betting study.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from ufc.evaluate import scores, walk_forward
from ufc.model import feature_sets, logit, residual_lgbm, Stacker, OffsetRidgeLogit

period = sys.argv[1] if len(sys.argv) > 1 else "dev"
last = 2020 if period == "dev" else 2024
P = pd.read_parquet(f"data/oos_{period}.parquet")
d = pd.read_parquet("data/model_table.parquet").loc[P.index]
d["ens"] = P["ens"]
d = d[d.p_close.notna() & d.ens.notna()].copy()
fs = feature_sets(d)

Lm, Lo, Lmod = logit(d.p_close.values), logit(d.p_open.fillna(d.p_close).values), logit(d.ens.values)
def anti(col):
    return (d[f"a_{col}"] - d[f"b_{col}"]).astype(float).values if f"a_{col}" in d else np.zeros(len(d))
main = (d.bout_order <= 4).values
pa, pb = d.p_close.values, 1 - d.p_close.values
Z = pd.DataFrame(index=d.index)
Z["def_skill"] = anti("sk_d_str")
Z["adj_sapm"] = anti("d_adj_sapm")
Z["sig_def"] = anti("d_sig_def")
Z["tda15"] = anti("c_tda15")
Z["age"] = anti("age")
Z["replacement"] = anti("replacement")
Z["missed_weight"] = anti("missed_weight")
Z["debut"] = anti("debut")
Z["main_dog_band"] = (main & (pa >= .375) & (pa <= .45)).astype(float) - (main & (pb >= .375) & (pb <= .45)).astype(float)
Z["longshot"] = (pa <= .2).astype(float) - (pb <= .2).astype(float)
Z["steam"] = Lm - Lo
Z["disagree_x_debut"] = (Lmod - Lm) * ((d.a_debut + d.b_debut) > 0).values
Z = Z.fillna(0)
yr = d.date.dt.year.values
y = d.y.values

def design_glm(idx, mu, sd):
    return np.column_stack([Lm[idx], Lmod[idx] - Lm[idx], ((Z[idx] - mu) / sd).values])

def fit_glm(tr, mu, sd, C):
    # C is an inverse penalty; corrections (incl. the market-slope correction) shrink toward the market itself
    return OffsetRidgeLogit(lam=1.0 / C).fit(design_glm(tr, mu, sd), y[tr], Lm[tr])

def choose_C(tr, Y, grid=(0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)):
    """nested time split inside the training window: fit on < Y-3, score on Y-3..Y-1."""
    inner_tr, inner_va = tr & (yr < Y - 3), tr & (yr >= Y - 3)
    if inner_tr.sum() < 800 or inner_va.sum() < 300:
        return 0.003
    mu_i, sd_i = Z[inner_tr].mean(), Z[inner_tr].std().replace(0, 1)
    best, bl = None, np.inf
    for C in grid:
        g = fit_glm(inner_tr, mu_i, sd_i, C)
        p = np.clip(g.predict(design_glm(inner_va, mu_i, sd_i), Lm[inner_va]), 1e-6, 1 - 1e-6)
        l = -np.mean(y[inner_va] * np.log(p) + (1 - y[inner_va]) * np.log(1 - p))
        if l < bl:
            best, bl = C, l
    return best

out = pd.DataFrame(index=d.index, columns=["stack", "edge_glm", "resid_lgbm"], dtype=float)
coefs = []
for Y in range(2011, last + 1):
    tr = (yr < Y) & ~np.isnan(y); te = yr == Y
    if te.sum() == 0:
        continue
    mu, sd = Z[tr].mean(), Z[tr].std().replace(0, 1)
    st = Stacker().fit(d.p_close.values[tr], d.ens.values[tr], y[tr])
    out.loc[te, "stack"] = st.predict(d.p_close.values[te], d.ens.values[te])
    C_best = choose_C(tr, Y)
    g = fit_glm(tr, mu, sd, C_best)
    out.loc[te, "edge_glm"] = g.predict(design_glm(te, mu, sd), Lm[te])
    coefs.append(pd.Series(list(g.coef_) + [C_best], index=["market_slope_corr", "model-market"] + list(Z.columns) + ["C"], name=Y))
keys = fs["ufc"] + fs["sherdog"] + fs["geo"] + fs["wiki"] + fs["skills"] + fs["fm"] + fs["mkt_hist"]
out["resid_lgbm"] = walk_forward(d, lambda tr, te: residual_lgbm(tr, te, keys, fs["context"]), start="2011-01-01", end=f"{last + 1}-01-01").reindex(d.index)
out["ens3"] = (out["stack"] + out["edge_glm"] + out["resid_lgbm"]) / 3
C = pd.DataFrame(coefs).round(3)
print("edge GLM coefficients by training cutoff year (standardised):"); print(C.T.to_string())

lo = 2013 if period == "dev" else 2021
m = (yr >= lo) & ~np.isnan(y)
rows = []
for col, p in [("market_close", d.p_close), ("stats_model", d.ens)] + [(c, out[c]) for c in out]:
    s = scores(pd.Series(y[m]), pd.Series(np.asarray(p, float)[m])); s["model"] = col; rows.append(s)
R = pd.DataFrame(rows).set_index("model"); print(R.to_string()); R.to_csv(f"reports/14_edge_models_{period}.csv")
res = out.join(d[["date", "y", "p_close", "p_open", "ens", "outcome", "a_close_mean_dec", "b_close_mean_dec", "a_close_best_dec", "b_close_best_dec", "a_open_dec", "b_open_dec"]])
res.to_parquet(f"data/edge_{period}.parquet")
C.to_csv(f"reports/14_edge_glm_coefs_{period}.csv")

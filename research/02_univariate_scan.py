"""Univariate signal scan. Each pre-fight difference (a-b) gets a symmetric spline-logistic fit on the
DISCOVERY window (2005-2014) and is scored on the REPLICATION window (2015-2020).
A signal counts only if it improves out-of-sample log loss over a coin flip."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from ufc.evaluate import make_pairs, side_cols

f = pd.read_parquet("data/fights.parquet"); X = pd.read_parquet("data/features_ufcstats.parquet"); prof = pd.read_parquet("data/profiles.parquet")
d = make_pairs(f, X, prof); d = d[d.y.notna()]
disc = d[(d.date >= "2005-01-01") & (d.date < "2015-01-01")]
rep = d[(d.date >= "2015-01-01") & (d.date < "2021-01-01")]
LN2 = np.log(2)

def fit_score(col):
    x_tr = (disc[f"a_{col}"] - disc[f"b_{col}"]).values
    x_te = (rep[f"a_{col}"] - rep[f"b_{col}"]).values
    ok_tr, ok_te = np.isfinite(x_tr), np.isfinite(x_te)
    if ok_tr.sum() < 300 or ok_te.sum() < 300 or np.nanstd(x_tr) == 0:
        return None
    lo, hi = np.nanpercentile(np.abs(x_tr), [0, 98])
    clip = lambda v: np.clip(v, -hi, hi)
    xt = clip(np.concatenate([x_tr[ok_tr], -x_tr[ok_tr]]))[:, None]
    yt = np.concatenate([disc.y.values[ok_tr], 1 - disc.y.values[ok_tr]])
    st = SplineTransformer(n_knots=5, degree=2).fit(xt)
    m = LogisticRegression(C=1.0, max_iter=1000, fit_intercept=False).fit(st.transform(xt), yt)
    pe = m.predict_proba(st.transform(clip(x_te[ok_te])[:, None]))[:, 1]
    pe2 = m.predict_proba(st.transform(clip(-x_te[ok_te])[:, None]))[:, 1]
    p = np.clip((pe + 1 - pe2) / 2, 1e-4, 1 - 1e-4)
    y = rep.y.values[ok_te]
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    # linear direction + discovery-window correlation for readability
    r_disc = np.corrcoef(np.sign(x_tr[ok_tr]) * 0 + x_tr[ok_tr], disc.y.values[ok_tr])[0, 1]
    r_rep = np.corrcoef(x_te[ok_te], y)[0, 1]
    return dict(feature=col, n_rep=int(ok_te.sum()), gain_bits_per_100=round((LN2 - ll) * 100 / LN2, 3),
                r_disc=round(r_disc, 3), r_rep=round(r_rep, 3), same_sign=np.sign(r_disc) == np.sign(r_rep),
                acc_rep=round(np.mean((p > 0.5) == (y == 1)), 3))

rows = [r for c in side_cols(d) if (r := fit_score(c))]
out = pd.DataFrame(rows).sort_values("gain_bits_per_100", ascending=False)
out.to_csv("reports/02_univariate_scan.csv", index=False)
pd.set_option("display.width", 200)
print(out.head(45).to_string(index=False))
print("...\nweakest:"); print(out.tail(15).to_string(index=False))
print("features with positive OOS gain:", (out.gain_bits_per_100 > 0).sum(), "of", len(out))

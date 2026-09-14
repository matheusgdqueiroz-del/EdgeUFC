"""Sanity controls for the betting results: are prices realistic, and does the edge come from the model?"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ufc.betting import walk_forward_policy, summary, simulate
from ufc.model import logit, Stacker
from sklearn.linear_model import LogisticRegression

period = sys.argv[1] if len(sys.argv) > 1 else "dev"
first, last = (2013, 2020) if period == "dev" else (2021, 2024)
E = pd.read_parquet(f"data/edge_{period}.parquet")
E = E[E.date.dt.year <= last]
yr = E.date.dt.year

# 1. price realism: overround implied by the prices we bet at
for a, b in (("a_close_mean_dec", "b_close_mean_dec"), ("a_close_best_dec", "b_close_best_dec"), ("a_open_dec", "b_open_dec")):
    ov = 1 / E[a] + 1 / E[b]
    print(f"{a[2:]}: overround median {ov.median():.4f}, mean {ov.mean():.4f}, share < 1.00 (arb-like) {np.mean(ov < 1):.3f}, by period:",
          ov.groupby((yr // 4) * 4).median().round(4).to_dict())

# 2. controls
L = logit(E.p_close.values)
E["recal_market"] = np.nan
E["random_model"] = np.nan
rng = np.random.default_rng(0)
noise = rng.normal(0, 0.35, len(E))
for Y in range(2011, last + 1):
    tr, te = ((yr < Y) & E.y.notna()).values, (yr == Y).values
    X = np.concatenate([L[tr], -L[tr]])[:, None]; yy = np.concatenate([E.y.values[tr], 1 - E.y.values[tr]])
    m = LogisticRegression(C=100, fit_intercept=False).fit(X, yy)
    E.loc[te, "recal_market"] = m.predict_proba(L[te][:, None])[:, 1]
    fake = 1 / (1 + np.exp(-(L + noise)))  # a "model" that is market + pure noise
    st = Stacker().fit(E.p_close.values[tr], fake[tr], E.y.values[tr])
    E.loc[te, "random_model"] = st.predict(E.p_close.values[te], fake[te])
    if Y == last:
        print("market recalibration slope:", round(m.coef_[0][0], 3), "| stack weight on noise model:", np.round(st.coef, 3))
# a mediocre single book: fair closing probability with a 5% / 7% margin applied proportionally
for v in (5, 7):
    E[f"a_vig{v}"] = 1 / (E.p_close * (1 + v / 100)); E[f"b_vig{v}"] = 1 / ((1 - E.p_close) * (1 + v / 100))
rows = []
for name, col, a, b in [("edge_ens3 @mean", "ens3", "a_close_mean_dec", "b_close_mean_dec"),
                        ("CONTROL recalibrated market only @mean", "recal_market", "a_close_mean_dec", "b_close_mean_dec"),
                        ("CONTROL market + random noise model @mean", "random_model", "a_close_mean_dec", "b_close_mean_dec"),
                        ("edge_ens3 @synthetic 5% vig book", "ens3", "a_vig5", "b_vig5"),
                        ("edge_ens3 @synthetic 7% vig book", "ens3", "a_vig7", "b_vig7"),
                        ("stack(model+close) @synthetic 5% vig", "stack", "a_vig5", "b_vig5")]:
    bets, ch = walk_forward_policy(E, col, a, b, first, last)
    s = summary(bets); s["scenario"] = name; rows.append(s)
    if name.startswith("edge_ens3 @mean"):
        print(ch[["year", "min_edge", "frac", "max_units", "bets", "profit", "roi"]].to_string(index=False))
        bb = bets.assign(p_mkt=np.where(bets.bet_side == "a", E.loc[bets.index, "p_close"], 1 - E.loc[bets.index, "p_close"]))
        bb["band"] = pd.cut(bb.p_mkt, [0, .2, .35, .45, .55, .65, .8, 1])
        print("by market probability of the side we bet:"); print(bb.groupby("band", observed=True).agg(bets=("profit", "size"), roi=("profit", lambda s: s.sum()), staked=("stake", "sum")).assign(roi=lambda t: (t.roi / t.staked).round(3)))
print(pd.DataFrame(rows).set_index("scenario").to_string())

# 3. independent prices: settle the SAME model decisions at prices from other archives (2010-2020 only)
if period == "dev":
    X = pd.read_parquet("data/odds_external.parquet").set_index("fight_id")
    T = pd.read_parquet("data/model_table.parquet")[["fight_id"]].loc[E.index]
    for src in ("uf_mean", "um"):
        pe = T.fight_id.map(X[src])
        # external files store P(first-listed fighter wins) in our a-orientation already (see research/05)
        E[f"a_{src}5"], E[f"b_{src}5"] = 1 / (pe * 1.05), 1 / ((1 - pe) * 1.05)
        sub = E[pe.notna() & ((E.p_close - pe).abs() < 0.5)]
        bets, _ = walk_forward_policy(sub, "ens3", f"a_{src}5", f"b_{src}5", first, last)
        print(f"settled at {src} prices (+5% vig), model still fed BFO close:", summary(bets))
    small_move = E[(E.p_close - E.p_open).abs() < 0.12]
    bets, _ = walk_forward_policy(small_move, "ens3", "a_vig5", "b_vig5", first, last)
    print("excluding bouts whose line moved > 12 points (news / possible bad data):", summary(bets))

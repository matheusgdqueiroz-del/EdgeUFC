"""How a fight ends (v2): finish vs decision, how the winner wins, and fight length, from pre-fight features only.

Targets (bouts with standard 5-minute rounds):
  itd            fight ends inside the distance (KO/TKO, doctor stoppage, submission, DQ) vs goes to decision
  method|winner  KO (KO/TKO/DOC) / SUB / DEC, given which fighter wins (trained with the winner as side a)
  under_k        fight ends before k.5 rounds (k = 1..4; only k < scheduled rounds)
Market-free. Combined with a winner probability p(a wins):
  P(a by KO) = p * P(KO | a wins),  P(fight goes to decision) = 1 - P(itd), ...
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from ufc.evaluate import design

METHODS = ["KO", "SUB", "DEC"]
# per-side raw values (finishing tendencies need levels, not only differences)
FIN = ["ko_win_rate", "sub_win_rate", "ko_loss_rate", "sub_loss_rate", "went_dist_rate", "dec_win_rate", "dec_loss_rate", "finish_win_share",
       "finished_loss_share", "c_kd15", "c_kd_abs15", "c_sub15", "c_o_sub15", "c_td15", "c_slpm", "c_sapm", "c_sig_acc", "c_sig_def", "c_ctrl_share",
       "sk_o_kof", "sk_d_kof", "sk_o_sbf", "sk_d_sbf", "sk_o_kd", "sk_d_kd", "sk_o_sub", "sk_d_sub", "sk_o_str", "sk_d_str", "sk_o_td", "sk_d_td",
       "pro_finish_rate", "pro_ko_wins", "pro_sub_wins", "pro_ko_losses", "pro_sub_losses", "pro_n", "n_fights", "age", "reach", "height", "ko_loss_n",
       "avg_mins", "five_rate", "fm_r1", "gelo", "elo"]
CTX = ["sched_rounds", "title", "women", "weight_lbs", "bout_order", "main_event", "year"]
LGB = dict(n_estimators=500, learning_rate=0.02, num_leaves=8, min_child_samples=60, colsample_bytree=0.3, subsample=0.7,
           subsample_freq=1, reg_lambda=10, verbose=-1)


def targets(d):
    t = pd.DataFrame(index=d.index)
    std = d.time_format.isin(["3 Rnd (5-5-5)", "5 Rnd (5-5-5-5-5)"])
    m = d.method
    fin = m.isin(["KO", "SUB", "DOC", "DQ"])
    dec = m.isin(["U-DEC", "S-DEC", "M-DEC"])
    t["itd"] = np.where(std & (fin | dec), fin.astype(float), np.nan)
    meth = np.select([m.isin(["KO", "DOC"]), m.eq("SUB"), dec], ["KO", "SUB", "DEC"], "")
    t["meth3"] = np.where(std & d.y.notna() & (meth != ""), meth, None)
    for k in (1, 2, 3, 4):
        ok = std & (fin | dec) & (d.sched_rounds > k)
        t[f"under_{k}"] = np.where(ok, (fin & (d.fight_secs < k * 300 + 150)).astype(float), np.nan)
    return t


def sym_design(d, keys, raw):
    return design(d, keys, raw, CTX), design(d, keys, raw, CTX, swap=True)


def _lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.003, max_iter=3000))


def fit_binary(train, test, keys, raw, target):
    """Orientation-free binary target: train on both orientations, average both predictions."""
    tr = train[train[target].notna()]
    X1, X2 = sym_design(tr, keys, raw)
    X, y = pd.concat([X1, X2]), np.concatenate([tr[target].values] * 2)
    T1, T2 = sym_design(test, keys, raw)
    g = lgb.LGBMClassifier(**LGB).fit(X, y)
    l = _lr().fit(X, y)
    pg = (g.predict_proba(T1)[:, 1] + g.predict_proba(T2)[:, 1]) / 2
    pl = (l.predict_proba(T1)[:, 1] + l.predict_proba(T2)[:, 1]) / 2
    return (pg + pl) / 2


def fit_method(train, test, keys, raw):
    """P(method | a wins) and P(method | b wins) for test rows; trained with the winner as side a."""
    tr = train[train.meth3.notna() & train.y.notna()]
    X1, X2 = sym_design(tr, keys, raw)
    X = pd.concat([X1[tr.y.values == 1], X2[tr.y.values == 0]])  # winner always side a
    y = np.concatenate([tr.meth3.values[tr.y.values == 1], tr.meth3.values[tr.y.values == 0]])
    T1, T2 = sym_design(test, keys, raw)
    g = lgb.LGBMClassifier(objective="multiclass", **LGB).fit(X, y)
    l = _lr().fit(X, y)
    order = lambda m: [list(m.classes_).index(c) for c in METHODS]
    pa = (g.predict_proba(T1)[:, order(g)] + l.predict_proba(T1)[:, order(l)]) / 2  # a wins
    pb = (g.predict_proba(T2)[:, order(g)] + l.predict_proba(T2)[:, order(l)]) / 2  # b wins
    return pd.DataFrame(np.hstack([pa, pb]), index=test.index, columns=[f"a_{c}" for c in METHODS] + [f"b_{c}" for c in METHODS])


def predict_all(train, test, keys, raw):
    out = fit_method(train, test, keys, raw)
    out["itd"] = fit_binary(train, test, keys, raw, "itd")
    for k in (1, 2, 3, 4):
        if train[f"under_{k}"].notna().sum() > 300:
            out[f"under_{k}"] = fit_binary(train, test, keys, raw, f"under_{k}")
    return out

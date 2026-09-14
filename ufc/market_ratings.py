"""Market-implied fighter strength from the history of betting lines (Kalman filter).

Each past closing line is a noisy measurement of the strength gap:  logit(p_close) = s_A - s_B + e,  e~N(0,R).
Strengths drift as a random walk (q per year). Before bout t we only use lines from bouts dated before t,
so the rating is what the market's own past pricing implies. Features derived later:
  mr        : market-implied strength before the bout, mr_var its uncertainty
  line_gap  : logit(opening line now) - (mr_A - mr_B)   -> how far the new line departs from history
Hyperparameters (P0, q, R) are tuned on predicting the NEXT closing line (not results), 2008-2014.
"""
import math
import numpy as np
import pandas as pd


class MarketRatings:
    def __init__(self, p0=0.8, q=0.25, R=0.05):
        self.p0, self.q, self.R = p0, q, R
        self.m, self.v, self.last = {}, {}, {}

    def get(self, f, date):
        m = self.m.get(f, 0.0)
        v = self.v.get(f, self.p0)
        if f in self.last:
            v = min(v + self.q * (date - self.last[f]).days / 365.25, self.p0 * 2)
        return m, v

    def update(self, a, b, L, date):
        ma, va = self.get(a, date); mb, vb = self.get(b, date)
        S = va + vb + self.R
        innov = L - (ma - mb)
        self.m[a], self.m[b] = ma + va / S * innov, mb - vb / S * innov
        self.v[a], self.v[b] = va - va * va / S, vb - vb * vb / S
        self.last[a] = self.last[b] = date
        return innov, S


def run(fights, odds, p0=0.8, q=0.25, R=0.05, score=None):
    """fights sorted chronologically; odds has fight_id, p_close. Returns features and mean predictive log-lik."""
    o = odds.set_index("fight_id").p_close
    mr = MarketRatings(p0, q, R)
    out, ll, n = [], 0.0, 0
    for date, day in fights.groupby("date", sort=True):
        pend = []
        for f in day.itertuples():
            ma, va = mr.get(f.f1_id, date); mb, vb = mr.get(f.f2_id, date)
            out.append(dict(fight_id=f.fight_id, a_mr=ma, b_mr=mb, a_mr_var=va, b_mr_var=vb))
            p = o.get(f.fight_id)
            if p is not None and p == p:
                pend.append((f.f1_id, f.f2_id, math.log(p / (1 - p)), f.fight_id))
        for a, b, L, fid in pend:  # lines are folded in only after every bout on this date was snapshotted
            innov, S = mr.update(a, b, L, date)
            if score and score[0] <= date < score[1]:
                ll += -0.5 * (math.log(2 * math.pi * S) + innov * innov / S); n += 1
    return pd.DataFrame(out), ll / max(n, 1)


if __name__ == "__main__":
    import itertools
    f = pd.read_parquet("data/fights.parquet"); od = pd.read_parquet("data/odds.parquet")
    rows = []
    for p0, q, R in itertools.product([0.08, 0.15, 0.3], [0.005, 0.02, 0.05], [0.2, 0.35, 0.6]):
        _, ll = run(f, od, p0, q, R, score=(pd.Timestamp("2009-01-01"), pd.Timestamp("2015-01-01")))
        rows.append(dict(p0=p0, q=q, R=R, ll=round(ll, 4)))
    r = pd.DataFrame(rows).sort_values("ll", ascending=False); print(r.head(6).to_string(index=False))
    b = r.iloc[0]
    import json; json.dump(dict(p0=b.p0, q=b.q, R=b.R), open("data/market_ratings_cfg.json", "w"))

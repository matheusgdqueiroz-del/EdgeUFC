"""Level of competition before and outside the UFC, from Sherdog records (v2).

Promotion strength is data-driven and walk-forward: the average pre-fight global-Elo rating of everyone who fought
in that promotion during the 3 years before the cut-off (shrunk towards the overall average).
Per fighter (pro fights dated before the bout minus 3 days only):
  pr_str_last5    mean strength of the promotions of the last 5 non-UFC pro fights
  pr_str_max_win  strongest promotion beaten someone in (non-UFC)
  pr_strong_wins  sum over non-UFC wins of max(strength - 1500, 0) / 100
  pr_major_n      non-UFC pro fights in established promotions (Bellator, PFL, ONE, ...)
  pr_dwcs_win     won on Dana White's Contender Series
  pr_tuf_n        Ultimate Fighter house fights (exhibitions)
  pr_am_n, pr_am_winrate_s  amateur record
"""
import re
import numpy as np
import pandas as pd
from ufc.fetch import ROOT
from ufc.sources.sherdog_features import global_elo, load_records

SD = ROOT / "data" / "sherdog"
ALIAS = {"Bellator MMA": "Bellator", "Pride FC": "Pride", "CW": "CWFC", "Cage Warriors": "CWFC", "One Championship": "ONE",
         "ONE Championship": "ONE", "ONE FC": "ONE", "WSOF": "Professional Fighters League", "PFL": "Professional Fighters League",
         "Rizin FF": "Rizin", "RFA": "LFA", "Legacy FC": "LFA", "Invicta FC": "Invicta", "M-1 Challenge": "M-1", "M-1 Global": "M-1"}
MAJOR = {"Bellator", "Professional Fighters League", "ONE", "Strikeforce", "WEC", "Pride", "Rizin", "Dream", "KSW", "ACB", "ACA",
         "M-1", "CWFC", "LFA", "Invicta", "Pancrase", "Shooto", "Deep", "Jungle Fight", "Titan FC", "CFFC", "Brave CF", "Oktagon MMA",
         "UAE Warriors", "Dana White's Contender Series", "IFL", "Affliction", "EliteXC", "Road FC", "Fury FC"}
WINDOW = pd.Timedelta(days=3 * 365)


def promotion(event):
    k = re.sub(r"\s*(#?\d+(\.\d+)?|[IVX]+)\s*$", "", str(event).split(" - ")[0]).strip()
    return ALIAS.get(k, k)


def all_records():
    r = pd.read_csv(SD / "records.csv", parse_dates=["date"])
    r = r[r.date.notna()].drop_duplicates(["sd_url", "date", "opp_url", "result", "kind"]).copy()
    r["res"] = r.result.map({"win": 1.0, "loss": 0.0, "draw": 0.5})
    r["promo"] = r.event.map(promotion)
    r["ufc"] = r.event.fillna("").str.contains(r"\bUFC\b|Ultimate Fighter", regex=True) & ~r.promo.eq("Dana White's Contender Series")
    return r.sort_values(["date", "sd_url", "opp_url"], kind="mergesort")


class Strength:
    """strength(promo, date): shrunk mean participant rating over fights in (date - 3y, date)."""

    def __init__(self, pro, rating, k=10):
        # promotions with few fights in the window are pulled toward 1500
        pro = pro[pro.opp_url.notna()].copy()
        a = pro.sd_url.where(pro.sd_url < pro.opp_url, pro.opp_url)
        b = pro.opp_url.where(pro.sd_url < pro.opp_url, pro.sd_url)
        pro["a"], pro["b"] = a, b
        g = pro.drop_duplicates(["date", "a", "b"]).sort_values(["date", "a", "b"], kind="mergesort")
        g["r"] = [(rating(x, d)[0] + rating(y, d)[0]) / 2 for x, y, d in zip(g.a, g.b, g.date)]
        self.prior = 1500.0  # shrinkage anchor = the Elo starting rating (a data-derived mean would include future fights)
        self.k = k
        self.by = {p: (x.date.values, np.concatenate([[0.0], np.cumsum(x.r.values)])) for p, x in g.groupby("promo")}

    def __call__(self, promo, date):
        v = self.by.get(promo)
        if v is None:
            return self.prior
        dates, cs = v
        lo, hi = np.searchsorted(dates, np.datetime64(date - WINDOW)), np.searchsorted(dates, np.datetime64(date))
        n = hi - lo
        return (cs[hi] - cs[lo] + self.k * self.prior) / (n + self.k)


def promotion_features(fights, link):
    r = all_records()
    pro = r[r.kind == "pro"]
    rating = global_elo(load_records())
    strength = Strength(pro, rating)
    by = {u: g.sort_values(["date", "opp_url", "kind", "result"], kind="mergesort") for u, g in r.groupby("sd_url")}  # "last 5" must not depend on input order
    out = []
    for f in fights.itertuples():
        rec = {"fight_id": f.fight_id}
        cutoff = f.date - pd.Timedelta(days=3)
        for side, fid in (("a", f.f1_id), ("b", f.f2_id)):
            g = by.get(link.get(fid))
            if g is None:
                continue
            h = g[g.date < cutoff]
            p = h[(h.kind == "pro") & ~h.ufc]
            s = np.array([strength(x, cutoff) for x in p.promo]) if len(p) else np.array([])
            wins = (p.res == 1).values
            am = h[h.kind == "amateur"]
            tuf = h[(h.kind == "exhibition") & h.event.fillna("").str.contains("Ultimate Fighter")]
            rec.update({
                f"{side}_pr_str_last5": s[-5:].mean() if len(s) else np.nan,
                f"{side}_pr_str_max_win": s[wins].max() if wins.any() else np.nan,
                f"{side}_pr_strong_wins": float(np.clip(s[wins] - 1500, 0, None).sum() / 100) if wins.any() else 0.0,
                f"{side}_pr_major_n": float(p.promo.isin(MAJOR).sum()),
                f"{side}_pr_dwcs_win": float(((p.promo == "Dana White's Contender Series") & (p.res == 1)).any()),
                f"{side}_pr_tuf_n": float(len(tuf)),
                f"{side}_pr_am_n": float(len(am)),
                f"{side}_pr_am_winrate_s": ((am.res == 1).sum() + 1) / (am.res.notna().sum() + 2),
            })
        out.append(rec)
    return pd.DataFrame(out)


if __name__ == "__main__":
    from ufc.sources.ufcstats import load_fights
    f = load_fights()
    link = dict(pd.read_csv(ROOT / "data" / "sherdog_link.csv").values)
    X = promotion_features(f, link)
    (ROOT / "data" / "v2").mkdir(exist_ok=True)
    X.to_parquet(ROOT / "data" / "v2" / "features_promo.parquet")
    print(X.shape); print(X.describe().T.round(3))

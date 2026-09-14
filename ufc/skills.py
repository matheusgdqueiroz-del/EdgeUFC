"""Dynamic Bayesian multi-skill model (Poisson state-space, iterated EKF updates).

For every bout and each direction (A attacking B) and stat k, over `t` minutes of fighting:

    count_k(A->B) ~ Poisson( t * exp( mu_k[group, time] + off_k[A] - def_k[B] ) )

Each fighter carries Gaussian beliefs (mean, variance) on every off_k / def_k. Between bouts the variance
grows (random walk, q_k per year) so layoffs and age-related change widen uncertainty. After a bout the
Poisson likelihood is folded in with an iterated extended-Kalman (IRLS) update, which shrinks small samples
toward the prior automatically and adjusts every number for the quality of the opponent.

Stats (k): sig strikes, takedowns, knockdowns, submission attempts, control time (per 10s),
           KO/TKO finishes, submission finishes.
Snapshots taken before each fight date provide: skill means, uncertainties, and model-EXPECTED matchup
stats (e.g. expected strike differential per minute if these two meet), all strictly pre-fight.
Hyperparameters are tuned on the predictive likelihood of the STATS (not on who won), 2005-2014 only.
"""
import math
from collections import defaultdict
import numpy as np
import pandas as pd

# typical league rates per minute, used as pseudo-data (500 minutes) until real history accumulates
PRIOR_RATE = {"str": 3.6, "td": 0.1, "kd": 0.025, "sub": 0.045, "ctl": 1.6, "kof": 0.02, "sbf": 0.012}
STATS = {  # name: (own column in perf, count transform)
    "str": ("sig_l", 1.0), "td": ("td_l", 1.0), "kd": ("kd", 1.0), "sub": ("sub_att", 1.0),
    "ctl": ("ctrl", 0.1), "kof": ("ko_win", 1.0), "sbf": ("sub_win", 1.0),
}
DEFAULT = dict(p0={"str": 0.12, "td": 0.5, "kd": 0.5, "sub": 0.5, "ctl": 0.4, "kof": 0.6, "sbf": 0.6},
               q={"str": 0.03, "td": 0.08, "kd": 0.08, "sub": 0.08, "ctl": 0.08, "kof": 0.1, "sbf": 0.1},
               phi={"str": 1.0, "td": 1.0, "kd": 1.0, "sub": 1.0, "ctl": 1.0, "kof": 1.0, "sbf": 1.0},
               iters=25, league_halflife_days=730)


def group_of(weight_lbs, women):
    if women:
        return "W"
    if weight_lbs != weight_lbs:
        return "M"
    return "L" if weight_lbs <= 145 else ("M" if weight_lbs <= 185 else "H")


class SkillModel:
    def __init__(self, **cfg):
        c = {**DEFAULT, **cfg}
        self.p0, self.q, self.iters, self.hl = c["p0"], c["q"], c["iters"], c["league_halflife_days"]
        self.phi = c["phi"]  # over-dispersion: each observation counts as 1/phi of a Poisson draw
        self.m = defaultdict(float)             # (fighter, 'o'|'d', stat) -> mean
        self.v = {}                              # (fighter, 'o'|'d', stat) -> variance
        self.last = {}                           # fighter -> last bout date
        self.lc = defaultdict(lambda: 0.0)       # league decayed counts  (group, stat)
        self.lm = defaultdict(lambda: 0.0)       # league decayed minutes (group)
        self.ldate = None

    # ---- league intercepts (rolling, past bouts only) ----
    def mu(self, g, k):
        c, mins = self.lc[(g, k)], self.lm[g]
        return math.log((c + PRIOR_RATE[k] * 500) / (mins + 500))

    def _decay_league(self, date):
        if self.ldate is not None:
            f = 0.5 ** ((date - self.ldate).days / self.hl)
            for key in list(self.lc): self.lc[key] *= f
            for key in list(self.lm): self.lm[key] *= f
        self.ldate = date

    # ---- beliefs ----
    def var(self, f, side, k, date):
        v = self.v.get((f, side, k), self.p0[k])
        if f in self.last:
            v = min(v + self.q[k] * (date - self.last[f]).days / 365.25, self.p0[k] * 1.5)
        return v

    def snapshot(self, a, b, date, g, minutes=15.0):
        """Pre-fight skill features for a vs b (a_ and b_ keyed dicts) + expected matchup stats."""
        out = {}
        for s, x, y in (("a", a, b), ("b", b, a)):
            for k in STATS:
                out[f"{s}_sk_o_{k}"] = self.m[(x, "o", k)]
                out[f"{s}_sk_d_{k}"] = self.m[(x, "d", k)]
                out[f"{s}_sk_ov_{k}"] = self.var(x, "o", k, date)
                out[f"{s}_sk_dv_{k}"] = self.var(x, "d", k, date)
                eta = self.mu(g, k) + self.m[(x, "o", k)] - self.m[(y, "d", k)]
                vv = self.var(x, "o", k, date) + self.var(y, "d", k, date)
                out[f"{s}_sk_exp_{k}"] = math.exp(eta + vv / 2)  # lognormal mean rate per minute
        for k in STATS:
            ra, rb = out[f"a_sk_exp_{k}"], out[f"b_sk_exp_{k}"]
            out[f"a_sk_share_{k}"] = ra / (ra + rb)
            out[f"b_sk_share_{k}"] = rb / (ra + rb)
        return out

    def _update_dir(self, x, y, k, count, t, g, date):
        """x attacking y produced `count` over t minutes. Exact Laplace update along eta = off - def."""
        mo, md = self.m[(x, "o", k)], self.m[(y, "d", k)]
        vo, vd = self.var(x, "o", k, date), self.var(y, "d", k, date)
        base = self.mu(g, k) + math.log(t)
        eta0, S = base + mo - md, vo + vd
        eta = eta0
        for _ in range(self.iters):  # damped Newton on the 1-D log posterior (concave -> converges)
            lam = math.exp(min(eta, 12))
            ph = self.phi[k]
            step = ((count - lam) / ph - (eta - eta0) / S) / (lam / ph + 1.0 / S)
            step = max(-1.0, min(1.0, step))
            eta += step
            if abs(step) < 1e-6:
                break
        lam = math.exp(min(eta, 12))
        delta = eta - eta0
        shrink = 1.0 / (1.0 + S * lam / self.phi[k])  # posterior/prior variance ratio of the projection
        self.m[(x, "o", k)] = mo + vo / S * delta
        self.m[(y, "d", k)] = md - vd / S * delta
        self.v[(x, "o", k)] = vo - vo * vo / S * (1 - shrink)
        self.v[(y, "d", k)] = vd - vd * vd / S * (1 - shrink)
        return eta0, S  # prior eta (incl. exposure) and variance, for scoring

    def update(self, bouts, date):
        """bouts: list of dicts with a, b, group, mins, and counts ca[k], cb[k]. Returns predictive log-lik."""
        self._decay_league(date)
        ll, n = defaultdict(float), defaultdict(int)
        for bt in bouts:
            t = bt["mins"]
            if not (t and t == t and t > 0.1):
                continue
            for k in STATS:
                ca, cb = bt["ca"].get(k), bt["cb"].get(k)
                if ca is None or cb is None or ca != ca or cb != cb:
                    continue
                for x, y, c in ((bt["a"], bt["b"], ca), (bt["b"], bt["a"], cb)):
                    eta0, S = self._update_dir(x, y, k, c, t, bt["group"], date)
                    lam = math.exp(min(eta0 + S / 2, 12))
                    ll[k] += c * math.log(lam + 1e-12) - lam - math.lgamma(c + 1); n[k] += 1
        for bt in bouts:
            t = bt["mins"]
            if t and t == t and t > 0.1:
                for k in STATS:
                    ca, cb = bt["ca"].get(k), bt["cb"].get(k)
                    if ca == ca and cb == cb and ca is not None and cb is not None:
                        self.lc[(bt["group"], k)] += ca + cb
                self.lm[bt["group"]] += 2 * t
            self.last[bt["a"]] = date; self.last[bt["b"]] = date
        return ll, n


def run(fights, perf, cfg=None, score_from=None, score_to=None):
    """Chronological pass. Returns (features DataFrame per fight, predictive log-lik per stat-observation)."""
    sm = SkillModel(**(cfg or {}))
    p = perf.set_index(["fight_id", "fid"])
    feats, ll_tot, n_tot = [], defaultdict(float), defaultdict(int)
    for date, day in fights.groupby("date", sort=True):
        bouts = []
        for f in day.itertuples():
            g = group_of(f.weight_lbs, f.women)
            snap = sm.snapshot(f.f1_id, f.f2_id, date, g)
            snap["fight_id"] = f.fight_id
            feats.append(snap)
            if (f.fight_id, f.f1_id) not in p.index or (f.fight_id, f.f2_id) not in p.index:
                continue
            ra, rb = p.loc[(f.fight_id, f.f1_id)], p.loc[(f.fight_id, f.f2_id)]
            if ra.sig_a != ra.sig_a:  # no round stats for this bout
                continue
            ca = {k: float(ra[col]) * sc if ra[col] == ra[col] else np.nan for k, (col, sc) in STATS.items()}
            cb = {k: float(rb[col]) * sc if rb[col] == rb[col] else np.nan for k, (col, sc) in STATS.items()}
            bouts.append(dict(a=f.f1_id, b=f.f2_id, group=g, mins=ra.mins, ca=ca, cb=cb))
        ll, n = sm.update(bouts, date)
        if score_from is not None and score_from <= date < score_to:
            for k in ll:
                ll_tot[k] += ll[k]; n_tot[k] += n[k]
    return pd.DataFrame(feats), {k: ll_tot[k] / max(n_tot[k], 1) for k in ll_tot}


if __name__ == "__main__":
    # sanity: a fighter who out-lands everyone should end with higher offence than one who is out-landed
    sm = SkillModel()
    d0 = pd.Timestamp("2020-01-01")
    for i in range(6):
        sm.update([dict(a="good", b=f"opp{i}", group="M", mins=15, ca={"str": 90}, cb={"str": 30}),
                   dict(a="bad", b=f"opq{i}", group="M", mins=15, ca={"str": 25}, cb={"str": 80})], d0 + pd.Timedelta(days=90 * i))
    s = sm.snapshot("good", "bad", d0 + pd.Timedelta(days=700), "M")
    assert s["a_sk_o_str"] > 0 > s["b_sk_o_str"] and s["a_sk_share_str"] > 0.6, s
    assert s["a_sk_ov_str"] < DEFAULT["p0"]["str"]
    print("skills ok", round(s["a_sk_share_str"], 3), round(s["a_sk_o_str"], 3), round(s["b_sk_d_str"], 3))

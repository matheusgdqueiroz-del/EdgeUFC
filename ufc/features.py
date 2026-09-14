"""Chronological feature engine.

For each fight date D (in order):
  1. snapshot every fighter on that date using ONLY bouts dated < D
  2. then fold the date's results into the fighter states.
No per-fight statistic, result, rating or league average from D or later can reach a snapshot.

Snapshot-type inputs (tale of the tape) are handled conservatively:
  * height / reach / DOB / stance are physical constants (DOB) or near-constants; they come from a current
    snapshot, so their *missingness* correlates with career length (old short-career fighters never got a
    reach entry). We therefore never expose missingness indicators and impute missing reach from height.
  * the tott "weight" column is current weight -> never used (bout weight class is used instead).
"""
import math
from collections import defaultdict
import numpy as np
import pandas as pd

from ufc.ratings import Elo, Glicko2, ShareElo, elo_expect

OWN = ["kd", "sig_l", "sig_a", "tot_l", "tot_a", "td_l", "td_a", "sub_att", "rev", "ctrl",
       "head_l", "head_a", "body_l", "body_a", "leg_l", "leg_a", "dist_l", "dist_a",
       "clinch_l", "clinch_a", "ground_l", "ground_a"]


# ----------------------------------------------------------------------------------------------
# 1. per-fighter per-fight performance rows (these are POST-fight facts)
# ----------------------------------------------------------------------------------------------
def build_perf(fights, rounds):
    r = rounds.copy()
    fl = fights.set_index("fight_id")
    # seconds spent in each round (full rounds = their scheduled length; last round = end time)
    fmt_len = fights.set_index("fight_id").time_format.str.extract(r"\(([\d\-]+)\)")[0].str.split("-")
    def round_secs(fid, rnd):
        L = fmt_len.get(fid)
        er, et = fl.at[fid, "end_round"], fl.at[fid, "end_time"]
        if rnd != rnd:
            return np.nan
        if rnd == er:
            return et
        if isinstance(L, list) and rnd <= len(L):
            return int(L[int(rnd) - 1]) * 60
        return 300.0
    r["secs"] = [round_secs(f, k) for f, k in zip(r.fight_id, r["round"])]
    tot = r.groupby(["fight_id", "fighter_id"])[OWN].sum(min_count=1)
    early = r[r["round"] == 1].groupby(["fight_id", "fighter_id"]).agg(e_sig_l=("sig_l", "sum"), e_secs=("secs", "sum"))
    late = r[r["round"] >= 3].groupby(["fight_id", "fighter_id"]).agg(l_sig_l=("sig_l", "sum"), l_secs=("secs", "sum"))
    tot = tot.join(early).join(late).reset_index()

    rows = []
    for side, opp in (("f1", "f2"), ("f2", "f1")):
        d = fights[["fight_id", "date", "event_id", "location", "method", "method_detail", "end_round", "fight_secs",
                    "sched_rounds", "title", "weight_lbs", "weightclass", "women", "bout_order", "winner", "outcome"]].copy()
        d["fid"], d["oid"] = fights[f"{side}_id"], fights[f"{opp}_id"]
        w = fights.winner if side == "f1" else 1 - fights.winner
        d["res"] = np.where(fights.outcome == "D/D", 0.5, w)  # NaN = no contest
        rows.append(d)
    p = pd.concat(rows, ignore_index=True)
    p = p.merge(tot.rename(columns={"fighter_id": "fid"}), on=["fight_id", "fid"], how="left")
    opp_cols = OWN + ["e_sig_l", "l_sig_l"]
    p = p.merge(tot[["fight_id", "fighter_id"] + opp_cols].rename(columns={"fighter_id": "oid", **{c: "o_" + c for c in opp_cols}}),
                on=["fight_id", "oid"], how="left")
    p["mins"] = p.fight_secs / 60.0
    dec = p.method.isin(["U-DEC", "S-DEC", "M-DEC"])
    fin = p.method.isin(["KO", "DOC", "SUB"])
    p["win"] = (p.res == 1).astype(float); p["loss"] = (p.res == 0).astype(float)
    p["ko_win"] = ((p.res == 1) & p.method.isin(["KO", "DOC"])).astype(float)
    p["sub_win"] = ((p.res == 1) & (p.method == "SUB")).astype(float)
    p["dec_win"] = ((p.res == 1) & dec).astype(float)
    p["ko_loss"] = ((p.res == 0) & p.method.isin(["KO", "DOC"])).astype(float)
    p["sub_loss"] = ((p.res == 0) & (p.method == "SUB")).astype(float)
    p["dec_loss"] = ((p.res == 0) & dec).astype(float)
    p["split_win"] = ((p.res == 1) & (p.method == "S-DEC")).astype(float)
    p["split_loss"] = ((p.res == 0) & (p.method == "S-DEC")).astype(float)
    p["went_dist"] = dec.astype(float)
    p["finish"] = fin.astype(float)
    # judges: UFCStats lists each card as "loser - winner"; margin in points from the winner's view
    def margin(s):
        sc = [tuple(map(int, m)) for m in __import__("re").findall(r"(\d+) - (\d+)", str(s))]
        return np.mean([b - a for a, b in sc]) if sc else np.nan
    m = p.method_detail.map(margin)
    p["dec_margin"] = np.where(dec & (p.res == 1), m, np.where(dec & (p.res == 0), -m, np.nan))
    return p.sort_values(["date", "fight_id", "fid"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# 2. helpers
# ----------------------------------------------------------------------------------------------
def _safe(a, b, prior, strength):
    return (a + prior * strength) / (b + strength)


class League:
    """Running league-wide averages from bouts already completed (used as shrinkage priors)."""

    def __init__(self):
        self.s = defaultdict(float)

    def add(self, row):
        if row["mins"] == row["mins"] and row["sig_a"] == row["sig_a"]:
            for c in OWN:
                v = row[c]
                if v == v:
                    self.s[c] += v
            self.s["mins"] += row["mins"]

    def rate(self, c, default):
        return self.s[c] / self.s["mins"] if self.s["mins"] > 100 else default

    def acc(self, l, a, default):
        return self.s[l] / self.s[a] if self.s[a] > 100 else default


HIST = ["t", "res", "mins", "method_fin", "win", "loss", "ko_win", "sub_win", "dec_win", "ko_loss", "sub_loss", "dec_loss",
        "split_win", "split_loss", "went_dist", "title", "five", "weight", "opp_elo", "exp", "has_stats",
        "dec_margin", "early_rate", "late_rate", "e_secs", "l_secs"] + OWN + ["o_" + c for c in OWN] + \
       ["adj_slpm", "adj_sapm", "adj_td", "adj_tda", "adj_ctrl", "adj_kd", "adj_acc", "adj_def", "adj_sub"]


class Fighter:
    def __init__(self):
        self.h = {k: [] for k in HIST}
        self.dates = []
        self.last_weight = np.nan
        self.first_date = None
        self.countries = defaultdict(int)
        self.opps = []  # (opponent id, result) in chronological order


def _arr(fs, k):
    return np.asarray(fs.h[k], dtype=float)


# ----------------------------------------------------------------------------------------------
# 3. snapshot: everything about a fighter known before `date`
# ----------------------------------------------------------------------------------------------
def snapshot(fs, date, league, weight, country):
    out = {}
    n = len(fs.dates)
    out["n_fights"] = n
    out["debut"] = float(n == 0)
    t = np.array([(date - d).days for d in fs.dates], dtype=float)
    if n == 0:
        out.update(days_since_last=np.nan, tenure_days=0.0, fights_365=0, fights_730=0)
    else:
        out["days_since_last"] = t[-1]
        out["tenure_days"] = t[0]
        out["fights_365"] = float((t <= 365).sum())
        out["fights_730"] = float((t <= 730).sum())
        out["avg_gap_days"] = t[0] / n if n > 1 else np.nan
    res = _arr(fs, "res")
    valid = ~np.isnan(res)
    wins, losses = _arr(fs, "win").sum(), _arr(fs, "loss").sum()
    out.update(wins=wins, losses=losses, win_rate_s=(wins + 1) / (wins + losses + 2))
    # streak
    streak = 0
    for v in res[valid][::-1]:
        if v == 1 and streak >= 0:
            streak += 1
        elif v == 0 and streak <= 0:
            streak -= 1
        else:
            break
    out["streak"] = streak
    rv = res[valid]
    out["last_res"] = rv[-1] if len(rv) else np.nan
    out["last3_winrate"] = rv[-3:].mean() if len(rv) else np.nan
    for k in ["ko_win", "sub_win", "dec_win", "ko_loss", "sub_loss", "dec_loss", "split_win", "split_loss", "went_dist", "title", "five"]:
        a = _arr(fs, k)
        out[k + "_n"] = a.sum()
        out[k + "_rate"] = (a.sum() + 0.0) / (n + 2) if n else np.nan
    out["finish_win_share"] = (out["ko_win_n"] + out["sub_win_n"]) / wins if wins else np.nan
    out["finished_loss_share"] = (out["ko_loss_n"] + out["sub_loss_n"]) / losses if losses else np.nan
    if n:
        recent = t <= 730
        out["ko_loss_2y"] = _arr(fs, "ko_loss")[recent].sum()
        out["last_ko_loss"] = _arr(fs, "ko_loss")[-1]
        out["losses_last2"] = _arr(fs, "loss")[-2:].sum()
    mins = _arr(fs, "mins")
    out["cage_mins"] = np.nansum(mins)
    out["avg_mins"] = np.nanmean(mins) if n else np.nan
    dm = _arr(fs, "dec_margin")
    out["dec_margin_avg"] = np.nanmean(dm) if np.isfinite(dm).any() else np.nan
    # weight movement
    out["weight_change"] = weight - fs.last_weight if weight == weight and fs.last_weight == fs.last_weight else 0.0
    w_hist = _arr(fs, "weight")
    out["fights_at_weight"] = float((w_hist == weight).sum()) if weight == weight else np.nan
    out["country_fights"] = fs.countries.get(country, 0)
    # strength of schedule & performance vs expectation (uses opponent ratings as they were at the time)
    oe = _arr(fs, "opp_elo")
    ex = _arr(fs, "exp")
    if n:
        out["sos_elo"] = np.nanmean(oe)
        wmask = _arr(fs, "win") == 1
        lmask = _arr(fs, "loss") == 1
        out["best_win_elo"] = oe[wmask].max() if wmask.any() else np.nan
        out["worst_loss_elo"] = oe[lmask].min() if lmask.any() else np.nan
        over = res - ex
        out["over_exp_all"] = np.nanmean(over)
        out["over_exp_last3"] = np.nanmean(over[-3:])
    # ---- performance statistics (only bouts with round stats) ----
    hs = _arr(fs, "has_stats") == 1
    decay = 0.5 ** (t / 548.0) if n else np.array([])
    for tag, w in (("c", np.ones(n)), ("d", decay)):
        w = w * hs
        M = np.nansum(w * mins)
        S = lambda k: np.nansum(w * _arr(fs, k))
        m0 = 12.0  # prior minutes
        out[f"{tag}_stat_mins"] = M
        for c, lab in (("sig_l", "slpm"), ("o_sig_l", "sapm"), ("tot_l", "tlpm"), ("o_tot_l", "tapm"), ("sig_a", "sig_att_pm"),
                       ("o_sig_a", "o_sig_att_pm"), ("head_l", "head_pm"), ("body_l", "body_pm"), ("leg_l", "leg_pm"),
                       ("dist_l", "dist_pm"), ("clinch_l", "clinch_pm"), ("ground_l", "ground_pm"),
                       ("o_head_l", "o_head_pm"), ("o_body_l", "o_body_pm"), ("o_leg_l", "o_leg_pm"),
                       ("o_dist_l", "o_dist_pm"), ("o_clinch_l", "o_clinch_pm"), ("o_ground_l", "o_ground_pm")):
            base = c[2:] if c.startswith("o_") else c
            out[f"{tag}_{lab}"] = _safe(S(c), M, league.rate(base, 0.0), m0)
        for c, lab, prior_n in (("kd", "kd15", 30), ("o_kd", "kd_abs15", 30), ("td_l", "td15", 30), ("o_td_l", "td_abs15", 30),
                                ("td_a", "tda15", 30), ("o_td_a", "o_tda15", 30), ("sub_att", "sub15", 30), ("o_sub_att", "o_sub15", 30),
                                ("rev", "rev15", 30), ("o_rev", "o_rev15", 30)):
            base = c[2:] if c.startswith("o_") else c
            out[f"{tag}_{lab}"] = 15 * _safe(S(c), M, league.rate(base, 0.0), prior_n)
        out[f"{tag}_ctrl_share"] = _safe(S("ctrl") / 60, M, league.rate("ctrl", 0.0) / 60, m0)
        out[f"{tag}_ctrl_abs_share"] = _safe(S("o_ctrl") / 60, M, league.rate("ctrl", 0.0) / 60, m0)
        out[f"{tag}_sig_acc"] = _safe(S("sig_l"), S("sig_a"), league.acc("sig_l", "sig_a", 0.45), 40)
        out[f"{tag}_sig_def"] = 1 - _safe(S("o_sig_l"), S("o_sig_a"), league.acc("sig_l", "sig_a", 0.45), 40)
        out[f"{tag}_head_acc"] = _safe(S("head_l"), S("head_a"), league.acc("head_l", "head_a", 0.35), 40)
        out[f"{tag}_head_def"] = 1 - _safe(S("o_head_l"), S("o_head_a"), league.acc("head_l", "head_a", 0.35), 40)
        out[f"{tag}_td_acc"] = _safe(S("td_l"), S("td_a"), league.acc("td_l", "td_a", 0.35), 6)
        out[f"{tag}_td_def"] = 1 - _safe(S("o_td_l"), S("o_td_a"), league.acc("td_l", "td_a", 0.35), 6)
        out[f"{tag}_sig_share"] = _safe(S("sig_l"), S("sig_l") + S("o_sig_l"), 0.5, 20)
        out[f"{tag}_kd_share"] = _safe(S("kd"), S("kd") + S("o_kd"), 0.5, 2)
        out[f"{tag}_td_share"] = _safe(S("td_l"), S("td_l") + S("o_td_l"), 0.5, 3)
        out[f"{tag}_slpm_minus_sapm"] = out[f"{tag}_slpm"] - out[f"{tag}_sapm"]
        tot_sig = S("sig_l")
        out[f"{tag}_head_frac"] = _safe(S("head_l"), tot_sig, 0.6, 20)
        out[f"{tag}_leg_frac"] = _safe(S("leg_l"), tot_sig, 0.15, 20)
        out[f"{tag}_body_frac"] = _safe(S("body_l"), tot_sig, 0.2, 20)
        out[f"{tag}_dist_frac"] = _safe(S("dist_l"), tot_sig, 0.7, 20)
        out[f"{tag}_ground_frac"] = _safe(S("ground_l"), tot_sig, 0.12, 20)
        out[f"{tag}_clinch_frac"] = _safe(S("clinch_l"), tot_sig, 0.12, 20)
        # opponent-adjusted residuals (minute weighted, shrunk to 0)
        for k in ("adj_slpm", "adj_sapm", "adj_td", "adj_tda", "adj_ctrl", "adj_kd", "adj_acc", "adj_def", "adj_sub"):
            a = _arr(fs, k)
            ok = np.isfinite(a) & (w > 0)
            out[f"{tag}_{k}"] = np.nansum((w * mins * a)[ok]) / (np.nansum((w * mins)[ok]) + m0) if n else 0.0
    # cardio: late-round output relative to round 1 output
    es, ls = _arr(fs, "e_secs"), _arr(fs, "l_secs")
    er, lr = _arr(fs, "early_rate"), _arr(fs, "late_rate")
    ok = np.isfinite(lr) & np.isfinite(er) & (ls > 60)
    out["late_early_ratio"] = (np.sum(lr[ok] * ls[ok]) / np.sum(ls[ok]) + 1) / (np.sum(er[ok] * es[ok]) / np.sum(es[ok]) + 1) if ok.sum() else np.nan
    out["late_fights"] = float(ok.sum())
    return out


# ----------------------------------------------------------------------------------------------
# 4. engine
# ----------------------------------------------------------------------------------------------
def pair_features(opps_a, opps_b, a, b):
    """Head-to-head history and common-opponent comparison (from a's perspective, antisymmetric)."""
    out = {}
    h2h = [r for o, r in opps_a if o == b and r == r]
    out["h2h_n"] = float(len(h2h))
    out["h2h_score"] = np.mean(h2h) - 0.5 if h2h else 0.0
    ra, rb = defaultdict(list), defaultdict(list)
    for o, r in opps_a:
        if r == r and o != b: ra[o].append(r)
    for o, r in opps_b:
        if r == r and o != a: rb[o].append(r)
    common = set(ra) & set(rb)
    out["common_n"] = float(len(common))
    out["common_score"] = float(np.mean([np.mean(ra[c]) - np.mean(rb[c]) for c in common])) if common else 0.0
    return out


def country_of(loc):
    return str(loc).split(",")[-1].strip() if isinstance(loc, str) else "Unknown"


def run(fights, perf, tott=None, elo_k=None, verbose=True):
    """Returns one row per fight with pre-fight features for f1 (`a_*`) and f2 (`b_*`)."""
    league = League()
    F = defaultdict(Fighter)
    elo = Elo(k=40)
    elo_mov = Elo(k=40, finish_mult=1.5)
    elo_fast = Elo(k=80)
    elo_new = Elo(k=32, k_decay_fights=2.0)
    elo_tuned = Elo(k=90, k_decay_fights=2.0)  # best on 2005-2014 tuning window (research/03)
    glk = Glicko2()
    share = {k: ShareElo(k=v) for k, v in (("strike", 80), ("grapple", 80), ("kd", 40), ("sub", 40))}
    p_by_fight = {k: g for k, g in perf.groupby("fight_id")}
    out = []
    for date, day in fights.groupby("date", sort=True):
        snaps = {}
        for f in day.itertuples():
            ctry = country_of(f.location)
            rec = dict(fight_id=f.fight_id)
            for side, fid, oid in (("a", f.f1_id, f.f2_id), ("b", f.f2_id, f.f1_id)):
                s = snapshot(F[fid], date, league, f.weight_lbs, ctry)
                s["elo"] = elo.get(fid); s["elo_mov"] = elo_mov.get(fid); s["elo_fast"] = elo_fast.get(fid); s["elo_new"] = elo_new.get(fid)
                s["elo_tuned"] = elo_tuned.get(fid)
                s["glicko"], s["glicko_rd"] = glk.get(fid, date)
                # strength of schedule re-rated with opponents' CURRENT ratings (as of this date)
                fo = F[fid].opps
                if fo:
                    now = np.array([elo_tuned.get(o) for o, _ in fo]); rr = np.array([r for _, r in fo], dtype=float)
                    s["sos_now"] = now.mean()
                    s["best_win_now"] = now[rr == 1].max() if (rr == 1).any() else np.nan
                    s["worst_loss_now"] = now[rr == 0].min() if (rr == 0).any() else np.nan
                    s["win_quality_now"] = np.nanmean(np.where(rr == 1, now - 1500, np.where(rr == 0, -(1500 - now).clip(min=0) - 50, 0)))
                for k, se in share.items():
                    s[f"selo_{k}"] = se.get(fid)
                snaps[(f.fight_id, fid)] = s
                rec.update({f"{side}_{k}": v for k, v in s.items()})
            rec["glicko_p"] = glk.expect(f.f1_id, f.f2_id, date)
            rec.update(pair_features(F[f.f1_id].opps, F[f.f2_id].opps, f.f1_id, f.f2_id))
            out.append(rec)
        # ---- update states with this date's results ----
        for f in day.itertuples():
            rows = p_by_fight.get(f.fight_id)
            if rows is None:
                continue
            res_a = 0.5 if f.outcome == "D/D" else f.winner
            fin = f.method in ("KO", "DOC", "SUB")
            ea = elo_expect(elo.get(f.f1_id), elo.get(f.f2_id))
            pre = {f.f1_id: (elo.get(f.f2_id), ea), f.f2_id: (elo.get(f.f1_id), 1 - ea)}
            for r in rows.to_dict("records"):
                fs = F[r["fid"]]
                opp = snaps[(f.fight_id, r["oid"])]
                has = float(r["sig_a"] == r["sig_a"] and r["mins"] == r["mins"] and r["mins"] > 0)
                m = r["mins"] if r["mins"] == r["mins"] and r["mins"] > 0 else np.nan
                h = fs.h
                h["t"].append(0); h["res"].append(r["res"]); h["mins"].append(m)
                for k in ("win", "loss", "ko_win", "sub_win", "dec_win", "ko_loss", "sub_loss", "dec_loss", "split_win", "split_loss", "went_dist", "dec_margin"):
                    h[k].append(r[k])
                h["method_fin"].append(r["finish"]); h["title"].append(float(r["title"])); h["five"].append(float(r["sched_rounds"] == 5))
                h["weight"].append(r["weight_lbs"]); h["opp_elo"].append(pre[r["fid"]][0]); h["exp"].append(pre[r["fid"]][1])
                h["has_stats"].append(has)
                for c in OWN:
                    h[c].append(r[c]); h["o_" + c].append(r["o_" + c])
                es, ls = r.get("e_secs"), r.get("l_secs")
                h["e_secs"].append(es); h["l_secs"].append(ls)
                h["early_rate"].append(r["e_sig_l"] / (es / 60) if es and es == es and es > 0 else np.nan)
                h["late_rate"].append(r["l_sig_l"] / (ls / 60) if ls and ls == ls and ls > 0 else np.nan)
                if has:
                    # performance relative to what this opponent usually allows (opponent's PRE-fight profile, career window)
                    h["adj_slpm"].append(r["sig_l"] / m - opp["c_sapm"])
                    h["adj_sapm"].append(r["o_sig_l"] / m - opp["c_slpm"])
                    h["adj_td"].append(15 * r["td_l"] / m - opp["c_td_abs15"])
                    h["adj_tda"].append(15 * r["td_a"] / m - opp["c_o_tda15"])
                    h["adj_ctrl"].append(r["ctrl"] / 60 / m - opp["c_ctrl_abs_share"] if r["ctrl"] == r["ctrl"] else np.nan)
                    h["adj_kd"].append(15 * r["kd"] / m - opp["c_kd_abs15"])
                    h["adj_acc"].append((r["sig_l"] / r["sig_a"] - (1 - opp["c_sig_def"])) if r["sig_a"] > 0 else np.nan)
                    h["adj_def"].append(((1 - r["o_sig_l"] / r["o_sig_a"]) - (1 - opp["c_sig_acc"])) if r["o_sig_a"] > 0 else np.nan)
                    h["adj_sub"].append(15 * r["sub_att"] / m - opp["c_o_sub15"])
                else:
                    for k in ("adj_slpm", "adj_sapm", "adj_td", "adj_tda", "adj_ctrl", "adj_kd", "adj_acc", "adj_def", "adj_sub"):
                        h[k].append(np.nan)
                fs.dates.append(date)
                fs.last_weight = r["weight_lbs"] if r["weight_lbs"] == r["weight_lbs"] else fs.last_weight
                fs.countries[country_of(r["location"])] += 1
                league.add(r)
            F[f.f1_id].opps.append((f.f2_id, res_a)); F[f.f2_id].opps.append((f.f1_id, 1 - res_a if res_a == res_a else res_a))
            if res_a == res_a:  # skip no contests for win ratings
                elo_tuned.update(f.f1_id, f.f2_id, res_a)
                elo.update(f.f1_id, f.f2_id, res_a); elo_fast.update(f.f1_id, f.f2_id, res_a)
                elo_mov.update(f.f1_id, f.f2_id, res_a, finish=fin); elo_new.update(f.f1_id, f.f2_id, res_a)
                glk.update(f.f1_id, f.f2_id, res_a, date)
            ra = rows.set_index("fid")
            if f.f1_id in ra.index and f.f2_id in ra.index:
                a, b = ra.loc[f.f1_id], ra.loc[f.f2_id]
                tot = a.sig_l + b.sig_l
                share["strike"].update(f.f1_id, f.f2_id, a.sig_l / tot if tot and tot == tot and tot > 4 else None)
                ga = (a.ctrl if a.ctrl == a.ctrl else 0) + 30 * (a.td_l if a.td_l == a.td_l else 0)
                gb = (b.ctrl if b.ctrl == b.ctrl else 0) + 30 * (b.td_l if b.td_l == b.td_l else 0)
                share["grapple"].update(f.f1_id, f.f2_id, ga / (ga + gb) if ga + gb > 30 else None)
                kt = (a.kd or 0) + (b.kd or 0)
                share["kd"].update(f.f1_id, f.f2_id, a.kd / kt if kt and kt == kt and kt > 0 else None)
                st = (a.sub_att or 0) + (b.sub_att or 0)
                share["sub"].update(f.f1_id, f.f2_id, a.sub_att / st if st and st == st and st > 0 else None)
        if verbose and date.month == 1 and date.day < 8:
            print("features through", date.date(), flush=True)
    return pd.DataFrame(out)

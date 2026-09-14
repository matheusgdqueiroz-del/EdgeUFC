"""FightMatrix pre-fight ratings per UFC bout (a_fm_*, b_fm_*).

Primary: the bout's own row on a FightMatrix profile (pre-fight ratings of both men as they were then).
Fallback: the fighter's most recent POST-fight rating from bouts at least 3 days earlier.
"""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from ufc.fetch import ROOT
from ufc.sources.ufcstats import slug

FM = ROOT / "data" / "fightmatrix"


def _rank(txt):
    m = re.match(r"#(\d+)", str(txt))
    return float(m.group(1)) if m else np.nan


def fm_features(fights):
    prof = pd.read_csv(FM / "profiles.csv")
    rows = pd.read_csv(FM / "fights.csv", parse_dates=["date"])
    link = pd.read_csv(ROOT / "data" / "sherdog_link.csv")
    sd2fid = dict(zip(link.sd_url, link.fid))
    prof["sd"] = prof.sherdog.fillna("").str.replace(r"^https?://(www\.)?sherdog\.com", "", regex=True).str.strip()
    prof["fid"] = prof.sd.map(sd2fid)
    # fall back to name matching when a profile has no Sherdog link
    names = pd.concat([fights[["f1_id", "f1_name"]].set_axis(["fid", "name"], axis=1), fights[["f2_id", "f2_name"]].set_axis(["fid", "name"], axis=1)]).drop_duplicates("fid")
    by_slug = dict(zip(names.name.map(slug), names.fid))
    prof["fid"] = prof.fid.fillna(prof.name.map(lambda n: by_slug.get(slug(n))))
    p2f = dict(zip(prof.fm_path, prof.fid))
    rows["fid"], rows["opp_fid"] = rows.fm_path.map(p2f), rows.opp_path.map(p2f)
    rows = rows[rows.fid.notna() & rows.date.notna()].sort_values("date")
    by_f = {f: g for f, g in rows.groupby("fid")}
    out = []
    for f in fights.itertuples():
        rec = {"fight_id": f.fight_id}
        got = None
        for fid, oid, flip in ((f.f1_id, f.f2_id, False), (f.f2_id, f.f1_id, True)):
            g = by_f.get(fid)
            if g is None:
                continue
            c = g[(g.date - f.date).abs() <= pd.Timedelta(days=2)]
            c = c[(c.opp_fid == oid) | c.opp.fillna("").map(lambda n: fuzz.ratio(slug(n), slug(f.f2_name if not flip else f.f1_name)) >= 85)]
            if len(c):
                r = c.iloc[0]
                me = [r.r1_pre, r.r2_pre, r.r3_pre, _rank(r.rank_pre)]
                op = [r.o1_pre, r.o2_pre, r.o3_pre, _rank(r.opp_rank_pre)]
                got = (op, me) if flip else (me, op)
                break
        for side, fid, i in (("a", f.f1_id, 0), ("b", f.f2_id, 1)):
            if got is not None:
                v = got[i]
                rec[f"{side}_fm_src_bout"] = 1.0
            else:
                g = by_f.get(fid)
                prev = g[g.date < f.date - pd.Timedelta(days=3)] if g is not None else None
                if prev is not None and len(prev):
                    r = prev.iloc[-1]
                    v = [r.r1_post, r.r2_post, r.r3_post, np.nan]
                else:
                    v = [np.nan] * 4
                rec[f"{side}_fm_src_bout"] = 0.0
            rec[f"{side}_fm_r1"], rec[f"{side}_fm_r2"], rec[f"{side}_fm_r3"], rec[f"{side}_fm_rank"] = v
            rec[f"{side}_fm_ranked"] = float(np.isfinite(v[3])) if got is not None else np.nan
        out.append(rec)
    return pd.DataFrame(out)

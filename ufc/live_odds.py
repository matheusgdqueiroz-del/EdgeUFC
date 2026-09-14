"""Current BestFightOdds lines for upcoming bouts (same fighter pages as the historical crawl)."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from ufc.sources import bfo
from ufc.sources.ufcstats import slug
from ufc.sources.odds import dec


def fighter_page(name, max_age_days=0.25):
    hits = bfo.search(name)
    exact = [h for n, h in hits if slug(n) == slug(name)]
    if not exact and hits and fuzz.ratio(slug(hits[0][0]), slug(name)) >= 85:
        exact = [hits[0][1]]
    if not exact:
        return []
    return bfo.parse_fighter_page(exact[0], max_age_days=max_age_days)


def lines_for(upcoming, max_age_days=0.25):
    """upcoming: rows with f1_name, f2_name, date. Returns market columns oriented to f1 (side a)."""
    out = []
    for f in upcoming.itertuples():
        rows = fighter_page(f.f1_name, max_age_days) or fighter_page(f.f2_name, max_age_days)
        best, sc_best = None, 0
        for r in rows:
            if r["date"] is not None and pd.notna(r["date"]) and abs((r["date"] - f.date).days) > 4:
                continue
            a = (fuzz.ratio(slug(r["f1"]), slug(f.f1_name)) + fuzz.ratio(slug(r["f2"]), slug(f.f2_name))) / 2
            b = (fuzz.ratio(slug(r["f1"]), slug(f.f2_name)) + fuzz.ratio(slug(r["f2"]), slug(f.f1_name))) / 2
            if max(a, b) > sc_best:
                best, sc_best, flip = r, max(a, b), b > a
        rec = {"fight_id": f.fight_id}
        if best is not None and sc_best >= 80:
            s1, s2 = ("f2", "f1") if flip else ("f1", "f2")
            lo_a, hi_a = dec(best[f"{s1}_close_lo"]), dec(best[f"{s1}_close_hi"])
            lo_b, hi_b = dec(best[f"{s2}_close_lo"]), dec(best[f"{s2}_close_hi"])
            spark = pd.to_numeric(str(best.get("f1_spark")).split(",")[-1], errors="coerce")
            pa_raw = (1 / lo_a + 1 / hi_a) / 2
            pb_raw = (1 / lo_b + 1 / hi_b) / 2
            if np.isfinite(spark):  # sparkline belongs to the page's main-row fighter (f1 of the row)
                if flip: pb_raw = 1 / spark
                else: pa_raw = 1 / spark
            rec.update(p_close=pa_raw / (pa_raw + pb_raw), a_close_mean_dec=1 / pa_raw, b_close_mean_dec=1 / pb_raw,
                       a_close_best_dec=float(np.fmax(lo_a, hi_a)), b_close_best_dec=float(np.fmax(lo_b, hi_b)),
                       a_open_dec=float(dec(best[f"{s1}_open"])), b_open_dec=float(dec(best[f"{s2}_open"])))
            oa, ob = 1 / rec["a_open_dec"], 1 / rec["b_open_dec"]
            rec["p_open"] = oa / (oa + ob)
        out.append(rec)
    return pd.DataFrame(out)

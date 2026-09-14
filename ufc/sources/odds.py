"""Turn BestFightOdds fighter-page lines into per-fight market probabilities matched to UFCStats bouts.

Per matchup we get, for each fighter: opening line, closing range [lo, hi] across sportsbooks, and (for the
page owner) an average-odds sparkline whose last point is the mean closing price.
"""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from ufc.fetch import ROOT
from ufc.sources.ufcstats import slug


def dec(american):
    a = pd.to_numeric(american, errors="coerce")
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def load_lines():
    b = pd.read_csv(ROOT / "data" / "bfo" / "fighter_page_lines.csv")
    b["date"] = pd.to_datetime(b.date, format="ISO8601", errors="coerce")  # incremental updates can mix date formats
    b = b[b.f1_url.notna() & b.f2_url.notna()]  # undated rows = BFO "Future Events", used only for upcoming bouts
    # orient every row alphabetically by url so both fighters' pages collapse into one matchup
    swap = b.f1_url.str.lower() > b.f2_url.str.lower()
    cols = ["", "_url", "_open", "_close_lo", "_close_hi"]
    for c in cols:
        b.loc[swap, [f"f1{c}", f"f2{c}"]] = b.loc[swap, [f"f2{c}", f"f1{c}"]].values
    b["spark_side"] = np.where(swap, 2, 1)
    last_spark = pd.to_numeric(b.f1_spark.astype(str).str.split(",").str[-1], errors="coerce")
    b["f1_spark_last"] = np.where(b.spark_side == 1, last_spark, np.nan)
    b["f2_spark_last"] = np.where(b.spark_side == 2, last_spark, np.nan)
    b["f1_spark_all"] = b.f1_spark.where(b.spark_side == 1)
    b["f2_spark_all"] = b.f1_spark.where(b.spark_side == 2)
    first_spark = pd.to_numeric(b.f1_spark.astype(str).str.split(",").str[0], errors="coerce")
    b["n_spark"] = b.f1_spark.astype(str).str.count(",") + 1
    agg = {c: "first" for c in ["event", "event_url", "f1", "f2", "f1_open", "f2_open", "f1_close_lo", "f1_close_hi", "f2_close_lo", "f2_close_hi"]}
    agg.update(f1_spark_last="max", f2_spark_last="max", n_spark="max", f1_spark_all="first", f2_spark_all="first")
    m = b.groupby(["date", "f1_url", "f2_url"], as_index=False, dropna=False).agg(agg)
    return m


def implied(m):
    out = pd.DataFrame(index=m.index)
    for s in ("f1", "f2"):
        out[f"{s}_open_dec"] = dec(m[f"{s}_open"])
        lo, hi = dec(m[f"{s}_close_lo"]), dec(m[f"{s}_close_hi"])
        out[f"{s}_close_best_dec"] = np.fmax(lo, hi)
        out[f"{s}_close_worst_dec"] = np.fmin(lo, hi)
        mid_prob = (1 / lo + 1 / hi) / 2
        spark = m[f"{s}_spark_last"].values
        out[f"{s}_close_prob_raw"] = np.where(np.isfinite(spark), 1 / spark, mid_prob)
        out[f"{s}_close_mean_dec"] = 1 / out[f"{s}_close_prob_raw"]
    ov = out.f1_close_prob_raw + out.f2_close_prob_raw
    out["close_overround"] = ov
    out["p1_close"] = out.f1_close_prob_raw / ov
    oo = 1 / out.f1_open_dec + 1 / out.f2_open_dec
    out["p1_open"] = (1 / out.f1_open_dec) / oo
    out["open_overround"] = oo
    return out


STAGES = 10  # BFO sparklines hold up to 10 average-price points from the opening line (0) to the close (9)


def stage_prices(spark):
    """Average decimal price at each of STAGES evenly spaced points of a sparkline string (NaN if absent)."""
    try:
        v = np.array([float(x) for x in str(spark).split(",") if x.strip()])
    except ValueError:
        v = np.array([])
    if len(v) < 2 or not np.all(v > 1):
        return np.full(STAGES, np.nan)
    return np.interp(np.linspace(0, 1, STAGES), np.linspace(0, 1, len(v)), v)


def _sim(x, y):
    """Name similarity robust to given/family name order (e.g. 'Wang Cong' vs 'Cong Wang')."""
    return max(fuzz.ratio(x, y), fuzz.token_sort_ratio(x.replace("-", " "), y.replace("-", " ")))


def match_to_fights(fights, m):
    """Match BFO matchups to UFCStats bouts by date (+-3 days) and fighter names.
    Upcoming bouts may also match BFO's undated "Future Events" lines."""
    m = m.copy()
    m["s1"], m["s2"] = m.f1.map(slug), m.f2.map(slug)
    fx = fights[["fight_id", "date", "f1_name", "f2_name"]].copy()
    fx["s1"], fx["s2"] = fx.f1_name.map(slug), fx.f2_name.map(slug)
    by_date = {d: g for d, g in m[m.date.notna()].groupby("date")}
    undated = m[m.date.isna()]
    today = pd.Timestamp.today().normalize()
    rows = []
    for f in fx.itertuples():
        best, best_score, flip = None, 0, False
        for off in (0, -1, 1, -2, 2, -3, 3, None):
            g = by_date.get(f.date + pd.Timedelta(days=off)) if off is not None else (undated if f.date >= today and len(undated) else None)
            if g is None:
                continue
            for r in g.itertuples():
                a = (_sim(f.s1, r.s1) + _sim(f.s2, r.s2)) / 2
                b = (_sim(f.s1, r.s2) + _sim(f.s2, r.s1)) / 2
                sc, fl = (a, False) if a >= b else (b, True)
                sc -= abs(off) * 2 if off is not None else 5  # dated lines win over undated ones
                if sc > best_score:
                    best, best_score, flip = r.Index, sc, fl
            if best_score >= 97:
                break
        if best is not None and best_score >= 80:
            rows.append((f.fight_id, best, flip, best_score))
    return pd.DataFrame(rows, columns=["fight_id", "bfo_idx", "flip", "match_score"])


def build(fights):
    m = load_lines()
    imp = implied(m)
    mm = pd.concat([m, imp], axis=1)
    link = match_to_fights(fights, m)
    link = link.sort_values("match_score", ascending=False).drop_duplicates("bfo_idx")  # one market per bout
    x = link.join(mm, on="bfo_idx")
    fl = x.flip.values
    def oriented(c1, c2):
        return np.where(fl, x[c2], x[c1]), np.where(fl, x[c1], x[c2])
    o = pd.DataFrame({"fight_id": x.fight_id, "match_score": x.match_score, "bfo_event": x.event})
    o["p_close"] = np.where(fl, 1 - x.p1_close, x.p1_close)
    o["p_open"] = np.where(fl, 1 - x.p1_open, x.p1_open)
    o["a_open_dec"], o["b_open_dec"] = oriented("f1_open_dec", "f2_open_dec")
    o["a_close_best_dec"], o["b_close_best_dec"] = oriented("f1_close_best_dec", "f2_close_best_dec")
    o["a_close_worst_dec"], o["b_close_worst_dec"] = oriented("f1_close_worst_dec", "f2_close_worst_dec")
    o["a_close_mean_dec"], o["b_close_mean_dec"] = oriented("f1_close_mean_dec", "f2_close_mean_dec")
    o["close_overround"], o["open_overround"], o["n_spark"] = x.close_overround.values, x.open_overround.values, x.n_spark.values
    # line stages: vig-free probability and average prices along the sparkline (stage 0 = open, 9 = close)
    s1 = np.vstack([stage_prices(v) for v in x.f1_spark_all]) if len(x) else np.empty((0, STAGES))
    s2 = np.vstack([stage_prices(v) for v in x.f2_spark_all]) if len(x) else np.empty((0, STAGES))
    sa, sb = np.where(fl[:, None], s2, s1), np.where(fl[:, None], s1, s2)
    for k in range(STAGES):
        o[f"a_s{k}_dec"], o[f"b_s{k}_dec"] = sa[:, k], sb[:, k]
        o[f"p_s{k}"] = (1 / sa[:, k]) / (1 / sa[:, k] + 1 / sb[:, k])
    # sanity: drop implausible markets (bad parses, props)
    ok = o.close_overround.between(0.98, 1.25) & o.p_close.between(0.01, 0.99)
    return o[ok].reset_index(drop=True)


if __name__ == "__main__":
    from ufc.sources.ufcstats import load_fights
    f = load_fights()
    o = build(f)
    o.to_parquet(ROOT / "data" / "odds.parquet")
    cov = f.merge(o, on="fight_id", how="left").groupby(f.date.dt.year).p_close.apply(lambda s: round(s.notna().mean(), 3))
    print(len(o)); print(cov.to_string())

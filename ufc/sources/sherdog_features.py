"""Link UFCStats fighters to Sherdog profiles and derive pre-fight features from full pro records.

All record-derived features for a bout on date D use Sherdog rows dated < D only.
Bio fields used: birth date (constant), nationality / home town (static) -> home-country & travel features.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz
from ufc.fetch import ROOT
from ufc.sources.ufcstats import slug
from ufc.ratings import Elo, elo_expect

SD = ROOT / "data" / "sherdog"


def link_fighters(fights):
    """Map UFCStats fighter ids -> Sherdog URL by matching bouts on (date, names)."""
    ev = pd.read_csv(SD / "events.csv", parse_dates=["date"])
    ef = pd.read_csv(SD / "event_fights.csv").merge(ev[["sd_event_url", "date"]], on="sd_event_url")
    ef["s1"], ef["s2"] = ef.f1.map(slug), ef.f2.map(slug)
    by_date = {d: g for d, g in ef.groupby("date")}
    votes = defaultdict(lambda: defaultdict(float))
    for f in fights.itertuples():
        s1, s2 = slug(f.f1_name), slug(f.f2_name)
        best, sc_best = None, 0
        for off in (0, -1, 1):
            g = by_date.get(f.date + pd.Timedelta(days=off))
            if g is None:
                continue
            for r in g.itertuples():
                a = (fuzz.token_sort_ratio(s1, r.s1) + fuzz.token_sort_ratio(s2, r.s2)) / 2
                b = (fuzz.token_sort_ratio(s1, r.s2) + fuzz.token_sort_ratio(s2, r.s1)) / 2
                sc, pair = (a, (r.f1_url, r.f2_url)) if a >= b else (b, (r.f2_url, r.f1_url))
                if sc > sc_best:
                    best, sc_best = pair, sc
        if best and sc_best >= 70:
            votes[f.f1_id][best[0]] += sc_best
            votes[f.f2_id][best[1]] += sc_best
    return {fid: max(v, key=v.get) for fid, v in votes.items()}


def load_records():
    r = pd.read_csv(SD / "records.csv", parse_dates=["date"])
    r = r[(r.kind == "pro") & r.date.notna()].copy()
    r["res"] = r.result.map({"win": 1.0, "loss": 0.0, "draw": 0.5})  # nc -> NaN
    m = r.method.fillna("").str.upper()
    r["is_ko"] = m.str.startswith("KO") | m.str.startswith("TKO")
    r["is_sub"] = m.str.startswith("SUBMISSION")
    r["is_dec"] = m.str.startswith("DECISION")
    r["ufc"] = r.event.fillna("").str.contains(r"\bUFC\b|Ultimate Fighter", regex=True)
    return r.drop_duplicates(["sd_url", "date", "opp_url", "result"]).sort_values(["date", "sd_url", "opp_url"], kind="mergesort")


def global_elo(records, k=80.0):
    """Elo over every pro fight appearing in any linked fighter's record (dedup both perspectives).
    Returns a function (sd_url, date) -> rating using only fights before date."""
    r = records[records.res.notna() & records.opp_url.notna()].copy()
    a = r.sd_url.where(r.sd_url < r.opp_url, r.opp_url)
    b = r.opp_url.where(r.sd_url < r.opp_url, r.sd_url)
    r["score_a"] = np.where(r.sd_url < r.opp_url, r.res, 1 - r.res)
    r["a"], r["b"] = a, b
    # canonical, order-independent fight list: one row per (date, pair), conflicting perspectives averaged,
    # processed in a fully deterministic order (unstable sorts made ratings depend on unrelated rows)
    g = r.groupby(["date", "a", "b"], as_index=False).score_a.mean().sort_values(["date", "a", "b"], kind="mergesort")
    e = Elo(k=k, k_decay_fights=2.0)
    hist = defaultdict(list)  # url -> list of (date, rating_after)
    for x in g.itertuples():
        e.update(x.a, x.b, x.score_a)
        hist[x.a].append((x.date, e.get(x.a)))
        hist[x.b].append((x.date, e.get(x.b)))
    def rating(url, date):
        h = hist.get(url)
        if not h:
            return 1500.0, 0
        i = np.searchsorted(np.array([d for d, _ in h], dtype="datetime64[ns]"), np.datetime64(date), side="left")
        return (h[i - 1][1] if i else 1500.0), int(i)
    return rating


def record_features(fights, link, records, bios):
    rating = global_elo(records)
    by = {u: g.sort_values(["date", "opp_url", "result"], kind="mergesort") for u, g in records.groupby("sd_url")}
    bio = bios.set_index("sd_url")
    ufc_debut = {}  # sd_url -> first UFC fight date (from Sherdog), used to credit wins over UFC-level opponents
    for u, g in records[records.ufc].groupby("sd_url"):
        ufc_debut[u] = g.date.min()
    out = []
    for f in fights.itertuples():
        rec = {"fight_id": f.fight_id}
        for side, fid in (("a", f.f1_id), ("b", f.f2_id)):
            u = link.get(fid)
            g = by.get(u)
            if g is None:
                rec[f"{side}_sd_linked"] = 0.0
                continue
            # strictly before the bout, with a 3-day guard: Sherdog sometimes dates overseas cards a day
            # earlier/later than UFCStats, which would otherwise leak the bout's own result
            cutoff = f.date - pd.Timedelta(days=3)
            h = g[g.date < cutoff]
            n = len(h)
            w = (h.res == 1).sum(); l = (h.res == 0).sum()
            nonufc = h[~h.ufc]
            rec.update({
                f"{side}_sd_linked": 1.0,
                f"{side}_pro_n": n, f"{side}_pro_wins": w, f"{side}_pro_losses": l,
                f"{side}_pro_winrate_s": (w + 1) / (w + l + 2),
                f"{side}_pro_ko_wins": (h.is_ko & (h.res == 1)).sum(), f"{side}_pro_sub_wins": (h.is_sub & (h.res == 1)).sum(),
                f"{side}_pro_ko_losses": (h.is_ko & (h.res == 0)).sum(), f"{side}_pro_sub_losses": (h.is_sub & (h.res == 0)).sum(),
                f"{side}_pro_finish_rate": ((h.is_ko | h.is_sub) & (h.res == 1)).sum() / max(w, 1),
                f"{side}_nonufc_n": len(nonufc), f"{side}_nonufc_winrate_s": ((nonufc.res == 1).sum() + 1) / (len(nonufc.res.dropna()) + 2),
                f"{side}_pro_years": (f.date - h.date.min()).days / 365.25 if n else 0.0,
                f"{side}_pro_days_since_last": (f.date - h.date.max()).days if n else np.nan,
                f"{side}_pro_fights_2y": (h.date > f.date - pd.Timedelta(days=730)).sum(),
            })
            streak = 0
            for v in h.res.dropna().values[::-1]:
                if v == 1 and streak >= 0: streak += 1
                elif v == 0 and streak <= 0: streak -= 1
                else: break
            rec[f"{side}_pro_streak"] = streak
            last5 = h.res.dropna().values[-5:]
            rec[f"{side}_pro_last5"] = last5.mean() if len(last5) else np.nan
            # wins over opponents who had already fought in the UFC before this bout's date
            beat = h[h.res == 1].opp_url.map(lambda o: ufc_debut.get(o, pd.Timestamp.max) < cutoff)
            rec[f"{side}_wins_over_ufc_level"] = float(beat.astype(bool).sum()) if len(beat) else 0.0
            ge, gn = rating(u, cutoff)
            rec[f"{side}_gelo"], rec[f"{side}_gelo_n"] = ge, gn
            if u in bio.index:
                b = bio.loc[u]
                rec[f"{side}_nationality"] = b.nationality
                rec[f"{side}_locality"] = b.locality
                rec[f"{side}_sd_dob"] = b.birth_date
        out.append(rec)
    return pd.DataFrame(out)


if __name__ == "__main__":
    from ufc.sources.ufcstats import load_fights
    f = load_fights()
    link = link_fighters(f)
    pd.Series(link, name="sd_url").rename_axis("fid").reset_index().to_csv(ROOT / "data" / "sherdog_link.csv", index=False)
    ids = set(f.f1_id) | set(f.f2_id)
    print("linked", len(link), "of", len(ids))
    recs = load_records(); bios = pd.read_csv(SD / "fighters.csv")
    X = record_features(f, link, recs, bios)
    X.to_parquet(ROOT / "data" / "features_sherdog.parquet")
    print(X.shape); print(X.describe().T.head(30))

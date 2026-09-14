"""Fighter popularity / hype from English Wikipedia pageviews (Wikimedia REST API, available from 2015-07).

Hypothesis being tested: recreational money follows famous fighters, so pre-fight attention may predict
market mispricing. Features use ONLY views from windows ending 2 days before the bout
(pageviews spike on fight night and after, which would otherwise leak).
Future-obtainable: the API serves yesterday's counts every day.
"""
import json, re
from urllib.parse import quote
import numpy as np
import pandas as pd
from ufc.fetch import fetch, ROOT
from ufc.sources.ufcstats import slug
from ufc.sources import wiki

API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{}/daily/20150701/{}"
OUT = ROOT / "data" / "pageviews.parquet"


def article_map():
    """fighter slug -> Wikipedia article title, harvested from links on UFC event articles."""
    m = {}
    for t in wiki.event_titles():
        raw = fetch(wiki.RAW + t.replace(" ", "_"), min_len=10, delay=1.5)
        if not raw:
            continue
        res = raw.split("==Results==")[-1] if "==Results==" in raw else raw
        for target, label in re.findall(r"\[\[([^\]|#]+)(?:\|([^\]]+))?\]\]", res):
            name = (label or re.sub(r"\s*\(.*?\)\s*$", "", target)).strip()
            if len(name.split()) >= 2 and not re.search(r"UFC|Championship|weight|Arena|Center|decision|File:", target, re.I):
                m.setdefault(slug(name), target.strip())
    return m


def fetch_views(article, end, max_age_days=None):
    url = API.format(quote(article.replace(" ", "_"), safe=""), end)
    txt = fetch(url, min_len=2, delay=0.25, max_age_days=max_age_days, retries=6)
    if not txt:
        return pd.Series(dtype=float)
    items = json.loads(txt).get("items", [])
    return pd.Series({pd.Timestamp(i["timestamp"][:8]): i["views"] for i in items}, dtype=float)


def build(fights, refresh_ids=(), today=None):
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    amap = article_map()
    ids = pd.concat([fights[["f1_id", "f1_name"]].set_axis(["fid", "name"], axis=1),
                     fights[["f2_id", "f2_name"]].set_axis(["fid", "name"], axis=1)]).drop_duplicates("fid")
    last = pd.concat([fights[["f1_id", "date"]].set_axis(["fid", "date"], axis=1), fights[["f2_id", "date"]].set_axis(["fid", "date"], axis=1)]).groupby("fid").date.max()
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for r in ids.itertuples():
        if last.get(r.fid, today) < pd.Timestamp("2015-08-01"):
            continue
        art = amap.get(slug(r.name))
        if art:
            jobs.append((r.fid, art))
    def one(job):
        fid, art = job
        # historical series fetched once; fighters on upcoming cards get a fresh copy
        end = "20260913" if fid not in refresh_ids else today.strftime("%Y%m%d")
        s = fetch_views(art, end, max_age_days=0.5 if fid in refresh_ids else None)
        return pd.DataFrame({"fid": fid, "date": s.index, "views": s.values}) if len(s) else None
    with ThreadPoolExecutor(1) as ex:  # Wikimedia throttles concurrent per-article queries (429s); stay serial
        rows = [x for x in ex.map(one, jobs) if x is not None]
    print("articles", len(jobs), "with views", len(rows), flush=True)
    pv = pd.concat(rows, ignore_index=True)
    pv.to_parquet(OUT)
    return pv


def features(fights, pv):
    """a_/b_ log views in the 30 days ending 2 days pre-fight, 1-year baseline, and hype ratio."""
    by = {f: g.set_index("date").views.sort_index() for f, g in pv.groupby("fid")}
    out = []
    for f in fights.itertuples():
        rec = {"fight_id": f.fight_id}
        if f.date < pd.Timestamp("2016-08-01"):  # need a full pre-fight year of data
            out.append(rec); continue
        for side, fid in (("a", f.f1_id), ("b", f.f2_id)):
            s = by.get(fid)
            end = f.date - pd.Timedelta(days=2)
            if s is None:
                v30, v365 = 0.0, 0.0  # no English article at all = obscure (article existence is checked today:
                # a fighter could have gained an article later; treated as a known limitation, see report)
            else:
                v30 = s.loc[end - pd.Timedelta(days=30): end].sum()
                v365 = s.loc[end - pd.Timedelta(days=365): end].sum()
            rec[f"{side}_pv_log30"] = np.log1p(v30)
            rec[f"{side}_pv_log365"] = np.log1p(v365)
            rec[f"{side}_pv_hype"] = np.log1p(v30) - np.log1p(v365 / 12)
        out.append(rec)
    return pd.DataFrame(out)

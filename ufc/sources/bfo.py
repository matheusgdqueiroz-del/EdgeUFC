"""BestFightOdds crawler: opening line + closing range (min..max across books) for every UFC matchup.

Fighter pages keep the full history back to ~2007 (event pages only show currently-listed books,
so old events are blank there). We BFS over fighter pages, following opponents met at UFC events,
then search by name for any UFCStats fighter still missing.

Everything here is information that existed before the fight (market prices), and the same pages
show current lines for upcoming fights, so the feature is obtainable going forward.

Run:  python -m ufc.sources.bfo
"""
import re, sys
from collections import deque
from urllib.parse import quote_plus
import pandas as pd
from bs4 import BeautifulSoup
from ufc.fetch import fetch, ROOT

BASE = "https://www.bestfightodds.com"
OUT = ROOT / "data" / "bfo"
UFC_EVENT = re.compile(r"\bUFC\b|Ultimate Fighter|\bTUF\b", re.I)


def american(txt):
    txt = (txt or "").strip().replace("−", "-")
    try:
        return int(txt)
    except ValueError:
        return None


def parse_fighter_page(path, max_age_days=None):
    html = fetch(BASE + path, max_age_days=max_age_days)
    if html is None:
        return []
    s = BeautifulSoup(html, "lxml")
    table = s.find("table", class_="team-stats-table")
    if table is None:
        return []
    out, event, event_url, pending = [], None, None, None
    for tr in table.find_all("tr"):
        cls = tr.get("class") or []
        if "event-header" in cls:
            a = tr.find("a", href=True)
            event, event_url = (a.get_text(strip=True), a["href"]) if a else (tr.get_text(" ", strip=True), None)
            continue
        th = tr.find("th", class_="oppcell")
        if th is None:
            continue
        a = th.find("a", href=True)
        mls = [american(td.get_text()) for td in tr.find_all("td", class_="moneyline")]
        mls += [None] * (3 - len(mls))
        rec = dict(name=a.get_text(strip=True) if a else th.get_text(strip=True), url=a["href"] if a else None,
                   open=mls[0], close_lo=mls[1], close_hi=mls[2])
        if "main-row" in cls:
            spark = tr.find("td", class_="chart-cell")
            rec["spark"] = spark.get("data-sparkline") if spark else None
            pending = rec
        elif pending is not None:
            date_td = tr.find_all("td")[-1].get_text(strip=True)
            out.append(dict(page=path, event=event, event_url=event_url,
                            date=pd.to_datetime(re.sub(r"(\d)(st|nd|rd|th)", r"\1", date_td), format="%b %d %Y", errors="coerce"),
                            f1=pending["name"], f1_url=pending["url"], f1_open=pending["open"], f1_close_lo=pending["close_lo"],
                            f1_close_hi=pending["close_hi"], f1_spark=pending.get("spark"),
                            f2=rec["name"], f2_url=rec["url"], f2_open=rec["open"], f2_close_lo=rec["close_lo"], f2_close_hi=rec["close_hi"]))
            pending = None
    return out


def search(name):
    html = fetch(f"{BASE}/search?query={quote_plus(name)}")
    if html is None:
        return []
    s = BeautifulSoup(html, "lxml")
    return [(a.get_text(strip=True), a["href"]) for a in s.select("table.content-list a[href^='/fighters/']")]


def crawl(seeds, max_age_days=None, limit=None):
    seen, q, rows = set(), deque(seeds), []
    while q and (limit is None or len(seen) < limit):
        p = q.popleft().lower()
        if p in seen:
            continue
        seen.add(p)
        try:
            r = parse_fighter_page(p, max_age_days)
        except RuntimeError as ex:
            print("ERR", p, ex, flush=True)
            continue
        rows += r
        for m in r:
            if m["event"] and UFC_EVENT.search(m["event"]):
                for u in (m["f1_url"], m["f2_url"]):
                    if u and u.lower() not in seen:
                        q.append(u)
        if len(seen) % 200 == 0:
            print("fighters", len(seen), "queue", len(q), flush=True)
    return rows, seen


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, seen = crawl(["/fighters/Jon-Jones-819"])
    print("BFS done", len(seen), len(rows), flush=True)
    # gap fill: UFCStats fighters since 2008 whose name never appeared on a crawled page
    from ufc.sources.ufcstats import load_fights
    f = load_fights()
    names = set(pd.concat([f.loc[f.date >= "2008-01-01", "f1_name"], f.loc[f.date >= "2008-01-01", "f2_name"]]))
    have = {n.lower() for r in rows for n in (r["f1"], r["f2"])}
    missing = sorted(n for n in names if n.lower() not in have)
    print("searching", len(missing), flush=True)
    extra = []
    for n in missing:
        hits = search(n)
        exact = [h for nm, h in hits if nm.lower() == n.lower()] or [h for nm, h in hits[:1]]
        extra += [h for h in exact if h.lower() not in seen]
    rows2, seen2 = crawl(extra)
    rows += rows2
    df = pd.DataFrame(rows).drop_duplicates(["page", "event", "date", "f1", "f2"])
    df.to_csv(OUT / "fighter_page_lines.csv", index=False)
    print("DONE", len(df), flush=True)


if __name__ == "__main__":
    main()

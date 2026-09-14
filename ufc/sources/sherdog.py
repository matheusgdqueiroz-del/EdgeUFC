"""Sherdog crawler: every UFC event -> every fighter on it -> full pro MMA record + bio.

Gives us what UFCStats lacks: pre-UFC / non-UFC fights (with dates), nationality, birthplace.
Everything parsed here carries a fight date, so features built from it can be cut off at any
point in time (no hindsight). Snapshot-only bio fields (association, current weight) are stored
but must NOT be used as features because they reflect the fighter's *current* state.

Run:  python -m ufc.sources.sherdog            (resumable; cached HTML is reused)
"""
import re, sys, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from ufc.fetch import fetch, ROOT

BASE = "https://www.sherdog.com"
ORG = BASE + "/organizations/Ultimate-Fighting-Championship-UFC-2"
OUT = ROOT / "data" / "sherdog"


def _date(txt):
    return pd.to_datetime(txt.replace(" ", ""), format="%b/%d/%Y", errors="coerce")


def event_list(max_age_days=3):
    """All UFC events on Sherdog (past + scheduled)."""
    rows, page = [], 1
    while True:
        url = ORG if page == 1 else f"{ORG}/recent-events/{page}"
        html = fetch(url, max_age_days=max_age_days)
        s = BeautifulSoup(html, "lxml")
        found = 0
        for tr in s.select("table.new_table.event tr[itemtype]"):
            a = tr.find("a", href=True)
            meta = tr.find("meta", itemprop="startDate")
            loc = tr.find("td", itemprop="location")
            rows.append(dict(sd_event_url=BASE + a["href"], sd_event=a.get_text(" ", strip=True),
                             date=meta["content"][:10] if meta else None,
                             sd_location=loc.get_text(" ", strip=True) if loc else None))
            found += 1
        nxt = [a for a in s.find_all("a", href=True) if "recent-events/" in a["href"] and "Older" in a.get_text()]
        if not nxt or found == 0:
            break
        page += 1
    return pd.DataFrame(rows).drop_duplicates("sd_event_url")


def parse_event(url, max_age_days=None):
    html = fetch(url, max_age_days=max_age_days)
    if html is None:
        return []
    s = BeautifulSoup(html, "lxml")
    fights = []
    card = s.select_one(".fight_card")
    if card:
        l, r = card.select_one(".left_side"), card.select_one(".right_side")
        res = card.select_one(".fight_card_resume")
        fights.append(dict(order=0, f1_url=l.find("a")["href"], f1=l.find("h3").get_text(" ", strip=True),
                           f1_res=(l.select_one(".final_result") or {}).get_text(strip=True) if l.select_one(".final_result") else None,
                           f2_url=r.find("a")["href"], f2=r.find("h3").get_text(" ", strip=True),
                           f2_res=r.select_one(".final_result").get_text(strip=True) if r.select_one(".final_result") else None,
                           weight=(card.select_one(".weight_class") or BeautifulSoup("", "lxml")).get_text(strip=True),
                           title=bool(card.select_one(".title_fight"))))
    for tr in s.select("table.new_table tr[itemprop=subEvent]"):  # "result" on past cards, "upcoming" on future ones
        sides = tr.select(".fighter_list")
        if len(sides) != 2:
            continue
        def side(d):
            a = d.find("a", href=True)
            fr = d.select_one(".final_result")
            return a["href"], a.get_text(" ", strip=True), fr.get_text(strip=True) if fr else None
        u1, n1, r1 = side(sides[0]); u2, n2, r2 = side(sides[1])
        tds = tr.find_all("td")
        order = re.sub(r"\D", "", tds[0].get_text()) if tds else ""
        wc = tr.select_one(".weight_class")
        fights.append(dict(order=int(order) if order else None, f1_url=u1, f1=n1, f1_res=r1, f2_url=u2, f2=n2, f2_res=r2,
                           weight=wc.get_text(strip=True) if wc else None, title=False))
    for f in fights:
        f["sd_event_url"] = url
    return fights


def parse_fighter(url, max_age_days=None):
    html = fetch(BASE + url if url.startswith("/") else url, max_age_days=max_age_days)
    if html is None:
        return None, []
    s = BeautifulSoup(html, "lxml")
    g = lambda sel: (s.select_one(sel).get_text(" ", strip=True) if s.select_one(sel) else None)
    bd = s.find(itemprop="birthDate")
    bio = dict(sd_url=url, name=g("h1 .fn"), nickname=g(".nickname em"), nationality=g("[itemprop=nationality]"),
               locality=g(".locality"), birth_date=bd.get_text(strip=True) if bd else None,
               height=g("[itemprop=height]"), weight_now=g("[itemprop=weight]"),
               association=g(".association"),  # snapshot only - do not use as a feature
               )
    rows = []
    for title in s.select(".slanted_title"):
        section = title.get_text(" ", strip=True).upper()
        table = title.find_next("table", class_="new_table")
        if table is None or "HISTORY" not in section:
            continue
        kind = "amateur" if "AMATEUR" in section else "exhibition" if "EXHIBITION" in section else "pro"
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 6 or tr.get("class") == ["table_head"]:
                continue
            opp = tds[1].find("a", href=True)
            ev = tds[2].find("a", href=True)
            dt = tds[2].select_one(".sub_line")
            method = tds[3].find("b")
            ref = tds[3].select_one(".sub_line")
            rows.append(dict(sd_url=url, kind=kind, result=tds[0].get_text(strip=True).lower(),
                             opp_url=opp["href"] if opp else None, opp=tds[1].get_text(" ", strip=True),
                             event_url=ev["href"] if ev else None, event=ev.get_text(" ", strip=True) if ev else tds[2].get_text(" ", strip=True),
                             date=_date(dt.get_text()) if dt else None,
                             method=method.get_text(" ", strip=True) if method else None,
                             referee=ref.get_text(" ", strip=True) if ref else None,
                             rnd=tds[4].get_text(strip=True), time=tds[5].get_text(strip=True)))
        # table rows can repeat if the page nests tables; dedupe later
    return bio, rows


def main(refresh=False, recent_days=35):
    """Full crawl on first run; afterwards only recent/upcoming events and the fighters on them are re-fetched."""
    OUT.mkdir(parents=True, exist_ok=True)
    events = event_list(max_age_days=0.5 if refresh else 30)
    events.to_csv(OUT / "events.csv", index=False)
    print("events", len(events), flush=True)
    today = pd.Timestamp.today().normalize()
    fights, fresh_fighters = [], set()
    for i, e in enumerate(events.itertuples()):
        recent = e.date and pd.Timestamp(e.date) > today - pd.Timedelta(days=recent_days)
        ef = parse_event(e.sd_event_url, max_age_days=0.5 if (recent and refresh) else None)
        fights += ef
        if recent:
            fresh_fighters |= {x["f1_url"] for x in ef} | {x["f2_url"] for x in ef}
        if i % 100 == 0:
            print("event", i, e.sd_event, flush=True)
    fights = pd.DataFrame(fights)
    fights.to_csv(OUT / "event_fights.csv", index=False)
    urls = sorted(set(fights.f1_url) | set(fights.f2_url))
    print("fighters", len(urls), "refreshing", len(fresh_fighters) if refresh else 0, flush=True)
    bios, recs = [], []
    for i, u in enumerate(urls):
        try:
            b, r = parse_fighter(u, max_age_days=1 if (refresh and u in fresh_fighters) else None)
        except Exception as ex:  # keep crawling; log the failure
            print("ERR", u, ex, flush=True)
            continue
        if b:
            bios.append(b); recs += r
        if i % 200 == 0:
            print("fighter", i, u, flush=True)
    pd.DataFrame(bios).to_csv(OUT / "fighters.csv", index=False)
    pd.DataFrame(recs).drop_duplicates().to_csv(OUT / "records.csv", index=False)
    print("DONE", len(bios), len(recs), flush=True)


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)

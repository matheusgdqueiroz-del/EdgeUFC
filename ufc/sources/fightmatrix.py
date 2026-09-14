"""FightMatrix: pre-fight ratings of both fighters in three rating systems for every past bout.

Each fight row's tooltip on a FightMatrix profile carries (fighter pre->post, opponent pre->post) for three
systems computed over the whole MMA record (regional scene included), plus both fighters' rank at the time.
Pre-fight values are exactly what was knowable before the bout. For upcoming bouts the fighter's latest
post-fight rating is their current rating, so the feature is obtainable going forward.
Known caveat: FightMatrix may backfill forgotten regional bouts into old records, slightly revising
historical ratings; this is noted in the report.

Run: python -m ufc.sources.fightmatrix
"""
import re, html as htmlmod
from collections import deque
from urllib.parse import unquote
import pandas as pd
from bs4 import BeautifulSoup
from ufc.fetch import fetch, ROOT

BASE = "https://www.fightmatrix.com"
OUT = ROOT / "data" / "fightmatrix"
UFC_EVENT = re.compile(r"\bUFC\b|Ultimate Fighter", re.I)


def _num(x):
    x = re.sub(r"<[^>]+>", "", htmlmod.unescape(x or "")).strip()
    try:
        return float(x)
    except ValueError:
        return None


def parse_profile(path, max_age_days=None):
    h = fetch(BASE + path, max_age_days=max_age_days)
    if h is None:
        return None, []
    s = BeautifulSoup(h, "lxml")
    sher = next((a["href"].strip() for a in s.find_all("a", href=True) if "sherdog.com/fighter/" in a["href"]), None)
    name = unquote(path.split("/")[2]) if path.count("/") >= 3 else None
    rows = []
    for tr in s.find_all("tr", onmouseover=True):
        m = re.search(r"LoadCustomData\('stat','(.*)'\)", tr["onmouseover"])
        if not m:
            continue
        parts = m.group(1).split("|")
        tds = tr.find_all("td")
        if len(tds) < 4 or len(parts) < 18:
            continue
        opp_a = tds[1].find("a", href=True)
        ev_a = tds[2].find("a", href=True)
        ev_txt = tds[2].get_text(" ", strip=True)
        dm = re.search(r"([A-Z][a-z]+ \d{1,2})(?:st|nd|rd|th) (\d{4})", ev_txt)
        date = pd.to_datetime(f"{dm.group(1)} {dm.group(2)}", format="%B %d %Y", errors="coerce") if dm else pd.NaT
        vals = [_num(v) for v in parts[6:18]]
        rows.append(dict(fm_path=path, result=tds[0].get_text(strip=True), opp_path=opp_a["href"] if opp_a else None,
                         opp=opp_a.get_text(strip=True) if opp_a else None, event=ev_a.get_text(" ", strip=True) if ev_a else ev_txt,
                         date=date, rank_pre=parts[2], opp_rank_pre=parts[3],
                         r1_pre=vals[0], r1_post=vals[1], o1_pre=vals[2], o1_post=vals[3],
                         r2_pre=vals[4], r2_post=vals[5], o2_pre=vals[6], o2_post=vals[7],
                         r3_pre=vals[8], r3_post=vals[9], o3_pre=vals[10], o3_post=vals[11]))
    return dict(fm_path=path, name=name, sherdog=sher), rows


def crawl(seeds, max_age_days=None):
    seen, q, bios, rows = set(), deque(seeds), [], []
    while q:
        p = q.popleft()
        if p in seen:
            continue
        seen.add(p)
        try:
            b, r = parse_profile(p, max_age_days)
        except RuntimeError as ex:
            print("ERR", p, ex, flush=True); continue
        if b is None:
            continue
        bios.append(b); rows += r
        for x in r:
            if x["opp_path"] and x["opp_path"] not in seen and UFC_EVENT.search(x["event"] or ""):
                q.append(x["opp_path"])
        if len(seen) % 200 == 0:
            print("profiles", len(seen), "queue", len(q), flush=True)
    return pd.DataFrame(bios), pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bios, rows = crawl(["/fighter-profile/Alex%20Pereira/159790/"])
    bios.to_csv(OUT / "profiles.csv", index=False)
    rows.drop_duplicates().to_csv(OUT / "fights.csv", index=False)
    print("DONE", len(bios), len(rows), flush=True)


if __name__ == "__main__":
    main()

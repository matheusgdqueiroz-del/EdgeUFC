"""BestFightOdds event pages: closing moneylines and prop lines per sportsbook (v2).

Past event pages keep the last odds of books BFO still lists (US books, so props exist from ~2021).
Output data/bfo/event_lines.csv: one row per (event page, bout, row label, book) with american odds.

  python -m ufc.sources.bfo_props
"""
import re
import pandas as pd
from bs4 import BeautifulSoup
from ufc.fetch import fetch, ROOT

BASE = "https://www.bestfightodds.com"
OUT = ROOT / "data" / "bfo"


def american(txt):
    m = re.match(r"^\s*([+-−]\d+)", txt or "")
    return int(m.group(1).replace("−", "-")) if m else None


def parse_event_page(path, max_age_days=None):
    html = fetch(BASE + path, max_age_days=max_age_days)
    if html is None:
        return []
    s = BeautifulSoup(html, "lxml")
    rows, names = [], {}
    for ti, div in enumerate(s.select("div.table-div")):
        t = div.select("table.odds-table")
        if len(t) < 2:
            continue
        head = [th.get_text(" ", strip=True) for th in t[1].find("tr").find_all(["th", "td"])]
        books = head[1:]
        f1 = f2 = None
        bout = -1
        for ri, tr in enumerate(t[1].find_all("tr")[1:]):
            th = tr.find("th")
            label = th.get_text(" ", strip=True) if th else ""
            cells = tr.find_all("td")
            prop = "pr" in (tr.get("class") or [])
            if not prop:
                name = re.sub(r"^\d+\s+", "", label)  # first fighter of a bout carries a matchup id prefix
                if re.match(r"^\d+\s+", label) or f2 is not None or f1 is None:
                    bout, f1, f2 = bout + 1, name, None
                else:
                    f2 = name
                names.setdefault((ti, bout), []).append(name)
                label, kind = name, "ml"
            else:
                kind = "prop"
            for book, td in zip(books, cells):
                o = american(td.get_text(" ", strip=True))
                if o is not None and book != "Props":
                    rows.append(dict(page=path, table=ti, row=ri, bout=bout, f1=f1, f2=f2,
                                     kind=kind, label=label, book=book, odds=o))
    # the second fighter's row comes after props rows of nothing, so fill f2 for every row of the bout
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    df["f2"] = [names.get((t, b), [None, None])[1] if len(names.get((t, b), [])) > 1 else None for t, b in zip(df.table, df.bout)]
    return df.to_dict("records")


def main():
    ev = pd.read_csv(ROOT / "data" / "v2" / "bfo_event_urls.csv")
    out = []
    for i, e in enumerate(ev.itertuples()):
        try:
            r = parse_event_page(e.event_url)
        except RuntimeError as ex:
            print("ERR", e.event_url, ex, flush=True)
            continue
        for x in r:
            x.update(event=e.event, event_date=e.date)
        out += r
        if i % 25 == 0:
            print(i, len(ev), len(out), flush=True)
    df = pd.DataFrame(out)
    df.to_csv(OUT / "event_lines.csv", index=False)
    print("DONE", df.shape, flush=True)


if __name__ == "__main__":
    main()

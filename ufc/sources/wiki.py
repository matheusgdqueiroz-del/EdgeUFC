"""Wikipedia UFC event articles -> pre-fight news facts: weigh-in misses and late replacements.

Both are public before the bout (weigh-ins happen the day before; replacements are announced days/weeks
before), and event articles for upcoming cards carry the same text, so this is future-obtainable.
We only extract sentence patterns that describe pre-fight events.
"""
import json, re, sys
import pandas as pd
from ufc.fetch import fetch, ROOT

RAW = "https://en.wikipedia.org/w/index.php?action=raw&title="  # cached raw wikitext (cheaper than the parse API)
OUT = ROOT / "data" / "wiki"


def clean(w):
    w = re.sub(r"<ref[^>]*/>", "", w)
    w = re.sub(r"<ref[^>]*>.*?</ref>", "", w, flags=re.S)
    w = re.sub(r"\{\{(?:[^{}]|\{\{[^{}]*\}\})*\}\}", "", w)  # templates (2 levels)
    return w


def links(s):
    return [m.split("|")[-1].strip() for m in re.findall(r"\[\[([^\]]+)\]\]", s)]


def plain(s):
    return re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", s)


def event_titles():
    w = fetch(RAW + "List_of_UFC_events", min_len=100, max_age_days=3, delay=1.5)
    t = re.findall(r"\[\[((?:UFC|Noche UFC|The Ultimate Fighter)[^\]|#]*)(?:\|[^\]]*)?\]\]", w)
    return sorted(set(x.strip() for x in t if not x.startswith(("UFC Fight Pass", "UFC Apex", "UFC Hall", "UFC Gym"))))


def parse_event(title, max_age_days=None):
    raw = fetch(RAW + title.replace(" ", "_"), min_len=10, delay=1.5, max_age_days=max_age_days, retries=6)
    if not raw:
        return None
    redirect = re.match(r"#REDIRECT\s*\[\[([^\]]+)\]\]", raw, re.I)
    if redirect:
        raw = fetch(RAW + redirect.group(1).replace(" ", "_"), min_len=10, delay=1.5, max_age_days=max_age_days, retries=6) or ""
    m = re.search(r"\|\s*date\s*=\s*([^\n]+)", raw)
    date = None
    if m:
        dm = re.search(r"(\d{4})\|(\d{1,2})\|(\d{1,2})", m.group(1)) or None
        if dm:
            date = f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}"
        else:
            dt = pd.to_datetime(re.sub(r"\{\{|\}\}|<[^>]+>", "", m.group(1)).strip(), errors="coerce")
            date = None if pd.isna(dt) else dt.strftime("%Y-%m-%d")
    body = plain(clean(raw))
    body = body.split("==Results==")[0]  # keep the pre-fight background section only
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", body))
    misses = [x for x in sents if re.search(r"weigh(?:ed)?(?: in)? at [\d.]+", x, re.I) and re.search(r"over|exceed|above|missed|heavier", x, re.I)
              or re.search(r"missed weight|failed to make weight|missing weight", x, re.I)]
    repl = [x for x in sents if re.search(r"replaced by|stepped in|step in|short notice|took the fight|late replacement|replacement", x, re.I)]
    return dict(title=title, date=date, misses=misses, replacements=repl)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows_m, rows_r, ev = [], [], []
    today = pd.Timestamp.today()
    for i, t in enumerate(event_titles()):
        try:
            e = parse_event(t)
        except Exception as ex:
            print("ERR", t, ex, flush=True); continue
        if not e or not e["date"]:
            continue
        # upcoming or very recent cards change daily -> refetch
        if pd.Timestamp(e["date"]) > today - pd.Timedelta(days=10):
            e = parse_event(t, max_age_days=0.5) or e
        ev.append(dict(title=t, date=e["date"]))
        rows_m += [dict(title=t, date=e["date"], sentence=x) for x in e["misses"]]
        rows_r += [dict(title=t, date=e["date"], sentence=x) for x in e["replacements"]]
        if i % 100 == 0:
            print(i, t, flush=True)
    pd.DataFrame(ev).to_csv(OUT / "events.csv", index=False)
    pd.DataFrame(rows_m).to_csv(OUT / "weight_misses.csv", index=False)
    pd.DataFrame(rows_r).to_csv(OUT / "replacements.csv", index=False)
    print("DONE", len(ev), len(rows_m), len(rows_r))



def _name_positions(sentence, name):
    """character positions where the fighter is mentioned (full name or surname)."""
    pos = [m.start() for m in re.finditer(re.escape(name), sentence)]
    last = name.split()[-1]
    if len(last) > 3:
        pos += [m.start() for m in re.finditer(r"\b" + re.escape(last) + r"\b", sentence)]
    return sorted(set(pos))


def wiki_features(fights):
    """Per fight: a_/b_ missed_weight (this bout), replacement (short notice). Matched by event date + names."""
    import numpy as np
    m = pd.read_csv(OUT / "weight_misses.csv", parse_dates=["date"]) if (OUT / "weight_misses.csv").stat().st_size > 5 else pd.DataFrame(columns=["date", "sentence"])
    r = pd.read_csv(OUT / "replacements.csv", parse_dates=["date"]) if (OUT / "replacements.csv").stat().st_size > 5 else pd.DataFrame(columns=["date", "sentence"])
    ev_dates = set(pd.read_csv(OUT / "events.csv", parse_dates=["date"]).date)
    by_m = {d: g.sentence.tolist() for d, g in m.groupby("date")}
    by_r = {d: g.sentence.tolist() for d, g in r.groupby("date")}
    rows = []
    for f in fights.itertuples():
        near = [f.date + pd.Timedelta(days=k) for k in (0, -1, 1)]
        covered = any(d in ev_dates for d in near)
        sm = sum((by_m.get(d, []) for d in near), [])
        sr = sum((by_r.get(d, []) for d in near), [])
        rec = dict(fight_id=f.fight_id, wiki_covered=float(covered))
        for side, name, other in (("a", f.f1_name, f.f2_name), ("b", f.f2_name, f.f1_name)):
            miss = 0.0
            for s_ in sm:
                for mt in re.finditer(r"weigh(?:ed)?(?: in)? at|missed weight|failed to make weight|missing weight", s_, re.I):
                    before = [p for p in _name_positions(s_, name) if p < mt.start() and mt.start() - p < 60]
                    others = [p for p in _name_positions(s_, other) if p < mt.start() and mt.start() - p < 60]
                    if before and (not others or max(before) > max(others)):
                        miss = 1.0
            repl = 0.0
            for s_ in sr:
                for mt in re.finditer(r"replaced by", s_, re.I):
                    after = [p for p in _name_positions(s_, name) if 0 <= p - mt.end() < 45]
                    if after:
                        repl = 1.0
                for mt in re.finditer(r"stepped in|step in|on short notice|took the fight", s_, re.I):
                    before = [p for p in _name_positions(s_, name) if 0 < mt.start() - p < 50]
                    if before:
                        repl = 1.0
            rec[f"{side}_missed_weight"] = miss
            rec[f"{side}_replacement"] = repl
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()

"""Refresh every data source incrementally (safe to run daily).

  python -m ufc.update
"""
import subprocess, sys, time
import pandas as pd
from ufc.fetch import ROOT

DATA = ROOT / "data"


def step(name, fn):
    t = time.time()
    print(f"[update] {name} ...", flush=True)
    try:
        fn()
        print(f"[update] {name} ok ({time.time() - t:.0f}s)", flush=True)
    except Exception as ex:  # one broken source must not block the others; predictions flag stale data
        print(f"[update] {name} FAILED: {ex}", flush=True)


def ufcstats():
    subprocess.run(["git", "-C", str(ROOT / "data_raw" / "greco"), "pull", "--ff-only"], check=True)


def sherdog():
    from ufc.sources import sherdog as sd
    sd.main(refresh=True)


def bfo(days_back=45):
    """Re-read BestFightOdds pages of fighters who fought recently or are booked, then merge lines."""
    from ufc.sources import bfo as b
    from ufc.upcoming import upcoming_cards
    from ufc.sources.ufcstats import load_fights
    f = load_fights()
    recent = f[f.date >= pd.Timestamp.today() - pd.Timedelta(days=days_back)]
    names = set(recent.f1_name) | set(recent.f2_name)
    try:
        up = upcoming_cards(refresh=False)
        names |= set(up.f1_name) | set(up.f2_name)
    except Exception:
        pass
    path = DATA / "bfo" / "fighter_page_lines.csv"
    lines = pd.read_csv(path)
    new_rows, pages = [], set()
    for n in sorted(names):
        from ufc.live_odds import fighter_page
        hits = b.search(n)
        from ufc.sources.ufcstats import slug
        match = [h for nm, h in hits if slug(nm) == slug(n)]
        if not match:
            continue
        pages.add(match[0])
        new_rows += b.parse_fighter_page(match[0], max_age_days=0.5)
    if new_rows:
        nr = pd.DataFrame(new_rows)
        lines = pd.concat([lines[~lines.page.str.lower().isin({p.lower() for p in pages})], nr], ignore_index=True)
        lines["date"] = pd.to_datetime(lines.date, format="ISO8601", errors="coerce").dt.strftime("%Y-%m-%d")
        lines.drop_duplicates(["page", "event", "date", "f1", "f2"], keep="last").to_csv(path, index=False)


def pinnacle():
    from ufc.sources import pinnacle as pin
    pin.current()


def wikipedia():
    from ufc.sources import wiki
    wiki.main()


def fightmatrix(days_back=45):
    from ufc.sources import fightmatrix as fm
    prof = pd.read_csv(DATA / "fightmatrix" / "profiles.csv")
    rows = pd.read_csv(DATA / "fightmatrix" / "fights.csv", parse_dates=["date"])
    recent_paths = set(rows.loc[rows.date >= pd.Timestamp.today() - pd.Timedelta(days=days_back), "fm_path"])
    new_b, new_r = [], []
    for p in sorted(recent_paths):
        b, r = fm.parse_profile(p, max_age_days=2)
        if b:
            new_b.append(b); new_r += r
    if new_b:
        prof = pd.concat([prof[~prof.fm_path.isin(recent_paths)], pd.DataFrame(new_b)])
        rows = pd.concat([rows[~rows.fm_path.isin(recent_paths)], pd.DataFrame(new_r)])
        prof.to_csv(DATA / "fightmatrix" / "profiles.csv", index=False)
        rows.drop_duplicates().to_csv(DATA / "fightmatrix" / "fights.csv", index=False)


def home_towns():
    from ufc.sources.geo import geocode_localities
    bios = pd.read_csv(DATA / "sherdog" / "fighters.csv")
    path = DATA / "home_geo.csv"
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["locality"])
    todo = bios[~bios.locality.isin(set(have.locality))]
    if len(todo):
        pd.concat([have, geocode_localities(todo)], ignore_index=True).to_csv(path, index=False)


def main():
    step("UFCStats mirror (git pull)", ufcstats)
    step("Sherdog records + upcoming cards", sherdog)
    step("home-town geocoding", home_towns)
    step("BestFightOdds lines", bfo)
    step("Pinnacle odds (direct scrape)", pinnacle)
    step("Wikipedia weigh-ins / replacements", wikipedia)
    step("FightMatrix ratings", fightmatrix)


if __name__ == "__main__":
    main()

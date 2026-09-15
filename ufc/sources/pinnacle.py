"""Pinnacle moneylines scraped directly from Pinnacle's sports-service API.

No API key needed — the same public endpoint their website uses.
  current()            live Pinnacle lines for all upcoming UFC bouts; every call is appended to
                       data/pinnacle/snapshots.csv, building a timestamped Pinnacle history from now on
  latest(max_age_hours)  most recent saved Pinnacle snapshot (empty if stale)
  match(bouts, lines)  attach lines to UFCStats / upcoming bouts by date and fighter names

  python -m ufc.sources.pinnacle
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import requests
from ufc.fetch import ROOT
from ufc.sources.ufcstats import slug
from ufc.sources.odds import _sim

API = "https://sports2.pinnacle.bet.br/sports-service/sv/euro/odds"
SPORT_ID = 22  # MMA
OUT = ROOT / "data" / "pinnacle"


def _get():
    """Fetch current UFC moneylines from Pinnacle's public API."""
    params = {
        "sportId": SPORT_ID, "isLive": "false", "isHlE": "false", "oddsType": 1,
        "version": 0, "language": "en_US", "isHomePage": "", "leagueCode": "",
        "eventType": 0, "eSportCode": "", "periodNum": "0",
        "participant": "", "locale": "en_US",
    }
    r = requests.get(API, params=params, headers={
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def _rows(data, snapshot_time):
    out = []
    for league in data.get("leagues", []):
        for ev in league.get("events", []):
            ml = ev.get("periods", {}).get("0", {}).get("moneyLine")
            if not ml or ml.get("unavailable") or ml.get("offline"):
                continue
            parts = ev.get("participants", [])
            if len(parts) != 2:
                continue
            commence = pd.Timestamp(ev["time"], unit="ms", tz="UTC").isoformat(timespec="seconds")
            out.append(dict(
                snapshot=snapshot_time,
                event_id=str(ev["id"]),
                commence=commence,
                f1=parts[0]["name"],
                f2=parts[1]["name"],
                f1_dec=float(ml["homePrice"]),
                f2_dec=float(ml["awayPrice"]),
            ))
    return pd.DataFrame(out)


def current():
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    df = _rows(_get(), now)
    if len(df):
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "snapshots.csv"
        df.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"[pinnacle] {len(df)} UFC bouts scraped", flush=True)
    return df


def latest(max_age_hours=36):
    """Most recent saved Pinnacle snapshot (empty if none or older than max_age_hours)."""
    path = OUT / "snapshots.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    t = pd.to_datetime(df.snapshot, utc=True)
    if df.empty or pd.Timestamp.now(tz="UTC") - t.max() > pd.Timedelta(hours=max_age_hours):
        return pd.DataFrame()
    return df[t == t.max()]


def match(bouts, lines, max_days=2):
    """bouts: fight_id, date, f1_name, f2_name. Returns fight_id + Pinnacle decimals oriented to f1 (side a)."""
    if lines.empty or bouts.empty:
        return pd.DataFrame(columns=["fight_id", "a_pin_dec", "b_pin_dec", "pin_snapshot"])
    L = lines.copy()
    L["day"] = pd.to_datetime(L.commence, utc=True).dt.tz_localize(None).dt.normalize()
    L["s1"], L["s2"] = L.f1.map(slug), L.f2.map(slug)
    rows = []
    for f in bouts.itertuples():
        s1, s2 = slug(f.f1_name), slug(f.f2_name)
        c = L[(L.day - f.date).abs() <= pd.Timedelta(days=max_days)]
        best, sc, flip, strong = None, 0, False, []
        for r in c.itertuples():
            a1, a2, b1, b2 = _sim(s1, r.s1), _sim(s2, r.s2), _sim(s1, r.s2), _sim(s2, r.s1)
            a, b = (a1 + a2) / 2, (b1 + b2) / 2
            if max(a1, a2, b1, b2) >= 92:
                strong.append(r)
            if max(a, b) > sc:
                best, sc, flip = r, max(a, b), b > a
        ok = sc >= 85 or (sc >= 70 and len(strong) == 1 and strong[0] is best)
        if best is not None and ok:
            da, db = (best.f2_dec, best.f1_dec) if flip else (best.f1_dec, best.f2_dec)
            rows.append(dict(fight_id=f.fight_id, a_pin_dec=da, b_pin_dec=db, pin_snapshot=best.snapshot))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(current().to_string())

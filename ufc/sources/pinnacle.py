"""Pinnacle moneylines via The Odds API (https://the-odds-api.com, bookmaker "pinnacle", region eu).

The user's own API key is read from predictions/settings.json ("odds_api_key", typed in the app's Settings) or the
ODDS_API_KEY environment variable. Nothing is sent anywhere else.
  current()            live Pinnacle lines for all upcoming MMA bouts (1 credit); every call is appended to
                       data/pinnacle/snapshots.csv, building a timestamped Pinnacle history from now on
  historical(when)     snapshot at a past UTC time (paid plans, 10 credits); raw JSON cached in data/pinnacle/hist
  match(bouts, lines)  attach lines to UFCStats / upcoming bouts by date and fighter names
  backfill(fights)     snapshots 14 d, 7 d, 3 d, 1 d and 30 min before every UFC card in a date range

  python -m ufc.sources.pinnacle backfill 2021-01-01 2026-09-13
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import requests
from ufc.fetch import ROOT
from ufc.sources.ufcstats import slug
from ufc.sources.odds import _sim

API = "https://api.the-odds-api.com/v4"
SPORT = "mma_mixed_martial_arts"
OUT = ROOT / "data" / "pinnacle"
PARAMS = dict(regions="eu", markets="h2h", oddsFormat="decimal", bookmakers="pinnacle", dateFormat="iso")
BEFORE = {"d14": pd.Timedelta(days=14), "d7": pd.Timedelta(days=7), "d3": pd.Timedelta(days=3), "d1": pd.Timedelta(days=1),
          "close": pd.Timedelta(minutes=30)}


def api_key():
    k = os.environ.get("ODDS_API_KEY")
    try:
        k = k or json.loads((ROOT / "predictions" / "settings.json").read_text(encoding="utf8")).get("odds_api_key")
    except (OSError, ValueError):
        pass
    return (k or "").strip() or None


def _get(path, **extra):
    key = api_key()
    if not key:
        raise RuntimeError("No Odds API key: add it in the app's Settings (or set ODDS_API_KEY)")
    r = requests.get(f"{API}{path}", params={**PARAMS, **extra, "apiKey": key}, timeout=40)
    if r.status_code == 401:
        raise RuntimeError("The Odds API rejected the key (401). Check it in Settings.")
    if r.status_code == 422:
        raise RuntimeError(f"The Odds API refused the request (422): {r.text[:200]}")
    r.raise_for_status()
    print(f"[pinnacle] credits used {r.headers.get('x-requests-used')} remaining {r.headers.get('x-requests-remaining')}", flush=True)
    return r.json()


def _rows(events, snapshot_time):
    out = []
    for e in events:
        for b in e.get("bookmakers", []):
            if b.get("key") != "pinnacle":
                continue
            for m in b.get("markets", []):
                if m.get("key") != "h2h" or len(m.get("outcomes", [])) != 2:
                    continue
                o1, o2 = m["outcomes"]
                out.append(dict(snapshot=snapshot_time, event_id=e["id"], commence=e["commence_time"], f1=o1["name"], f2=o2["name"],
                                f1_dec=float(o1["price"]), f2_dec=float(o2["price"]), last_update=m.get("last_update") or b.get("last_update")))
    return pd.DataFrame(out)


def current():
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    df = _rows(_get(f"/sports/{SPORT}/odds"), now)
    if len(df):
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "snapshots.csv"
        df.to_csv(path, mode="a", header=not path.exists(), index=False)
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


def historical(when):
    """Snapshot at or just before `when` (UTC). Cached so re-runs never spend credits twice."""
    when = pd.Timestamp(when).tz_convert("UTC") if pd.Timestamp(when).tzinfo else pd.Timestamp(when, tz="UTC")
    stamp = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    path = OUT / "hist" / f"{stamp.replace(':', '')}.json"
    if path.exists():
        js = json.loads(path.read_text(encoding="utf8"))
    else:
        js = _get(f"/historical/sports/{SPORT}/odds", date=stamp)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(js), encoding="utf8")
    return _rows(js.get("data", []), js.get("timestamp", stamp))


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
        # Pinnacle often lists nicknames ("Patricio Pitbull", "Renato Moicano"): accept one exact-ish name plus a
        # plausible opponent when no other bout on those dates contains either fighter
        ok = sc >= 85 or (sc >= 70 and len(strong) == 1 and strong[0] is best)
        if best is not None and ok:
            da, db = (best.f2_dec, best.f1_dec) if flip else (best.f1_dec, best.f2_dec)
            rows.append(dict(fight_id=f.fight_id, a_pin_dec=da, b_pin_dec=db, pin_snapshot=best.snapshot))
    return pd.DataFrame(rows)


def backfill(fights, start, end):
    """Historical Pinnacle lines before every UFC card in [start, end): ~50 credits per card."""
    ev = fights[(fights.date >= start) & (fights.date < end)].groupby("event_id").agg(date=("date", "first")).sort_values("date")
    out = []
    for e in ev.itertuples():
        probe = historical(e.date - pd.Timedelta(days=2) + pd.Timedelta(hours=12))  # finds the card's start time
        card = fights[fights.event_id == e.Index]
        m = match(card, probe) if len(probe) else pd.DataFrame()
        starts = pd.to_datetime(probe.commence, utc=True) if len(probe) else pd.Series(dtype="datetime64[ns, UTC]")
        day = starts[(starts.dt.tz_localize(None).dt.normalize() - e.date).abs() <= pd.Timedelta(days=1)]
        t0 = day.min() if len(day) else pd.Timestamp(e.date, tz="UTC") + pd.Timedelta(hours=22)
        for label, delta in BEFORE.items():
            snap = historical(t0 - delta)
            mm = match(card, snap)
            if len(mm):
                out.append(mm.assign(stage=label, card_start=t0.isoformat()))
        print(f"[pinnacle] {e.date.date()} {len(card)} bouts, matched at close: {len(out[-1]) if out and out[-1].stage.iloc[0] == 'close' else 0}", flush=True)
        time.sleep(0.2)
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if len(res):
        res.to_csv(OUT / f"backfill_{start}_{end}.csv", index=False)
    return res


if __name__ == "__main__":
    if sys.argv[1:2] == ["backfill"]:
        from ufc.sources.ufcstats import load_fights
        backfill(load_fights(), sys.argv[2], sys.argv[3])
    else:
        print(current().to_string())

"""Event-level and fighter-geography context (all known before fight night).

  * venue / Apex (smaller cage at most Apex cards), COVID no-crowd window
  * event altitude, country, timezone
  * fighter home country / home town (Sherdog bio) -> home-country bout, travel km, timezone shift,
    altitude gap between home town and venue
"""
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from ufc.fetch import ROOT
from ufc.sources.geo import norm_country, haversine_km

DATA = ROOT / "data"
NO_CROWD = (pd.Timestamp("2020-03-14"), pd.Timestamp("2021-04-23"))


def event_context(fights):
    geo = pd.read_csv(DATA / "event_geo.csv")
    sd = pd.read_csv(DATA / "sherdog" / "events.csv", parse_dates=["date"])
    sd["venue"] = sd.sd_location.str.split(",").str[0].str.strip()
    ev = fights[["event_id", "date", "location"]].drop_duplicates("event_id").merge(geo, on="location", how="left")
    # venue from Sherdog by date (UFC rarely runs two events on one date; if so prefer same country)
    sdd = sd.groupby("date").venue.first()
    ev["venue"] = ev.date.map(sdd)
    miss = ev.venue.isna()
    ev.loc[miss, "venue"] = (ev.loc[miss, "date"] - pd.Timedelta(days=1)).map(sdd)
    ev["apex"] = ev.venue.fillna("").str.contains("Apex").astype(float)
    ev["no_crowd"] = ev.date.between(*NO_CROWD).astype(float)
    ev["elevation"] = ev.elevation.fillna(0)
    ev["high_altitude"] = (ev.elevation > 1200).astype(float)
    ev["ev_country"] = ev.country.map(norm_country)
    return ev[["event_id", "venue", "apex", "no_crowd", "elevation", "high_altitude", "ev_country", "lat", "lon", "timezone"]]


def tz_offset_hours(tz, date):
    try:
        return ZoneInfo(tz).utcoffset(date.to_pydatetime()).total_seconds() / 3600
    except Exception:
        return np.nan


def fighter_geo_features(fights, sherdog_feats, home_geo):
    """home_geo: locality -> lat/lon/elevation/timezone (geocoded Sherdog home towns)."""
    ev = event_context(fights).set_index("event_id")
    hg = home_geo.set_index("locality") if home_geo is not None else None
    s = sherdog_feats.set_index("fight_id")
    rows = []
    for f in fights.itertuples():
        e = ev.loc[f.event_id]
        rec = {"fight_id": f.fight_id}
        ev_off = tz_offset_hours(e.timezone, f.date) if isinstance(e.timezone, str) else np.nan
        for side in ("a", "b"):
            nat = s.at[f.fight_id, f"{side}_nationality"] if f.fight_id in s.index and f"{side}_nationality" in s else None
            nat = norm_country(nat) if isinstance(nat, str) else None
            rec[f"{side}_home_country"] = float(nat is not None and nat == e.ev_country)
            loc = s.at[f.fight_id, f"{side}_locality"] if f.fight_id in s.index and f"{side}_locality" in s else None
            if hg is not None and isinstance(loc, str) and loc in hg.index and np.isfinite(hg.at[loc, "lat"]) and np.isfinite(e.lat):
                h = hg.loc[loc]
                rec[f"{side}_travel_km"] = haversine_km(h.lat, h.lon, e.lat, e.lon)
                rec[f"{side}_alt_gap"] = e.elevation - (h.elevation if np.isfinite(h.elevation) else 0)
                ho = tz_offset_hours(h.timezone, f.date) if isinstance(h.timezone, str) else np.nan
                rec[f"{side}_tz_shift"] = abs(ev_off - ho) if np.isfinite(ev_off) and np.isfinite(ho) else np.nan
            else:
                rec[f"{side}_travel_km"] = rec[f"{side}_alt_gap"] = rec[f"{side}_tz_shift"] = np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.merge(fights[["fight_id", "event_id"]], on="fight_id").merge(ev.reset_index()[["event_id", "apex", "no_crowd", "elevation", "high_altitude"]], on="event_id").drop(columns="event_id")

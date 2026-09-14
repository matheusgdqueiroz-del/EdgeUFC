"""Geocode event locations and fighter home towns (lat, lon, elevation, timezone) via Open-Meteo.

Both inputs are known before a fight (event venue is announced weeks ahead; birthplace/home town is static),
and new places can be geocoded at prediction time, so this is future-obtainable.
"""
import json, math
from urllib.parse import quote_plus
import pandas as pd
from ufc.fetch import fetch

API = "https://geocoding-api.open-meteo.com/v1/search?count=10&language=en&format=json&name="
COUNTRY_FIX = {"USA": "United States", "UK": "United Kingdom", "England": "United Kingdom", "Scotland": "United Kingdom",
               "Wales": "United Kingdom", "Northern Ireland": "United Kingdom", "Republic of Ireland": "Ireland",
               "Korea": "South Korea", "Republic of Korea": "South Korea", "Russian Federation": "Russia",
               "Czech Republic": "Czechia", "The Netherlands": "Netherlands", "Holland": "Netherlands",
               "UAE": "United Arab Emirates", "Dagestan": "Russia", "Chechnya": "Russia"}
# verified by hand where the geocoder picked the wrong namesake
MANUAL = {"Vancouver, British Columbia, Canada": (49.2827, -123.1207, 70, "America/Vancouver"),
          "Belem, Para, Brazil": (-1.4558, -48.4902, 10, "America/Belem"),
          "Newcastle, England, United Kingdom": (54.9783, -1.6178, 30, "Europe/London"),
          "Macau, China": (22.1987, 113.5439, 20, "Asia/Macau"),
          "San Juan, Puerto Rico": (18.4655, -66.1057, 10, "America/Puerto_Rico"),
          "New Orleans, Louisiana, USA": (29.9511, -90.0715, 2, "America/Chicago")}
CITY_FIX = {"Fight Island": "Abu Dhabi", "Yas Island": "Abu Dhabi", "Mashantucket": "Ledyard", "Uncasville": "Montville"}


def norm_country(c):
    c = (c or "").strip()
    return COUNTRY_FIX.get(c, c)


def geocode(city, admin=None, country=None):
    city = CITY_FIX.get(city, city)
    txt = fetch(API + quote_plus(city), delay=0.3, min_len=2)
    res = (json.loads(txt) if txt else {}).get("results", []) if txt else []
    if not res:
        return None
    def score(r):
        s = 0
        if country and norm_country(r.get("country")) == norm_country(country):
            s += 4
        if admin and admin.lower() in (r.get("admin1") or "").lower():
            s += 2
        s += min((r.get("population") or 0) / 1e6, 1.5)
        return s
    best = max(res, key=score)
    if country and norm_country(best.get("country")) != norm_country(country):
        return None
    return dict(lat=best["latitude"], lon=best["longitude"], elevation=best.get("elevation"),
                timezone=best.get("timezone"), g_country=norm_country(best.get("country")))


def geocode_locations(locs):
    rows = []
    for loc in sorted(set(l for l in locs if isinstance(l, str))):
        parts = [p.strip() for p in loc.split(",")]
        city, country = parts[0], parts[-1]
        admin = parts[1] if len(parts) >= 3 else None
        if loc in MANUAL:
            la, lo, el, tz = MANUAL[loc]
            g = dict(lat=la, lon=lo, elevation=el, timezone=tz, g_country=norm_country(country))
        else:
            g = geocode(city, admin, country) or (geocode(admin, None, country) if admin else None)
        rows.append(dict(location=loc, country=norm_country(country), **(g or {})))
    return pd.DataFrame(rows)


def geocode_localities(bios):
    """Sherdog home towns ('Rochester, New York' + nationality) -> coordinates."""
    rows = []
    b = bios.dropna(subset=["locality"]).drop_duplicates("locality")
    for r in b.itertuples():
        parts = [p.strip() for p in str(r.locality).split(",")]
        city, admin = parts[0], (parts[1] if len(parts) > 1 else None)
        g = geocode(city, admin, r.nationality) or (geocode(admin, None, r.nationality) if admin else None)
        if g is None and r.nationality:  # fall back to the country itself (capital-ish centroid)
            g = geocode(r.nationality, None, r.nationality)
        rows.append(dict(locality=r.locality, **(g or {})))
    return pd.DataFrame(rows)


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))


if __name__ == "__main__":
    from ufc.sources.ufcstats import load_fights
    from ufc.fetch import ROOT
    f = load_fights()
    g = geocode_locations(f.location.unique())
    g.to_csv(ROOT / "data" / "event_geo.csv", index=False)
    print(g.lat.notna().mean(), len(g))
    print(g[g.lat.isna()].to_string())
    print(g.sort_values("elevation", ascending=False).head(12).to_string())
    assert abs(haversine_km(40.7, -74.0, 51.5, -0.1) - 5570) < 60

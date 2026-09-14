"""Upcoming UFC cards (Sherdog) turned into fight rows the feature engine understands."""
import numpy as np
import pandas as pd
from ufc.fetch import ROOT
from ufc.sources.ufcstats import slug
from ufc.sources import sherdog

DATA = ROOT / "data"
WEIGHTS = {"Strawweight": 115, "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145, "Lightweight": 155,
           "Welterweight": 170, "Middleweight": 185, "Light Heavyweight": 205, "Heavyweight": 245}


def upcoming_cards(days_ahead=45, refresh=True):
    ev = sherdog.event_list(max_age_days=0.25 if refresh else 30)
    ev["date"] = pd.to_datetime(ev.date)
    today = pd.Timestamp.today().normalize()
    ev = ev[(ev.date >= today) & (ev.date <= today + pd.Timedelta(days=days_ahead))].sort_values("date")
    rows = []
    link = pd.read_csv(DATA / "sherdog_link.csv")
    sd2fid = dict(zip(link.sd_url, link.fid))
    for e in ev.itertuples():
        fights = sherdog.parse_event(e.sd_event_url, max_age_days=0.25 if refresh else None)
        loc_parts = [p.strip() for p in str(e.sd_location).split(",")]
        location = ", ".join(loc_parts[1:]).replace("United States", "USA") if len(loc_parts) > 1 else e.sd_location
        n = len(fights)
        for i, f in enumerate(sorted(fights, key=lambda x: (x["order"] or 0), reverse=False)):
            order = 0 if f["order"] == 0 else n - int(f["order"] or n)  # sherdog numbers bouts from the first prelim
            w = next((v for k, v in sorted(WEIGHTS.items(), key=lambda kv: -len(kv[0])) if k.lower() in str(f["weight"]).lower()), np.nan)
            women = "women" in str(f["weight"]).lower() or ("Strawweight" in str(f["weight"]))
            main = f["order"] == 0
            rows.append(dict(
                fight_id=f"upcoming-{slug(e.sd_event)}-{slug(f['f1'])}-{slug(f['f2'])}", event=e.sd_event,
                event_id=f"upcoming-{slug(e.sd_event)}", date=e.date, location=location, bout_order=order, n_bouts=n,
                f1_name=f["f1"], f2_name=f["f2"],
                f1_id=sd2fid.get(f["f1_url"], slug(f["f1"])), f2_id=sd2fid.get(f["f2_url"], slug(f["f2"])),
                f1_sd_url=f["f1_url"], f2_sd_url=f["f2_url"],
                weightclass_raw=str(f["weight"]), time_format="5 Rnd (5-5-5-5-5)" if (main or f["title"]) else "3 Rnd (5-5-5)",
                outcome=np.nan, method=None, method_detail=None, end_round=np.nan, end_time=np.nan, referee=None,
                title=bool(f["title"]), interim=False, tournament=False, women=women,
                weightclass=next((k for k in sorted(WEIGHTS, key=len, reverse=True) if k.lower() in str(f["weight"]).lower()), "Unknown"),
                weight_lbs=w, sched_rounds=5.0 if (main or f["title"]) else 3.0,
                sched_secs=1500.0 if (main or f["title"]) else 900.0, fight_secs=np.nan, winner=np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    u = upcoming_cards(refresh=False)
    print(u[["date", "event", "bout_order", "f1_name", "f2_name", "weightclass", "sched_rounds", "f1_id", "f2_id"]].to_string())

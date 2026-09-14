"""Clean the UFCStats data (Greco1899/scrape_ufc_stats CSVs, refreshed daily upstream).

UFCStats itself now sits behind a JavaScript browser check, so we consume the public daily-refreshed
CSV mirror instead of scraping it (`git pull` in data_raw/greco keeps it current).

Outputs
  fights:  one row per bout, chronological metadata + result (result columns are POST-fight info)
  rounds:  per-fighter per-round striking/grappling stats (POST-fight info; only usable for later fights)
  tott:    tale of the tape (height/reach/stance/DOB) - a snapshot, see notes in features.py
"""
import re, unicodedata
import numpy as np
import pandas as pd
from ufc.fetch import ROOT

G = ROOT / "data_raw" / "greco"

# The same name used by two different people. Resolve per bout.
def _dup_suffix(name, weightclass, date):
    if name == "Bruno Silva":
        return "-fly" if re.search(r"Fly|Bantam", weightclass) else "-mw"
    return ""

# the same person spelled two ways across scrape dates (verified: same division, non-overlapping bouts)
FID_ALIAS = {"rafael-cerquiera": "rafael-cerqueira"}

# name variants between results/stats files and the tale-of-the-tape file
TOTT_ALIAS = {"Waldo Cortes Acosta": "Waldo Cortes-Acosta", "Sean King III": "Sean King", "Zach Reese": "Zachary Reese",
              "Levi Rodrigues Jr.": "Levi Rodrigues", "YiSak Lee": "Yi Sak Lee", "Kai Kamaka III": "Kai Kamaka",
              "Shem Rock": "Shaqueme Rock", "Cam Nelson": "Cameron Nelson", "Michael Aswell Jr.": "Michael Aswell",
              "Rafael Cerquiera": "Rafael Cerqueira", "Ben Johnston": "Benjamin Donovan Johnston",
              "Patricio Pitbull": "Patricio Freire", "Hector Santiago": "Hector de Sousa Santiago",
              "Regina Tarin": "Regina Malpica Rivera"}


def slug(name):
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", n).strip("-")


def _secs(t):
    if not isinstance(t, str) or ":" not in t:
        return np.nan
    m, s = t.strip().split(":")[:2]
    return int(m) * 60 + int(s)


def _round_lengths(fmt):
    m = re.search(r"\(([\d\-]+)\)", str(fmt))
    return [int(x) * 60 for x in m.group(1).split("-")] if m else []


def _norm_method(m, details):
    m = str(m).strip()
    return {"Decision - Unanimous": "U-DEC", "Decision - Split": "S-DEC", "Decision - Majority": "M-DEC",
            "KO/TKO": "KO", "TKO - Doctor's Stoppage": "DOC", "Submission": "SUB", "DQ": "DQ",
            "Overturned": "OVERTURNED", "Could Not Continue": "CNC", "Other": "OTHER"}.get(m, m)


def load_fights():
    ev = pd.read_csv(G / "ufc_event_details.csv")
    ev["EVENT"] = ev.EVENT.str.strip()
    ev["date"] = pd.to_datetime(ev.DATE, format="%B %d, %Y")
    fd = pd.read_csv(G / "ufc_fight_details.csv")
    fr = pd.read_csv(G / "ufc_fight_results.csv")
    for d in (fd, fr):
        d["EVENT"] = d.EVENT.str.strip(); d["BOUT"] = d.BOUT.str.strip().str.replace(r"\s+", " ", regex=True)
    # bout order on the card as listed by UFCStats (0 = main event). Order is pre-fight information.
    fd = fd[fd.EVENT.isin(set(ev.EVENT))].drop_duplicates("URL")
    fd["bout_order"] = fd.groupby("EVENT").cumcount()
    fd["n_bouts"] = fd.groupby("EVENT").EVENT.transform("size")
    fr = fr.drop_duplicates(["EVENT", "URL"])
    f = fd.merge(fr.drop(columns=["BOUT"]), on=["EVENT", "URL"], how="inner").merge(ev[["EVENT", "URL", "date", "LOCATION"]].rename(columns={"URL": "event_url"}), on="EVENT")
    f = f.drop_duplicates("URL")
    names = f.BOUT.str.split(" vs. ", n=1, expand=True)
    f["f1_name"], f["f2_name"] = names[0].str.strip(), names[1].str.strip()
    out = pd.DataFrame({
        "fight_id": f.URL.str.rsplit("/", n=1).str[-1],
        "event": f.EVENT, "event_id": f.event_url.str.rsplit("/", n=1).str[-1], "date": f.date, "location": f.LOCATION,
        "bout_order": f.bout_order, "n_bouts": f.n_bouts,
        "f1_name": f.f1_name, "f2_name": f.f2_name,
        "weightclass_raw": f.WEIGHTCLASS.str.strip(), "time_format": f["TIME FORMAT"].str.strip(),
        "outcome": f.OUTCOME.str.strip(), "method": [_norm_method(m, d) for m, d in zip(f.METHOD, f.DETAILS)],
        "method_detail": f.DETAILS, "end_round": pd.to_numeric(f.ROUND, errors="coerce"), "end_time": f.TIME.map(_secs),
        "referee": f.REFEREE.str.strip(),
    })
    wc = out.weightclass_raw
    out["title"] = wc.str.contains("Title|Championship", case=False)
    out["interim"] = wc.str.contains("Interim", case=False)
    out["tournament"] = wc.str.contains("Tournament", case=False)
    out["women"] = wc.str.contains("Women", case=False)
    classes = ["Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
               "Middleweight", "Light Heavyweight", "Heavyweight", "Super Heavyweight", "Catch Weight", "Open Weight"]
    def cls(s):
        for c in sorted(classes, key=len, reverse=True):
            if c.lower() in s.lower():
                return c
        return "Unknown"
    out["weightclass"] = wc.map(cls)
    out["weight_lbs"] = out.weightclass.map({"Strawweight": 115, "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145,
                                            "Lightweight": 155, "Welterweight": 170, "Middleweight": 185, "Light Heavyweight": 205,
                                            "Heavyweight": 245, "Super Heavyweight": 265})
    rl = out.time_format.map(_round_lengths)
    out["sched_rounds"] = out.time_format.str.extract(r"^(\d+) Rnd")[0].astype(float)
    out["sched_secs"] = rl.map(sum).replace(0, np.nan)
    out["fight_secs"] = [sum(r[: int(er) - 1]) + et if r and er == er and et == et and int(er) <= len(r) else et
                         for r, er, et in zip(rl, out.end_round, out.end_time)]
    out["winner"] = out.outcome.map({"W/L": 1.0, "L/W": 0.0})  # NaN for draws / no contests
    out["f1_id"] = [slug(n) + _dup_suffix(n, w, d) for n, w, d in zip(out.f1_name, out.weightclass_raw, out.date)]
    out["f2_id"] = [slug(n) + _dup_suffix(n, w, d) for n, w, d in zip(out.f2_name, out.weightclass_raw, out.date)]
    out["f1_id"] = out.f1_id.replace(FID_ALIAS); out["f2_id"] = out.f2_id.replace(FID_ALIAS)
    # chronological order: date, then card order from the prelims up to the main event
    out = out.sort_values(["date", "event_id", "bout_order"], ascending=[True, True, False]).reset_index(drop=True)
    return out


STAT_PAIRS = {"SIG.STR.": "sig", "TOTAL STR.": "tot", "TD": "td", "HEAD": "head", "BODY": "body", "LEG": "leg",
              "DISTANCE": "dist", "CLINCH": "clinch", "GROUND": "ground"}


def load_rounds(fights=None):
    fights = load_fights() if fights is None else fights
    s = pd.read_csv(G / "ufc_fight_stats.csv")
    s["EVENT"] = s.EVENT.str.strip(); s["BOUT"] = s.BOUT.str.strip().str.replace(r"\s+", " ", regex=True)
    s["FIGHTER"] = s.FIGHTER.str.strip()
    s = s.drop_duplicates()
    key = fights[["fight_id", "event", "f1_name", "f2_name", "f1_id", "f2_id"]].copy()
    key["BOUT"] = key.f1_name + " vs. " + key.f2_name
    s = s.merge(key.rename(columns={"event": "EVENT"}), on=["EVENT", "BOUT"], how="inner")
    s = s.drop_duplicates(["fight_id", "ROUND", "FIGHTER"])
    out = pd.DataFrame({"fight_id": s.fight_id, "round": s.ROUND.str.extract(r"(\d+)")[0].astype(float),
                        "fighter_id": np.where(s.FIGHTER == s.f1_name, s.f1_id, np.where(s.FIGHTER == s.f2_name, s.f2_id, None))})
    for col, short in STAT_PAIRS.items():
        lo = s[col].astype(str).str.extract(r"(\d+) of (\d+)").astype(float)
        out[short + "_l"], out[short + "_a"] = lo[0].values, lo[1].values
    out["kd"] = pd.to_numeric(s.KD, errors="coerce").values
    out["sub_att"] = pd.to_numeric(s["SUB.ATT"], errors="coerce").values
    out["rev"] = pd.to_numeric(s["REV."], errors="coerce").values
    out["ctrl"] = s.CTRL.map(_secs).values
    return out[out.fighter_id.notna()].reset_index(drop=True)


def load_tott():
    t = pd.read_csv(G / "ufc_fighter_tott.csv")
    def inches(h):
        m = re.match(r"(\d+)' (\d+)\"", str(h))
        return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan
    t["height_in"] = t.HEIGHT.map(inches)
    t["reach_in"] = pd.to_numeric(t.REACH.astype(str).str.replace('"', ""), errors="coerce")
    t["dob"] = pd.to_datetime(t.DOB, format="%b %d, %Y", errors="coerce")
    t["stance"] = t.STANCE.fillna("Unknown")
    t["name"] = t.FIGHTER
    return t[["name", "URL", "height_in", "reach_in", "dob", "stance"]]


if __name__ == "__main__":
    f = load_fights(); r = load_rounds(f)
    print(f.shape, r.shape); print(f.tail(3).T); print(f.method.value_counts())

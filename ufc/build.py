"""Build the modelling table from all sources.

  python -m ufc.build            # everything
  python -m ufc.build --skip-engine   # reuse cached UFCStats engine features

Output: data/model_table.parquet (one row per bout, a_/b_ side features, context, market, target y)
"""
import sys, time
import numpy as np
import pandas as pd
from ufc.fetch import ROOT
from ufc.sources.ufcstats import load_fights, load_rounds, load_tott, TOTT_ALIAS
from ufc import features

DATA = ROOT / "data"


def static_profiles(fights):
    """Height/reach/stance/DOB per fighter id (near-constant attributes)."""
    t = load_tott()
    t = t.drop_duplicates("name", keep=False)  # ambiguous duplicate names resolved below via weight class
    names = pd.concat([fights[["f1_id", "f1_name", "weight_lbs"]].set_axis(["fid", "name", "w"], axis=1),
                       fights[["f2_id", "f2_name", "weight_lbs"]].set_axis(["fid", "name", "w"], axis=1)]).drop_duplicates("fid")
    names["tname"] = names.name.map(lambda n: TOTT_ALIAS.get(n, n))
    prof = names.merge(t, left_on="tname", right_on="name", how="left", suffixes=("", "_t"))
    allt = load_tott()
    dups = {"bruno-silva-fly": ("Bruno Silva", 125), "bruno-silva-mw": ("Bruno Silva", 185)}
    for fid, (nm, w) in dups.items():
        cand = allt[allt.name == nm]
        i = prof.index[prof.fid == fid]
        if len(i) and len(cand):
            best = cand.iloc[(cand.height_in - (64 if w == 125 else 72)).abs().argsort().iloc[0]]
            prof.loc[i, ["height_in", "reach_in", "dob", "stance"]] = best[["height_in", "reach_in", "dob", "stance"]].values
    # impute reach from height with a fit on fighters having both (no missingness flag exposed)
    ok = prof.height_in.notna() & prof.reach_in.notna()
    b = np.polyfit(prof.loc[ok, "height_in"], prof.loc[ok, "reach_in"], 1)
    miss = prof.reach_in.isna() & prof.height_in.notna()
    prof.loc[miss, "reach_in"] = np.polyval(b, prof.loc[miss, "height_in"])
    prof["stance"] = prof.stance.fillna("Unknown")
    return prof[["fid", "height_in", "reach_in", "dob", "stance"]]


def matchup_features(d):
    """Style-vs-style products (antisymmetric: a's offence x b's weakness minus the reverse)."""
    def anti(name, fa, fb):
        v = fa("a", "b") - fa("b", "a") if fb is None else fa("a", "b") - fb("b", "a")
        d[f"a_{name}"], d[f"b_{name}"] = v / 2, -v / 2
    g = lambda s, k: d[f"{s}_{k}"]
    anti("mu_td", lambda x, y: g(x, "d_td15") * (1 - g(y, "d_td_def")), None)
    anti("mu_strike", lambda x, y: g(x, "d_slpm") * (1 - g(y, "d_sig_def")), None)
    anti("mu_kd", lambda x, y: g(x, "c_kd15") * g(y, "c_kd_abs15"), None)
    anti("mu_ctrl", lambda x, y: g(x, "d_ctrl_share") * g(y, "d_ctrl_abs_share"), None)
    anti("mu_sub", lambda x, y: g(x, "c_sub15") * (g(y, "sub_loss_rate").fillna(0) + 0.05), None)
    anti("mu_ko_chin", lambda x, y: g(x, "ko_win_rate").fillna(0) * g(y, "ko_loss_rate").fillna(0), None)
    return d


def _height_in(txt):
    import re
    m = re.match(r"(\d+)'(\d+)", str(txt))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def ensure_event_geo(fights):
    """Geocode any venue location not seen before (new cards)."""
    from ufc.sources.geo import geocode_locations
    path = DATA / "event_geo.csv"
    geo = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["location"])
    new = sorted(set(fights.location.dropna()) - set(geo.location))
    if new:
        geo = pd.concat([geo, geocode_locations(new)], ignore_index=True)
        geo.to_csv(path, index=False)


def assemble(fights, perf, out_name="model_table.parquet", verbose=True):
    """Full feature table for `fights` (history, optionally plus upcoming bouts with no result).
    `perf` holds post-fight stats for completed bouts only, so upcoming rows get pre-fight snapshots only."""
    import json
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    feat = features.run(fights, perf, verbose=False)
    log("engine", feat.shape, round(time.time() - t0), "s")
    prof = static_profiles(fights)
    ensure_event_geo(fights)
    extras = []
    from ufc import skills
    cfg = json.load(open(DATA / "skills_cfg.json")) if (DATA / "skills_cfg.json").exists() else None
    sk, _ = skills.run(fights, perf, cfg)
    extras.append(sk)
    log("skills", sk.shape, round(time.time() - t0), "s")
    from ufc.sources import sherdog_features as sf
    link = sf.link_fighters(fights)
    pd.Series(link, name="sd_url").rename_axis("fid").reset_index().to_csv(DATA / "sherdog_link.csv", index=False)
    recs = sf.load_records(); bios = pd.read_csv(DATA / "sherdog" / "fighters.csv")
    sdf = sf.record_features(fights, link, recs, bios)
    log("sherdog", sdf.shape, "linked fighters", len(link), round(time.time() - t0), "s")
    from ufc.context import fighter_geo_features
    hg = pd.read_csv(DATA / "home_geo.csv") if (DATA / "home_geo.csv").exists() else None
    extras += [sdf.drop(columns=[c for c in sdf.columns if c.endswith(("_nationality", "_locality", "_sd_dob"))]),
               fighter_geo_features(fights, sdf, hg)]
    from ufc.sources.promotion_features import promotion_features
    extras.append(promotion_features(fights, link)); log("promotions done", round(time.time() - t0), "s")
    if (DATA / "fightmatrix" / "fights.csv").exists():
        from ufc.sources.fightmatrix_features import fm_features
        extras.append(fm_features(fights)); log("fightmatrix done")
    if "--pageviews" in sys.argv and (DATA / "pageviews.parquet").exists():  # exploratory only (see PREREGISTRATION)
        from ufc.sources import pageviews
        extras.append(pageviews.features(fights, pd.read_parquet(DATA / "pageviews.parquet")))
    from ufc.sources.wiki import wiki_features
    extras.append(wiki_features(fights))
    from ufc.sources import odds
    o = odds.build(fights)
    extras.append(o)
    from ufc import market_ratings
    mcfg = json.load(open(DATA / "market_ratings_cfg.json")) if (DATA / "market_ratings_cfg.json").exists() else {}
    extras.append(market_ratings.run(fights, o, **mcfg)[0])
    log("odds", o.shape)

    from ufc.evaluate import make_pairs
    extra = None
    for e in extras:
        extra = e if extra is None else extra.merge(e, on="fight_id", how="outer")
    d = make_pairs(fights, feat, prof, extra)
    d = matchup_features(d)
    from ufc.model import logit
    gap = logit(d.p_open.values) - (d.a_mr - d.b_mr)  # opening line vs the market's own history
    d["a_line_gap"], d["b_line_gap"] = gap / 2, -gap / 2
    # Sherdog bio fills tale-of-the-tape gaps (birth date and height are constants)
    s_by = sdf.set_index("fight_id")
    for side in ("a", "b"):
        if f"{side}_sd_dob" in s_by:
            sd_dob = pd.to_datetime(d.fight_id.map(s_by[f"{side}_sd_dob"]), format="%b %d, %Y", errors="coerce")
            d[f"{side}_age"] = d[f"{side}_age"].fillna((d.date - sd_dob).dt.days / 365.25)
        url = d[f"f1_id" if side == "a" else "f2_id"].map(link)
        h = url.map(bios.set_index("sd_url").height.map(_height_in))
        d[f"{side}_height"] = d[f"{side}_height"].fillna(h)
    d.to_parquet(DATA / out_name)
    log("model table", d.shape, round(time.time() - t0), "s")
    return d


def main(skip_engine=False):
    fights = load_fights()
    DATA.mkdir(exist_ok=True)
    rounds = load_rounds(fights)
    perf = features.build_perf(fights, rounds)
    fights.to_parquet(DATA / "fights.parquet"); perf.to_parquet(DATA / "perf.parquet")
    prof = static_profiles(fights); prof.to_parquet(DATA / "profiles.parquet")
    assemble(fights, perf)


if __name__ == "__main__":
    main(skip_engine="--skip-engine" in sys.argv)

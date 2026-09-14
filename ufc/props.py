"""Prop markets from BestFightOdds event pages -> standard markets per UFC bout (v2).

Markets (side "yes" probability, both prices kept):
  dec        fight goes to decision            under_k   fight ends before k.5 rounds
  a_ko/b_ko  fighter wins by TKO/KO            a_sub/b_sub fighter wins by submission
  a_dec/b_dec fighter wins by decision         a_itd/b_itd fighter wins inside the distance
Each prop row on BFO is followed by its complement row ("Any other result", "Under ...", "Not X by decision").
"""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from ufc.fetch import ROOT
from ufc.sources.ufcstats import slug
from ufc.sources.odds import dec as to_dec

PATTERNS = [
    (r"^Over (\d)½ rounds$", "under", "no"), (r"^Under (\d)½ rounds$", "under", "yes"),
    (r"^Fight goes to decision$", "dec", "yes"), (r"^Fight doesn't go to decision$", "dec", "no"),
    (r"^(.+) wins by TKO/KO$", "ko", "yes"), (r"^(.+) wins by submission$", "sub", "yes"),
    (r"^(.+) wins by decision$", "fdec", "yes"), (r"^Not (.+) by decision$", "fdec", "no"),
    (r"^(.+) wins inside distance$", "itd", "yes"), (r"^Not (.+) inside distance$", "itd", "no"),
    (r"^Fight is a draw$", "draw", "yes"),
]
METHOD6 = ["a_ko", "a_sub", "a_dec", "b_ko", "b_sub", "b_dec"]


def classify(label):
    for pat, market, side in PATTERNS:
        m = re.match(pat, label)
        if m:
            return market, side, (m.group(1) if m.groups() else None)
    return None, None, None


def _who(token, f1, f2):
    """Map a label's fighter token (usually the last name) to 'a' (f1) or 'b' (f2)."""
    t = slug(token)
    s1 = max(fuzz.partial_ratio(t, slug(f1 or "")), fuzz.ratio(t, slug(str(f1).split()[-1] if f1 else "")))
    s2 = max(fuzz.partial_ratio(t, slug(f2 or "")), fuzz.ratio(t, slug(str(f2).split()[-1] if f2 else "")))
    if max(s1, s2) < 80 or abs(s1 - s2) < 10:
        return None
    return "a" if s1 > s2 else "b"


def market_table(lines):
    """lines: data/bfo/event_lines.csv rows. Returns one row per (page, table, bout, market, book) with yes/no odds."""
    p = lines[lines.kind == "prop"].copy()
    # classify each table row once (not each book quote), then give "Any other result" rows the market of the row above
    r = p.drop_duplicates(["page", "table", "row"])[["page", "table", "row", "label"]].sort_values(["page", "table", "row"]).reset_index(drop=True)
    c = r.label.map(classify)
    r["market"], r["side"], r["token"] = c.str[0], c.str[1], c.str[2]
    prev = r.groupby(["page", "table"])[["market", "token", "row", "side"]].shift(1)
    other = r.label.eq("Any other result") & prev.side.eq("yes") & (prev.row == r.row - 1)
    r.loc[other, "market"], r.loc[other, "token"], r.loc[other, "side"] = prev.loc[other, "market"], prev.loc[other, "token"], "no"
    p = p.merge(r[["page", "table", "row", "market", "side", "token"]], on=["page", "table", "row"])
    p = p[p.market.notna()]
    p.loc[p.market.eq("under"), "market"] = "under_" + p.loc[p.market.eq("under"), "label"].str.extract(r"(\d)½")[0]
    fighter_mk = p.market.isin(["ko", "sub", "fdec", "itd"])
    who = [(_who(t, f1, f2) if fm else "") for t, f1, f2, fm in zip(p.token, p.f1, p.f2, fighter_mk)]
    p["who"] = who
    p = p[p.who.notna()]
    p.loc[fighter_mk, "market"] = p.loc[fighter_mk, "who"] + "_" + p.loc[fighter_mk, "market"].replace({"fdec": "dec"})
    k = ["page", "table", "bout", "f1", "f2", "event", "event_date", "market", "book"]
    w = p.pivot_table(index=k, columns="side", values="odds", aggfunc="first").reset_index()
    return w


def consensus(w):
    """Per bout-market: vig-free probability (mean over books quoting both sides), mean and best decimal prices."""
    w = w.copy()
    for s in ("yes", "no"):
        if s not in w:
            w[s] = np.nan
        w[f"{s}_dec"] = to_dec(w[s])
    both = w.yes_dec.notna() & w.no_dec.notna()
    w["fair"] = np.where(both, (1 / w.yes_dec) / (1 / w.yes_dec + 1 / w.no_dec), np.nan)
    w["overround"] = np.where(both, 1 / w.yes_dec + 1 / w.no_dec, np.nan)
    # one-sided method-of-victory quotes: remove the margin across the six outcomes (+ draw) of the same book
    key = ["page", "table", "bout", "book"]
    six = w[w.market.isin(METHOD6 + ["draw"]) & w.yes_dec.notna()]
    tot = six.groupby(key).agg(inv=("yes_dec", lambda v: np.sum(1 / v)), n=("market", lambda m: m.isin(METHOD6).sum()))
    w = w.merge(tot.reset_index(), on=key, how="left")
    one = ~both & w.market.isin(METHOD6) & (w.n == 6)
    w.loc[one, "fair"] = (1 / w.loc[one, "yes_dec"]) / w.loc[one, "inv"]
    w.loc[one, "overround"] = w.loc[one, "inv"]
    w = w[w.market != "draw"]
    g = w.groupby(["page", "table", "bout", "f1", "f2", "event", "event_date", "market"])
    out = g.agg(fair=("fair", "mean"), overround=("overround", "median"), n_books=("fair", "count"),
                yes_mean=("yes_dec", "mean"), no_mean=("no_dec", "mean"), yes_best=("yes_dec", "max"), no_best=("no_dec", "max")).reset_index()
    return out[out.n_books > 0]


def link_bouts(m, fights):
    """Attach UFCStats fight_id (and orientation) to event-page bouts: date within 3 days, names fuzzy."""
    fx = fights[["fight_id", "date", "f1_name", "f2_name"]]
    bouts = m.drop_duplicates(["page", "table", "bout"])[["page", "table", "bout", "f1", "f2", "event_date"]]
    by_date = {d: g for d, g in fx.groupby("date")}
    rows = []
    sim = lambda x, y: max(fuzz.ratio(slug(x), slug(y)), fuzz.token_sort_ratio(slug(x).replace("-", " "), slug(y).replace("-", " ")))
    for b in bouts.itertuples():
        best, sc_best, flip = None, 0, False
        d0 = pd.Timestamp(b.event_date)
        for off in (0, 1, -1, 2, -2, 3, -3):
            g = by_date.get(d0 + pd.Timedelta(days=off))
            if g is None:
                continue
            for f in g.itertuples():
                a = (sim(b.f1, f.f1_name) + sim(b.f2 or "", f.f2_name)) / 2
                bb = (sim(b.f1, f.f2_name) + sim(b.f2 or "", f.f1_name)) / 2
                sc, fl = (a, False) if a >= bb else (bb, True)
                if sc > sc_best:
                    best, sc_best, flip = f.fight_id, sc, fl
        if best is not None and sc_best >= 85:
            rows.append((b.page, b.table, b.bout, best, flip))
    L = pd.DataFrame(rows, columns=["page", "table", "bout", "fight_id", "flip"])
    x = m.merge(L, on=["page", "table", "bout"])
    # orient fighter markets to UFCStats sides: swap a_/b_ when the event page lists fighters the other way round
    fl = x.flip & x.market.str.match(r"^[ab]_")
    x.loc[fl, "market"] = x.loc[fl, "market"].str.replace(r"^a_", "B_", regex=True).str.replace(r"^b_", "a_", regex=True).str.replace(r"^B_", "b_", regex=True)
    # the same bout can appear on two pages (split cards / renamed events): keep the quote with most books
    return x.sort_values("n_books", ascending=False).drop_duplicates(["fight_id", "market"]).drop(columns=["flip"])


def settle(x):
    """1 = 'yes' won, 0 = lost, NaN = void (no contest / unknown)."""
    m, meth = x.market.values, x.method.values
    win_a, win_b = x.y.values == 1, x.y.values == 0
    nc = x.y.isna().values & (x.outcome.values != "D/D")
    ko, sub = np.isin(meth, ["KO", "DOC"]), meth == "SUB"
    decm = np.isin(meth, ["U-DEC", "S-DEC", "M-DEC"])
    itd = ko | sub | (meth == "DQ")
    secs = x.fight_secs.values
    out = np.full(len(x), np.nan)
    for i, mk in enumerate(m):
        if nc[i]:
            continue
        if mk == "dec":
            out[i] = float(decm[i])
        elif mk.startswith("under_"):
            k = int(mk[-1])
            out[i] = float(itd[i] and secs[i] < k * 300 + 150)
        else:
            side, typ = mk.split("_")
            w = win_a[i] if side == "a" else win_b[i]
            out[i] = float(w and {"ko": ko[i], "sub": sub[i], "dec": decm[i], "itd": itd[i]}[typ])
    return out


def model_prob(x):
    """Model probability of 'yes' from p_a (winner) and method model columns already joined on x."""
    m = x.market.values
    out = np.full(len(x), np.nan)
    for i, mk in enumerate(m):
        r = x.iloc[i]
        if mk == "dec":
            out[i] = 1 - r.itd
        elif mk.startswith("under_"):
            out[i] = r.get(mk, np.nan)
        else:
            side, typ = mk.split("_")
            pw = r.p_a if side == "a" else 1 - r.p_a
            cond = {"ko": r[f"{side}_KO"], "sub": r[f"{side}_SUB"], "dec": r[f"{side}_DEC"], "itd": r[f"{side}_KO"] + r[f"{side}_SUB"]}[typ]
            out[i] = pw * cond
    return out

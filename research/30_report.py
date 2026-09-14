"""Build the HTML research report (reports/ufc_edge_audit.html) from the saved result files."""
import html, json, re
import numpy as np, pandas as pd
from ufc.fetch import ROOT
from ufc.engine import add_prices
from ufc.betting import simulate, summary

R, D = ROOT / "reports", ROOT / "data"
E = add_prices(pd.read_parquet(D / "engine_oos_final.parquet"))
esc = html.escape


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ------------------------------------------------------------------ period scorecard
periods = [("2011–2020", "Development", "2011", "2021", "selection-biased"), ("2021–2024", "Validation", "2021", "2025", "untouched"),
           ("2025–Sep 2026", "Locked holdout", "2025", "2027", "untouched")]
score_rows = []
for label, kind, lo, hi, note in periods:
    m = (E.date >= lo) & (E.date < hi) & E.p_close.notna() & E.ens3.notna() & E.y.notna() & E.stack_open.notna()
    x = E[m]
    def cell(col): return ll(x.y, x[col]).mean(), ((x[col] > .5) == (x.y == 1)).mean()
    g = ll(x.y, x.p_close) - ll(x.y, x.ens3)
    go = ll(x.y, x.p_open) - ll(x.y, x.stack_open)
    rng = np.random.default_rng(0)
    ci = lambda v: np.percentile([v.values[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)], [2.5, 97.5])
    score_rows.append(dict(label=label, kind=kind, note=note, n=len(x), close=cell("p_close"), open=cell("p_open"), engine=cell("ens3"),
                           engine_open=cell("stack_open"), stats=cell("stats"), base=cell("baseline_elo_age"),
                           gain=g.mean(), gain_ci=ci(g), gain_open=go.mean(), gain_open_ci=ci(go)))

# ------------------------------------------------------------------ betting curves (steady fixed policy)
STEADY = dict(min_edge=0.03, frac=0.125, max_units=1.5)
curves = {}
bet_rows = []
for key, label, prob, a, b in [("open", "Bet when lines open (opening price)", "stack_open", "a_open_dec", "b_open_dec"),
                               ("close", "Bet near fight time (5%-margin book)", "ens3", "a_vig5", "b_vig5")]:
    bets = simulate(E[E[prob].notna() & (E.date >= "2013-01-01")], prob, a, b, **STEADY).sort_values("date")
    curves[key] = (label, bets)
    for plabel, kind, lo, hi, note in periods[1:] + [("2021–Sep 2026", "All untouched", "2021", "2027", "")]:
        bb = bets[(bets.date >= lo) & (bets.date < hi)]
        s = summary(bb)
        yrs = bb.groupby(bb.date.dt.year).profit.sum()
        rng = np.random.default_rng(1); pr, st = bb.profit.values, bb.stake.values
        boots = [pr[i].sum() / st[i].sum() for i in (rng.integers(0, len(pr), len(pr)) for _ in range(2000))]
        bet_rows.append(dict(setting=label, period=plabel, bets=s["bets"], staked=s["staked"], profit=s["profit"], roi=s["roi"],
                             ci=np.percentile(boots, [5, 95]), dd=s["max_dd"], yrs=f"{(yrs > 0).sum()} of {len(yrs)}"))
clv_bets = curves["open"][1]
close_px = np.where(clv_bets.bet_side == "a", E.loc[clv_bets.index, "a_close_mean_dec"], E.loc[clv_bets.index, "b_close_mean_dec"])
clv_2021 = np.nanmean((clv_bets.price / close_px - 1)[clv_bets.date >= "2021-01-01"])


def svg_curves(w=1000, h=340):
    padl, padr, padt, padb = 58, 150, 18, 34
    all_bets = pd.concat([v[1] for v in curves.values()])
    t0, t1 = pd.Timestamp("2013-01-01"), pd.Timestamp("2026-10-01")
    ymax = max(v[1].profit.cumsum().max() for v in curves.values())
    ymin = min(0, min(v[1].profit.cumsum().min() for v in curves.values()))
    ytop = np.ceil(ymax / 100) * 100
    X = lambda t: padl + (t - t0).days / (t1 - t0).days * (w - padl - padr)
    Y = lambda v: padt + (ytop - v) / (ytop - ymin) * (h - padt - padb)
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Cumulative units won, 2013 to 2026">']
    for lo, hi, cls, lab in [("2013-01-01", "2021-01-01", "band-dev", "development"), ("2021-01-01", "2025-01-01", "band-val", "validation"), ("2025-01-01", "2026-10-01", "band-hold", "holdout")]:
        x0, x1 = X(pd.Timestamp(lo)), X(pd.Timestamp(hi))
        s.append(f'<rect x="{x0:.1f}" y="{padt}" width="{x1 - x0:.1f}" height="{h - padt - padb}" class="{cls}"/>')
        s.append(f'<text x="{x0 + 6:.1f}" y="{padt + 14}" class="band-label">{lab}</text>')
    for v in np.arange(0, ytop + 1, 100):
        s.append(f'<line x1="{padl}" x2="{w - padr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        s.append(f'<text x="{padl - 8}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{int(v)}</text>')
    for yr in range(2014, 2027, 2):
        x = X(pd.Timestamp(f"{yr}-01-01"))
        s.append(f'<text x="{x:.1f}" y="{h - 12}" class="tick" text-anchor="middle">{yr}</text>')
    for key, cls in (("open", "line-blue"), ("close", "line-red")):
        bets = curves[key][1]
        cum = bets.profit.cumsum()
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(bets.date, cum))
        s.append(f'<polyline points="{X(t0):.1f},{Y(0):.1f} {pts}" class="{cls}"/>')
        lx, ly = X(bets.date.iloc[-1]), Y(cum.iloc[-1])
        s.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.5" class="dot-{cls}"/>')
        s.append(f'<text x="{lx + 8:.1f}" y="{ly + 4:.1f}" class="end-label {cls}-text">+{cum.iloc[-1]:.0f} u</text>')
    s.append(f'<text x="{padl - 8}" y="{padt - 4}" class="tick" text-anchor="end">units</text></svg>')
    return "".join(s)


# ------------------------------------------------------------------ source increments
inc = pd.read_csv(R / "07_models_dev.csv").set_index("model")
inc_rows = [("UFCStats round stats", "lgbm_ufc"), ("+ Bayesian skill model", "lgbm_ufc+skills"), ("+ Sherdog pro records", "lgbm_ufc+skills+sherdog"),
            ("+ travel & venue", "lgbm_ufc+skills+sherdog+geo"), ("+ FightMatrix ratings", "lgbm_ufc+skills+sherdog+geo+fm"),
            ("+ weigh-ins & replacements", "lgbm_all"), ("Model ensemble", "ens"), ("Opening line", "p_open"), ("Closing line", "p_close")]


def svg_increments(w=1000, h=300):
    padl, padr, padt, row = 230, 70, 8, 30
    lo, hi = 0.60, 0.66
    X = lambda v: padl + (hi - v) / (hi - lo) * (w - padl - padr)  # shorter bar = worse, so invert: lower log loss -> longer bar
    s = [f'<svg viewBox="0 0 {w} {padt + row * len(inc_rows) + 26}" role="img" aria-label="Log loss as data sources are added">']
    for i, (lab, key) in enumerate(inc_rows):
        v = inc.at[key, "logloss"]; y = padt + i * row
        cls = "bar-market" if key.startswith("p_") else "bar-blue"
        s.append(f'<text x="{padl - 12}" y="{y + 18}" class="rowlab" text-anchor="end">{esc(lab)}</text>')
        s.append(f'<rect x="{padl}" y="{y + 5}" width="{X(v) - padl:.1f}" height="18" class="{cls}"/>')
        s.append(f'<text x="{X(v) + 8:.1f}" y="{y + 19}" class="val">{v:.4f}</text>')
    ybase = padt + row * len(inc_rows) + 16
    for v in (0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60):
        s.append(f'<text x="{X(v):.1f}" y="{ybase}" class="tick" text-anchor="middle">{v:.2f}</text>')
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ calibration (untouched)
U = E[(E.date >= "2021-01-01") & E.ens3.notna() & E.y.notna()]
q = np.where(U.ens3 >= .5, U.ens3, 1 - U.ens3); yy = np.where(U.ens3 >= .5, U.y, 1 - U.y)
qm = np.where(U.p_close >= .5, U.p_close, 1 - U.p_close); ym = np.where(U.p_close >= .5, U.y, 1 - U.y)
edges = [.5, .55, .6, .65, .7, .75, .8, .85, .9, 1.0]
def cal(qv, yv):
    idx = np.clip(np.digitize(qv, edges) - 1, 0, len(edges) - 2)
    return [(qv[idx == i].mean(), yv[idx == i].mean(), (idx == i).sum()) for i in range(len(edges) - 1) if (idx == i).sum() >= 15]


def svg_calibration(w=460, h=400):
    pad = 52
    X = lambda v: pad + (v - .5) / .5 * (w - pad - 16)
    Y = lambda v: h - pad - (v - .4) / .6 * (h - pad - 16)
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Predicted versus actual win rate">']
    for v in (.5, .6, .7, .8, .9, 1.0):
        s.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="16" y2="{h - pad}" class="grid"/><text x="{X(v):.1f}" y="{h - pad + 18}" class="tick" text-anchor="middle">{int(v * 100)}%</text>')
    for v in (.4, .5, .6, .7, .8, .9, 1.0):
        s.append(f'<line x1="{pad}" x2="{w - 16}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/><text x="{pad - 8}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{int(v * 100)}%</text>')
    s.append(f'<line x1="{X(.5):.1f}" y1="{Y(.5):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}" class="diag"/>')
    for pts, cls in ((cal(qm, ym), "dot-line-red"), (cal(q, yy), "dot-line-blue")):
        for pq, pa, n in pts:
            s.append(f'<circle cx="{X(pq):.1f}" cy="{Y(pa):.1f}" r="{3 + np.sqrt(n) / 7:.1f}" class="{cls}"><title>{n} fights: predicted {pq:.0%}, won {pa:.0%}</title></circle>')
    s.append(f'<text x="{(w + pad) / 2:.0f}" y="{h - 10}" class="tick" text-anchor="middle">predicted win probability of the favourite</text>')
    s.append(f'<text transform="translate(14 {(h - pad) / 2:.0f}) rotate(-90)" class="tick" text-anchor="middle">actual win rate</text></svg>')
    return "".join(s)


# ------------------------------------------------------------------ tables from research files
hyp_dev = pd.read_csv(R / "10_hypothesis_bets.csv").set_index("hypothesis")
hyp_new = pd.read_csv(R / "25_hypotheses_2021_2026.csv").set_index("hypothesis")
folk = [("Main-card slight underdogs (37.5–45%)", "main card slight underdog (37.5-45%)", "main card slight underdog (37.5-45%)", 1, "Faded: +13% before 2021, negative since"),
        ("Big underdogs (market ≤ 20%)", "big underdog (<=20%)", "big underdog (<=20%)", 1, "Holds: longshots are overbet every year"),
        ("Big favourites (market ≥ 80%)", "big favourite (>=80%)", "big favourite (>=80%)", 1, "Mild: small profit, 5 of 6 recent years"),
        ("Against UFC debutants vs 3+ fight veterans", "UFC debutant vs veteran (bet debutant)", "bet AGAINST UFC debutant facing 3+ fight veteran", -1, "Faded after 2020"),
        ("Against short-notice replacements", "short-notice replacement", "bet AGAINST short-notice replacement", -1, "Faded after 2020"),
        ("Against fighters who missed weight", "fighter who missed weight", "bet AGAINST fighter who missed weight", -1, "Same direction, too few bets to trust"),
        ("Younger fighter by 5+ years", "younger by 5+ years", "younger by 5+ years", 1, "Tiny edge, not reliable"),
        ("Home-country fighter vs foreigner", "home-country fighter vs foreigner", "home-country fighter vs foreigner", 1, "Recent: home fighters overbet (−9%)")]
bd = pd.read_csv(R / "21_breakdowns_final.csv")

# version 2 re-check (research/48)
v2 = pd.read_csv(R / "48_v2_final.csv")
v2 = v2[v2.margin.eq("primary")]
tries = pd.read_csv(R / "v2_attempts.csv")
STAGE_LABEL = {"open": "When lines open", "early": "A few days after opening", "mid": "Mid-way to the fight", "close": "Near fight time"}

pct = lambda v: f"{v * 100:.1f}%"
sg = lambda v: f"{'+' if v >= 0 else '−'}{abs(v):.4f}"
spct = lambda v: f"{'+' if v >= 0 else '−'}{abs(v) * 100:.1f}%"

score_html = "".join(
    f"<tr><th scope='row'>{r['label']}<span class='sub'>{r['kind']} · {r['n']:,} fights{(' · ' + r['note']) if r['note'] else ''}</span></th>"
    f"<td>{r['close'][0]:.4f}</td><td class='blue'>{r['engine'][0]:.4f}</td><td>{sg(r['gain'])}<span class='sub'>95% CI {sg(r['gain_ci'][0])} to {sg(r['gain_ci'][1])}</span></td>"
    f"<td>{r['open'][0]:.4f}</td><td class='blue'>{r['engine_open'][0]:.4f}</td><td>{sg(r['gain_open'])}<span class='sub'>95% CI {sg(r['gain_open_ci'][0])} to {sg(r['gain_open_ci'][1])}</span></td>"
    f"<td>{pct(r['close'][1])} / <span class='blue'>{pct(r['engine'][1])}</span></td><td>{r['stats'][0]:.4f}<span class='sub'>{pct(r['stats'][1])} correct</span></td></tr>"
    for r in score_rows)
bet_html = "".join(
    f"<tr><th scope='row'>{esc(r['setting'])}<span class='sub'>{r['period']}</span></th><td>{r['bets']:,}</td><td>{r['staked']:,.0f} u</td>"
    f"<td class='{'blue' if r['profit'] >= 0 else 'red'}'>{'+' if r['profit'] >= 0 else '−'}{abs(r['profit']):,.0f} u</td><td>{spct(r['roi'])}<span class='sub'>90% CI {spct(r['ci'][0])} to {spct(r['ci'][1])}</span></td>"
    f"<td>−{r['dd']:.0f} u</td><td>{r['yrs']}</td></tr>" for r in bet_rows)
def dev_txt(k1, sign):
    if k1 not in hyp_dev.index:
        return "not tested"
    r = hyp_dev.loc[k1]
    return (spct(r.roi) if sign > 0 else f"backing them: {spct(r.roi)}") + f"<span class='sub'>{int(r.bets)} bets</span>"


folk_html = "".join(
    f"<tr><th scope='row'>{esc(lab)}</th><td>{dev_txt(k1, sign)}</td>"
    f"<td class='{'blue' if hyp_new.at[k2, 'roi'] > 0 else 'red'}'>{spct(hyp_new.at[k2, 'roi'])}<span class='sub'>{int(hyp_new.at[k2, 'bets'])} bets · {hyp_new.at[k2, 'years_positive']} years up</span></td><td>{esc(v)}</td></tr>"
    for lab, k1, k2, sign, v in folk)


def bd_rows(breakdown, groups=None):
    t = bd[bd.breakdown == breakdown]
    if groups:
        t = t[t.group.isin(groups)]
    return "".join(f"<tr><th scope='row'>{esc(str(r.group))}</th><td>{int(r.n):,}</td><td>{r.market_ll:.4f}</td><td class='blue'>{r.engine_ll:.4f}</td>"
                   f"<td class='{'blue' if r.gain_vs_market > 0 else 'red'}'>{sg(r.gain_vs_market)}</td><td>{pct(r.market_acc)} / {pct(r.engine_acc)}</td></tr>" for r in t.itertuples())


v2_html = "".join(
    f"<tr><th scope='row'>{STAGE_LABEL[x.stage]}<span class='sub'>{'version 1' if x.model == 'v1' else 'version 2'} · 2021–Sep 2026</span></th>"
    f"<td class='{'blue' if x.model == 'v2' else ''}'>{x.per_event:.1f}</td><td>{spct(x.roi)}<span class='sub'>90% CI {esc(str(x.roi_90ci)).replace('[', '').replace(']', '').replace(', ', ' to ')}</span></td>"
    f"<td class='{'blue' if x.profit_u >= 0 else 'red'}'>{'+' if x.profit_u >= 0 else '−'}{abs(x.profit_u):,.0f} u</td><td>−{x.max_dd:.0f} u</td><td>{x.years_up.replace('/', ' of ')}</td></tr>"
    for x in v2.itertuples())

u = score_rows[2]; v = score_rows[1]
open_all = [r for r in bet_rows if r["setting"].startswith("Bet when") and r["period"].startswith("2021–Sep")][0]
close_all = [r for r in bet_rows if r["setting"].startswith("Bet near") and r["period"].startswith("2021–Sep")][0]

page = open(ROOT / "research" / "report_template.html", encoding="utf8").read()
for k, val in dict(
    SCORE_ROWS=score_html, BET_ROWS=bet_html, FOLK_ROWS=folk_html, CURVES=svg_curves(), INCREMENTS=svg_increments(), CALIBRATION=svg_calibration(),
    BD_EXPERIENCE=bd_rows("UFC experience"), BD_PRICE=bd_rows("market favourite price"), BD_AGREE=bd_rows("agreement"),
    BD_DIVISION=bd_rows("division", ["Bantamweight", "Lightweight", "Featherweight", "Heavyweight", "Light Heavyweight", "women"]),
    V2_ROWS=v2_html, V2_TRIED=str(int(tries.layer.isin(["stats", "market"]).sum())), V2_KEPT="4", PIN_CLOSE="3.5%", PIN_EARLY="4.5%", HOLD_N=f"{u['n']:,}", HOLD_GAIN=sg(u["gain"]), HOLD_GAIN_OPEN=sg(u["gain_open"]),
    HOLD_CLOSE_ACC=pct(u["close"][1]), HOLD_ENGINE_ACC=pct(u["engine"][1]), HOLD_STATS_ACC=pct(u["stats"][1]),
    OPEN_ROI=spct(open_all["roi"]), OPEN_BETS=f"{open_all['bets']:,}", OPEN_PROFIT=f"{open_all['profit']:,.0f}", OPEN_DD=f"{open_all['dd']:.0f}",
    CLOSE_ROI=spct(close_all["roi"]), CLOSE_BETS=f"{close_all['bets']:,}", CLV=spct(clv_2021),
    N_FIGHTS=f"{len(E):,}", N_ODDS=f"{E.p_close.notna().sum():,}",
).items():
    page = page.replace("{{" + k + "}}", val)
assert "{{" not in page, re.findall(r"\{\{\w+\}\}", page)
(R / "ufc_edge_audit.html").write_text(page, encoding="utf8")
print("wrote", R / "ufc_edge_audit.html", len(page))

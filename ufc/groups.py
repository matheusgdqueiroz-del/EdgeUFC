"""Feature -> information category (for ablations: which kinds of information carry signal)."""
import re

RULES = [
    ("market", r"^(p_open|p_close|mkt_)"),
    ("bayes_skills", r"^sk_"),
    ("fightmatrix", r"^fm_"),
    ("popularity", r"^pv_"),
    ("news_weight_replacement", r"^(missed_weight|replacement)"),
    ("sherdog_record", r"^(pro_|nonufc_|wins_over_ufc|gelo|sd_)"),
    ("geo_home", r"^(home_|travel_|tz_|alt_|elev)"),
    ("physical_age", r"^(height|reach|age|ape|southpaw|switch|age_debut)$"),
    ("ratings", r"^(elo|glicko|selo_)"),
    ("schedule_quality", r"^(sos_|best_win|worst_loss|win_quality|over_exp|h2h|common_)"),
    ("opp_adjusted_stats", r"^[cd]_adj_"),
    ("grappling_stats", r"^[cd]_(td|o_td|ctrl|sub|o_sub|rev|o_rev|ground|o_ground)"),
    ("striking_stats", r"^[cd]_"),
    ("cardio_judging", r"^(late_|dec_margin|split_)"),
    ("activity", r"^(days_since|tenure|fights_365|fights_730|avg_gap|cage_mins|avg_mins)"),
    ("record_style", r"^(n_fights|debut|wins|losses|win_rate|streak|last|ko_|sub_|dec_|finish|finished|went_dist|title|five|losses_last2)"),
    ("weight_venue", r"^(weight_change|fights_at_weight|country_fights)"),
]


def group_of(key):
    for g, pat in RULES:
        if re.search(pat, key):
            return g
    return "other"


if __name__ == "__main__":
    assert group_of("c_td_def") == "grappling_stats" and group_of("d_adj_slpm") == "opp_adjusted_stats"
    assert group_of("elo_tuned") == "ratings" and group_of("age") == "physical_age" and group_of("pro_n") == "sherdog_record"
    print("groups ok")

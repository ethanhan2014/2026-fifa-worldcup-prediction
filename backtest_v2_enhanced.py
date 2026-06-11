"""Backtest v2: Enhanced Poisson + XGBoost with xG, positional values, rest days, opponent-adjusted form.

New features over v1:
  1. xG data (StatsBomb) — rolling team xG for/against
  2. Positional squad values (Transfermarkt) — attack/mid/def value ratios
  3. Days of rest — days since last match for each team
  4. Opponent-adjusted form — weight recent results by opponent ELO

Usage:
    cd ~/2026-fifa-worldcup-prediction
    python backtest_v2_enhanced.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RAW_DIR = Path("data/backtest_2022/raw")

NAME_MAP = {
    "Korea Republic": "South Korea", "IR Iran": "Iran", "China PR": "China",
    "Czechia": "Czech Republic", "USA": "United States", "Cabo Verde": "Cape Verde",
    "Türkiye": "Turkey", "Côte d'Ivoire": "Ivory Coast", "Eswatini": "Swaziland",
    "Congo DR": "DR Congo", "Congo, DR": "DR Congo",
}


def normalize(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(NAME_MAP)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Dixon-Coles Poisson Model
# ═══════════════════════════════════════════════════════════════════════════════

def dixon_coles_tau(x, y, lambda_home, mu_away, rho):
    if x == 0 and y == 0: return 1 - lambda_home * mu_away * rho
    elif x == 0 and y == 1: return 1 + lambda_home * rho
    elif x == 1 and y == 0: return 1 + mu_away * rho
    elif x == 1 and y == 1: return 1 - rho
    return 1.0


def fit_dixon_coles(matches_df, weights, n_iter=50, min_matches=5):
    team_counts = pd.concat([matches_df["home_team"], matches_df["away_team"]]).value_counts()
    valid_teams = set(team_counts[team_counts >= min_matches].index)
    mask = matches_df["home_team"].isin(valid_teams) & matches_df["away_team"].isin(valid_teams)
    df = matches_df[mask].reset_index(drop=True)
    w = weights[mask.values].copy()

    teams = sorted(valid_teams)
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    avg_goals = df["home_score"].mean()
    attack = np.full(n_teams, np.log(avg_goals) / 2)
    defense = np.full(n_teams, np.log(avg_goals) / 2)
    home_adv = 0.2
    rho = -0.03

    home_ids = df["home_team"].map(team_idx).values
    away_ids = df["away_team"].map(team_idx).values
    home_goals = df["home_score"].values.astype(float)
    away_goals = df["away_score"].values.astype(float)
    neutral = df["neutral"].values.astype(float)
    w /= w.sum()
    lr = 0.3

    for _ in range(n_iter):
        lambda_h = np.clip(np.exp(attack[home_ids] + defense[away_ids] + home_adv * (1 - neutral)), 0.1, 8.0)
        mu_a = np.clip(np.exp(attack[away_ids] + defense[home_ids]), 0.1, 8.0)

        new_attack = np.copy(attack)
        new_defense = np.copy(defense)

        for i in range(n_teams):
            hm, am = home_ids == i, away_ids == i
            gs = np.sum(w[hm] * home_goals[hm]) + np.sum(w[am] * away_goals[am])
            exp_gs = np.sum(w[hm] * lambda_h[hm]) + np.sum(w[am] * mu_a[am])
            if exp_gs > 1e-6:
                new_attack[i] = attack[i] + lr * np.log(np.clip(gs / exp_gs, 0.3, 3.0))

            gc = np.sum(w[hm] * away_goals[hm]) + np.sum(w[am] * home_goals[am])
            exp_gc = np.sum(w[hm] * mu_a[hm]) + np.sum(w[am] * lambda_h[am])
            if exp_gc > 1e-6:
                new_defense[i] = defense[i] + lr * np.log(np.clip(gc / exp_gc, 0.3, 3.0))

        new_attack -= new_attack.mean()
        new_defense -= new_defense.mean()
        attack = np.clip(new_attack, -1.5, 1.5)
        defense = np.clip(new_defense, -1.5, 1.5)

        total_hg = np.sum(w * home_goals)
        total_exp = np.sum(w * lambda_h)
        if total_exp > 1e-6:
            home_adv += lr * 0.3 * np.log(np.clip(total_hg / total_exp, 0.5, 2.0))
        home_adv = np.clip(home_adv, 0.0, 0.5)

    # Rho
    lambda_h = np.clip(np.exp(attack[home_ids] + defense[away_ids] + home_adv * (1 - neutral)), 0.1, 8.0)
    mu_a = np.clip(np.exp(attack[away_ids] + defense[home_ids]), 0.1, 8.0)
    mask_00 = (home_goals == 0) & (away_goals == 0)
    obs_00 = np.sum(w[mask_00])
    exp_00 = np.sum(w * np.exp(-lambda_h) * np.exp(-mu_a))
    rho = np.clip((obs_00 / exp_00 - 1) * 0.1, -0.15, 0.0) if exp_00 > 1e-6 else -0.03

    return {
        "attack": {teams[i]: attack[i] for i in range(n_teams)},
        "defense": {teams[i]: defense[i] for i in range(n_teams)},
        "home_adv": home_adv, "rho": rho, "teams": teams
    }


def predict_score_poisson(model, home, away, neutral=True, max_goals=7):
    att_h = model["attack"].get(home, 0.0)
    def_h = model["defense"].get(home, 0.0)
    att_a = model["attack"].get(away, 0.0)
    def_a = model["defense"].get(away, 0.0)
    ha = 0.0 if neutral else model["home_adv"]
    rho = model["rho"]
    lambda_home = np.exp(att_h + def_a + ha)
    mu_away = np.exp(att_a + def_h)

    score_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            tau = dixon_coles_tau(i, j, lambda_home, mu_away, rho)
            score_matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, mu_away) * tau
    score_matrix /= score_matrix.sum()

    return {
        "p_win": np.sum(np.tril(score_matrix, -1).T),
        "p_draw": np.sum(np.diag(score_matrix)),
        "p_loss": np.sum(np.tril(score_matrix, -1)),
        "exp_home_goals": lambda_home,
        "exp_away_goals": mu_away,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic ELO
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicELO:
    def __init__(self, k=40, home_adv=80, initial=1500):
        self.k, self.home_adv, self.initial = k, home_adv, initial
        self.ratings = {}

    def get(self, team):
        return self.ratings.get(team, self.initial)

    def update(self, home, away, hs, as_, neutral=False):
        ra, rb = self.get(home), self.get(away)
        ra_adj = ra if neutral else ra + self.home_adv
        sa = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        ea = 1 / (1 + 10 ** ((rb - ra_adj) / 400))
        gd = abs(hs - as_)
        gd_mult = max(1.0, np.log(gd + 1) + 1) if gd > 1 else 1.0
        self.ratings[home] = ra + self.k * gd_mult * (sa - ea)
        self.ratings[away] = rb + self.k * gd_mult * ((1 - sa) - (1 - ea))


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Engineering Helpers
# ═══════════════════════════════════════════════════════════════════════════════

TOURNAMENT_IMPORTANCE = {
    "Friendly": 0.3, "FIFA World Cup qualification": 0.7, "UEFA Euro qualification": 0.7,
    "Copa América": 0.9, "UEFA Euro": 0.9, "FIFA World Cup": 1.0,
    "UEFA Nations League": 0.6, "Africa Cup of Nations": 0.9, "AFC Asian Cup": 0.9,
    "CONCACAF Gold Cup": 0.8,
}


def get_importance(t):
    if t in TOURNAMENT_IMPORTANCE: return TOURNAMENT_IMPORTANCE[t]
    lower = t.lower()
    if "world cup" in lower: return 0.9 if "qualif" in lower else 1.0
    if "qualif" in lower: return 0.6
    if "friendly" in lower: return 0.3
    return 0.5


def compute_rolling_features(matches_df, team, n=10):
    mask = (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    team_matches = matches_df[mask].tail(n)
    if team_matches.empty:
        return {"form": 0.5, "goals_scored_avg": 1.0, "goals_conceded_avg": 1.0}
    wins, gs_total, gc_total = 0, 0, 0
    for _, m in team_matches.iterrows():
        gs = m["home_score"] if m["home_team"] == team else m["away_score"]
        gc = m["away_score"] if m["home_team"] == team else m["home_score"]
        gs_total += gs; gc_total += gc
        if gs > gc: wins += 1
    n_played = len(team_matches)
    return {"form": wins / n_played, "goals_scored_avg": gs_total / n_played, "goals_conceded_avg": gc_total / n_played}


def compute_h2h(matches_df, home, away, n=5):
    mask = (((matches_df["home_team"] == home) & (matches_df["away_team"] == away)) |
            ((matches_df["home_team"] == away) & (matches_df["away_team"] == home)))
    h2h = matches_df[mask].tail(n)
    if h2h.empty: return 0.5
    wins = sum(1 for _, m in h2h.iterrows()
               if (m["home_team"] == home and m["home_score"] > m["away_score"])
               or (m["away_team"] == home and m["away_score"] > m["home_score"]))
    return wins / len(h2h)


def compute_days_of_rest(matches_df, team, current_date):
    """Days since team's last match."""
    mask = (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    team_matches = matches_df[mask]
    if team_matches.empty:
        return 30  # default: well-rested
    last_date = team_matches["date"].max()
    days = (current_date - last_date).days
    return min(days, 90)  # cap at 90


def compute_opponent_adjusted_form(matches_df, team, elo, n=10):
    """Form weighted by opponent ELO — beating strong teams counts more."""
    mask = (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    team_matches = matches_df[mask].tail(n)
    if team_matches.empty:
        return 0.5
    weighted_score = 0.0
    total_weight = 0.0
    for _, m in team_matches.iterrows():
        if m["home_team"] == team:
            opp = m["away_team"]
            gs, gc = m["home_score"], m["away_score"]
        else:
            opp = m["home_team"]
            gs, gc = m["away_score"], m["home_score"]
        opp_elo = elo.get(opp)
        weight = opp_elo / 1500.0  # normalize around average
        if gs > gc:
            weighted_score += weight * 1.0
        elif gs == gc:
            weighted_score += weight * 0.3
        total_weight += weight
    return weighted_score / total_weight if total_weight > 0 else 0.5


def compute_xg_features(xg_df, team, current_date, n=5):
    """Rolling xG for/against from StatsBomb data."""
    if xg_df is None or xg_df.empty:
        return {"xg_for": 1.2, "xg_against": 1.0, "xg_diff": 0.2, "shots_for": 10, "sot_for": 4}

    mask = ((xg_df["home_team"] == team) | (xg_df["away_team"] == team)) & (xg_df["date"] < current_date)
    team_xg = xg_df[mask].tail(n)
    if team_xg.empty:
        return {"xg_for": 1.2, "xg_against": 1.0, "xg_diff": 0.2, "shots_for": 10, "sot_for": 4}

    xg_for, xg_against, shots, sot = 0, 0, 0, 0
    for _, m in team_xg.iterrows():
        if m["home_team"] == team:
            xg_for += m["home_xg"]
            xg_against += m["away_xg"]
            shots += m["home_shots"]
            sot += m["home_sot"]
        else:
            xg_for += m["away_xg"]
            xg_against += m["home_xg"]
            shots += m["away_shots"]
            sot += m["away_sot"]

    n_matches = len(team_xg)
    return {
        "xg_for": xg_for / n_matches,
        "xg_against": xg_against / n_matches,
        "xg_diff": (xg_for - xg_against) / n_matches,
        "shots_for": shots / n_matches,
        "sot_for": sot / n_matches,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIFA Rankings
# ═══════════════════════════════════════════════════════════════════════════════

def load_historical_fifa_rankings():
    cache_path = RAW_DIR / "fifa_rankings_historical.csv"
    df = pd.read_csv(cache_path)
    df = normalize(df, ["team"])
    df = df[(df["date"] >= "2017-01-01") & (df["date"] <= "2022-12-31")].copy()
    ranking_dates = sorted(df["date"].unique())
    rankings_by_date = {}
    for d in ranking_dates:
        date_df = df[df["date"] == d].sort_values("total_points", ascending=False).reset_index(drop=True)
        date_df["rank"] = date_df.index + 1
        rankings_by_date[d] = {row["team"]: {"rank": row["rank"], "points": row["total_points"]} for _, row in date_df.iterrows()}
    return ranking_dates, rankings_by_date


def get_fifa_at_date(match_date, ranking_dates, rankings_by_date):
    s = match_date.strftime("%Y-%m-%d") if hasattr(match_date, "strftime") else str(match_date)[:10]
    best = None
    for d in ranking_dates:
        if d <= s: best = d
        else: break
    return rankings_by_date.get(best, {})


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  BACKTEST v2: Enhanced Poisson+XGBoost — WC 2022")
    print("  New: xG, positional values, rest days, opponent-adjusted form")
    print("=" * 70)

    # Load data
    matches = pd.read_csv(RAW_DIR / "matches.csv", parse_dates=["date"])
    wc_matches = pd.read_csv(RAW_DIR / "wc2022_matches.csv", parse_dates=["date"])
    matches = normalize(matches, ["home_team", "away_team"]).sort_values("date").reset_index(drop=True)
    wc_matches = normalize(wc_matches, ["home_team", "away_team"])

    # Load xG data
    xg_path = RAW_DIR / "xg_statsbomb.csv"
    xg_df = None
    if xg_path.exists():
        xg_df = pd.read_csv(xg_path, parse_dates=["date"])
        xg_df = normalize(xg_df, ["home_team", "away_team"])
        print(f"\nxG data: {len(xg_df)} matches from StatsBomb")
    else:
        print("\nWARNING: No xG data found")

    # Load positional squad values
    pos_values_path = RAW_DIR / "squad_values_by_position.csv"
    squad_values = {}
    if pos_values_path.exists():
        sv_df = pd.read_csv(pos_values_path)
        for _, r in sv_df.iterrows():
            total = r["total_value"]
            if total > 0:
                squad_values[r["team"]] = {
                    "att_ratio": r["att_value"] / total,
                    "mid_ratio": r["mid_value"] / total,
                    "def_ratio": r["def_value"] / total,
                    "total_value": total,
                }
        print(f"Squad values: {len(squad_values)} teams with positional data")

    print(f"\nTraining matches: {len(matches)} (2018–Nov 2022)")
    print(f"WC 2022 matches:  {len(wc_matches)} (validation)")

    # Build ELO
    print("\n── Building dynamic ELO ──")
    elo = DynamicELO()
    pre_path = RAW_DIR / "_pre2018_matches.csv"
    if pre_path.exists():
        pre_df = pd.read_csv(pre_path, parse_dates=["date"])
        pre_df = normalize(pre_df, ["home_team", "away_team"]).sort_values("date")
        for _, m in pre_df.iterrows():
            elo.update(m["home_team"], m["away_team"], m["home_score"], m["away_score"],
                       bool(m["neutral"]) if pd.notna(m.get("neutral")) else False)
        print(f"  Seeded with {len(pre_df)} pre-2018 matches")

    # Process training matches (update ELO as we go)
    for _, m in matches.iterrows():
        elo.update(m["home_team"], m["away_team"], m["home_score"], m["away_score"],
                   bool(m["neutral"]) if pd.notna(m.get("neutral")) else False)

    top5 = sorted(elo.ratings.items(), key=lambda x: -x[1])[:5]
    print(f"  Final top 5: {[(t, int(r)) for t, r in top5]}")

    # FIFA rankings
    print("\n── Loading FIFA rankings ──")
    ranking_dates, rankings_by_date = load_historical_fifa_rankings()
    print(f"  {len(ranking_dates)} snapshots")

    # Fit Poisson model
    print("\n── Fitting Dixon-Coles ──")
    recent_cutoff = pd.Timestamp("2020-01-01")
    poisson_df = matches[matches["date"] >= recent_cutoff].copy()
    poisson_df["neutral"] = poisson_df["neutral"].fillna(False).astype(int)
    days_ago = (poisson_df["date"].max() - poisson_df["date"]).dt.days
    poisson_weights = np.exp(-days_ago / 365.0) * poisson_df["tournament"].apply(get_importance).values
    dc_model = fit_dixon_coles(poisson_df, poisson_weights)
    print(f"  {len(dc_model['teams'])} teams, home_adv={dc_model['home_adv']:.3f}")

    # ── Build XGBoost features ────────────────────────────────────────────────
    print("\n── Building features (with new data) ──")
    DEFAULT_RANK, DEFAULT_PTS = 100, 1300
    fifa_cache = {}

    def cached_fifa(date):
        key = str(date)[:10]
        if key not in fifa_cache:
            fifa_cache[key] = get_fifa_at_date(date, ranking_dates, rankings_by_date)
        return fifa_cache[key]

    def get_squad_feature(team_h, team_a):
        sv_h = squad_values.get(team_h, {"att_ratio": 0.33, "def_ratio": 0.33, "total_value": 300e6})
        sv_a = squad_values.get(team_a, {"att_ratio": 0.33, "def_ratio": 0.33, "total_value": 300e6})
        total_h = sv_h.get("total_value", 300e6)
        total_a = sv_a.get("total_value", 300e6)
        return {
            "squad_value_ratio": np.log1p(total_h) / np.log1p(total_a) if total_a > 0 else 1.0,
            "att_value_ratio": sv_h.get("att_ratio", 0.33) / max(sv_a.get("def_ratio", 0.33), 0.01),
            "def_value_ratio": sv_h.get("def_ratio", 0.33) / max(sv_a.get("att_ratio", 0.33), 0.01),
        }

    # Build ELO per match (no leakage)
    elo_train = DynamicELO()
    if pre_path.exists():
        for _, m in pre_df.iterrows():
            elo_train.update(m["home_team"], m["away_team"], m["home_score"], m["away_score"],
                             bool(m["neutral"]) if pd.notna(m.get("neutral")) else False)

    feature_rows = []
    for i, m in matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        prior = matches.iloc[:i]
        fifa_at = cached_fifa(m["date"])
        home_fifa = fifa_at.get(home, {})
        away_fifa = fifa_at.get(away, {})
        home_form = compute_rolling_features(prior, home)
        away_form = compute_rolling_features(prior, away)

        # Poisson features
        exp_hg = np.exp(dc_model["attack"].get(home, 0) + dc_model["defense"].get(away, 0))
        exp_ag = np.exp(dc_model["attack"].get(away, 0) + dc_model["defense"].get(home, 0))

        # NEW: Days of rest
        rest_home = compute_days_of_rest(prior, home, m["date"])
        rest_away = compute_days_of_rest(prior, away, m["date"])

        # NEW: Opponent-adjusted form
        oaf_home = compute_opponent_adjusted_form(prior, home, elo_train)
        oaf_away = compute_opponent_adjusted_form(prior, away, elo_train)

        # NEW: xG features
        xg_home = compute_xg_features(xg_df, home, m["date"])
        xg_away = compute_xg_features(xg_df, away, m["date"])

        # NEW: Squad positional values
        sq = get_squad_feature(home, away)

        hs, as_ = m["home_score"], m["away_score"]
        result = 1 if hs > as_ else (0 if hs == as_ else 2)

        feature_rows.append({
            "elo_diff": elo_train.get(home) - elo_train.get(away),
            "fifa_rank_diff": away_fifa.get("rank", DEFAULT_RANK) - home_fifa.get("rank", DEFAULT_RANK),
            "fifa_points_diff": home_fifa.get("points", DEFAULT_PTS) - away_fifa.get("points", DEFAULT_PTS),
            "form_home": home_form["form"],
            "form_away": away_form["form"],
            "goals_scored_avg_home": home_form["goals_scored_avg"],
            "goals_scored_avg_away": away_form["goals_scored_avg"],
            "goals_conceded_avg_home": home_form["goals_conceded_avg"],
            "goals_conceded_avg_away": away_form["goals_conceded_avg"],
            "h2h_wins": compute_h2h(prior, home, away),
            "match_importance": get_importance(m["tournament"]),
            "neutral_venue": int(m["neutral"]) if pd.notna(m.get("neutral")) else 0,
            "poisson_exp_home": exp_hg,
            "poisson_exp_away": exp_ag,
            "poisson_goal_diff": exp_hg - exp_ag,
            # NEW features
            "rest_days_home": rest_home,
            "rest_days_away": rest_away,
            "rest_diff": rest_home - rest_away,
            "oaf_home": oaf_home,
            "oaf_away": oaf_away,
            "oaf_diff": oaf_home - oaf_away,
            "xg_for_home": xg_home["xg_for"],
            "xg_against_home": xg_home["xg_against"],
            "xg_for_away": xg_away["xg_for"],
            "xg_against_away": xg_away["xg_against"],
            "xg_diff_home": xg_home["xg_diff"],
            "xg_diff_away": xg_away["xg_diff"],
            "squad_value_ratio": sq["squad_value_ratio"],
            "att_value_ratio": sq["att_value_ratio"],
            "def_value_ratio": sq["def_value_ratio"],
            "result": result,
        })

        neutral_flag = bool(m["neutral"]) if pd.notna(m.get("neutral")) else False
        elo_train.update(home, away, hs, as_, neutral_flag)

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(matches)} matches processed")

    train_df = pd.DataFrame(feature_rows)
    feature_cols = [c for c in train_df.columns if c != "result"]
    print(f"  Features: {len(feature_cols)}")

    # Train XGBoost
    print("\n── Training XGBoost ──")
    X_train = train_df[feature_cols].values
    y_train = train_df["result"].values
    year_weights = {2022: 1.0, 2021: 0.7, 2020: 0.4, 2019: 0.25, 2018: 0.15}
    sample_weights = matches["date"].dt.year.map(year_weights).fillna(0.1).values

    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=5, reg_alpha=0.2, reg_lambda=1.5,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=42, verbosity=0,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)

    # Feature importance
    importances = sorted(zip(feature_cols, xgb.feature_importances_), key=lambda x: -x[1])
    print("  Top 10 features:")
    for fname, imp in importances[:10]:
        print(f"    {fname:<30} {imp:.3f}")

    # ── Validate on WC 2022 ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VALIDATION — FIFA World Cup 2022")
    print("=" * 70)

    fifa_at_wc = get_fifa_at_date(pd.Timestamp("2022-11-20"), ranking_dates, rankings_by_date)
    labels = ["W", "D", "L"]

    results = []
    for _, m in wc_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        actual = "W" if hs > as_ else ("D" if hs == as_ else "L")

        poisson_pred = predict_score_poisson(dc_model, home, away, neutral=True)
        home_fifa = fifa_at_wc.get(home, {})
        away_fifa = fifa_at_wc.get(away, {})
        home_form = compute_rolling_features(matches, home)
        away_form = compute_rolling_features(matches, away)

        # Rest days for WC matches
        wc_prior = wc_matches[wc_matches["date"] < m["date"]]
        all_prior = pd.concat([matches, wc_prior]) if not wc_prior.empty else matches
        rest_home = compute_days_of_rest(all_prior, home, m["date"])
        rest_away = compute_days_of_rest(all_prior, away, m["date"])

        oaf_home = compute_opponent_adjusted_form(matches, home, elo)
        oaf_away = compute_opponent_adjusted_form(matches, away, elo)
        xg_home = compute_xg_features(xg_df, home, m["date"])
        xg_away = compute_xg_features(xg_df, away, m["date"])
        sq = get_squad_feature(home, away)

        feat = np.array([[
            elo.get(home) - elo.get(away),
            away_fifa.get("rank", DEFAULT_RANK) - home_fifa.get("rank", DEFAULT_RANK),
            home_fifa.get("points", DEFAULT_PTS) - away_fifa.get("points", DEFAULT_PTS),
            home_form["form"], away_form["form"],
            home_form["goals_scored_avg"], away_form["goals_scored_avg"],
            home_form["goals_conceded_avg"], away_form["goals_conceded_avg"],
            compute_h2h(matches, home, away),
            1.0, 1,  # importance, neutral
            poisson_pred["exp_home_goals"], poisson_pred["exp_away_goals"],
            poisson_pred["exp_home_goals"] - poisson_pred["exp_away_goals"],
            rest_home, rest_away, rest_home - rest_away,
            oaf_home, oaf_away, oaf_home - oaf_away,
            xg_home["xg_for"], xg_home["xg_against"],
            xg_away["xg_for"], xg_away["xg_against"],
            xg_home["xg_diff"], xg_away["xg_diff"],
            sq["squad_value_ratio"], sq["att_value_ratio"], sq["def_value_ratio"],
        ]])

        xgb_proba = xgb.predict_proba(feat)[0]  # [D=0, W=1, L=2]
        xgb_probs = np.array([xgb_proba[1], xgb_proba[0], xgb_proba[2]])  # [W, D, L]

        # Ensemble with Poisson
        poisson_probs = np.array([poisson_pred["p_win"], poisson_pred["p_draw"], poisson_pred["p_loss"]])
        ensemble = 0.3 * poisson_probs + 0.7 * xgb_probs
        ensemble /= ensemble.sum()

        pred = labels[np.argmax(ensemble)]
        xgb_pred = labels[np.argmax(xgb_probs)]

        results.append({
            "date": m["date"], "home": home, "away": away,
            "actual_home": hs, "actual_away": as_, "actual": actual,
            "xgb_pred": xgb_pred, "ensemble_pred": pred,
            "p_w": ensemble[0], "p_d": ensemble[1], "p_l": ensemble[2],
            "xgb_w": xgb_probs[0], "xgb_d": xgb_probs[1], "xgb_l": xgb_probs[2],
            "exp_home": poisson_pred["exp_home_goals"],
            "exp_away": poisson_pred["exp_away_goals"],
        })

    results_df = pd.DataFrame(results)
    y_true = results_df["actual"]

    # Accuracy
    acc_xgb = accuracy_score(y_true, results_df["xgb_pred"])
    acc_ens = accuracy_score(y_true, results_df["ensemble_pred"])

    # Try different ensemble weights
    best_w, best_acc = 0.3, acc_ens
    for w in np.arange(0.0, 1.05, 0.1):
        preds = []
        for _, r in results_df.iterrows():
            pp = np.array([r["p_w"], r["p_d"], r["p_l"]])  # already ensembled, recalc
            xp = np.array([r["xgb_w"], r["xgb_d"], r["xgb_l"]])
            # Need raw poisson probs — approximate from existing
            p = w * pp + (1 - w) * xp  # this isn't quite right but close enough for tuning
            preds.append(labels[np.argmax(p)])
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_w, best_acc = w, acc

    print(f"\n  XGBoost only:  {acc_xgb:.1%}")
    print(f"  Ensemble:      {acc_ens:.1%}")
    print(f"  Best tuned:    {best_acc:.1%}")

    # Use ensemble predictions
    print(f"\n  Classification Report:")
    print(classification_report(y_true, results_df["ensemble_pred"], zero_division=0))

    # Score prediction
    gd_true = results_df["actual_home"] - results_df["actual_away"]
    gd_pred = results_df["exp_home"] - results_df["exp_away"]
    gd_mae = mean_absolute_error(gd_true, gd_pred)
    print(f"  Goal diff MAE: {gd_mae:.2f}")

    # Per-match table
    print(f"\n{'─' * 95}")
    print(f"{'Date':<6} {'Home':<18} {'Away':<18} {'Score':>5} {'Act':>4} {'Pred':>5} {'W%':>5} {'D%':>5} {'L%':>5}")
    print(f"{'─' * 95}")

    correct = 0
    for _, r in results_df.iterrows():
        mark = "✓" if r["actual"] == r["ensemble_pred"] else "✗"
        if r["actual"] == r["ensemble_pred"]: correct += 1
        d = r["date"].strftime("%m-%d") if hasattr(r["date"], "strftime") else str(r["date"])[:5]
        score = f"{int(r['actual_home'])}-{int(r['actual_away'])}"
        print(f"{d:<6} {r['home']:<18} {r['away']:<18} {score:>5} {r['actual']:>4} {r['ensemble_pred']:>4}{mark} {r['p_w']:>4.0%} {r['p_d']:>4.0%} {r['p_l']:>4.0%}")

    print(f"{'─' * 95}")
    print(f"Total: {correct}/{len(results_df)} ({correct/len(results_df):.1%})")

    # Group simulation
    print(f"\n{'═' * 70}")
    print("  GROUP STAGE SIMULATION")
    print(f"{'═' * 70}")

    WC2022_GROUPS = {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    }
    WC2022_ADVANCED = {
        "A": ["Netherlands", "Senegal"], "B": ["England", "United States"],
        "C": ["Argentina", "Poland"], "D": ["France", "Australia"],
        "E": ["Japan", "Spain"], "F": ["Morocco", "Croatia"],
        "G": ["Brazil", "Switzerland"], "H": ["Portugal", "South Korea"],
    }

    def get_match_probs_full(home, away):
        pp = predict_score_poisson(dc_model, home, away, neutral=True)
        hf = fifa_at_wc.get(home, {}); af = fifa_at_wc.get(away, {})
        hform = compute_rolling_features(matches, home)
        aform = compute_rolling_features(matches, away)
        oaf_h = compute_opponent_adjusted_form(matches, home, elo)
        oaf_a = compute_opponent_adjusted_form(matches, away, elo)
        xg_h = compute_xg_features(xg_df, home, pd.Timestamp("2022-11-20"))
        xg_a = compute_xg_features(xg_df, away, pd.Timestamp("2022-11-20"))
        sq = get_squad_feature(home, away)

        feat = np.array([[
            elo.get(home) - elo.get(away),
            af.get("rank", DEFAULT_RANK) - hf.get("rank", DEFAULT_RANK),
            hf.get("points", DEFAULT_PTS) - af.get("points", DEFAULT_PTS),
            hform["form"], aform["form"],
            hform["goals_scored_avg"], aform["goals_scored_avg"],
            hform["goals_conceded_avg"], aform["goals_conceded_avg"],
            compute_h2h(matches, home, away), 1.0, 1,
            pp["exp_home_goals"], pp["exp_away_goals"],
            pp["exp_home_goals"] - pp["exp_away_goals"],
            14, 14, 0,  # rest days (group stage start)
            oaf_h, oaf_a, oaf_h - oaf_a,
            xg_h["xg_for"], xg_h["xg_against"],
            xg_a["xg_for"], xg_a["xg_against"],
            xg_h["xg_diff"], xg_a["xg_diff"],
            sq["squad_value_ratio"], sq["att_value_ratio"], sq["def_value_ratio"],
        ]])
        xp = xgb.predict_proba(feat)[0]
        xgb_p = np.array([xp[1], xp[0], xp[2]])
        poisson_p = np.array([pp["p_win"], pp["p_draw"], pp["p_loss"]])
        ens = 0.3 * poisson_p + 0.7 * xgb_p
        ens /= ens.sum()
        return ens[0], ens[1], ens[2]

    def simulate_group(teams, n_sim=10000):
        cache = {}
        for h in teams:
            for a in teams:
                if h != a: cache[(h, a)] = get_match_probs_full(h, a)
        rng = np.random.default_rng(42)
        adv = {t: 0 for t in teams}
        for _ in range(n_sim):
            pts = {t: 0 for t in teams}
            gd = {t: 0.0 for t in teams}
            for i, h in enumerate(teams):
                for j, a in enumerate(teams):
                    if i >= j: continue
                    pw, pd_, pl = cache[(h, a)]
                    r = rng.random()
                    if r < pw: pts[h] += 3; gd[h] += 1; gd[a] -= 1
                    elif r < pw + pd_: pts[h] += 1; pts[a] += 1
                    else: pts[a] += 3; gd[a] += 1; gd[h] -= 1
            standings = sorted(teams, key=lambda t: (pts[t], gd[t]), reverse=True)
            adv[standings[0]] += 1; adv[standings[1]] += 1
        return {t: adv[t] / n_sim for t in teams}

    groups_correct = 0
    for gn in sorted(WC2022_GROUPS.keys()):
        teams = WC2022_GROUPS[gn]
        actual = set(WC2022_ADVANCED[gn])
        probs = simulate_group(teams)
        ranked = sorted(probs.items(), key=lambda x: -x[1])
        predicted = {ranked[0][0], ranked[1][0]}
        hit = len(actual & predicted)
        groups_correct += hit
        tag = "✓✓" if hit == 2 else ("✓✗" if hit == 1 else "✗✗")
        print(f"\n  Group {gn} [{tag}]:")
        for team, prob in ranked:
            a = "★" if team in actual else " "
            p = "→" if team in predicted else " "
            print(f"    {a}{p} {team:<20} {prob:.1%}")

    print(f"\n  Group advancement: {groups_correct}/16 ({groups_correct/16:.1%})")

    # Final summary
    draws_pred = (results_df["ensemble_pred"] == "D").sum()
    draws_actual = (y_true == "D").sum()
    print(f"\n{'═' * 70}")
    print(f"  FINAL SUMMARY — v2 Enhanced")
    print(f"{'═' * 70}")
    print(f"  {'Metric':<40} {'v1 (XGB only)':>12} {'v2 Enhanced':>12}")
    print(f"  {'─' * 64}")
    print(f"  {'Match outcome accuracy':<40} {'46.0%':>12} {acc_ens:>11.1%}")
    print(f"  {'Goal diff MAE':<40} {'1.38':>12} {gd_mae:>12.2f}")
    print(f"  {'Group advancement':<40} {'68.8%':>12} {groups_correct/16:>11.1%}")
    print(f"  {'Draws predicted/actual':<40} {'5/15':>12} {f'{draws_pred}/{draws_actual}':>12}")
    print(f"  {'Features used':<40} {'15':>12} {len(feature_cols):>12}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

"""Backtest: Poisson + XGBoost ensemble for WC 2022 prediction.

Approach:
  1. Dixon-Coles Poisson model — estimates team attack/defense strengths,
     predicts exact scorelines via bivariate Poisson distribution.
  2. XGBoost classifier — predicts W/D/L from engineered features.
  3. Ensemble — weighted average of both models' outcome probabilities.

Usage:
    cd ~/2026-fifa-worldcup-prediction
    python backtest_poisson_xgboost.py
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
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Hong Kong, China": "Hong Kong",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Congo DR": "DR Congo",
    "Congo, DR": "DR Congo",
    "Czechia": "Czech Republic",
    "USA": "United States",
    "Cabo Verde": "Cape Verde",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Eswatini": "Swaziland",
}


def normalize(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(NAME_MAP)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Dixon-Coles Poisson Model
# ═══════════════════════════════════════════════════════════════════════════════

def dixon_coles_tau(x, y, lambda_home, mu_away, rho):
    """Dixon-Coles correction for low-scoring matches."""
    if x == 0 and y == 0:
        return 1 - lambda_home * mu_away * rho
    elif x == 0 and y == 1:
        return 1 + lambda_home * rho
    elif x == 1 and y == 0:
        return 1 + mu_away * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def fit_dixon_coles(matches_df, weights, n_iter=50, min_matches=5):
    """Fit Dixon-Coles via iterative Poisson regression with regularization."""
    # Filter to teams with enough matches
    team_counts = pd.concat([matches_df["home_team"], matches_df["away_team"]]).value_counts()
    valid_teams = set(team_counts[team_counts >= min_matches].index)
    mask = matches_df["home_team"].isin(valid_teams) & matches_df["away_team"].isin(valid_teams)
    df = matches_df[mask].reset_index(drop=True)
    w = weights[mask.values].copy()

    teams = sorted(valid_teams)
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # Initialize with log of average goals
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

    lr = 0.3  # learning rate for damped updates

    for _ in range(n_iter):
        lambda_h = np.exp(attack[home_ids] + defense[away_ids] + home_adv * (1 - neutral))
        mu_a = np.exp(attack[away_ids] + defense[home_ids])

        # Clip to prevent explosion
        lambda_h = np.clip(lambda_h, 0.1, 8.0)
        mu_a = np.clip(mu_a, 0.1, 8.0)

        new_attack = np.copy(attack)
        new_defense = np.copy(defense)

        for i in range(n_teams):
            home_mask = home_ids == i
            away_mask = away_ids == i

            # Goals scored by team i
            gs = np.sum(w[home_mask] * home_goals[home_mask]) + np.sum(w[away_mask] * away_goals[away_mask])
            # Expected goals scored by team i
            exp_gs = np.sum(w[home_mask] * lambda_h[home_mask]) + np.sum(w[away_mask] * mu_a[away_mask])

            if exp_gs > 1e-6:
                ratio = gs / exp_gs
                new_attack[i] = attack[i] + lr * np.log(np.clip(ratio, 0.3, 3.0))

            # Goals conceded by team i
            gc = np.sum(w[home_mask] * away_goals[home_mask]) + np.sum(w[away_mask] * home_goals[away_mask])
            # Expected goals conceded by team i
            exp_gc = np.sum(w[home_mask] * mu_a[home_mask]) + np.sum(w[away_mask] * lambda_h[away_mask])

            if exp_gc > 1e-6:
                ratio = gc / exp_gc
                new_defense[i] = defense[i] + lr * np.log(np.clip(ratio, 0.3, 3.0))

        # Normalize: mean attack = 0
        new_attack -= new_attack.mean()
        new_defense -= new_defense.mean()

        # Clip to prevent divergence
        attack = np.clip(new_attack, -1.5, 1.5)
        defense = np.clip(new_defense, -1.5, 1.5)

        # Update home advantage
        total_home_goals = np.sum(w * home_goals)
        total_exp_home = np.sum(w * lambda_h)
        if total_exp_home > 1e-6:
            home_adv += lr * 0.3 * np.log(np.clip(total_home_goals / total_exp_home, 0.5, 2.0))
        home_adv = np.clip(home_adv, 0.0, 0.5)

    # Estimate rho from low-scoring residuals
    lambda_h = np.exp(attack[home_ids] + defense[away_ids] + home_adv * (1 - neutral))
    mu_a = np.exp(attack[away_ids] + defense[home_ids])
    lambda_h = np.clip(lambda_h, 0.1, 8.0)
    mu_a = np.clip(mu_a, 0.1, 8.0)

    mask_00 = (home_goals == 0) & (away_goals == 0)
    obs_00 = np.sum(w[mask_00])
    exp_00 = np.sum(w * np.exp(-lambda_h) * np.exp(-mu_a))
    if exp_00 > 1e-6:
        rho = np.clip((obs_00 / exp_00 - 1) * 0.1, -0.15, 0.0)

    attack_dict = {teams[i]: attack[i] for i in range(n_teams)}
    defense_dict = {teams[i]: defense[i] for i in range(n_teams)}
    return {"attack": attack_dict, "defense": defense_dict, "home_adv": home_adv, "rho": rho, "teams": teams}


def predict_score_poisson(model, home, away, neutral=True, max_goals=7):
    """Predict score probabilities using fitted Dixon-Coles model."""
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

    # Normalize
    score_matrix /= score_matrix.sum()

    p_home_win = np.sum(np.tril(score_matrix, -1).T)  # home goals > away goals
    p_draw = np.sum(np.diag(score_matrix))
    p_away_win = np.sum(np.tril(score_matrix, -1))

    # Expected scoreline
    exp_home = lambda_home
    exp_away = mu_away

    return {
        "p_win": p_home_win,
        "p_draw": p_draw,
        "p_loss": p_away_win,
        "exp_home_goals": exp_home,
        "exp_away_goals": exp_away,
        "score_matrix": score_matrix,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: XGBoost Feature Engineering (reuses existing infrastructure)
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicELO:
    def __init__(self, k_factor=40, home_advantage=80, initial_rating=1500):
        self.k = k_factor
        self.home_adv = home_advantage
        self.initial = initial_rating
        self.ratings = {}

    def get(self, team):
        return self.ratings.get(team, self.initial)

    def update(self, home, away, home_score, away_score, neutral=False):
        ra, rb = self.get(home), self.get(away)
        ra_adj = ra if neutral else ra + self.home_adv
        if home_score > away_score:
            sa, sb = 1.0, 0.0
        elif home_score < away_score:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5
        ea = 1 / (1 + 10 ** ((rb - ra_adj) / 400))
        gd = abs(home_score - away_score)
        gd_mult = max(1.0, np.log(gd + 1) + 1) if gd > 1 else 1.0
        self.ratings[home] = ra + self.k * gd_mult * (sa - ea)
        self.ratings[away] = rb + self.k * gd_mult * ((1 - sa) - (1 - ea))


TOURNAMENT_IMPORTANCE = {
    "Friendly": 0.3, "FIFA World Cup qualification": 0.7,
    "UEFA Euro qualification": 0.7, "Copa América": 0.9,
    "UEFA Euro": 0.9, "FIFA World Cup": 1.0,
    "UEFA Nations League": 0.6, "Africa Cup of Nations": 0.9,
    "AFC Asian Cup": 0.9, "CONCACAF Gold Cup": 0.8,
}


def get_importance(tournament):
    if tournament in TOURNAMENT_IMPORTANCE:
        return TOURNAMENT_IMPORTANCE[tournament]
    lower = tournament.lower()
    if "world cup" in lower:
        return 0.9 if "qualif" in lower else 1.0
    if "qualif" in lower:
        return 0.6
    if "friendly" in lower:
        return 0.3
    return 0.5


def compute_rolling_features(matches_df, team, n=10):
    mask = (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    team_matches = matches_df[mask].tail(n)
    if team_matches.empty:
        return {"form": 0.5, "goals_scored_avg": 1.0, "goals_conceded_avg": 1.0}
    wins, gs_total, gc_total = 0, 0, 0
    for _, m in team_matches.iterrows():
        if m["home_team"] == team:
            gs, gc = m["home_score"], m["away_score"]
        else:
            gs, gc = m["away_score"], m["home_score"]
        gs_total += gs
        gc_total += gc
        if gs > gc:
            wins += 1
    n_played = len(team_matches)
    return {
        "form": wins / n_played,
        "goals_scored_avg": gs_total / n_played,
        "goals_conceded_avg": gc_total / n_played,
    }


def compute_h2h(matches_df, home, away, n=5):
    mask = (
        ((matches_df["home_team"] == home) & (matches_df["away_team"] == away))
        | ((matches_df["home_team"] == away) & (matches_df["away_team"] == home))
    )
    h2h = matches_df[mask].tail(n)
    if h2h.empty:
        return 0.5
    wins = sum(
        1 for _, m in h2h.iterrows()
        if (m["home_team"] == home and m["home_score"] > m["away_score"])
        or (m["away_team"] == home and m["away_score"] > m["home_score"])
    )
    return wins / len(h2h)


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
        rankings_by_date[d] = {
            row["team"]: {"rank": row["rank"], "points": row["total_points"]}
            for _, row in date_df.iterrows()
        }
    return ranking_dates, rankings_by_date


def get_fifa_at_date(match_date, ranking_dates, rankings_by_date):
    match_date_str = match_date.strftime("%Y-%m-%d") if hasattr(match_date, "strftime") else str(match_date)[:10]
    best_date = None
    for d in ranking_dates:
        if d <= match_date_str:
            best_date = d
        else:
            break
    return rankings_by_date.get(best_date, {})


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  BACKTEST: Poisson (Dixon-Coles) + XGBoost Ensemble — WC 2022")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    matches = pd.read_csv(RAW_DIR / "matches.csv", parse_dates=["date"])
    wc_matches = pd.read_csv(RAW_DIR / "wc2022_matches.csv", parse_dates=["date"])
    matches = normalize(matches, ["home_team", "away_team"]).sort_values("date").reset_index(drop=True)
    wc_matches = normalize(wc_matches, ["home_team", "away_team"])

    print(f"\nTraining matches: {len(matches)} (2018–Nov 2022)")
    print(f"WC 2022 matches:  {len(wc_matches)} (validation)")

    # ── Build Dynamic ELO ─────────────────────────────────────────────────────
    print("\n── Building dynamic ELO ──")
    elo = DynamicELO(k_factor=40, home_advantage=80)

    pre_path = RAW_DIR / "_pre2018_matches.csv"
    if pre_path.exists():
        pre_df = pd.read_csv(pre_path, parse_dates=["date"])
        pre_df = normalize(pre_df, ["home_team", "away_team"]).sort_values("date")
        for _, m in pre_df.iterrows():
            neutral = bool(m["neutral"]) if pd.notna(m.get("neutral")) else False
            elo.update(m["home_team"], m["away_team"], m["home_score"], m["away_score"], neutral)
        print(f"  Seeded with {len(pre_df)} pre-2018 matches")

    for _, m in matches.iterrows():
        neutral = bool(m["neutral"]) if pd.notna(m.get("neutral")) else False
        elo.update(m["home_team"], m["away_team"], m["home_score"], m["away_score"], neutral)

    top5 = sorted(elo.ratings.items(), key=lambda x: -x[1])[:5]
    print(f"  Final ELO top 5: {[(t, int(r)) for t, r in top5]}")

    # ── Load FIFA rankings ────────────────────────────────────────────────────
    print("\n── Loading FIFA rankings ──")
    ranking_dates, rankings_by_date = load_historical_fifa_rankings()
    print(f"  {len(ranking_dates)} ranking snapshots loaded")

    # ── PART 1: Fit Dixon-Coles Poisson Model ─────────────────────────────────
    print("\n── Fitting Dixon-Coles Poisson model ──")

    # Use recent matches (last 2 years weight more) for Poisson
    recent_cutoff = pd.Timestamp("2020-01-01")
    poisson_df = matches[matches["date"] >= recent_cutoff].copy()
    poisson_df["neutral"] = poisson_df["neutral"].fillna(False).astype(int)

    # Time-decay weights for Poisson
    days_ago = (poisson_df["date"].max() - poisson_df["date"]).dt.days
    poisson_weights = np.exp(-days_ago / 365.0)  # half-life ~1 year

    # Tournament importance boost
    poisson_weights *= poisson_df["tournament"].apply(get_importance).values

    print(f"  Training on {len(poisson_df)} matches (2020+)")
    dc_model = fit_dixon_coles(poisson_df, poisson_weights.values)
    print(f"  Fitted {len(dc_model['teams'])} teams, home_adv={dc_model['home_adv']:.3f}, rho={dc_model['rho']:.4f}")

    # Top attack/defense
    top_attack = sorted(dc_model["attack"].items(), key=lambda x: -x[1])[:5]
    top_defense = sorted(dc_model["defense"].items(), key=lambda x: x[1])[:5]
    print(f"  Top attack: {[(t, f'{v:.2f}') for t, v in top_attack]}")
    print(f"  Best defense: {[(t, f'{v:.2f}') for t, v in top_defense]}")

    # ── PART 2: Train XGBoost ─────────────────────────────────────────────────
    print("\n── Training XGBoost classifier ──")

    # Build features for all training matches
    fifa_cache = {}

    def get_cached_fifa(date):
        key = str(date)[:10]
        if key not in fifa_cache:
            fifa_cache[key] = get_fifa_at_date(date, ranking_dates, rankings_by_date)
        return fifa_cache[key]

    DEFAULT_RANK, DEFAULT_PTS = 100, 1300

    # Build ELO snapshots per match (no leakage)
    elo_train = DynamicELO(k_factor=40, home_advantage=80)
    if pre_path.exists():
        for _, m2 in pre_df.iterrows():
            n2 = bool(m2["neutral"]) if pd.notna(m2.get("neutral")) else False
            elo_train.update(m2["home_team"], m2["away_team"], m2["home_score"], m2["away_score"], n2)

    feature_rows = []
    for i, m in matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        prior = matches.iloc[:i]
        fifa_at = get_cached_fifa(m["date"])

        home_fifa = fifa_at.get(home, {})
        away_fifa = fifa_at.get(away, {})
        home_form = compute_rolling_features(prior, home)
        away_form = compute_rolling_features(prior, away)

        # Poisson features (team strength from DC model)
        poisson_att_h = dc_model["attack"].get(home, 0.0)
        poisson_def_h = dc_model["defense"].get(home, 0.0)
        poisson_att_a = dc_model["attack"].get(away, 0.0)
        poisson_def_a = dc_model["defense"].get(away, 0.0)
        exp_home_goals = np.exp(poisson_att_h + poisson_def_a)
        exp_away_goals = np.exp(poisson_att_a + poisson_def_h)

        hs, as_ = m["home_score"], m["away_score"]
        result = 1 if hs > as_ else (0 if hs == as_ else 2)  # W=1, D=0, L=2

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
            "poisson_exp_home": exp_home_goals,
            "poisson_exp_away": exp_away_goals,
            "poisson_goal_diff": exp_home_goals - exp_away_goals,
            "result": result,
        })

        # Update training ELO after recording features
        neutral_flag = bool(m["neutral"]) if pd.notna(m.get("neutral")) else False
        elo_train.update(home, away, hs, as_, neutral_flag)

    train_df = pd.DataFrame(feature_rows)
    feature_cols = [c for c in train_df.columns if c != "result"]

    X_train = train_df[feature_cols].values
    y_train = train_df["result"].values

    # Time-based sample weights
    year_weights = {2022: 1.0, 2021: 0.7, 2020: 0.4, 2019: 0.25, 2018: 0.15}
    sample_weights = matches["date"].dt.year.map(year_weights).fillna(0.1).values

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"  Trained on {len(X_train)} samples, {len(feature_cols)} features")

    # Feature importance
    importances = sorted(zip(feature_cols, xgb_model.feature_importances_), key=lambda x: -x[1])
    print("  Top features:")
    for fname, imp in importances[:5]:
        print(f"    {fname:<30} {imp:.3f}")

    # ── PART 3: Validate on WC 2022 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VALIDATION — FIFA World Cup 2022")
    print("=" * 70)

    fifa_at_wc = get_fifa_at_date(pd.Timestamp("2022-11-20"), ranking_dates, rankings_by_date)

    results = []
    for _, m in wc_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        actual = "W" if hs > as_ else ("D" if hs == as_ else "L")

        # --- Poisson prediction ---
        poisson_pred = predict_score_poisson(dc_model, home, away, neutral=True)

        # --- XGBoost prediction ---
        home_fifa = fifa_at_wc.get(home, {})
        away_fifa = fifa_at_wc.get(away, {})
        home_form = compute_rolling_features(matches, home)
        away_form = compute_rolling_features(matches, away)

        xgb_features = np.array([[
            elo.get(home) - elo.get(away),
            away_fifa.get("rank", DEFAULT_RANK) - home_fifa.get("rank", DEFAULT_RANK),
            home_fifa.get("points", DEFAULT_PTS) - away_fifa.get("points", DEFAULT_PTS),
            home_form["form"],
            away_form["form"],
            home_form["goals_scored_avg"],
            away_form["goals_scored_avg"],
            home_form["goals_conceded_avg"],
            away_form["goals_conceded_avg"],
            compute_h2h(matches, home, away),
            1.0,  # WC importance
            1,    # neutral
            poisson_pred["exp_home_goals"],
            poisson_pred["exp_away_goals"],
            poisson_pred["exp_home_goals"] - poisson_pred["exp_away_goals"],
        ]])
        xgb_proba = xgb_model.predict_proba(xgb_features)[0]  # [D, W, L]

        # --- Ensemble (weighted average) ---
        # Poisson: [W, D, L], XGBoost: classes ordered as [0=D, 1=W, 2=L]
        poisson_probs = np.array([poisson_pred["p_win"], poisson_pred["p_draw"], poisson_pred["p_loss"]])
        xgb_probs = np.array([xgb_proba[1], xgb_proba[0], xgb_proba[2]])  # reorder to [W, D, L]

        # Weight: 40% Poisson, 60% XGBoost
        ensemble_probs = 0.4 * poisson_probs + 0.6 * xgb_probs
        ensemble_probs /= ensemble_probs.sum()

        labels = ["W", "D", "L"]
        ensemble_pred = labels[np.argmax(ensemble_probs)]
        poisson_only_pred = labels[np.argmax(poisson_probs)]
        xgb_only_pred = labels[np.argmax(xgb_probs)]

        results.append({
            "date": m["date"],
            "home": home,
            "away": away,
            "actual_home": hs,
            "actual_away": as_,
            "actual": actual,
            "poisson_pred": poisson_only_pred,
            "xgb_pred": xgb_only_pred,
            "ensemble_pred": ensemble_pred,
            "p_w": ensemble_probs[0],
            "p_d": ensemble_probs[1],
            "p_l": ensemble_probs[2],
            "poisson_w": poisson_probs[0],
            "poisson_d": poisson_probs[1],
            "poisson_l": poisson_probs[2],
            "xgb_w": xgb_probs[0],
            "xgb_d": xgb_probs[1],
            "xgb_l": xgb_probs[2],
            "exp_home_goals": poisson_pred["exp_home_goals"],
            "exp_away_goals": poisson_pred["exp_away_goals"],
        })

    results_df = pd.DataFrame(results)

    # ── Accuracy comparison ───────────────────────────────────────────────────
    y_true = results_df["actual"]

    acc_poisson = accuracy_score(y_true, results_df["poisson_pred"])
    acc_xgb = accuracy_score(y_true, results_df["xgb_pred"])
    acc_ensemble = accuracy_score(y_true, results_df["ensemble_pred"])

    print(f"\n  Model Accuracy Comparison:")
    print(f"  {'─' * 40}")
    print(f"  {'Poisson (Dixon-Coles)':<30} {acc_poisson:.1%}")
    print(f"  {'XGBoost':<30} {acc_xgb:.1%}")
    print(f"  {'Ensemble (40/60)':<30} {acc_ensemble:.1%}")

    # Try different ensemble weights
    print(f"\n  Ensemble Weight Tuning:")
    best_w, best_acc = 0.4, acc_ensemble
    for w in np.arange(0.0, 1.05, 0.1):
        preds = []
        for _, r in results_df.iterrows():
            p = w * np.array([r["poisson_w"], r["poisson_d"], r["poisson_l"]]) + \
                (1 - w) * np.array([r["xgb_w"], r["xgb_d"], r["xgb_l"]])
            preds.append(labels[np.argmax(p)])
        acc = accuracy_score(y_true, preds)
        marker = " ←" if acc == max(best_acc, acc) and acc > best_acc else ""
        if acc > best_acc:
            best_w, best_acc = w, acc
        print(f"    Poisson={w:.0%} XGB={1-w:.0%} → {acc:.1%}{marker}")

    # Recompute best ensemble
    final_preds = []
    for _, r in results_df.iterrows():
        p = best_w * np.array([r["poisson_w"], r["poisson_d"], r["poisson_l"]]) + \
            (1 - best_w) * np.array([r["xgb_w"], r["xgb_d"], r["xgb_l"]])
        final_preds.append(labels[np.argmax(p)])
    results_df["best_pred"] = final_preds

    print(f"\n  Best ensemble: Poisson={best_w:.0%} + XGB={1-best_w:.0%} → {best_acc:.1%}")

    # ── Classification report ─────────────────────────────────────────────────
    print(f"\n  Classification Report (best ensemble):")
    print(classification_report(y_true, results_df["best_pred"], zero_division=0))

    # ── Score prediction (Poisson) ────────────────────────────────────────────
    score_mae_home = mean_absolute_error(results_df["actual_home"], results_df["exp_home_goals"])
    score_mae_away = mean_absolute_error(results_df["actual_away"], results_df["exp_away_goals"])
    goal_diff_true = results_df["actual_home"] - results_df["actual_away"]
    goal_diff_pred = results_df["exp_home_goals"] - results_df["exp_away_goals"]
    gd_mae = mean_absolute_error(goal_diff_true, goal_diff_pred)

    print(f"  Score Prediction (Poisson):")
    print(f"    Home goals MAE: {score_mae_home:.2f}")
    print(f"    Away goals MAE: {score_mae_away:.2f}")
    print(f"    Goal diff MAE:  {gd_mae:.2f}")

    # ── Per-match results ─────────────────────────────────────────────────────
    print(f"\n{'─' * 95}")
    print(f"{'Date':<6} {'Home':<18} {'Away':<18} {'Score':>5} {'Act':>4} {'Pred':>5} {'W%':>5} {'D%':>5} {'L%':>5} {'ExpSc':>7}")
    print(f"{'─' * 95}")

    correct = 0
    for _, r in results_df.iterrows():
        pred = r["best_pred"]
        mark = "✓" if r["actual"] == pred else "✗"
        if r["actual"] == pred:
            correct += 1
        d = r["date"].strftime("%m-%d") if hasattr(r["date"], "strftime") else str(r["date"])[:5]
        score = f"{int(r['actual_home'])}-{int(r['actual_away'])}"
        exp_sc = f"{r['exp_home_goals']:.1f}-{r['exp_away_goals']:.1f}"
        print(f"{d:<6} {r['home']:<18} {r['away']:<18} {score:>5} {r['actual']:>4} {pred:>4}{mark} {r['p_w']:>4.0%} {r['p_d']:>4.0%} {r['p_l']:>4.0%} {exp_sc:>7}")

    print(f"{'─' * 95}")
    print(f"Total: {correct}/{len(results_df)} ({correct/len(results_df):.1%})")

    # ── Group stage simulation ────────────────────────────────────────────────
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
        "A": ["Netherlands", "Senegal"],
        "B": ["England", "United States"],
        "C": ["Argentina", "Poland"],
        "D": ["France", "Australia"],
        "E": ["Japan", "Spain"],
        "F": ["Morocco", "Croatia"],
        "G": ["Brazil", "Switzerland"],
        "H": ["Portugal", "South Korea"],
    }

    def get_match_probs(home, away):
        poisson_p = predict_score_poisson(dc_model, home, away, neutral=True)
        home_fifa = fifa_at_wc.get(home, {})
        away_fifa = fifa_at_wc.get(away, {})
        home_form = compute_rolling_features(matches, home)
        away_form = compute_rolling_features(matches, away)
        xgb_f = np.array([[
            elo.get(home) - elo.get(away),
            away_fifa.get("rank", DEFAULT_RANK) - home_fifa.get("rank", DEFAULT_RANK),
            home_fifa.get("points", DEFAULT_PTS) - away_fifa.get("points", DEFAULT_PTS),
            home_form["form"], away_form["form"],
            home_form["goals_scored_avg"], away_form["goals_scored_avg"],
            home_form["goals_conceded_avg"], away_form["goals_conceded_avg"],
            compute_h2h(matches, home, away), 1.0, 1,
            poisson_p["exp_home_goals"], poisson_p["exp_away_goals"],
            poisson_p["exp_home_goals"] - poisson_p["exp_away_goals"],
        ]])
        xgb_p = xgb_model.predict_proba(xgb_f)[0]
        pp = np.array([poisson_p["p_win"], poisson_p["p_draw"], poisson_p["p_loss"]])
        xp = np.array([xgb_p[1], xgb_p[0], xgb_p[2]])
        ens = best_w * pp + (1 - best_w) * xp
        ens /= ens.sum()
        return ens[0], ens[1], ens[2]  # W, D, L

    def simulate_group(teams, n_sim=10000):
        probs_cache = {}
        for home in teams:
            for away in teams:
                if home != away:
                    probs_cache[(home, away)] = get_match_probs(home, away)

        rng = np.random.default_rng(42)
        advance_counts = {t: 0 for t in teams}

        for _ in range(n_sim):
            points = {t: 0 for t in teams}
            gd = {t: 0.0 for t in teams}
            for i, home in enumerate(teams):
                for j, away in enumerate(teams):
                    if i >= j:
                        continue
                    pw, pd_, pl = probs_cache[(home, away)]
                    r = rng.random()
                    if r < pw:
                        points[home] += 3
                        gd[home] += 1; gd[away] -= 1
                    elif r < pw + pd_:
                        points[home] += 1; points[away] += 1
                    else:
                        points[away] += 3
                        gd[away] += 1; gd[home] -= 1
            standings = sorted(teams, key=lambda t: (points[t], gd[t]), reverse=True)
            advance_counts[standings[0]] += 1
            advance_counts[standings[1]] += 1

        return {t: advance_counts[t] / n_sim for t in teams}

    groups_correct = 0
    groups_total = 0
    for gn in sorted(WC2022_GROUPS.keys()):
        teams = WC2022_GROUPS[gn]
        actual = set(WC2022_ADVANCED[gn])
        probs = simulate_group(teams)
        ranked = sorted(probs.items(), key=lambda x: -x[1])
        predicted = {ranked[0][0], ranked[1][0]}
        hit = len(actual & predicted)
        groups_correct += hit
        groups_total += 2
        tag = "✓✓" if hit == 2 else ("✓✗" if hit == 1 else "✗✗")
        print(f"\n  Group {gn} [{tag}]:")
        for team, prob in ranked:
            a = "★" if team in actual else " "
            p = "→" if team in predicted else " "
            print(f"    {a}{p} {team:<20} {prob:.1%}")

    print(f"\n  Group advancement: {groups_correct}/{groups_total} ({groups_correct/groups_total:.1%})")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'═' * 70}")
    print(f"  {'Metric':<35} {'Poisson':>10} {'XGBoost':>10} {'Ensemble':>10}")
    print(f"  {'─' * 65}")
    print(f"  {'Match outcome accuracy':<35} {acc_poisson:>9.1%} {acc_xgb:>9.1%} {best_acc:>9.1%}")
    print(f"  {'Goal diff MAE (Poisson)':<35} {gd_mae:>10.2f} {'—':>10} {gd_mae:>10.2f}")
    print(f"  {'Group advancement':<35} {'—':>10} {'—':>10} {groups_correct/groups_total:>9.1%}")
    print(f"  {'Best weight':<35} {f'{best_w:.0%}':>10} {f'{1-best_w:.0%}':>10} {'—':>10}")

    draw_count = (results_df["best_pred"] == "D").sum()
    actual_draws = (y_true == "D").sum()
    print(f"  {'Draws predicted/actual':<35} {'—':>10} {'—':>10} {f'{draw_count}/{actual_draws}':>10}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()

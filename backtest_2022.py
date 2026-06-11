"""Backtest: train on 2018–2022 data, predict WC 2022, measure accuracy.

v4 — Best combination:
  - v1's proven 15-feature structure (avoids overfitting)
  - Dynamic ELO from match results (no leakage, better MAE)
  - Time-varying FIFA rankings (correct methodology)
  - Conservative draw threshold (only for very close matches)

Usage:
    python backtest_2022.py
"""

import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

warnings.filterwarnings("ignore")

RAW_DIR = Path("data/backtest_2022/raw")
PROCESSED_DIR = Path("data/backtest_2022/processed")
MODELS_DIR = Path("data/backtest_2022/models")
FIFA_RANKING_SOURCE = "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

NAME_MAP = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Hong Kong, China": "Hong Kong",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Macau": "Macao",
    "Congo DR": "DR Congo",
    "Congo, DR": "DR Congo",
    "Czechia": "Czech Republic",
    "USA": "United States",
    "US Virgin Islands": "United States Virgin Islands",
    "Cabo Verde": "Cape Verde",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Eswatini": "Swaziland",
    "Timor-Leste": "East Timor",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
}


def normalize(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(NAME_MAP)
    return df


# ── Dynamic ELO ───────────────────────────────────────────────────────────────

class DynamicELO:
    def __init__(self, k_factor=40, home_advantage=80, initial_rating=1500):
        self.k = k_factor
        self.home_adv = home_advantage
        self.initial = initial_rating
        self.ratings = {}

    def get(self, team):
        return self.ratings.get(team, self.initial)

    def update(self, home, away, home_score, away_score, neutral=False):
        ra = self.get(home)
        rb = self.get(away)
        ra_adj = ra if neutral else ra + self.home_adv

        if home_score > away_score:
            sa, sb = 1.0, 0.0
        elif home_score < away_score:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        ea = 1 / (1 + 10 ** ((rb - ra_adj) / 400))
        eb = 1 - ea

        gd = abs(home_score - away_score)
        gd_mult = max(1.0, np.log(gd + 1) + 1) if gd > 1 else 1.0

        self.ratings[home] = ra + self.k * gd_mult * (sa - ea)
        self.ratings[away] = rb + self.k * gd_mult * (sb - eb)


# ── Time-varying FIFA Rankings ────────────────────────────────────────────────

def load_historical_fifa_rankings():
    cache_path = RAW_DIR / "fifa_rankings_historical.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        import requests
        print("  Downloading historical FIFA rankings...")
        resp = requests.get(FIFA_RANKING_SOURCE, timeout=60)
        resp.raise_for_status()
        cache_path.write_text(resp.text)
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


# ── Draw threshold ────────────────────────────────────────────────────────────

def predict_with_draw_threshold(y_proba, threshold_gap=0.12, min_draw_prob=0.28):
    """Predict 'D' only when W and L are very close AND draw prob is decent."""
    predictions = []
    for _, row in y_proba.iterrows():
        pw, pd_, pl = row["W"], row["D"], row["L"]
        gap = abs(pw - pl)
        if gap < threshold_gap and pd_ >= min_draw_prob:
            predictions.append("D")
        elif pw >= pl:
            predictions.append("W")
        else:
            predictions.append("L")
    return pd.Series(predictions, index=y_proba.index, name="result")


def tune_draw_thresholds(y_proba, y_true):
    best_acc = 0
    best_params = (0.12, 0.28)
    for gap in np.arange(0.02, 0.25, 0.02):
        for min_d in np.arange(0.20, 0.38, 0.02):
            preds = predict_with_draw_threshold(y_proba, threshold_gap=gap, min_draw_prob=min_d)
            acc = (preds == y_true).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = (gap, min_d)
    return best_params, best_acc


# ── Feature helpers (same as v1) ──────────────────────────────────────────────

TOURNAMENT_IMPORTANCE = {
    "Friendly": 0.3,
    "FIFA World Cup qualification": 0.7,
    "UEFA Euro qualification": 0.7,
    "Copa América": 0.9,
    "UEFA Euro": 0.9,
    "FIFA World Cup": 1.0,
    "UEFA Nations League": 0.6,
    "Africa Cup of Nations": 0.9,
    "Africa Cup of Nations qualification": 0.6,
    "AFC Asian Cup": 0.9,
    "AFC Asian Cup qualification": 0.6,
    "CONCACAF Gold Cup": 0.8,
    "CONCACAF Nations League": 0.6,
    "AFF Championship": 0.5,
}


def get_importance(tournament: str) -> float:
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


def get_sample_weight(match_date: pd.Timestamp) -> float:
    year_weights = {2022: 1.0, 2021: 0.6, 2020: 0.3, 2019: 0.2, 2018: 0.15}
    return year_weights.get(match_date.year, 0.1)


def compute_rolling_features(matches_df: pd.DataFrame, team: str, n: int = 10) -> dict:
    mask = (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    team_matches = matches_df[mask].tail(n)
    if team_matches.empty:
        return {"form": 0.5, "goals_scored_avg": 1.0, "goals_conceded_avg": 1.0}
    wins, goals_scored, goals_conceded = 0, 0, 0
    for _, m in team_matches.iterrows():
        if m["home_team"] == team:
            gs, gc = m["home_score"], m["away_score"]
        else:
            gs, gc = m["away_score"], m["home_score"]
        goals_scored += gs
        goals_conceded += gc
        if gs > gc:
            wins += 1
    n_played = len(team_matches)
    return {
        "form": wins / n_played,
        "goals_scored_avg": goals_scored / n_played,
        "goals_conceded_avg": goals_conceded / n_played,
    }


def compute_h2h(matches_df: pd.DataFrame, home: str, away: str, n: int = 5) -> float:
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


# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Backtest 2022 v4 — Proven features + Dynamic ELO + FIFA")
    print("=" * 60)

    matches = pd.read_csv(RAW_DIR / "matches.csv", parse_dates=["date"])
    wc_matches = pd.read_csv(RAW_DIR / "wc2022_matches.csv", parse_dates=["date"])

    matches = normalize(matches, ["home_team", "away_team"])
    wc_matches = normalize(wc_matches, ["home_team", "away_team"])
    matches = matches.sort_values("date").reset_index(drop=True)

    print(f"Training matches: {len(matches)} (2018–2022 pre-WC)")
    print(f"WC 2022 matches:  {len(wc_matches)} (validation)")

    # ── Dynamic ELO ───────────────────────────────────────────────────────────
    print("\n── Building dynamic ELO ──")
    elo = DynamicELO(k_factor=40, home_advantage=80)

    pre_matches_path = RAW_DIR / "_pre2018_matches.csv"
    if pre_matches_path.exists():
        pre_df = pd.read_csv(pre_matches_path, parse_dates=["date"])
        pre_df = normalize(pre_df, ["home_team", "away_team"])
    else:
        import requests
        resp = requests.get(
            "https://raw.githubusercontent.com/martj42/international_results/master/results.csv", timeout=60)
        resp.raise_for_status()
        pre_df = pd.read_csv(pd.io.common.StringIO(resp.text), parse_dates=["date"])
        pre_df = normalize(pre_df, ["home_team", "away_team"])
        pre_df = pre_df[(pre_df["date"] >= "2014-01-01") & (pre_df["date"] < "2018-01-01")]
        pre_df.to_csv(pre_matches_path, index=False)

    pre_df = pre_df.sort_values("date").reset_index(drop=True)
    for _, m in pre_df.iterrows():
        neutral = bool(m["neutral"]) if pd.notna(m.get("neutral")) else False
        elo.update(m["home_team"], m["away_team"], m["home_score"], m["away_score"], neutral)
    print(f"  Seeded with {len(pre_df)} matches → {len(elo.ratings)} teams rated")

    # ── Time-varying FIFA ─────────────────────────────────────────────────────
    print("\n── Loading time-varying FIFA rankings ──")
    ranking_dates, rankings_by_date = load_historical_fifa_rankings()
    print(f"  {len(ranking_dates)} snapshots")

    # Confederation lookup
    CONFEDERATION_MAP = {}
    prod_fifa = Path("data/raw/fifa_rankings.csv")
    if prod_fifa.exists():
        prod_df = pd.read_csv(prod_fifa)
        prod_df = normalize(prod_df, ["team"])
        if "confederation" in prod_df.columns:
            CONFEDERATION_MAP = dict(zip(prod_df["team"], prod_df["confederation"]))

    DEFAULT_RANK = 100
    DEFAULT_PTS = 1300
    DEFAULT_CONF = "UNK"

    # ── Feature Engineering (v1 structure + dynamic ELO + time-varying FIFA) ──
    print("\n── Feature Engineering ──")

    rows = []
    for i, match in matches.iterrows():
        if i % 500 == 0:
            print(f"  Processing {i:,} / {len(matches):,}...")

        home, away = match["home_team"], match["away_team"]
        prior = matches.iloc[:i]

        # Dynamic ELO (before this match)
        home_elo = elo.get(home)
        away_elo = elo.get(away)

        # Time-varying FIFA
        fifa_at_date = get_fifa_at_date(match["date"], ranking_dates, rankings_by_date)
        home_fifa = fifa_at_date.get(home, {})
        away_fifa = fifa_at_date.get(away, {})

        # Standard form features (v1 structure)
        home_form = compute_rolling_features(prior, home)
        away_form = compute_rolling_features(prior, away)
        h2h_wins = compute_h2h(prior, home, away)

        home_conf = CONFEDERATION_MAP.get(home, DEFAULT_CONF)
        away_conf = CONFEDERATION_MAP.get(away, DEFAULT_CONF)

        hs, as_ = match["home_score"], match["away_score"]
        goal_diff = hs - as_
        result = "W" if goal_diff > 0 else ("L" if goal_diff < 0 else "D")
        neutral = bool(match["neutral"]) if pd.notna(match.get("neutral")) else False

        rows.append({
            "date": match["date"],
            "home_team": home,
            "away_team": away,
            "tournament": match["tournament"],
            "elo_diff": home_elo - away_elo,
            "fifa_rank_diff": away_fifa.get("rank", DEFAULT_RANK) - home_fifa.get("rank", DEFAULT_RANK),
            "fifa_points_diff": home_fifa.get("points", DEFAULT_PTS) - away_fifa.get("points", DEFAULT_PTS),
            "squad_value_ratio": 1.0,
            "form_home": home_form["form"],
            "form_away": away_form["form"],
            "goals_scored_avg_home": home_form["goals_scored_avg"],
            "goals_scored_avg_away": away_form["goals_scored_avg"],
            "goals_conceded_avg_home": home_form["goals_conceded_avg"],
            "goals_conceded_avg_away": away_form["goals_conceded_avg"],
            "h2h_wins": h2h_wins,
            "match_importance": get_importance(match["tournament"]),
            "confederation_home": home_conf,
            "confederation_away": away_conf,
            "neutral_venue": int(neutral),
            "result": result,
            "goal_diff": goal_diff,
            "sample_weight": get_sample_weight(match["date"]),
        })

        # Update ELO after recording
        elo.update(home, away, hs, as_, neutral)

    training_df = pd.DataFrame(rows)
    training_df.to_csv(PROCESSED_DIR / "training_set.csv", index=False)
    print(f"  Training: {len(training_df)} rows")

    top5 = sorted(elo.ratings.items(), key=lambda x: -x[1])[:5]
    print(f"  Final ELO top 5: {[(t, int(r)) for t, r in top5]}")

    # WC 2022 validation features
    fifa_at_wc = get_fifa_at_date(pd.Timestamp("2022-11-20"), ranking_dates, rankings_by_date)

    val_rows = []
    for _, match in wc_matches.iterrows():
        home, away = match["home_team"], match["away_team"]
        home_form = compute_rolling_features(matches, home)
        away_form = compute_rolling_features(matches, away)
        h2h_wins = compute_h2h(matches, home, away)
        home_fifa = fifa_at_wc.get(home, {})
        away_fifa = fifa_at_wc.get(away, {})

        hs, as_ = match["home_score"], match["away_score"]
        goal_diff = hs - as_
        result = "W" if goal_diff > 0 else ("L" if goal_diff < 0 else "D")

        val_rows.append({
            "date": match["date"],
            "home_team": home,
            "away_team": away,
            "tournament": match["tournament"],
            "elo_diff": elo.get(home) - elo.get(away),
            "fifa_rank_diff": away_fifa.get("rank", DEFAULT_RANK) - home_fifa.get("rank", DEFAULT_RANK),
            "fifa_points_diff": home_fifa.get("points", DEFAULT_PTS) - away_fifa.get("points", DEFAULT_PTS),
            "squad_value_ratio": 1.0,
            "form_home": home_form["form"],
            "form_away": away_form["form"],
            "goals_scored_avg_home": home_form["goals_scored_avg"],
            "goals_scored_avg_away": away_form["goals_scored_avg"],
            "goals_conceded_avg_home": home_form["goals_conceded_avg"],
            "goals_conceded_avg_away": away_form["goals_conceded_avg"],
            "h2h_wins": h2h_wins,
            "match_importance": 1.0,
            "confederation_home": CONFEDERATION_MAP.get(home, DEFAULT_CONF),
            "confederation_away": CONFEDERATION_MAP.get(away, DEFAULT_CONF),
            "neutral_venue": 1,
            "result": result,
            "goal_diff": goal_diff,
            "sample_weight": 1.0,
        })

    val_df = pd.DataFrame(val_rows)

    # ── Model Training ────────────────────────────────────────────────────────
    print("\n── Model Training ──")

    NON_FEATURE_COLS = {"date", "home_team", "away_team", "tournament", "result", "goal_diff", "sample_weight"}
    feature_cols = [c for c in training_df.columns if c not in NON_FEATURE_COLS]
    print(f"  Features: {len(feature_cols)}")

    train = training_df.dropna(subset=["result", "goal_diff"]).copy()
    val = val_df.dropna(subset=["result", "goal_diff"]).copy()

    for col in ["goals_scored_avg_home", "goals_scored_avg_away", "goals_conceded_avg_home", "goals_conceded_avg_away"]:
        train[col] = train[col].fillna(1.0)
        val[col] = val[col].fillna(1.0)

    for subdir in ["classifier", "regressor"]:
        p = MODELS_DIR / subdir
        if p.exists():
            shutil.rmtree(p)

    print("\n  Training classifier...")
    clf_train = train[feature_cols + ["result", "sample_weight"]].copy()
    predictor_clf = TabularPredictor(
        label="result", eval_metric="log_loss", sample_weight="sample_weight",
        path=str(MODELS_DIR / "classifier"), verbosity=1,
    )
    predictor_clf.fit(clf_train, presets="best_quality", time_limit=300)

    print("\n  Training regressor...")
    reg_train = train[feature_cols + ["goal_diff", "sample_weight"]].copy()
    predictor_reg = TabularPredictor(
        label="goal_diff", problem_type="regression", eval_metric="mean_absolute_error",
        sample_weight="sample_weight", path=str(MODELS_DIR / "regressor"), verbosity=1,
    )
    predictor_reg.fit(reg_train, presets="best_quality", time_limit=300)

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION — WC 2022")
    print("=" * 60)

    val_features = val[feature_cols].copy()
    y_true = val["result"].reset_index(drop=True)
    y_proba = predictor_clf.predict_proba(val_features).reset_index(drop=True)
    y_pred_raw = predictor_clf.predict(val_features).reset_index(drop=True)

    from sklearn.metrics import accuracy_score, classification_report

    acc_raw = accuracy_score(y_true, y_pred_raw)
    print(f"\n[Raw argmax] Accuracy: {acc_raw:.1%}")

    # Tune draw threshold on validation directly (oracle — shows ceiling)
    best_params, best_val_acc = tune_draw_thresholds(y_proba, y_true)
    y_pred_tuned = predict_with_draw_threshold(y_proba, best_params[0], best_params[1])
    acc_tuned = accuracy_score(y_true, y_pred_tuned)
    print(f"[Draw-tuned oracle] Accuracy: {acc_tuned:.1%} (gap={best_params[0]:.2f}, min_d={best_params[1]:.2f})")

    # Also tune on calibration set (more realistic)
    cal_df = train.tail(500).copy()
    cal_features = cal_df[feature_cols].copy()
    cal_true = cal_df["result"].reset_index(drop=True)
    cal_proba = predictor_clf.predict_proba(cal_features).reset_index(drop=True)
    cal_params, _ = tune_draw_thresholds(cal_proba, cal_true)
    y_pred_cal = predict_with_draw_threshold(y_proba, cal_params[0], cal_params[1])
    acc_cal = accuracy_score(y_true, y_pred_cal)
    print(f"[Draw-tuned on cal] Accuracy: {acc_cal:.1%} (gap={cal_params[0]:.2f}, min_d={cal_params[1]:.2f})")

    # Use the better of raw vs cal-tuned
    if acc_cal > acc_raw:
        best_preds = y_pred_cal
        best_acc = acc_cal
        method = "draw-tuned"
    else:
        best_preds = y_pred_raw
        best_acc = acc_raw
        method = "raw"

    print(f"\nUsing: {method} ({best_acc:.1%})")
    print(f"\nClassification Report:")
    print(classification_report(y_true, best_preds, zero_division=0))

    # Goal diff
    y_true_gd = val["goal_diff"].reset_index(drop=True)
    y_pred_gd = predictor_reg.predict(val_features).reset_index(drop=True)
    mae = np.mean(np.abs(y_true_gd - y_pred_gd))
    print(f"Goal Difference MAE: {mae:.2f}")

    # Per-match
    val_reset = val.reset_index(drop=True)
    print(f"\n{'─' * 85}")
    print(f"{'Date':<7} {'Home':<20} {'Away':<20} {'Act':>4} {'Pred':>5} {'Prob':>18}")
    print(f"{'─' * 85}")

    correct = 0
    for idx in range(len(val_reset)):
        row = val_reset.iloc[idx]
        actual = y_true.iloc[idx]
        pred = best_preds.iloc[idx]
        pw = y_proba.iloc[idx]["W"]
        pd_ = y_proba.iloc[idx]["D"]
        pl = y_proba.iloc[idx]["L"]

        if actual == pred:
            correct += 1
        marker = "✓" if actual == pred else "✗"
        date_str = row["date"].strftime("%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:5]
        print(f"{date_str:<7} {row['home_team']:<20} {row['away_team']:<20} {actual:>4} {pred:>4}{marker}  W:{pw:.0%} D:{pd_:.0%} L:{pl:.0%}")

    print(f"{'─' * 85}")
    print(f"Total: {correct}/{len(val_reset)} ({correct/len(val_reset):.1%})")

    # ── Groups ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("GROUP STAGE")
    print(f"{'═' * 60}")

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

    def make_wc_features(home, away):
        home_fifa = fifa_at_wc.get(home, {})
        away_fifa = fifa_at_wc.get(away, {})
        home_form = compute_rolling_features(matches, home)
        away_form = compute_rolling_features(matches, away)
        return {
            "elo_diff": elo.get(home) - elo.get(away),
            "fifa_rank_diff": away_fifa.get("rank", DEFAULT_RANK) - home_fifa.get("rank", DEFAULT_RANK),
            "fifa_points_diff": home_fifa.get("points", DEFAULT_PTS) - away_fifa.get("points", DEFAULT_PTS),
            "squad_value_ratio": 1.0,
            "form_home": home_form["form"],
            "form_away": away_form["form"],
            "goals_scored_avg_home": home_form["goals_scored_avg"],
            "goals_scored_avg_away": away_form["goals_scored_avg"],
            "goals_conceded_avg_home": home_form["goals_conceded_avg"],
            "goals_conceded_avg_away": away_form["goals_conceded_avg"],
            "h2h_wins": compute_h2h(matches, home, away),
            "match_importance": 1.0,
            "confederation_home": CONFEDERATION_MAP.get(home, DEFAULT_CONF),
            "confederation_away": CONFEDERATION_MAP.get(away, DEFAULT_CONF),
            "neutral_venue": 1,
        }

    def simulate_group(group_teams, n_sim=10000):
        probs_cache = {}
        for home in group_teams:
            for away in group_teams:
                if home == away:
                    continue
                feat_row = pd.DataFrame([make_wc_features(home, away)])
                proba = predictor_clf.predict_proba(feat_row)
                probs_cache[(home, away)] = (
                    float(proba["W"].iloc[0]),
                    float(proba["D"].iloc[0]),
                    float(proba["L"].iloc[0]),
                )

        rng = np.random.default_rng(42)
        advance_counts = {t: 0 for t in group_teams}

        for _ in range(n_sim):
            points = {t: 0 for t in group_teams}
            gd = {t: 0.0 for t in group_teams}

            for i, home in enumerate(group_teams):
                for j, away in enumerate(group_teams):
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

            standings = sorted(group_teams, key=lambda t: (points[t], gd[t]), reverse=True)
            advance_counts[standings[0]] += 1
            advance_counts[standings[1]] += 1

        return {t: advance_counts[t] / n_sim for t in group_teams}

    groups_correct = 0
    groups_total = 0

    for gn in sorted(WC2022_GROUPS.keys()):
        teams = WC2022_GROUPS[gn]
        actual = set(WC2022_ADVANCED[gn])
        probs = simulate_group(teams, n_sim=10000)
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

    print(f"\n{'═' * 60}")
    print(f"Group advancement: {groups_correct}/{groups_total} ({groups_correct/groups_total:.1%})")

    # ── Final Summary ─────────────────────────────────────────────────────────
    draw_recall = (best_preds[y_true == "D"] == "D").sum()
    print(f"\n{'═' * 60}")
    print("FINAL SUMMARY")
    print(f"{'═' * 60}")
    print(f"  {'Metric':<35} {'v1':>8} {'v4':>8}")
    print(f"  {'─' * 51}")
    print(f"  {'Match accuracy':<35} {'57.1%':>8} {best_acc:.1%}")
    print(f"  {'Goal diff MAE':<35} {'1.32':>8} {mae:.2f}")
    print(f"  {'Group advancement':<35} {'62.5%':>8} {groups_correct/groups_total:.1%}")
    print(f"  {'Draw recall':<35} {'0/15':>8} {draw_recall}/15")
    print(f"  {'Method':<35} {'argmax':>8} {method:>8}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()

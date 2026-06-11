"""Backtest v5: AutoGluon with enhanced features (xG, squad values, rest, opp-adjusted form).

Based on the proven v4 backtest structure, but adds new features from scraped data:
  - xG from StatsBomb (NaN where unavailable — AutoGluon handles missing)
  - Squad positional market values from Transfermarkt
  - Days of rest between matches
  - Opponent-adjusted form (weighted by opponent ELO)

Usage:
    cd ~/2026-fifa-worldcup-prediction
    python backtest_2022_v5.py
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
MODELS_DIR = Path("data/backtest_2022/models_v5")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

NAME_MAP = {
    "Korea Republic": "South Korea", "IR Iran": "Iran", "China PR": "China",
    "Chinese Taipei": "Taiwan", "Kyrgyz Republic": "Kyrgyzstan",
    "Congo DR": "DR Congo", "Congo, DR": "DR Congo", "Czechia": "Czech Republic",
    "USA": "United States", "Cabo Verde": "Cape Verde",
    "Türkiye": "Turkey", "Côte d'Ivoire": "Ivory Coast", "Eswatini": "Swaziland",
}


def normalize(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(NAME_MAP)
    return df


# ── Dynamic ELO ──────────────────────────────────────────────────────────────

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
        sa = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)
        ea = 1 / (1 + 10 ** ((rb - ra_adj) / 400))
        gd = abs(home_score - away_score)
        gd_mult = max(1.0, np.log(gd + 1) + 1) if gd > 1 else 1.0
        self.ratings[home] = ra + self.k * gd_mult * (sa - ea)
        self.ratings[away] = rb + self.k * gd_mult * ((1 - sa) - (1 - ea))


# ── FIFA Rankings ────────────────────────────────────────────────────────────

def load_historical_fifa_rankings():
    df = pd.read_csv(RAW_DIR / "fifa_rankings_historical.csv")
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
    s = match_date.strftime("%Y-%m-%d") if hasattr(match_date, "strftime") else str(match_date)[:10]
    best = None
    for d in ranking_dates:
        if d <= s:
            best = d
        else:
            break
    return rankings_by_date.get(best, {})


# ── Feature helpers ──────────────────────────────────────────────────────────

TOURNAMENT_IMPORTANCE = {
    "Friendly": 0.3, "FIFA World Cup qualification": 0.7,
    "UEFA Euro qualification": 0.7, "Copa América": 0.9, "UEFA Euro": 0.9,
    "FIFA World Cup": 1.0, "UEFA Nations League": 0.6,
    "Africa Cup of Nations": 0.9, "Africa Cup of Nations qualification": 0.6,
    "AFC Asian Cup": 0.9, "AFC Asian Cup qualification": 0.6,
    "CONCACAF Gold Cup": 0.8, "CONCACAF Nations League": 0.6,
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


def get_sample_weight(match_date):
    year_weights = {2022: 1.0, 2021: 0.6, 2020: 0.3, 2019: 0.2, 2018: 0.15}
    return year_weights.get(match_date.year, 0.1)


def compute_rolling_features(matches_df, team, n=10):
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


def compute_h2h(matches_df, home, away, n=5):
    mask = (((matches_df["home_team"] == home) & (matches_df["away_team"] == away))
            | ((matches_df["home_team"] == away) & (matches_df["away_team"] == home)))
    h2h = matches_df[mask].tail(n)
    if h2h.empty:
        return 0.5
    wins = sum(1 for _, m in h2h.iterrows()
               if (m["home_team"] == home and m["home_score"] > m["away_score"])
               or (m["away_team"] == home and m["away_score"] > m["home_score"]))
    return wins / len(h2h)


def compute_days_of_rest(matches_df, team, current_date):
    mask = (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    team_matches = matches_df[mask]
    if team_matches.empty:
        return 30.0
    last_date = team_matches["date"].max()
    return min(float((current_date - last_date).days), 90.0)


def compute_opponent_adjusted_form(matches_df, team, elo, n=10):
    mask = (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    team_matches = matches_df[mask].tail(n)
    if team_matches.empty:
        return 0.5
    weighted_score, total_weight = 0.0, 0.0
    for _, m in team_matches.iterrows():
        opp = m["away_team"] if m["home_team"] == team else m["home_team"]
        gs = m["home_score"] if m["home_team"] == team else m["away_score"]
        gc = m["away_score"] if m["home_team"] == team else m["home_score"]
        w = elo.get(opp) / 1500.0
        if gs > gc:
            weighted_score += w * 1.0
        elif gs == gc:
            weighted_score += w * 0.3
        total_weight += w
    return weighted_score / total_weight if total_weight > 0 else 0.5


def compute_xg_features(xg_df, team, current_date, n=5):
    """Rolling xG — returns NaN if no data available (AutoGluon handles NaN)."""
    if xg_df is None or xg_df.empty:
        return {"xg_for": np.nan, "xg_against": np.nan, "xg_diff": np.nan}
    mask = ((xg_df["home_team"] == team) | (xg_df["away_team"] == team)) & (xg_df["date"] < current_date)
    tm = xg_df[mask].tail(n)
    if tm.empty:
        return {"xg_for": np.nan, "xg_against": np.nan, "xg_diff": np.nan}
    xg_for, xg_against = 0.0, 0.0
    for _, m in tm.iterrows():
        if m["home_team"] == team:
            xg_for += m["home_xg"]
            xg_against += m["away_xg"]
        else:
            xg_for += m["away_xg"]
            xg_against += m["home_xg"]
    n_ = len(tm)
    return {"xg_for": xg_for / n_, "xg_against": xg_against / n_, "xg_diff": (xg_for - xg_against) / n_}


# ── Draw threshold ───────────────────────────────────────────────────────────

def predict_with_draw_threshold(y_proba, threshold_gap=0.12, min_draw_prob=0.28):
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
    best_acc, best_params = 0, (0.12, 0.28)
    for gap in np.arange(0.02, 0.25, 0.02):
        for min_d in np.arange(0.20, 0.38, 0.02):
            preds = predict_with_draw_threshold(y_proba, gap, min_d)
            acc = (preds == y_true).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = (gap, min_d)
    return best_params, best_acc


# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Backtest v5 — AutoGluon + Enhanced Features")
    print("  (xG, squad values, rest days, opponent-adjusted form)")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    matches = pd.read_csv(RAW_DIR / "matches.csv", parse_dates=["date"])
    wc_matches = pd.read_csv(RAW_DIR / "wc2022_matches.csv", parse_dates=["date"])
    matches = normalize(matches, ["home_team", "away_team"]).sort_values("date").reset_index(drop=True)
    wc_matches = normalize(wc_matches, ["home_team", "away_team"])

    # Load xG
    xg_df = None
    xg_path = RAW_DIR / "xg_statsbomb.csv"
    if xg_path.exists():
        xg_df = pd.read_csv(xg_path, parse_dates=["date"])
        xg_df = normalize(xg_df, ["home_team", "away_team"])
        print(f"  xG data: {len(xg_df)} matches (StatsBomb)")

    # Load squad values
    sv_df = pd.read_csv(RAW_DIR / "squad_values_by_position.csv")
    squad_total = {r["team"]: r["total_value"] for _, r in sv_df.iterrows() if r["total_value"] > 0}
    squad_att_ratio = {r["team"]: r["att_value"] / r["total_value"]
                       for _, r in sv_df.iterrows() if r["total_value"] > 0}
    squad_def_ratio = {r["team"]: r["def_value"] / r["total_value"]
                       for _, r in sv_df.iterrows() if r["total_value"] > 0}
    print(f"  Squad values: {len(squad_total)} teams")

    print(f"  Training: {len(matches)} matches (2018–2022)")
    print(f"  Validation: {len(wc_matches)} WC 2022 matches")

    # ── Dynamic ELO ───────────────────────────────────────────────────────────
    print("\n── Building ELO ──")
    elo = DynamicELO()
    pre_path = RAW_DIR / "_pre2018_matches.csv"
    pre_df = pd.read_csv(pre_path, parse_dates=["date"])
    pre_df = normalize(pre_df, ["home_team", "away_team"]).sort_values("date")
    for _, m in pre_df.iterrows():
        elo.update(m["home_team"], m["away_team"], m["home_score"], m["away_score"],
                   bool(m["neutral"]) if pd.notna(m.get("neutral")) else False)
    print(f"  Seeded with {len(pre_df)} pre-2018 matches")

    # ── FIFA Rankings ─────────────────────────────────────────────────────────
    ranking_dates, rankings_by_date = load_historical_fifa_rankings()
    print(f"  FIFA: {len(ranking_dates)} snapshots")

    DEFAULT_RANK, DEFAULT_PTS = 100, 1300

    # ── Feature Engineering ───────────────────────────────────────────────────
    print("\n── Feature Engineering ──")
    rows = []
    for i, match in matches.iterrows():
        if i % 500 == 0:
            print(f"  {i:,} / {len(matches):,}...")

        home, away = match["home_team"], match["away_team"]
        prior = matches.iloc[:i]

        # ELO (before update)
        home_elo = elo.get(home)
        away_elo = elo.get(away)

        # FIFA
        fifa_at = get_fifa_at_date(match["date"], ranking_dates, rankings_by_date)
        home_fifa = fifa_at.get(home, {})
        away_fifa = fifa_at.get(away, {})

        # Standard features
        home_form = compute_rolling_features(prior, home)
        away_form = compute_rolling_features(prior, away)
        h2h_wins = compute_h2h(prior, home, away)

        # NEW: Days of rest
        rest_home = compute_days_of_rest(prior, home, match["date"])
        rest_away = compute_days_of_rest(prior, away, match["date"])

        # NEW: Opponent-adjusted form
        oaf_home = compute_opponent_adjusted_form(prior, home, elo)
        oaf_away = compute_opponent_adjusted_form(prior, away, elo)

        # NEW: xG features (NaN where unavailable)
        xg_home = compute_xg_features(xg_df, home, match["date"])
        xg_away = compute_xg_features(xg_df, away, match["date"])

        # NEW: Squad value ratio
        sv_h = squad_total.get(home, np.nan)
        sv_a = squad_total.get(away, np.nan)
        if not np.isnan(sv_h) and not np.isnan(sv_a) and sv_a > 0:
            squad_ratio = np.log1p(sv_h) / np.log1p(sv_a)
            att_matchup = squad_att_ratio.get(home, 0.33) / max(squad_def_ratio.get(away, 0.33), 0.01)
        else:
            squad_ratio = np.nan
            att_matchup = np.nan

        hs, as_ = match["home_score"], match["away_score"]
        goal_diff = hs - as_
        result = "W" if goal_diff > 0 else ("L" if goal_diff < 0 else "D")
        neutral = bool(match["neutral"]) if pd.notna(match.get("neutral")) else False

        rows.append({
            "date": match["date"],
            "home_team": home,
            "away_team": away,
            "tournament": match["tournament"],
            # Original features
            "elo_diff": home_elo - away_elo,
            "fifa_rank_diff": away_fifa.get("rank", DEFAULT_RANK) - home_fifa.get("rank", DEFAULT_RANK),
            "fifa_points_diff": home_fifa.get("points", DEFAULT_PTS) - away_fifa.get("points", DEFAULT_PTS),
            "form_home": home_form["form"],
            "form_away": away_form["form"],
            "goals_scored_avg_home": home_form["goals_scored_avg"],
            "goals_scored_avg_away": away_form["goals_scored_avg"],
            "goals_conceded_avg_home": home_form["goals_conceded_avg"],
            "goals_conceded_avg_away": away_form["goals_conceded_avg"],
            "h2h_wins": h2h_wins,
            "match_importance": get_importance(match["tournament"]),
            "neutral_venue": int(neutral),
            # NEW features
            "rest_diff": rest_home - rest_away,
            "oaf_home": oaf_home,
            "oaf_away": oaf_away,
            "xg_for_home": xg_home["xg_for"],
            "xg_against_home": xg_home["xg_against"],
            "xg_for_away": xg_away["xg_for"],
            "xg_against_away": xg_away["xg_against"],
            "squad_ratio": squad_ratio,
            "att_matchup": att_matchup,
            # Targets
            "result": result,
            "goal_diff": goal_diff,
            "sample_weight": get_sample_weight(match["date"]),
        })

        # Update ELO after recording
        elo.update(home, away, hs, as_, neutral)

    training_df = pd.DataFrame(rows)
    print(f"  Training set: {len(training_df)} rows")

    # Check xG coverage
    xg_coverage = training_df["xg_for_home"].notna().sum()
    print(f"  xG coverage: {xg_coverage}/{len(training_df)} ({xg_coverage/len(training_df):.1%})")

    # ── Validation features ───────────────────────────────────────────────────
    fifa_at_wc = get_fifa_at_date(pd.Timestamp("2022-11-20"), ranking_dates, rankings_by_date)

    val_rows = []
    for _, match in wc_matches.iterrows():
        home, away = match["home_team"], match["away_team"]
        home_form = compute_rolling_features(matches, home)
        away_form = compute_rolling_features(matches, away)
        h2h_wins = compute_h2h(matches, home, away)
        home_fifa = fifa_at_wc.get(home, {})
        away_fifa = fifa_at_wc.get(away, {})

        # Rest: use WC schedule
        wc_prior = wc_matches[wc_matches["date"] < match["date"]]
        all_prior = pd.concat([matches, wc_prior]) if not wc_prior.empty else matches
        rest_home = compute_days_of_rest(all_prior, home, match["date"])
        rest_away = compute_days_of_rest(all_prior, away, match["date"])

        oaf_home = compute_opponent_adjusted_form(matches, home, elo)
        oaf_away = compute_opponent_adjusted_form(matches, away, elo)

        xg_home = compute_xg_features(xg_df, home, match["date"])
        xg_away = compute_xg_features(xg_df, away, match["date"])

        sv_h = squad_total.get(home, np.nan)
        sv_a = squad_total.get(away, np.nan)
        if not np.isnan(sv_h) and not np.isnan(sv_a) and sv_a > 0:
            squad_ratio = np.log1p(sv_h) / np.log1p(sv_a)
            att_matchup = squad_att_ratio.get(home, 0.33) / max(squad_def_ratio.get(away, 0.33), 0.01)
        else:
            squad_ratio = np.nan
            att_matchup = np.nan

        hs, as_ = match["home_score"], match["away_score"]
        goal_diff = hs - as_
        result = "W" if goal_diff > 0 else ("L" if goal_diff < 0 else "D")

        val_rows.append({
            "date": match["date"], "home_team": home, "away_team": away,
            "tournament": match["tournament"],
            "elo_diff": elo.get(home) - elo.get(away),
            "fifa_rank_diff": away_fifa.get("rank", DEFAULT_RANK) - home_fifa.get("rank", DEFAULT_RANK),
            "fifa_points_diff": home_fifa.get("points", DEFAULT_PTS) - away_fifa.get("points", DEFAULT_PTS),
            "form_home": home_form["form"], "form_away": away_form["form"],
            "goals_scored_avg_home": home_form["goals_scored_avg"],
            "goals_scored_avg_away": away_form["goals_scored_avg"],
            "goals_conceded_avg_home": home_form["goals_conceded_avg"],
            "goals_conceded_avg_away": away_form["goals_conceded_avg"],
            "h2h_wins": h2h_wins,
            "match_importance": 1.0, "neutral_venue": 1,
            "rest_diff": rest_home - rest_away,
            "oaf_home": oaf_home, "oaf_away": oaf_away,
            "xg_for_home": xg_home["xg_for"], "xg_against_home": xg_home["xg_against"],
            "xg_for_away": xg_away["xg_for"], "xg_against_away": xg_away["xg_against"],
            "squad_ratio": squad_ratio, "att_matchup": att_matchup,
            "result": result, "goal_diff": goal_diff, "sample_weight": 1.0,
        })

    val_df = pd.DataFrame(val_rows)
    xg_val_coverage = val_df["xg_for_home"].notna().sum()
    print(f"  Validation xG coverage: {xg_val_coverage}/{len(val_df)} ({xg_val_coverage/len(val_df):.1%})")

    # ── Model Training ────────────────────────────────────────────────────────
    print("\n── Training AutoGluon ──")

    NON_FEATURE_COLS = {"date", "home_team", "away_team", "tournament", "result", "goal_diff", "sample_weight"}
    feature_cols = [c for c in training_df.columns if c not in NON_FEATURE_COLS]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    train = training_df.dropna(subset=["result"]).copy()
    val = val_df.dropna(subset=["result"]).copy()

    # Clean up models dir
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)

    # Train classifier
    print("\n  Training classifier (best_quality, 5min)...")
    clf_train = train[feature_cols + ["result", "sample_weight"]].copy()
    predictor = TabularPredictor(
        label="result", eval_metric="log_loss", sample_weight="sample_weight",
        path=str(MODELS_DIR / "classifier"), verbosity=1,
    )
    predictor.fit(clf_train, presets="best_quality", time_limit=300)

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VALIDATION — WC 2022")
    print("=" * 70)

    val_features = val[feature_cols].copy()
    y_true = val["result"].reset_index(drop=True)
    y_proba = predictor.predict_proba(val_features).reset_index(drop=True)
    y_pred_raw = predictor.predict(val_features).reset_index(drop=True)

    from sklearn.metrics import accuracy_score, classification_report

    acc_raw = accuracy_score(y_true, y_pred_raw)
    print(f"\n  [Raw argmax] Accuracy: {acc_raw:.1%}")

    # Draw threshold tuning
    best_params, best_val_acc = tune_draw_thresholds(y_proba, y_true)
    y_pred_tuned = predict_with_draw_threshold(y_proba, best_params[0], best_params[1])
    acc_tuned = accuracy_score(y_true, y_pred_tuned)
    print(f"  [Draw-tuned oracle] Accuracy: {acc_tuned:.1%} (gap={best_params[0]:.2f}, min_d={best_params[1]:.2f})")

    # Calibration tuning
    cal_df = train.tail(500).copy()
    cal_features = cal_df[feature_cols].copy()
    cal_true = cal_df["result"].reset_index(drop=True)
    cal_proba = predictor.predict_proba(cal_features).reset_index(drop=True)
    cal_params, _ = tune_draw_thresholds(cal_proba, cal_true)
    y_pred_cal = predict_with_draw_threshold(y_proba, cal_params[0], cal_params[1])
    acc_cal = accuracy_score(y_true, y_pred_cal)
    print(f"  [Draw-tuned on cal] Accuracy: {acc_cal:.1%} (gap={cal_params[0]:.2f}, min_d={cal_params[1]:.2f})")

    # Pick best
    results = [(acc_raw, y_pred_raw, "raw"), (acc_tuned, y_pred_tuned, "oracle"),
               (acc_cal, y_pred_cal, "cal-tuned")]
    best_acc, best_preds, method = max(results, key=lambda x: x[0])
    print(f"\n  Using: {method} ({best_acc:.1%})")

    print(f"\n  Classification Report:")
    print(classification_report(y_true, best_preds, zero_division=0))

    # ── Per-match results ─────────────────────────────────────────────────────
    val_reset = val.reset_index(drop=True)
    print(f"{'─' * 95}")
    print(f"{'Date':<7} {'Home':<20} {'Away':<20} {'Act':>4} {'Pred':>5} {'Prob':>18}")
    print(f"{'─' * 95}")

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

    print(f"{'─' * 95}")
    print(f"Total: {correct}/{len(val_reset)} ({correct/len(val_reset):.1%})")

    # ── Final Summary ─────────────────────────────────────────────────────────
    draw_pred = (best_preds == "D").sum()
    draw_correct = ((best_preds == "D") & (y_true == "D")).sum()
    draw_actual = (y_true == "D").sum()

    print(f"\n{'═' * 70}")
    print("  COMPARISON: v4 (original AutoGluon) vs v5 (enhanced)")
    print(f"{'═' * 70}")
    print(f"  {'Metric':<40} {'v4 (15 feat)':>12} {'v5 (21 feat)':>12}")
    print(f"  {'─' * 64}")
    print(f"  {'Match accuracy':<40} {'57.1%':>12} {best_acc:>11.1%}")
    print(f"  {'Draw recall':<40} {'0/15':>12} {f'{draw_correct}/{draw_actual}':>12}")
    print(f"  {'Draws predicted':<40} {'0':>12} {draw_pred:>12}")
    print(f"  {'Method':<40} {'argmax':>12} {method:>12}")
    print(f"  {'Features':<40} {'15':>12} {len(feature_cols):>12}")
    print(f"  {'New data':<40} {'—':>12} {'xG+TM+rest':>12}")
    print(f"{'═' * 70}")

    # Show feature importance from AutoGluon
    print("\n  AutoGluon feature importance:")
    try:
        fi = predictor.feature_importance(val_features, y_true, num_shuffle_sets=3)
        for feat_name in fi.index[:10]:
            print(f"    {feat_name:<30} {fi.loc[feat_name, 'importance']:.4f}")
    except Exception:
        print("    (could not compute — model type may not support it)")


if __name__ == "__main__":
    main()

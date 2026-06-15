"""Build training_set.csv from raw data with updated sample weights.

Weight scheme:
- 2023 matches: 0.15
- 2024 matches: 0.3
- 2025 matches: 0.6
- 2026 non-WC matches: 1.0
- 2026 FIFA World Cup matches: 2.0  (double weight for actual WC games)
- June 2026 non-WC matches: 1.5  (boost for this month's form)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "North Macedonia": "North Macedonia",
    "USA": "United States",
    "US Virgin Islands": "United States Virgin Islands",
    "Cabo Verde": "Cape Verde",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Eswatini": "Swaziland",
    "Timor-Leste": "East Timor",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
}

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


def normalize_names(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(NAME_MAP)
    return df


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


def get_sample_weight(match_date: pd.Timestamp, tournament: str) -> float:
    """Recency + tournament weight. World Cup and June 2026 get extra boost."""
    year = match_date.year
    month = match_date.month

    if year <= 2022:
        return 0.0
    elif year == 2023:
        base = 0.15
    elif year == 2024:
        base = 0.3
    elif year == 2025:
        base = 0.6
    else:  # 2026
        base = 1.0

    # Boost for June 2026 (current month form)
    if year == 2026 and month == 6:
        base = 1.5

    # Extra boost for actual FIFA World Cup matches
    if "FIFA World Cup" == tournament:
        base = max(base, 2.0)

    return base


def compute_rolling_features(matches_df: pd.DataFrame, team: str, n: int = 10) -> dict:
    team_mask = (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    team_matches = matches_df[team_mask].tail(n)

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


def compute_h2h(matches_df: pd.DataFrame, home_team: str, away_team: str, n: int = 5) -> float:
    mask = (
        ((matches_df["home_team"] == home_team) & (matches_df["away_team"] == away_team))
        | ((matches_df["home_team"] == away_team) & (matches_df["away_team"] == home_team))
    )
    h2h_matches = matches_df[mask].tail(n)

    if h2h_matches.empty:
        return 0.5

    wins = 0
    for _, m in h2h_matches.iterrows():
        if m["home_team"] == home_team:
            hs, as_ = m["home_score"], m["away_score"]
        else:
            hs, as_ = m["away_score"], m["home_score"]
        if hs > as_:
            wins += 1

    return wins / len(h2h_matches)


def main():
    print("Loading raw data...")
    matches = pd.read_csv(DATA_DIR / "matches.csv", parse_dates=["date"])
    fifa_rankings = pd.read_csv(DATA_DIR / "fifa_rankings.csv")
    elo_ratings = pd.read_csv(DATA_DIR / "elo_ratings.csv")
    market_values = pd.read_csv(DATA_DIR / "market_values.csv")

    # Filter to 2023+ only
    matches = matches[matches["date"] >= "2023-01-01"].copy()

    # Normalize names
    matches = normalize_names(matches, ["home_team", "away_team"])
    fifa_rankings = normalize_names(fifa_rankings, ["team"])
    elo_ratings = normalize_names(elo_ratings, ["team"])
    market_values = normalize_names(market_values, ["team"])

    print(f"Matches: {len(matches):,} (2023-2026)")
    print(f"By year:\n{matches['date'].dt.year.value_counts().sort_index()}")

    # Build lookups
    fifa_lookup = {
        row["team"]: {"rank": row["rank"], "points": row["points"], "confederation": row["confederation"]}
        for _, row in fifa_rankings.iterrows()
    }
    elo_lookup = {row["team"]: row["elo_rating"] for _, row in elo_ratings.iterrows()}
    market_lookup = {row["team"]: row["squad_total_value_eur"] for _, row in market_values.iterrows()}

    # Sort chronologically
    matches = matches.sort_values("date").reset_index(drop=True)

    DEFAULT_RANK = 100
    DEFAULT_PTS = 1300
    DEFAULT_ELO = 1500
    DEFAULT_CONF = "UNK"

    rows = []
    total = len(matches)

    for i, match in matches.iterrows():
        if i % 500 == 0:
            print(f"  processing {i:,} / {total:,}...")

        home, away = match["home_team"], match["away_team"]
        prior = matches.iloc[:i]

        home_form = compute_rolling_features(prior, home)
        away_form = compute_rolling_features(prior, away)
        h2h_wins = compute_h2h(prior, home, away)

        home_fifa = fifa_lookup.get(home, {})
        away_fifa = fifa_lookup.get(away, {})

        home_rank = home_fifa.get("rank", DEFAULT_RANK)
        away_rank = away_fifa.get("rank", DEFAULT_RANK)
        home_pts = home_fifa.get("points", DEFAULT_PTS)
        away_pts = away_fifa.get("points", DEFAULT_PTS)
        home_conf = home_fifa.get("confederation", DEFAULT_CONF)
        away_conf = away_fifa.get("confederation", DEFAULT_CONF)

        home_elo = elo_lookup.get(home, DEFAULT_ELO)
        away_elo = elo_lookup.get(away, DEFAULT_ELO)

        home_mkt = market_lookup.get(home, None)
        away_mkt = market_lookup.get(away, None)
        if home_mkt and away_mkt and away_mkt > 0:
            squad_value_ratio = home_mkt / away_mkt
        else:
            squad_value_ratio = 1.0

        hs, as_ = match["home_score"], match["away_score"]
        goal_diff = hs - as_
        if goal_diff > 0:
            result = "W"
        elif goal_diff < 0:
            result = "L"
        else:
            result = "D"

        rows.append({
            "date": match["date"],
            "home_team": home,
            "away_team": away,
            "tournament": match["tournament"],
            "elo_diff": home_elo - away_elo,
            "fifa_rank_diff": away_rank - home_rank,
            "fifa_points_diff": home_pts - away_pts,
            "squad_value_ratio": squad_value_ratio,
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
            "neutral_venue": int(match["neutral"]) if pd.notna(match["neutral"]) else 0,
            "result": result,
            "goal_diff": goal_diff,
            "sample_weight": get_sample_weight(match["date"], match["tournament"]),
        })

    training_df = pd.DataFrame(rows)

    out_path = OUTPUT_DIR / "training_set.csv"
    training_df.to_csv(out_path, index=False)

    print(f"\nSaved → {out_path}")
    print(f"Shape: {training_df.shape}")
    print(f"\nResult distribution:\n{training_df['result'].value_counts()}")
    print(f"\nSample weight distribution:\n{training_df['sample_weight'].value_counts().sort_index()}")
    print(f"\nWorld Cup matches in training set:")
    wc = training_df[training_df["tournament"] == "FIFA World Cup"]
    print(f"  Total: {len(wc)}, With scores: {wc['goal_diff'].notna().sum()}")


if __name__ == "__main__":
    main()

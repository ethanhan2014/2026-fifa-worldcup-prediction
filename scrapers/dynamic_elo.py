"""Compute dynamic ELO ratings from international match results.

Seeds from 2014 onwards, then updates through all available matches.
Produces elo_ratings.csv compatible with the existing pipeline format.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = DATA_DIR / "elo_ratings.csv"
MATCH_SOURCE = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

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

K_FACTOR = 40
HOME_ADVANTAGE = 80
INITIAL_RATING = 1500
SEED_START = "2014-01-01"


class ELO:
    def __init__(self):
        self.ratings = {}

    def get(self, team):
        return self.ratings.get(team, INITIAL_RATING)

    def update(self, home, away, home_score, away_score, neutral=False):
        ra = self.get(home)
        rb = self.get(away)
        ra_adj = ra if neutral else ra + HOME_ADVANTAGE

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

        self.ratings[home] = ra + K_FACTOR * gd_mult * (sa - ea)
        self.ratings[away] = rb + K_FACTOR * gd_mult * (sb - eb)


def scrape() -> pd.DataFrame:
    """Compute ELO ratings from all international matches since 2014."""
    print("Downloading international match results...")
    resp = requests.get(MATCH_SOURCE, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(pd.io.common.StringIO(resp.text), parse_dates=["date"])

    # Normalize team names
    for col in ["home_team", "away_team"]:
        df[col] = df[col].replace(NAME_MAP)

    # Use matches from 2014 onwards for ELO computation
    df = df[df["date"] >= SEED_START].sort_values("date").reset_index(drop=True)
    print(f"Processing {len(df)} matches ({df['date'].min().date()} to {df['date'].max().date()})...")

    elo = ELO()
    for _, m in df.iterrows():
        neutral = bool(m["neutral"]) if pd.notna(m.get("neutral")) else False
        elo.update(m["home_team"], m["away_team"], m["home_score"], m["away_score"], neutral)

    # Build output sorted by rating descending
    rows = []
    for rank, (team, rating) in enumerate(
        sorted(elo.ratings.items(), key=lambda x: -x[1]), start=1
    ):
        rows.append({
            "rank": rank,
            "team": team,
            "elo_rating": int(round(rating)),
            "date": df["date"].max().strftime("%Y-%m-%d"),
        })

    result = pd.DataFrame(rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"ELO ratings: {len(result)} teams")
    print(f"Top 10:")
    for _, r in result.head(10).iterrows():
        print(f"  {r['rank']:>3}. {r['team']:<25} {r['elo_rating']}")

    return result


if __name__ == "__main__":
    scrape()

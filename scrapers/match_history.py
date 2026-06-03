"""Download international match results and filter to 2022-01-01 onwards."""

from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
SOURCE_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
OUTPUT_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score",
                  "tournament", "city", "country", "neutral"]


def scrape() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = DATA_DIR / "_results_full.csv"

    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()
    tmp_path.write_bytes(response.content)

    df = pd.read_csv(tmp_path, parse_dates=["date"])
    tmp_path.unlink()

    filtered = df[df["date"] >= "2022-01-01"][OUTPUT_COLUMNS].copy()
    filtered.to_csv(DATA_DIR / "matches.csv", index=False)

    print(f"Matches: {len(filtered)}")
    print(f"Date range: {filtered['date'].min().date()} to {filtered['date'].max().date()}")

    return filtered


if __name__ == "__main__":
    scrape()

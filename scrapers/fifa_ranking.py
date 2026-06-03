"""Fetch FIFA Men's World Rankings from the official FIFA API."""

import csv
from pathlib import Path

import requests

API_URL = (
    "https://api.fifa.com/api/v3/fifarankings/rankings/live"
    "?gender=1&sportType=0&language=en"
)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = DATA_DIR / "fifa_rankings.csv"

COLUMNS = [
    "rank",
    "prev_rank",
    "team",
    "country_code",
    "confederation",
    "points",
    "prev_points",
    "ranking_movement",
    "rated_matches",
]


def _extract_team_name(team_name_list: list) -> str:
    for entry in team_name_list:
        if entry.get("Locale") == "en-GB":
            return entry["Description"]
    return team_name_list[0]["Description"] if team_name_list else ""


def scrape() -> list[dict]:
    print(f"Fetching FIFA rankings from {API_URL} ...")
    headers = {"User-Agent": "curl/8.5.0"}
    response = requests.get(API_URL, headers=headers, timeout=30)
    response.raise_for_status()

    results = response.json()["Results"]
    print(f"Received {len(results)} teams")

    rows = []
    for item in results:
        rows.append(
            {
                "rank": item["Rank"],
                "prev_rank": item["PrevRank"],
                "team": _extract_team_name(item["TeamName"]),
                "country_code": item["IdCountry"],
                "confederation": item["ConfederationName"],
                "points": item["TotalPoints"],
                "prev_points": item["PrevPoints"],
                "ranking_movement": item["RankingMovement"],
                "rated_matches": item["RatedMatches"],
            }
        )

    return rows


def save(rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    rows = scrape()
    save(rows)
    print(f"Top 5: {[r['team'] for r in rows[:5]]}")

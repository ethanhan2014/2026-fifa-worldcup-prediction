"""Fetch xG data from StatsBomb open data for international matches."""

import json
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "backtest_2022" / "raw"
OUTPUT_FILE = DATA_DIR / "xg_statsbomb.csv"

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

COMPETITIONS = [
    (43, 106, "FIFA World Cup 2022"),
    (43, 3, "FIFA World Cup 2018"),
    (55, 43, "UEFA Euro 2020"),
    (223, 282, "Copa America 2024"),  # bonus for future
    (55, 282, "UEFA Euro 2024"),       # bonus for future
]

NAME_MAP = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
}


def normalize_name(name):
    return NAME_MAP.get(name, name)


def fetch_match_xg(match_id):
    """Fetch events for a match and compute team xG totals."""
    url = f"{BASE}/events/{match_id}.json"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    events = json.loads(r.text)
    shots = [e for e in events if e.get("type", {}).get("name") == "Shot"]

    home_xg, away_xg = 0.0, 0.0
    home_shots, away_shots = 0, 0
    home_sot, away_sot = 0, 0

    for shot in shots:
        xg = shot.get("shot", {}).get("statsbomb_xg", 0)
        team = shot.get("team", {}).get("name", "")
        outcome = shot.get("shot", {}).get("outcome", {}).get("name", "")
        is_on_target = outcome in ("Goal", "Saved", "Saved to Post")

        # Determine if home or away based on possession_team
        if shot.get("possession_team", {}).get("name") == shot.get("team", {}).get("name"):
            pass  # already using team name to split

        # We need match context to know home/away — return per-team
        if "home_xg" not in dir():
            pass

    # Split by team name — we'll map later
    team_xg = {}
    team_shots_count = {}
    team_sot_count = {}

    for shot in shots:
        team = shot.get("team", {}).get("name", "")
        xg = shot.get("shot", {}).get("statsbomb_xg", 0)
        outcome = shot.get("shot", {}).get("outcome", {}).get("name", "")
        is_on_target = outcome in ("Goal", "Saved", "Saved to Post")

        team_xg[team] = team_xg.get(team, 0) + xg
        team_shots_count[team] = team_shots_count.get(team, 0) + 1
        if is_on_target:
            team_sot_count[team] = team_sot_count.get(team, 0) + 1

    return {
        "team_xg": team_xg,
        "team_shots": team_shots_count,
        "team_sot": team_sot_count,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for comp_id, season_id, comp_name in COMPETITIONS:
        print(f"\n── {comp_name} ──")
        url = f"{BASE}/matches/{comp_id}/{season_id}.json"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            print(f"  Failed to fetch matches: {r.status_code}")
            continue

        matches = json.loads(r.text)
        print(f"  {len(matches)} matches")

        for i, match in enumerate(matches):
            match_id = match["match_id"]
            home_name = normalize_name(match["home_team"]["home_team_name"])
            away_name = normalize_name(match["away_team"]["away_team_name"])
            date = match["match_date"]
            home_score = match["home_score"]
            away_score = match["away_score"]

            result = fetch_match_xg(match_id)
            if result is None:
                continue

            # Map team names to home/away
            home_xg = 0.0
            away_xg = 0.0
            home_shots = 0
            away_shots = 0
            home_sot = 0
            away_sot = 0

            for team, xg in result["team_xg"].items():
                norm_team = normalize_name(team)
                if norm_team == home_name:
                    home_xg = xg
                    home_shots = result["team_shots"].get(team, 0)
                    home_sot = result["team_sot"].get(team, 0)
                elif norm_team == away_name:
                    away_xg = xg
                    away_shots = result["team_shots"].get(team, 0)
                    away_sot = result["team_sot"].get(team, 0)

            rows.append({
                "date": date,
                "home_team": home_name,
                "away_team": away_name,
                "home_score": home_score,
                "away_score": away_score,
                "home_xg": round(home_xg, 3),
                "away_xg": round(away_xg, 3),
                "home_shots": home_shots,
                "away_shots": away_shots,
                "home_sot": home_sot,
                "away_sot": away_sot,
                "tournament": comp_name,
            })

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(matches)} processed")
            time.sleep(0.3)

        print(f"  Done: {len([r for r in rows if r['tournament'] == comp_name])} matches")

    # Write CSV
    import csv
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} matches to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

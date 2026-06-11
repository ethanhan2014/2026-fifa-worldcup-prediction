"""Fetch 2018-2022 data for backtesting the model against WC 2022 results.

Datasets:
- matches.csv: International matches from 2018-01-01 to 2022-11-20 (training)
- wc2022_matches.csv: WC 2022 group stage + knockouts (validation)
- fifa_rankings.csv: FIFA rankings from 2022-10-06 (last before WC 2022)
- elo_ratings.csv: ELO ratings from 2022-11-20 (day before WC 2022 start)
"""

import csv
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "backtest_2022" / "raw"

MATCH_SOURCE = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FIFA_RANKING_SOURCE = "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv"

WC2022_START = "2022-11-21"
WC2022_END = "2022-12-18"
TRAINING_START = "2018-01-01"
TRAINING_END = "2022-11-20"
FIFA_RANKING_DATE = "2022-10-06"
ELO_DATE = date(2022, 11, 20)


def fetch_matches():
    """Download matches, split into training (2018-2022 pre-WC) and validation (WC 2022)."""
    print("Fetching match history...")
    response = requests.get(MATCH_SOURCE, timeout=60)
    response.raise_for_status()

    tmp_path = DATA_DIR / "_results_full.csv"
    tmp_path.write_bytes(response.content)
    df = pd.read_csv(tmp_path, parse_dates=["date"])
    tmp_path.unlink()

    columns = ["date", "home_team", "away_team", "home_score", "away_score",
               "tournament", "city", "country", "neutral"]

    # Training set: 2018-01-01 to 2022-11-20
    training = df[(df["date"] >= TRAINING_START) & (df["date"] <= TRAINING_END)][columns].copy()
    training.to_csv(DATA_DIR / "matches.csv", index=False)
    print(f"  Training matches: {len(training)} ({training['date'].min().date()} to {training['date'].max().date()})")

    # Validation set: WC 2022 matches only
    wc2022 = df[(df["date"] >= WC2022_START) & (df["date"] <= WC2022_END) &
                (df["tournament"] == "FIFA World Cup")][columns].copy()
    wc2022.to_csv(DATA_DIR / "wc2022_matches.csv", index=False)
    print(f"  WC 2022 matches: {len(wc2022)} ({wc2022['date'].min().date()} to {wc2022['date'].max().date()})")

    return training, wc2022


def fetch_fifa_rankings():
    """Download FIFA rankings from 2022-10-06 (last ranking before WC 2022)."""
    print(f"Fetching FIFA rankings ({FIFA_RANKING_DATE})...")
    response = requests.get(FIFA_RANKING_SOURCE, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(pd.io.common.StringIO(response.text))
    ranking = df[df["date"] == FIFA_RANKING_DATE].copy()

    # Add rank column based on points
    ranking = ranking.sort_values("total_points", ascending=False).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1

    output = ranking[["rank", "team", "team_short", "total_points", "date"]].rename(
        columns={"total_points": "points", "team_short": "country_code"}
    )
    output.to_csv(DATA_DIR / "fifa_rankings.csv", index=False)
    print(f"  Rankings: {len(output)} teams")
    print(f"  Top 5: {output.head(5)['team'].tolist()}")

    return output


def fetch_elo_ratings():
    """Fetch ELO ratings from 2022-11-20 (day before WC 2022 start)."""
    print(f"Fetching ELO ratings ({ELO_DATE})...")

    _lib_path = os.path.expanduser("~/lib-deps/usr/lib/x86_64-linux-gnu")
    if _lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = _lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    from cloakbrowser import launch
    from scrapling.parser import Adaptor

    url = f"https://www.international-football.net/elo-ratings-table?year={ELO_DATE.year}&month={ELO_DATE.month:02d}&day={ELO_DATE.day:02d}&confed=&old-team="
    print(f"  URL: {url}")

    browser = launch(headless=True)
    try:
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
    finally:
        browser.close()

    parsed = Adaptor(html, auto_match=False)
    rows = []
    for tr in parsed.css("table tr"):
        th = tr.css("th")
        tds = tr.css("td")
        if not th or len(tds) < 3:
            continue
        rank_el = th[0].css("strong")
        if not rank_el:
            continue
        rank_text = rank_el[0].text.strip().rstrip(".")
        if not rank_text.isdigit():
            continue
        team = tds[1].text.strip()
        elo = tds[2].text.strip()
        if not team or not elo:
            continue
        rows.append({
            "rank": int(rank_text),
            "team": team,
            "elo_rating": elo,
            "date": ELO_DATE.isoformat(),
        })

    with open(DATA_DIR / "elo_ratings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "team", "elo_rating", "date"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ELO ratings: {len(rows)} teams")
    if rows:
        print(f"  Top 5: {[r['team'] for r in rows[:5]]}")

    return rows


def main():
    print("=" * 60)
    print("Backtest 2022 — Data Collection")
    print("Training: 2018-01-01 to 2022-11-20")
    print("Validation: WC 2022 (Nov 21 – Dec 18)")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fetch_matches()
    fetch_fifa_rankings()
    fetch_elo_ratings()

    print("\n" + "=" * 60)
    print("Done! Data saved to:", DATA_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()

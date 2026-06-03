"""Fetch World Cup 2026 injury data.

Primary source: API-Football (requires API_FOOTBALL_KEY env var).
Fallback: scrape ESPN injury tracker via CloakBrowser + Scrapling.
If both fail, writes an empty CSV with headers only.
"""

import csv
import os
import re
from pathlib import Path

import requests

API_FOOTBALL_URL = "https://v3.football.api-sports.io"
ESPN_URL = "https://www.espn.com/soccer/story/_/id/48572979/2026-fifa-world-cup-injuries-tracker"

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = DATA_DIR / "injuries.csv"

COLUMNS = ["team", "player", "position", "injury_type", "status"]


def _write_csv(rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _fetch_api_football(api_key: str) -> list[dict]:
    headers = {"x-apisports-key": api_key}
    resp = requests.get(
        f"{API_FOOTBALL_URL}/injuries",
        headers=headers,
        params={"league": "1", "season": "2026"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json().get("response", [])
    rows = []
    for item in data:
        player = item.get("player", {})
        team = item.get("team", {})
        rows.append({
            "team": team.get("name", ""),
            "player": player.get("name", ""),
            "position": player.get("position", ""),
            "injury_type": item.get("fixture", {}).get("reason", ""),
            "status": player.get("type", ""),
        })
    return rows


def _fetch_espn_fallback() -> list[dict]:
    from cloakbrowser import launch
    from scrapling.parser import Adaptor

    browser = launch(headless=True)
    try:
        page = browser.new_page()
        page.goto(ESPN_URL, timeout=45000, wait_until="domcontentloaded")
        html = page.content()
    finally:
        try:
            browser.close()
        except Exception:
            pass

    parsed = Adaptor(html, auto_match=False)

    rows = []
    # ESPN injury tracker articles list injuries in structured paragraphs.
    # Look for lines containing injury keywords like "hamstring", "knee", "out",
    # "doubtful", "questionable", "suspended" alongside a player name.
    INJURY_KEYWORDS = re.compile(
        r"\b(hamstring|knee|ankle|calf|thigh|muscle|back|hip|shoulder|"
        r"foot|groin|torn|fracture|strain|sprain|surgery|suspended|"
        r"out|doubtful|questionable|unavailable|injured|injury|fitness)\b",
        re.IGNORECASE,
    )
    # Player name: two or more Title-Case words (2-20 chars each)
    NAME_RE = re.compile(r"\b([A-Z][a-z]{1,19}(?:\s[A-Z][a-z]{1,19})+)\b")

    text_blocks = parsed.css("p, li")
    seen: set[str] = set()
    for el in text_blocks:
        text = el.text if hasattr(el, "text") else str(el)
        if not INJURY_KEYWORDS.search(text):
            continue
        for name_match in NAME_RE.finditer(text):
            player = name_match.group(1).strip()
            # Skip very common non-player title-case phrases
            if player in seen or player in {
                "World Cup", "FIFA World", "United States", "South Korea",
                "Saudi Arabia", "Costa Rica",
            }:
                continue
            seen.add(player)
            # Grab the injury keyword nearest to the name
            inj_match = INJURY_KEYWORDS.search(text)
            rows.append({
                "team": "",
                "player": player,
                "position": "",
                "injury_type": inj_match.group(1).lower() if inj_match else "",
                "status": "",
            })

    return rows


def scrape() -> None:
    api_key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    rows: list[dict] = []

    if api_key:
        try:
            rows = _fetch_api_football(api_key)
            print(f"API-Football: fetched {len(rows)} injury records")
        except Exception as exc:
            print(f"API-Football failed: {exc}; trying ESPN fallback")

    if not rows:
        try:
            rows = _fetch_espn_fallback()
            print(f"ESPN fallback: extracted {len(rows)} injury records")
        except Exception as exc:
            print(f"ESPN fallback failed: {exc}; writing empty CSV")

    _write_csv(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    scrape()

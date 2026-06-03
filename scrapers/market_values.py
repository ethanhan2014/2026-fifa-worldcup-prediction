"""Scrape national team squad market values from Transfermarkt."""

import csv
import re
import time
from pathlib import Path

from cloakbrowser import launch
from scrapling.parser import Adaptor

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = DATA_DIR / "market_values.csv"

BASE_URL = "https://www.transfermarkt.com"

TEAMS = [
    ("Argentina", "/argentinien/startseite/verein/3437"),
    ("Brazil", "/brasilien/startseite/verein/3439"),
    ("England", "/england/startseite/verein/3299"),
    ("France", "/frankreich/startseite/verein/3377"),
    ("Spain", "/spanien/startseite/verein/3375"),
    ("Germany", "/deutschland/startseite/verein/3262"),
    ("Portugal", "/portugal/startseite/verein/3300"),
    ("Netherlands", "/niederlande/startseite/verein/3379"),
    ("Belgium", "/belgien/startseite/verein/3382"),
    ("Italy", "/italien/startseite/verein/3376"),
    ("Croatia", "/kroatien/startseite/verein/3556"),
    ("Morocco", "/marokko/startseite/verein/3575"),
    ("Colombia", "/kolumbien/startseite/verein/3816"),
    ("Mexico", "/mexiko/startseite/verein/6303"),
    ("USA", "/vereinigte-staaten/startseite/verein/3505"),
    ("Uruguay", "/uruguay/startseite/verein/3449"),
    ("Japan", "/japan/startseite/verein/3435"),
    ("Senegal", "/senegal/startseite/verein/3499"),
    ("Switzerland", "/schweiz/startseite/verein/3384"),
    ("Denmark", "/danemark/startseite/verein/3436"),
]

COLUMNS = [
    "team",
    "squad_total_value_eur",
    "avg_player_value_eur",
    "top_player_value_eur",
    "num_players",
]


def _parse_value_string(text: str) -> float | None:
    """Convert strings like '€1.05bn', '€85.00m', '€750k' to EUR float."""
    if not text:
        return None
    text = text.strip().replace(",", ".")
    # Extract numeric part and suffix
    m = re.search(r"([\d.]+)\s*(bn|m|k)?", text, re.IGNORECASE)
    if not m:
        return None
    number = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    if suffix == "bn":
        return number * 1_000_000_000
    if suffix == "m":
        return number * 1_000_000
    if suffix == "k":
        return number * 1_000
    return number


def _extract_total_value(parsed: Adaptor) -> float | None:
    """Extract total squad value from the data-header market value wrapper."""
    elems = parsed.css(".data-header__market-value-wrapper")
    if not elems:
        return None
    raw = elems[0].get_all_text()
    # get_all_text returns lines like: '€\n799.50\nm\n\nTotal market value'
    # Reconstruct the value string from the parts
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    # lines[0] = currency symbol, lines[1] = number, lines[2] = suffix (m/bn/k)
    if len(lines) >= 3 and lines[2] in ("m", "bn", "k"):
        value_str = lines[1] + lines[2]
    elif len(lines) >= 2:
        value_str = lines[1]
    else:
        return None
    return _parse_value_string(value_str)


def _extract_player_values(parsed: Adaptor) -> list[float]:
    """Extract individual player market values from the squad table."""
    links = parsed.css("table.items td.rechts.hauptlink a")
    values = []
    for a in links:
        raw = a.get_all_text().strip()
        if not raw or raw == "-":
            continue
        val = _parse_value_string(raw)
        if val is not None:
            values.append(val)
    return values


def _scrape_team(page, team: str, path: str) -> dict:
    url = BASE_URL + path
    print(f"  Fetching {team}: {url}")
    try:
        page.goto(url, timeout=30000)
        time.sleep(2)
        html = page.content()
        parsed = Adaptor(html, auto_match=False)

        player_values = _extract_player_values(parsed)
        total_value = _extract_total_value(parsed)

        if total_value is None and player_values:
            total_value = sum(player_values)

        num_players = len(player_values) if player_values else None
        avg_value = (sum(player_values) / num_players) if player_values else None
        top_value = max(player_values) if player_values else None

        print(
            f"    total={total_value}, players={num_players}, top={top_value}"
        )
        return {
            "team": team,
            "squad_total_value_eur": total_value,
            "avg_player_value_eur": avg_value,
            "top_player_value_eur": top_value,
            "num_players": num_players,
        }
    except Exception as e:
        print(f"    ERROR scraping {team}: {e}")
        return {
            "team": team,
            "squad_total_value_eur": None,
            "avg_player_value_eur": None,
            "top_player_value_eur": None,
            "num_players": None,
        }


def scrape() -> list[dict]:
    print("Launching browser...")
    browser = launch(headless=True, humanize=False)
    try:
        page = browser.new_page()
        rows = []
        for i, (team, path) in enumerate(TEAMS):
            row = _scrape_team(page, team, path)
            rows.append(row)
            if i < len(TEAMS) - 1:
                time.sleep(3)
        return rows
    finally:
        browser.close()


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
    successful = sum(1 for r in rows if r["squad_total_value_eur"] is not None)
    print(f"Done: {successful}/{len(rows)} teams scraped successfully")

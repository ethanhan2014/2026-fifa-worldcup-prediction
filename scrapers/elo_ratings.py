"""Scrape ELO ratings from international-football.net."""

import csv
import os
import sys
from datetime import date
from pathlib import Path

# Add lib path for browser shared libs when running standalone
_LIB_PATH = os.path.expanduser("~/lib-deps/usr/lib/x86_64-linux-gnu")
if _LIB_PATH not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _LIB_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")

from cloakbrowser import launch
from scrapling.parser import Adaptor

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = DATA_DIR / "elo_ratings.csv"
DEBUG_FILE = DATA_DIR / "elo_debug.html"

BASE_URL = "https://www.international-football.net/elo-ratings-table"


def build_url(d: date) -> str:
    return f"{BASE_URL}?year={d.year}&month={d.month:02d}&day={d.day:02d}&confed=&old-team="


def parse_table(html: str, fetch_date: date) -> list[dict]:
    parsed = Adaptor(html, auto_match=False)
    rows = []

    # The page has many tables of 10 rows each.
    # Each row: <th><strong>RANK</strong>.</th> <td>(flag img)</td> <td>TEAM</td> <td>ELO</td>
    for tr in parsed.css("table tr"):
        th = tr.css("th")
        tds = tr.css("td")
        if not th or len(tds) < 3:
            continue
        # Rank is inside <strong> inside <th>
        rank_el = th[0].css("strong")
        if not rank_el:
            continue
        rank_text = rank_el[0].text.strip().rstrip(".")
        if not rank_text.isdigit():
            continue
        team = tds[1].text.strip()   # tds[0] is flag image
        elo = tds[2].text.strip()
        if not team or not elo:
            continue
        rows.append({
            "rank": int(rank_text),
            "team": team,
            "elo_rating": elo,
            "date": fetch_date.isoformat(),
        })

    return rows


def scrape(fetch_date: date | None = None) -> list[dict]:
    if fetch_date is None:
        fetch_date = date.today()

    url = build_url(fetch_date)
    print(f"Fetching: {url}", file=sys.stderr)

    browser = launch(headless=True)
    try:
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
    finally:
        browser.close()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = parse_table(html, fetch_date)

    if not rows:
        print("No rows parsed — saving debug HTML", file=sys.stderr)
        DEBUG_FILE.write_text(html, encoding="utf-8")
        print(f"Debug HTML saved to {DEBUG_FILE}", file=sys.stderr)

        # Try fallback: look for any table cells with numeric content
        parsed = Adaptor(html, auto_match=False)
        all_tables = parsed.css("table")
        print(f"Tables found: {len(all_tables)}", file=sys.stderr)
        for i, t in enumerate(all_tables[:3]):
            trs = t.css("tr")
            print(f"  table[{i}]: {len(trs)} rows", file=sys.stderr)
            if trs:
                first_tds = trs[0].css("td,th")
                print(f"    first row cells: {[c.text.strip() for c in first_tds]}", file=sys.stderr)

    return rows


def write_csv(rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "team", "elo_rating", "date"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    rows = scrape()
    if rows:
        write_csv(rows)
        print(f"Success: {len(rows)} teams scraped")
        print(f"Sample: {rows[:3]}")
    else:
        print("ERROR: No data scraped", file=sys.stderr)
        sys.exit(1)

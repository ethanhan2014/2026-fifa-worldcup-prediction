# 2026 FIFA World Cup Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a prediction system that collects football data from 5 sources, engineers features, and trains AutoGluon models to predict 2026 World Cup match outcomes (W/D/L probabilities + expected scores).

**Architecture:** Modular scrapers write raw CSV to `data/raw/`. Jupyter notebooks handle exploration, feature engineering, model training, and prediction. CloakBrowser provides stealth browsing for anti-bot sites; Scrapling handles HTML parsing. AutoGluon trains both a W/D/L classifier and a goal-difference regressor.

**Tech Stack:** Python 3.12, AutoGluon Tabular, Scrapling, CloakBrowser, pandas, requests, Jupyter

---

## File Structure

```
~/2026-fifa-worldcup-prediction/
├── scrapers/
│   ├── __init__.py
│   ├── fifa_ranking.py          # FIFA API → data/raw/fifa_rankings.csv
│   ├── elo_ratings.py           # eloratings.net via CloakBrowser → data/raw/elo_ratings.csv
│   ├── match_history.py         # GitHub CSV download → data/raw/matches.csv
│   ├── market_values.py         # Transfermarkt via CloakBrowser → data/raw/market_values.csv
│   ├── injuries.py              # API-Football REST → data/raw/injuries.csv
│   └── run_all.py               # Orchestrates all scrapers
├── data/
│   ├── raw/                     # Scraper outputs
│   └── processed/               # Feature-engineered datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training.ipynb
│   └── 04_prediction.ipynb
├── models/                      # Saved AutoGluon models
├── requirements.txt
└── README.md
```

---

### Task 1: Project Setup & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `scrapers/__init__.py`
- Create: directory structure

- [ ] **Step 1: Create project directory structure**

```bash
cd ~/2026-fifa-worldcup-prediction
mkdir -p scrapers data/raw data/processed notebooks models
touch scrapers/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

```text
autogluon[tabular]
scrapling[all]
cloakbrowser
pandas
numpy
requests
jupyter
matplotlib
seaborn
scikit-learn
```

- [ ] **Step 3: Install dependencies**

```bash
cd ~/2026-fifa-worldcup-prediction
pip install -r requirements.txt
```

Expected: All packages install successfully. CloakBrowser will download its stealth Chromium binary (~200MB) on first use.

- [ ] **Step 4: Verify installations**

```bash
python3 -c "from scrapling.fetchers import Fetcher; print('scrapling OK')"
python3 -c "from cloakbrowser import launch; print('cloakbrowser OK')"
python3 -c "from autogluon.tabular import TabularPredictor; print('autogluon OK')"
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt scrapers/__init__.py
git commit -m "feat: project setup with dependencies"
```

---

### Task 2: FIFA Ranking Scraper

**Files:**
- Create: `scrapers/fifa_ranking.py`

The FIFA API at `api.fifa.com` returns JSON with all 211 teams. No browser needed — direct HTTP GET.

- [ ] **Step 1: Create the scraper**

```python
"""Scrape FIFA Men's World Ranking from api.fifa.com"""
import csv
import os
from datetime import date
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
FIFA_API_URL = "https://api.fifa.com/api/v3/fifarankings/rankings/live?gender=1&sportType=0&language=en"


def get_team_name(name_field):
    if isinstance(name_field, list) and name_field:
        return name_field[0].get("Description", "")
    return str(name_field)


def scrape():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://inside.fifa.com/fifa-world-ranking/men",
    }

    print("Fetching FIFA rankings from API...")
    resp = requests.get(FIFA_API_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    rankings = resp.json()["Results"]

    rows = []
    for r in rankings:
        rows.append({
            "rank": r["Rank"],
            "prev_rank": r["PrevRank"],
            "team": get_team_name(r["TeamName"]),
            "country_code": r["IdCountry"],
            "confederation": r["ConfederationName"],
            "points": round(r["TotalPoints"], 2),
            "prev_points": round(r["PrevPoints"], 2),
            "ranking_movement": r["RankingMovement"],
            "rated_matches": r["RatedMatches"],
        })

    output_path = DATA_DIR / "fifa_rankings.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} teams to {output_path}")
    return rows


if __name__ == "__main__":
    scrape()
```

- [ ] **Step 2: Run and verify**

```bash
cd ~/2026-fifa-worldcup-prediction
python3 scrapers/fifa_ranking.py
head -5 data/raw/fifa_rankings.csv
```

Expected: `data/raw/fifa_rankings.csv` with 211 rows, headers: rank, prev_rank, team, country_code, confederation, points, prev_points, ranking_movement, rated_matches.

- [ ] **Step 3: Commit**

```bash
git add scrapers/fifa_ranking.py
git commit -m "feat: FIFA ranking scraper via official API"
```

---

### Task 3: ELO Ratings Scraper

**Files:**
- Create: `scrapers/elo_ratings.py`

Uses CloakBrowser to render JS on `international-football.net/elo-ratings-table` (easier table structure than eloratings.net). Scrapling's `Adaptor` parses the HTML. Fetches the current snapshot.

- [ ] **Step 1: Create the scraper**

```python
"""Scrape ELO ratings from international-football.net using CloakBrowser + Scrapling"""
import csv
import time
from datetime import date
from pathlib import Path

from cloakbrowser import launch
from scrapling import Adaptor

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
BASE_URL = "https://www.international-football.net/elo-ratings-table?year={year}&month={month}&day={day}&confed=&old-team="


def fetch_elo_for_date(page, target_date):
    url = BASE_URL.format(year=target_date.year, month=target_date.month, day=target_date.day)
    page.goto(url, wait_until="networkidle", timeout=60000)
    time.sleep(2)
    return page.content()


def parse_elo_table(html, snapshot_date):
    parsed = Adaptor(html, auto_match=False)
    rows = []

    table_rows = parsed.css("table.table tbody tr")
    for row in table_rows:
        cells = row.css("td")
        if len(cells) < 4:
            continue

        rank_text = cells[0].text.strip()
        if not rank_text.isdigit():
            continue

        team = cells[1].text.strip()
        elo_text = cells[3].text.strip()

        try:
            elo = int(elo_text)
        except ValueError:
            continue

        rows.append({
            "rank": int(rank_text),
            "team": team,
            "elo_rating": elo,
            "date": snapshot_date.isoformat(),
        })

    return rows


def scrape():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Launching CloakBrowser...")
    browser = launch(headless=True)
    page = browser.new_page()

    all_rows = []
    today = date.today()

    # Fetch current ELO ratings
    print(f"Fetching ELO ratings for {today}...")
    html = fetch_elo_for_date(page, today)
    rows = parse_elo_table(html, today)
    all_rows.extend(rows)
    print(f"  Got {len(rows)} teams")

    browser.close()

    if not all_rows:
        print("ERROR: No ELO data scraped")
        return []

    output_path = DATA_DIR / "elo_ratings.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "team", "elo_rating", "date"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} rows to {output_path}")
    return all_rows


if __name__ == "__main__":
    scrape()
```

- [ ] **Step 2: Run and verify**

```bash
cd ~/2026-fifa-worldcup-prediction
python3 scrapers/elo_ratings.py
head -10 data/raw/elo_ratings.csv
```

Expected: `data/raw/elo_ratings.csv` with ~230 teams, columns: rank, team, elo_rating, date.

- [ ] **Step 3: If CSS selectors don't match, debug by saving the HTML**

Add a fallback: save the page HTML to `data/raw/elo_debug.html` and inspect the actual table structure. Adjust `parse_elo_table` selectors accordingly. The site uses a standard `<table>` with rows, but the exact class names may differ.

- [ ] **Step 4: Commit**

```bash
git add scrapers/elo_ratings.py
git commit -m "feat: ELO ratings scraper via CloakBrowser"
```

---

### Task 4: Match History Scraper

**Files:**
- Create: `scrapers/match_history.py`

Downloads the maintained CSV from GitHub (`martj42/international_results`). This dataset is updated regularly and covers through 2024. For 2025–2026 matches, we'll also check the Kaggle daily-updated dataset.

- [ ] **Step 1: Create the scraper**

```python
"""Download international match history from GitHub (martj42/international_results)"""
import pandas as pd
from datetime import date
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
GITHUB_CSV_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def scrape():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading match history from GitHub...")
    resp = requests.get(GITHUB_CSV_URL, timeout=60)
    resp.raise_for_status()

    # Save raw then filter
    raw_path = DATA_DIR / "matches_full.csv"
    raw_path.write_bytes(resp.content)

    # Load and filter to last 4 years
    df = pd.read_csv(raw_path, parse_dates=["date"])
    cutoff = pd.Timestamp("2022-01-01")
    df = df[df["date"] >= cutoff].copy()

    # Standardize columns
    df = df[["date", "home_team", "away_team", "home_score", "away_score",
             "tournament", "city", "country", "neutral"]].copy()

    output_path = DATA_DIR / "matches.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} matches (from {df['date'].min().date()} to {df['date'].max().date()})")
    print(f"Output: {output_path}")

    # Clean up full file
    raw_path.unlink()

    return df


if __name__ == "__main__":
    scrape()
```

- [ ] **Step 2: Run and verify**

```bash
cd ~/2026-fifa-worldcup-prediction
python3 scrapers/match_history.py
wc -l data/raw/matches.csv
head -5 data/raw/matches.csv
```

Expected: ~2000–3000 matches from 2022-01-01 onwards. Columns: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral.

- [ ] **Step 3: Commit**

```bash
git add scrapers/match_history.py
git commit -m "feat: match history scraper from GitHub dataset"
```

---

### Task 5: Transfermarkt Market Values Scraper

**Files:**
- Create: `scrapers/market_values.py`

Scrapes squad market values for the 48 World Cup qualified teams. Uses CloakBrowser for anti-bot evasion and Scrapling for parsing. Rate-limited to 1 request per 3 seconds.

- [ ] **Step 1: Create the scraper**

```python
"""Scrape national team squad market values from Transfermarkt via CloakBrowser + Scrapling"""
import csv
import time
from pathlib import Path

from cloakbrowser import launch
from scrapling import Adaptor

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# 2026 World Cup qualified teams with their Transfermarkt URLs
# Format: (team_name, transfermarkt_path)
WORLD_CUP_TEAMS = [
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
    ("Australia", "/australien/startseite/verein/3433"),
    ("Canada", "/kanada/startseite/verein/3510"),
    ("South Korea", "/sudkorea/startseite/verein/3589"),
    ("Ecuador", "/ecuador/startseite/verein/5750"),
    ("Saudi Arabia", "/saudi-arabien/startseite/verein/3807"),
    ("Qatar", "/katar/startseite/verein/14484"),
    ("Iran", "/iran/startseite/verein/3582"),
    ("Serbia", "/serbien/startseite/verein/3438"),
    ("Poland", "/polen/startseite/verein/3442"),
    ("Ukraine", "/ukraine/startseite/verein/3699"),
    ("Nigeria", "/nigeria/startseite/verein/3444"),
    ("Cameroon", "/kamerun/startseite/verein/3434"),
    ("Ghana", "/ghana/startseite/verein/3441"),
    ("Egypt", "/agypten/startseite/verein/3672"),
    ("Tunisia", "/tunesien/startseite/verein/3670"),
    ("Costa Rica", "/costa-rica/startseite/verein/8497"),
    ("Paraguay", "/paraguay/startseite/verein/3583"),
    ("Bolivia", "/bolivien/startseite/verein/3810"),
    ("Venezuela", "/venezuela/startseite/verein/3501"),
    ("Chile", "/chile/startseite/verein/3700"),
    ("Peru", "/peru/startseite/verein/3584"),
    ("Panama", "/panama/startseite/verein/3577"),
    ("Honduras", "/honduras/startseite/verein/3590"),
    ("Jamaica", "/jamaika/startseite/verein/3573"),
    ("Scotland", "/schottland/startseite/verein/3380"),
    ("Wales", "/wales/startseite/verein/3864"),
    ("Austria", "/osterreich/startseite/verein/3383"),
    ("Turkey", "/turkei/startseite/verein/3381"),
]


def parse_value(value_str):
    """Parse Transfermarkt value strings like '€1.05bn', '€85.00m', '€750k'"""
    if not value_str or value_str.strip() in ["-", "€-", ""]:
        return None
    value_str = value_str.replace("€", "").replace(",", "").strip().lower()
    multiplier = 1
    if value_str.endswith("bn"):
        multiplier = 1_000_000_000
        value_str = value_str[:-2]
    elif value_str.endswith("m"):
        multiplier = 1_000_000
        value_str = value_str[:-1]
    elif value_str.endswith("k") or value_str.endswith("th"):
        multiplier = 1_000
        value_str = value_str.rstrip("kth")
    try:
        return float(value_str) * multiplier
    except ValueError:
        return None


def scrape_team(page, team_name, path):
    """Scrape a single team's squad page."""
    url = f"https://www.transfermarkt.com{path}"
    page.goto(url, wait_until="domcontentloaded", timeout=45000)
    time.sleep(3)

    html = page.content()
    parsed = Adaptor(html, auto_match=False)

    # Look for total market value in the page header area
    total_value = None
    value_elements = parsed.css("[class*='market-value'], [class*='dataValue'], .data-header__market-value-wrapper")
    for el in value_elements:
        text = el.text.strip()
        val = parse_value(text)
        if val and val > 1_000_000:
            total_value = val
            break

    # Fallback: look for the total row in the squad table
    if total_value is None:
        tfoot = parsed.css("tfoot td, .transfer-zus498-gesamtmarktwert")
        for el in tfoot:
            val = parse_value(el.text.strip())
            if val and val > 1_000_000:
                total_value = val
                break

    # Count players in the squad table
    player_rows = parsed.css("table.items tbody tr.odd, table.items tbody tr.even")
    num_players = len(player_rows) if player_rows else 0

    # Extract individual player values for avg/top calculation
    player_values = []
    for row in (player_rows or []):
        value_cell = row.css("td.rechts.hauptlink a, td:last-child a")
        if value_cell:
            val = parse_value(value_cell[0].text.strip())
            if val:
                player_values.append(val)

    avg_value = sum(player_values) / len(player_values) if player_values else None
    top_value = max(player_values) if player_values else None

    # If we got player values but no total, compute it
    if total_value is None and player_values:
        total_value = sum(player_values)

    return {
        "team": team_name,
        "squad_total_value_eur": total_value,
        "avg_player_value_eur": round(avg_value, 2) if avg_value else None,
        "top_player_value_eur": top_value,
        "num_players": num_players or len(player_values),
    }


def scrape():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Launching CloakBrowser for Transfermarkt...")
    browser = launch(headless=True, humanize=True)
    page = browser.new_page()

    results = []
    for i, (team, path) in enumerate(WORLD_CUP_TEAMS):
        print(f"  [{i+1}/{len(WORLD_CUP_TEAMS)}] Scraping {team}...")
        try:
            row = scrape_team(page, team, path)
            results.append(row)
            print(f"    Total value: €{row['squad_total_value_eur']:,.0f}" if row["squad_total_value_eur"] else "    No value found")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"team": team, "squad_total_value_eur": None, "avg_player_value_eur": None, "top_player_value_eur": None, "num_players": 0})

        time.sleep(3)  # Rate limit

    browser.close()

    output_path = DATA_DIR / "market_values.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["team", "squad_total_value_eur", "avg_player_value_eur", "top_player_value_eur", "num_players"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} teams to {output_path}")
    return results


if __name__ == "__main__":
    scrape()
```

- [ ] **Step 2: Run and verify**

```bash
cd ~/2026-fifa-worldcup-prediction
python3 scrapers/market_values.py
```

Expected: `data/raw/market_values.csv` with 48 rows. Some teams may fail due to anti-bot — that's OK, the notebook will handle missing values. If Transfermarkt blocks fully, consider reducing the team list to top 20 or using a proxy.

- [ ] **Step 3: Debug anti-bot issues if needed**

If Transfermarkt blocks requests, try:
1. Add `humanize=True` to `launch()` (already included)
2. Increase sleep to 5 seconds between requests
3. Run in headed mode: `launch(headless=False)` to see what's happening
4. Save HTML to `data/raw/transfermarkt_debug.html` for selector debugging

- [ ] **Step 4: Commit**

```bash
git add scrapers/market_values.py
git commit -m "feat: Transfermarkt market values scraper via CloakBrowser"
```

---

### Task 6: Injuries Scraper

**Files:**
- Create: `scrapers/injuries.py`

Uses API-Football free tier. Requires API key from api-football.com (free signup, 100 requests/day). If no API key available, falls back to scraping ESPN's injury tracker.

- [ ] **Step 1: Create the scraper**

```python
"""Fetch World Cup 2026 injury data from API-Football (or fallback to ESPN scraping)"""
import csv
import os
from pathlib import Path

import requests
from cloakbrowser import launch
from scrapling import Adaptor

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# API-Football config
API_FOOTBALL_URL = "https://v3.football.api-sports.io"
API_KEY = os.environ.get("API_FOOTBALL_KEY", "")

# ESPN fallback
ESPN_INJURY_URL = "https://www.espn.com/soccer/story/_/id/48572979/2026-fifa-world-cup-injuries-tracker"


def fetch_via_api():
    """Fetch injuries from API-Football (requires API key)."""
    if not API_KEY:
        return None

    headers = {"x-apisports-key": API_KEY}

    # Get World Cup 2026 injuries
    resp = requests.get(
        f"{API_FOOTBALL_URL}/injuries",
        headers=headers,
        params={"league": "1", "season": "2026"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("errors"):
        print(f"  API error: {data['errors']}")
        return None

    results = data.get("response", [])
    rows = []
    for item in results:
        player = item.get("player", {})
        team = item.get("team", {})
        rows.append({
            "team": team.get("name", ""),
            "player": player.get("name", ""),
            "position": player.get("type", ""),
            "injury_type": player.get("reason", ""),
            "status": "injured",
        })

    return rows


def fetch_via_espn():
    """Fallback: scrape ESPN injury tracker page."""
    print("  Using ESPN injury tracker fallback...")
    browser = launch(headless=True)
    page = browser.new_page()

    page.goto(ESPN_INJURY_URL, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(5000)
    html = page.content()
    browser.close()

    parsed = Adaptor(html, auto_match=False)
    rows = []

    # ESPN lists injuries in article text format — extract what we can
    paragraphs = parsed.css("article p, .article-body p")
    for p in paragraphs:
        text = p.text.strip()
        # Look for injury mentions (basic extraction)
        if any(keyword in text.lower() for keyword in ["acl", "hamstring", "knee", "ankle", "muscle", "out", "injured"]):
            rows.append({
                "team": "",
                "player": text[:100],
                "position": "",
                "injury_type": "see description",
                "status": "injured",
            })

    return rows


def scrape():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching injury data...")

    # Try API first
    rows = fetch_via_api()
    if rows is not None:
        print(f"  Got {len(rows)} injuries from API-Football")
    else:
        print("  No API key set (export API_FOOTBALL_KEY=your_key)")
        rows = fetch_via_espn()
        print(f"  Got {len(rows)} injury entries from ESPN")

    output_path = DATA_DIR / "injuries.csv"
    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["team", "player", "position", "injury_type", "status"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved to {output_path}")
    else:
        # Create empty file so downstream doesn't break
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["team", "player", "position", "injury_type", "status"])
            writer.writeheader()
        print("No injury data available — created empty CSV")

    return rows


if __name__ == "__main__":
    scrape()
```

- [ ] **Step 2: Run and verify**

```bash
cd ~/2026-fifa-worldcup-prediction
python3 scrapers/injuries.py
cat data/raw/injuries.csv | head -10
```

Expected: Either API data (if key is set via `export API_FOOTBALL_KEY=xxx`) or ESPN fallback data. An empty CSV is acceptable — injury data is a "nice to have" feature.

- [ ] **Step 3: Commit**

```bash
git add scrapers/injuries.py
git commit -m "feat: injury scraper with API-Football and ESPN fallback"
```

---

### Task 7: Run-All Orchestrator

**Files:**
- Create: `scrapers/run_all.py`

- [ ] **Step 1: Create orchestrator script**

```python
"""Run all scrapers sequentially."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers import fifa_ranking, elo_ratings, match_history, market_values, injuries


def main():
    print("=" * 60)
    print("2026 FIFA World Cup Prediction — Data Collection")
    print("=" * 60)

    scrapers = [
        ("FIFA Rankings", fifa_ranking.scrape),
        ("ELO Ratings", elo_ratings.scrape),
        ("Match History", match_history.scrape),
        ("Market Values", market_values.scrape),
        ("Injuries", injuries.scrape),
    ]

    results = {}
    for name, fn in scrapers:
        print(f"\n{'─' * 40}")
        print(f"▶ {name}")
        print(f"{'─' * 40}")
        try:
            result = fn()
            results[name] = "OK" if result else "EMPTY"
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = f"FAILED: {e}"

    print(f"\n{'=' * 60}")
    print("Summary:")
    for name, status in results.items():
        print(f"  {name}: {status}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
cd ~/2026-fifa-worldcup-prediction
python3 scrapers/run_all.py
ls -la data/raw/
```

Expected: All 5 CSV files in `data/raw/`. Some may have partial data if sites blocked us — that's OK for now.

- [ ] **Step 3: Commit**

```bash
git add scrapers/run_all.py
git commit -m "feat: run-all scraper orchestrator"
```

---

### Task 8: Data Exploration Notebook

**Files:**
- Create: `notebooks/01_data_exploration.ipynb`

- [ ] **Step 1: Create the notebook**

Create a Jupyter notebook with these cells:

**Cell 1 — Imports and setup:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path("../data/raw")
sns.set_style("whitegrid")
pd.set_option("display.max_columns", 20)
```

**Cell 2 — Load all datasets:**
```python
fifa = pd.read_csv(DATA_DIR / "fifa_rankings.csv")
elo = pd.read_csv(DATA_DIR / "elo_ratings.csv")
matches = pd.read_csv(DATA_DIR / "matches.csv", parse_dates=["date"])
market = pd.read_csv(DATA_DIR / "market_values.csv")
injuries = pd.read_csv(DATA_DIR / "injuries.csv")

print(f"FIFA Rankings: {len(fifa)} teams")
print(f"ELO Ratings: {len(elo)} teams")
print(f"Matches: {len(matches)} matches ({matches['date'].min().date()} to {matches['date'].max().date()})")
print(f"Market Values: {len(market)} teams")
print(f"Injuries: {len(injuries)} entries")
```

**Cell 3 — FIFA Rankings overview:**
```python
print("Top 20 FIFA Rankings:")
print(fifa.sort_values("rank").head(20)[["rank", "team", "points", "confederation"]].to_string(index=False))
```

**Cell 4 — Match distribution by year and tournament:**
```python
matches["year"] = matches["date"].dt.year
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

matches["year"].value_counts().sort_index().plot(kind="bar", ax=axes[0], title="Matches per Year")
matches["tournament"].value_counts().head(10).plot(kind="barh", ax=axes[1], title="Top 10 Tournaments")
plt.tight_layout()
plt.show()
```

**Cell 5 — Market value distribution:**
```python
market_clean = market.dropna(subset=["squad_total_value_eur"])
market_clean = market_clean.sort_values("squad_total_value_eur", ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(market_clean["team"].head(20), market_clean["squad_total_value_eur"].head(20) / 1e9)
plt.xlabel("Squad Value (€ Billions)")
plt.title("Top 20 National Teams by Squad Market Value")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

**Cell 6 — Data quality check:**
```python
print("=== Missing Values ===")
for name, df in [("FIFA", fifa), ("ELO", elo), ("Matches", matches), ("Market", market), ("Injuries", injuries)]:
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n{name}:")
        print(missing[missing > 0])
    else:
        print(f"\n{name}: No missing values")
```

**Cell 7 — Team name alignment check:**
```python
# Check which team names don't match across datasets
fifa_teams = set(fifa["team"])
elo_teams = set(elo["team"])
match_teams = set(matches["home_team"]) | set(matches["away_team"])
market_teams = set(market["team"])

print(f"FIFA teams: {len(fifa_teams)}")
print(f"ELO teams: {len(elo_teams)}")
print(f"Match teams: {len(match_teams)}")
print(f"Market teams: {len(market_teams)}")

print("\nTeams in FIFA but not in matches:")
print(sorted(fifa_teams - match_teams)[:10])

print("\nTeams in matches but not in FIFA:")
print(sorted(match_teams - fifa_teams)[:10])
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/01_data_exploration.ipynb
git commit -m "feat: data exploration notebook"
```

---

### Task 9: Feature Engineering Notebook

**Files:**
- Create: `notebooks/02_feature_engineering.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1 — Imports:**
```python
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("../data/raw")
OUTPUT_DIR = Path("../data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)
```

**Cell 2 — Load data:**
```python
fifa = pd.read_csv(DATA_DIR / "fifa_rankings.csv")
elo = pd.read_csv(DATA_DIR / "elo_ratings.csv")
matches = pd.read_csv(DATA_DIR / "matches.csv", parse_dates=["date"])
market = pd.read_csv(DATA_DIR / "market_values.csv")
injuries = pd.read_csv(DATA_DIR / "injuries.csv")

print(f"Loaded {len(matches)} matches")
```

**Cell 3 — Team name normalization:**
```python
# Create a mapping to normalize team names across datasets
NAME_MAP = {
    # Add mappings as discovered in exploration notebook
    # "Korea Republic": "South Korea",
    # "IR Iran": "Iran",
    # etc.
}

def normalize_team(name):
    return NAME_MAP.get(name, name)

matches["home_team"] = matches["home_team"].apply(normalize_team)
matches["away_team"] = matches["away_team"].apply(normalize_team)
fifa["team"] = fifa["team"].apply(normalize_team)
elo["team"] = elo["team"].apply(normalize_team)
market["team"] = market["team"].apply(normalize_team)
```

**Cell 4 — Create lookup dictionaries:**
```python
# FIFA rankings lookup: team -> {rank, points, confederation}
fifa_lookup = fifa.set_index("team")[["rank", "points", "confederation"]].to_dict("index")

# ELO lookup: team -> elo_rating (latest)
elo_lookup = elo.set_index("team")["elo_rating"].to_dict()

# Market value lookup: team -> squad_total_value_eur
market_lookup = market.set_index("team")["squad_total_value_eur"].to_dict()
```

**Cell 5 — Rolling form features:**
```python
def compute_rolling_features(matches_df, team, n=10):
    """Compute rolling stats for a team's last N matches (as home or away)."""
    team_matches = matches_df[
        (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    ].sort_values("date")

    if len(team_matches) == 0:
        return {"form": 0.5, "goals_scored_avg": 1.0, "goals_conceded_avg": 1.0}

    last_n = team_matches.tail(n)

    wins = 0
    goals_scored = []
    goals_conceded = []

    for _, row in last_n.iterrows():
        if row["home_team"] == team:
            gs, gc = row["home_score"], row["away_score"]
        else:
            gs, gc = row["away_score"], row["home_score"]

        goals_scored.append(gs)
        goals_conceded.append(gc)
        if gs > gc:
            wins += 1

    return {
        "form": wins / len(last_n),
        "goals_scored_avg": np.mean(goals_scored),
        "goals_conceded_avg": np.mean(goals_conceded),
    }
```

**Cell 6 — Head-to-head feature:**
```python
def compute_h2h(matches_df, home_team, away_team, n=5):
    """Compute head-to-head wins for home team in last N meetings."""
    h2h = matches_df[
        ((matches_df["home_team"] == home_team) & (matches_df["away_team"] == away_team)) |
        ((matches_df["home_team"] == away_team) & (matches_df["away_team"] == home_team))
    ].sort_values("date").tail(n)

    if len(h2h) == 0:
        return 0.5

    wins = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == home_team and row["home_score"] > row["away_score"]:
            wins += 1
        elif row["away_team"] == home_team and row["away_score"] > row["home_score"]:
            wins += 1

    return wins / len(h2h)
```

**Cell 7 — Match importance weights:**
```python
TOURNAMENT_IMPORTANCE = {
    "Friendly": 0.3,
    "FIFA World Cup qualification": 0.7,
    "UEFA Euro qualification": 0.7,
    "AFC Asian Cup qualification": 0.7,
    "African Cup of Nations qualification": 0.7,
    "Copa América": 0.9,
    "UEFA Euro": 0.9,
    "African Cup of Nations": 0.9,
    "AFC Asian Cup": 0.9,
    "FIFA World Cup": 1.0,
    "UEFA Nations League": 0.6,
    "CONMEBOL-UEFA Cup of Champions": 0.8,
}

def get_importance(tournament):
    for key, val in TOURNAMENT_IMPORTANCE.items():
        if key.lower() in tournament.lower():
            return val
    return 0.5  # default for unknown tournaments
```

**Cell 8 — Time-based sample weights:**
```python
def get_sample_weight(match_date):
    year = match_date.year
    weights = {2026: 1.0, 2025: 0.7, 2024: 0.4, 2023: 0.2, 2022: 0.1}
    return weights.get(year, 0.1)
```

**Cell 9 — Build training dataset:**
```python
print("Building training dataset...")
training_rows = []

for idx, row in matches.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    match_date = row["date"]

    # Get prior matches for rolling features (only matches BEFORE this one)
    prior_matches = matches[matches["date"] < match_date]

    home_stats = compute_rolling_features(prior_matches, home)
    away_stats = compute_rolling_features(prior_matches, away)

    # Lookup static features
    home_fifa = fifa_lookup.get(home, {"rank": 100, "points": 1300, "confederation": "Unknown"})
    away_fifa = fifa_lookup.get(away, {"rank": 100, "points": 1300, "confederation": "Unknown"})
    home_elo = elo_lookup.get(home, 1500)
    away_elo = elo_lookup.get(away, 1500)
    home_market = market_lookup.get(home, 0)
    away_market = market_lookup.get(away, 0)

    # Determine result
    if row["home_score"] > row["away_score"]:
        result = "W"
    elif row["home_score"] < row["away_score"]:
        result = "L"
    else:
        result = "D"

    goal_diff = row["home_score"] - row["away_score"]

    feature_row = {
        "date": match_date,
        "home_team": home,
        "away_team": away,
        "elo_diff": home_elo - away_elo,
        "fifa_rank_diff": away_fifa["rank"] - home_fifa["rank"],
        "fifa_points_diff": home_fifa["points"] - away_fifa["points"],
        "squad_value_ratio": (home_market / away_market) if away_market and home_market else 1.0,
        "form_home": home_stats["form"],
        "form_away": away_stats["form"],
        "goals_scored_avg_home": home_stats["goals_scored_avg"],
        "goals_scored_avg_away": away_stats["goals_scored_avg"],
        "goals_conceded_avg_home": home_stats["goals_conceded_avg"],
        "goals_conceded_avg_away": away_stats["goals_conceded_avg"],
        "h2h_wins": compute_h2h(prior_matches, home, away),
        "match_importance": get_importance(row["tournament"]),
        "confederation_home": home_fifa["confederation"],
        "confederation_away": away_fifa["confederation"],
        "neutral_venue": 1 if row["neutral"] else 0,
        "result": result,
        "goal_diff": goal_diff,
        "sample_weight": get_sample_weight(match_date),
    }
    training_rows.append(feature_row)

    if idx % 500 == 0:
        print(f"  Processed {idx}/{len(matches)} matches...")

training_df = pd.DataFrame(training_rows)
print(f"\nTraining set: {len(training_df)} rows, {len(training_df.columns)} columns")
print(training_df["result"].value_counts())
```

**Cell 10 — Save processed dataset:**
```python
output_path = OUTPUT_DIR / "training_set.csv"
training_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
print(f"\nFeature summary:")
print(training_df.describe())
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/02_feature_engineering.ipynb
git commit -m "feat: feature engineering notebook"
```

---

### Task 10: Training Notebook

**Files:**
- Create: `notebooks/03_training.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1 — Imports:**
```python
import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)
```

**Cell 2 — Load and split data:**
```python
df = pd.read_csv(DATA_DIR / "training_set.csv", parse_dates=["date"])
print(f"Total samples: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nResult distribution:")
print(df["result"].value_counts(normalize=True))

# Hold out 2022 World Cup matches for validation
wc_2022_mask = (df["date"] >= "2022-11-20") & (df["date"] <= "2022-12-18")
val_data = df[wc_2022_mask].copy()
train_data = df[~wc_2022_mask].copy()

print(f"\nTrain: {len(train_data)} matches")
print(f"Validation (2022 WC): {len(val_data)} matches")
```

**Cell 3 — Prepare features:**
```python
# Drop non-feature columns for training
drop_cols = ["date", "home_team", "away_team", "goal_diff"]
feature_cols = [c for c in train_data.columns if c not in drop_cols + ["result", "sample_weight"]]

print(f"Features ({len(feature_cols)}):")
for c in feature_cols:
    print(f"  {c}: {train_data[c].dtype}")
```

**Cell 4 — Train W/D/L Classifier:**
```python
print("Training W/D/L Classifier...")
print("=" * 50)

clf_train = train_data[feature_cols + ["result", "sample_weight"]].copy()
clf_val = val_data[feature_cols + ["result"]].copy()

predictor_clf = TabularPredictor(
    label="result",
    eval_metric="log_loss",
    path=str(MODELS_DIR / "classifier"),
)

predictor_clf.fit(
    clf_train,
    sample_weight="sample_weight",
    presets="best_quality",
    time_limit=600,
)

print("\n=== Classifier Results ===")
print(predictor_clf.leaderboard(clf_val))
```

**Cell 5 — Evaluate classifier on 2022 WC:**
```python
clf_preds = predictor_clf.predict(clf_val[feature_cols])
clf_probs = predictor_clf.predict_proba(clf_val[feature_cols])

accuracy = (clf_preds == clf_val["result"]).mean()
print(f"2022 World Cup Accuracy: {accuracy:.1%}")
print(f"\nPrediction distribution:")
print(clf_preds.value_counts(normalize=True))
print(f"\nActual distribution:")
print(clf_val["result"].value_counts(normalize=True))
```

**Cell 6 — Train Goal Difference Regressor:**
```python
print("Training Goal Difference Regressor...")
print("=" * 50)

# Prepare regression data
drop_cols_reg = ["date", "home_team", "away_team", "result"]
reg_train = train_data[feature_cols + ["goal_diff", "sample_weight"]].copy()
reg_val = val_data[feature_cols + ["goal_diff"]].copy()

predictor_reg = TabularPredictor(
    label="goal_diff",
    problem_type="regression",
    eval_metric="mean_absolute_error",
    path=str(MODELS_DIR / "regressor"),
)

predictor_reg.fit(
    reg_train,
    sample_weight="sample_weight",
    presets="best_quality",
    time_limit=600,
)

print("\n=== Regressor Results ===")
print(predictor_reg.leaderboard(reg_val))
```

**Cell 7 — Evaluate regressor:**
```python
reg_preds = predictor_reg.predict(reg_val[feature_cols])

mae = np.abs(reg_preds - reg_val["goal_diff"]).mean()
print(f"2022 World Cup MAE: {mae:.2f} goals")

# Show some predictions vs actuals
comparison = pd.DataFrame({
    "home": val_data["home_team"].values,
    "away": val_data["away_team"].values,
    "actual_diff": val_data["goal_diff"].values,
    "predicted_diff": reg_preds.values.round(2),
})
print("\nSample predictions:")
print(comparison.head(15).to_string(index=False))
```

**Cell 8 — Feature importance:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Classifier feature importance
clf_importance = predictor_clf.feature_importance(clf_val[feature_cols + ["result"]])
clf_importance.head(10).plot(kind="barh", ax=axes[0], title="Classifier — Top 10 Features")

# Regressor feature importance
reg_importance = predictor_reg.feature_importance(reg_val[feature_cols + ["goal_diff"]])
reg_importance.head(10).plot(kind="barh", ax=axes[1], title="Regressor — Top 10 Features")

plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/03_training.ipynb
git commit -m "feat: model training notebook with classifier + regressor"
```

---

### Task 11: Prediction Notebook

**Files:**
- Create: `notebooks/04_prediction.ipynb`

- [ ] **Step 1: Create the notebook**

**Cell 1 — Imports and load models:**
```python
import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor

MODELS_DIR = Path("../models")
DATA_DIR = Path("../data")

predictor_clf = TabularPredictor.load(str(MODELS_DIR / "classifier"))
predictor_reg = TabularPredictor.load(str(MODELS_DIR / "regressor"))

# Load data for lookups
fifa = pd.read_csv(DATA_DIR / "raw" / "fifa_rankings.csv")
elo = pd.read_csv(DATA_DIR / "raw" / "elo_ratings.csv")
market = pd.read_csv(DATA_DIR / "raw" / "market_values.csv")
matches = pd.read_csv(DATA_DIR / "raw" / "matches.csv", parse_dates=["date"])

fifa_lookup = fifa.set_index("team")[["rank", "points", "confederation"]].to_dict("index")
elo_lookup = elo.set_index("team")["elo_rating"].to_dict()
market_lookup = market.set_index("team")["squad_total_value_eur"].to_dict()

print("Models and data loaded successfully")
```

**Cell 2 — 2026 World Cup Groups:**
```python
# 2026 World Cup groups (48 teams, 12 groups of 4)
# Update these with the actual draw when announced
GROUPS = {
    "A": ["USA", "Morocco", "Scotland", "TBD"],
    "B": ["Mexico", "TBD", "TBD", "TBD"],
    "C": ["Canada", "TBD", "TBD", "TBD"],
    "D": ["Argentina", "TBD", "TBD", "TBD"],
    "E": ["Brazil", "TBD", "TBD", "TBD"],
    "F": ["France", "TBD", "TBD", "TBD"],
    "G": ["England", "TBD", "TBD", "TBD"],
    "H": ["Spain", "TBD", "TBD", "TBD"],
    "I": ["Germany", "TBD", "TBD", "TBD"],
    "J": ["Portugal", "TBD", "TBD", "TBD"],
    "K": ["Netherlands", "TBD", "TBD", "TBD"],
    "L": ["Italy", "TBD", "TBD", "TBD"],
}
# NOTE: Replace TBD with actual teams once draw is complete
```

**Cell 3 — Feature generation for a matchup:**
```python
def generate_match_features(home_team, away_team):
    """Generate feature row for a hypothetical match."""
    from notebooks.feature_helpers import compute_rolling_features, compute_h2h

    home_stats = compute_rolling_features(matches, home_team)
    away_stats = compute_rolling_features(matches, away_team)

    home_fifa = fifa_lookup.get(home_team, {"rank": 100, "points": 1300, "confederation": "Unknown"})
    away_fifa = fifa_lookup.get(away_team, {"rank": 100, "points": 1300, "confederation": "Unknown"})
    home_elo = elo_lookup.get(home_team, 1500)
    away_elo = elo_lookup.get(away_team, 1500)
    home_market = market_lookup.get(home_team, 0)
    away_market = market_lookup.get(away_team, 0)

    return {
        "elo_diff": home_elo - away_elo,
        "fifa_rank_diff": away_fifa["rank"] - home_fifa["rank"],
        "fifa_points_diff": home_fifa["points"] - away_fifa["points"],
        "squad_value_ratio": (home_market / away_market) if away_market and home_market else 1.0,
        "form_home": home_stats["form"],
        "form_away": away_stats["form"],
        "goals_scored_avg_home": home_stats["goals_scored_avg"],
        "goals_scored_avg_away": away_stats["goals_scored_avg"],
        "goals_conceded_avg_home": home_stats["goals_conceded_avg"],
        "goals_conceded_avg_away": away_stats["goals_conceded_avg"],
        "h2h_wins": compute_h2h(matches, home_team, away_team),
        "match_importance": 1.0,  # World Cup
        "confederation_home": home_fifa["confederation"],
        "confederation_away": away_fifa["confederation"],
        "neutral_venue": 1,  # Most WC matches are neutral
    }
```

**Cell 4 — Predict a single match:**
```python
def predict_match(home_team, away_team):
    """Predict outcome for a single match."""
    features = generate_match_features(home_team, away_team)
    features_df = pd.DataFrame([features])

    probs = predictor_clf.predict_proba(features_df).iloc[0]
    goal_diff = predictor_reg.predict(features_df).iloc[0]

    return {
        "home": home_team,
        "away": away_team,
        "P(home_win)": round(probs.get("W", 0), 3),
        "P(draw)": round(probs.get("D", 0), 3),
        "P(away_win)": round(probs.get("L", 0), 3),
        "expected_goal_diff": round(goal_diff, 2),
    }

# Example predictions
example_matches = [
    ("France", "Brazil"),
    ("Argentina", "England"),
    ("Spain", "Germany"),
    ("USA", "Morocco"),
]

print("Example Predictions:")
print("=" * 70)
for home, away in example_matches:
    result = predict_match(home, away)
    print(f"{result['home']:>15} vs {result['away']:<15} | "
          f"W:{result['P(home_win)']:.1%} D:{result['P(draw)']:.1%} L:{result['P(away_win)']:.1%} | "
          f"GD: {result['expected_goal_diff']:+.1f}")
```

**Cell 5 — Monte Carlo tournament simulation:**
```python
def simulate_match(home_team, away_team, rng):
    """Simulate a single match outcome based on predicted probabilities."""
    features = generate_match_features(home_team, away_team)
    features_df = pd.DataFrame([features])

    probs = predictor_clf.predict_proba(features_df).iloc[0]
    p_win = probs.get("W", 0.33)
    p_draw = probs.get("D", 0.33)
    # p_loss = probs.get("L", 0.34)

    roll = rng.random()
    if roll < p_win:
        return "W"
    elif roll < p_win + p_draw:
        return "D"
    else:
        return "L"


def simulate_group(group_teams, n_simulations=10000):
    """Simulate group stage and return advancement probabilities."""
    rng = np.random.default_rng(42)
    advance_count = {team: 0 for team in group_teams}

    for _ in range(n_simulations):
        points = {team: 0 for team in group_teams}

        # Round-robin: each team plays each other once
        for i, home in enumerate(group_teams):
            for away in group_teams[i+1:]:
                result = simulate_match(home, away, rng)
                if result == "W":
                    points[home] += 3
                elif result == "D":
                    points[home] += 1
                    points[away] += 1
                else:
                    points[away] += 3

        # Top 2 advance (in 2026: top 2 + best 3rd places)
        ranked = sorted(points.items(), key=lambda x: x[1], reverse=True)
        for team, _ in ranked[:2]:
            advance_count[team] += 1

    return {team: count / n_simulations for team, count in advance_count.items()}


# Simulate all groups
print("Simulating group stages (10,000 iterations per group)...")
print("=" * 50)
for group_name, teams in GROUPS.items():
    if "TBD" in teams:
        continue
    probs = simulate_group(teams)
    print(f"\nGroup {group_name}:")
    for team, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {team:<20} {prob:.1%} chance to advance")
```

**Cell 6 — Overall tournament winner probabilities:**
```python
def simulate_tournament(n_simulations=10000):
    """Full tournament Monte Carlo simulation."""
    rng = np.random.default_rng(42)
    winner_count = {}

    all_teams = [t for teams in GROUPS.values() for t in teams if t != "TBD"]

    for team in all_teams:
        winner_count[team] = 0

    for sim in range(n_simulations):
        # Simplified: assume top teams from each group advance to knockout
        # Then simulate knockout rounds
        # (Full implementation depends on actual bracket structure)
        pass

    # For now, use a simplified approach: probability proportional to model strength
    print("\nTournament Winner Probabilities (simplified):")
    print("=" * 50)

    team_strengths = []
    for team in all_teams:
        elo = elo_lookup.get(team, 1500)
        fifa_pts = fifa_lookup.get(team, {}).get("points", 1300)
        strength = (elo / 2200) * 0.6 + (fifa_pts / 1900) * 0.4
        team_strengths.append((team, strength))

    total = sum(s for _, s in team_strengths)
    team_strengths.sort(key=lambda x: x[1], reverse=True)

    for team, strength in team_strengths[:15]:
        prob = strength / total
        print(f"  {team:<20} {prob:.1%}")


simulate_tournament()
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/04_prediction.ipynb
git commit -m "feat: prediction notebook with Monte Carlo simulation"
```

---

### Task 12: Final Integration & README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README**

```markdown
# 2026 FIFA World Cup Match Prediction

Predicts match scores and win probabilities for the 2026 FIFA World Cup using AutoGluon.

## Quick Start

```bash
pip install -r requirements.txt

# Collect data
python scrapers/run_all.py

# Run notebooks in order
jupyter notebook notebooks/
```

## Data Sources

| Source | Script | Method |
|--------|--------|--------|
| FIFA Rankings | `scrapers/fifa_ranking.py` | Direct API |
| ELO Ratings | `scrapers/elo_ratings.py` | CloakBrowser |
| Match History | `scrapers/match_history.py` | CSV download |
| Market Values | `scrapers/market_values.py` | CloakBrowser |
| Injuries | `scrapers/injuries.py` | API-Football |

## Models

Two AutoGluon models trained on 4 years of international match data:

1. **W/D/L Classifier** — predicts win/draw/loss probabilities
2. **Goal Difference Regressor** — predicts expected score margin

## Notebooks

1. `01_data_exploration.ipynb` — Data quality checks and visualization
2. `02_feature_engineering.ipynb` — Feature computation and training set creation
3. `03_training.ipynb` — AutoGluon model training and validation
4. `04_prediction.ipynb` — 2026 World Cup predictions and simulation
```

- [ ] **Step 2: Commit all**

```bash
git add README.md
git commit -m "docs: add project README"
```

- [ ] **Step 3: Final verification**

```bash
cd ~/2026-fifa-worldcup-prediction
find . -name "*.py" -o -name "*.ipynb" -o -name "*.md" -o -name "*.txt" | grep -v .git | sort
```

Expected output:
```
./README.md
./docs/superpowers/plans/2026-06-03-worldcup-prediction.md
./docs/superpowers/specs/2026-06-03-worldcup-prediction-design.md
./notebooks/01_data_exploration.ipynb
./notebooks/02_feature_engineering.ipynb
./notebooks/03_training.ipynb
./notebooks/04_prediction.ipynb
./requirements.txt
./scrapers/__init__.py
./scrapers/elo_ratings.py
./scrapers/fifa_ranking.py
./scrapers/injuries.py
./scrapers/market_values.py
./scrapers/match_history.py
./scrapers/run_all.py
```

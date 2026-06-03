# 2026 FIFA World Cup Match Prediction System

## Overview

Predict match scores and win probabilities for the 2026 FIFA World Cup using AutoGluon. Combines five data sources (FIFA rankings, ELO ratings, match history, player market values, injury data) with time-weighted feature engineering to train two models: a W/D/L classifier and a goal-difference regressor.

## Project Structure

```
~/2026-fifa-worldcup-prediction/
├── scrapers/
│   ├── fifa_ranking.py          # FIFA API → data/raw/fifa_rankings.csv
│   ├── elo_ratings.py           # eloratings.net → data/raw/elo_ratings.csv
│   ├── match_history.py         # Kaggle dataset → data/raw/matches.csv
│   ├── market_values.py         # Transfermarkt → data/raw/market_values.csv
│   └── injuries.py              # API-Football → data/raw/injuries.csv
├── data/
│   ├── raw/                     # Scraper outputs (immutable)
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

## Data Sources

### 1. FIFA Rankings (fifa_ranking.py)

- **Source**: `https://api.fifa.com/api/v3/fifarankings/rankings/live?gender=1&sportType=0&language=en`
- **Method**: Direct HTTP GET (no browser needed)
- **Output**: `data/raw/fifa_rankings.csv`
- **Fields**: rank, prev_rank, team, country_code, confederation, points, prev_points, ranking_movement, rated_matches
- **Coverage**: All 211 FIFA member teams, latest official ranking

### 2. ELO Ratings (elo_ratings.py)

- **Source**: `https://www.eloratings.net` and `https://www.international-football.net/elo-ratings-table`
- **Method**: Scrapling + CloakBrowser (JS-rendered content)
- **Output**: `data/raw/elo_ratings.csv`
- **Fields**: team, elo_rating, date
- **Coverage**: All teams, monthly snapshots from 2022 to present

### 3. Match History (match_history.py)

- **Source**: Kaggle `patateriedata/all-international-football-results` (daily-updated, 50k+ matches)
- **Method**: Direct CSV download
- **Output**: `data/raw/matches.csv`
- **Fields**: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
- **Filter**: Only matches from 2022-01-01 onwards (4-year window)

### 4. Player Market Values (market_values.py)

- **Source**: Transfermarkt national team squad pages
- **Method**: Scrapling + CloakBrowser (anti-bot protection)
- **Output**: `data/raw/market_values.csv`
- **Fields**: team, squad_total_value_eur, avg_player_value, top_player_value, num_players
- **Coverage**: All 48 World Cup qualified teams

### 5. Injury Data (injuries.py)

- **Source**: API-Football (api-football.com) free tier
- **Method**: REST API with API key (free tier at api-football.com, requires signup)
- **Output**: `data/raw/injuries.csv`
- **Fields**: team, player, position, injury_type, market_value, expected_return
- **Coverage**: Current injuries for World Cup squads (league=1, season=2026)

## Scraping Technology

- **Scrapling** (https://github.com/d4vinci/Scrapling): Primary scraping framework for parsing and extraction
- **CloakBrowser** (https://github.com/CloakHQ/cloakbrowser): Stealth browser for JS-heavy sites with anti-bot protection (Transfermarkt, eloratings.net)
- **requests**: For simple API calls (FIFA API, API-Football, Kaggle download)

## Notebooks

### 01_data_exploration.ipynb

- Load all raw CSVs
- Inspect row counts, missing values, date ranges
- Visualize: team rankings over time, ELO distributions, match frequency
- Data quality checks and cleaning

### 02_feature_engineering.ipynb

**Per-match features** (computed for both home and away team):

| Feature | Source | Description |
|---------|--------|-------------|
| elo_diff | ELO ratings | Home ELO - Away ELO |
| fifa_rank_diff | FIFA ranking | Away rank - Home rank |
| fifa_points_diff | FIFA ranking | Home points - Away points |
| squad_value_ratio | Transfermarkt | Home value / Away value |
| form_home | Match history | Win % last 10 matches |
| form_away | Match history | Win % last 10 matches |
| goals_scored_avg_home | Match history | Rolling avg goals scored (last 10) |
| goals_scored_avg_away | Match history | Rolling avg goals scored (last 10) |
| goals_conceded_avg_home | Match history | Rolling avg conceded (last 10) |
| goals_conceded_avg_away | Match history | Rolling avg conceded (last 10) |
| h2h_wins | Match history | Head-to-head wins last 5 meetings |
| injury_impact_home | Injuries | % of squad value unavailable |
| injury_impact_away | Injuries | % of squad value unavailable |
| match_importance | Match history | Friendly=0.3, Qualifier=0.7, Tournament=1.0 |
| confederation_home | FIFA ranking | Categorical |
| confederation_away | FIFA ranking | Categorical |
| neutral_venue | Match history | Boolean |

**Time weighting** (applied as `sample_weight` column):
- 2026 matches: weight 1.0
- 2025 matches: weight 0.7
- 2024 matches: weight 0.4
- 2023 matches: weight 0.2
- 2022 matches: weight 0.1

**Output**: `data/processed/training_set.csv`

### 03_training.ipynb

**Model 1 — W/D/L Classifier:**
```python
from autogluon.tabular import TabularPredictor

predictor_clf = TabularPredictor(
    label='result',
    eval_metric='log_loss',
    path='models/classifier'
)
predictor_clf.fit(
    train_data,
    sample_weight='sample_weight',
    presets='best_quality',
    time_limit=600
)
```

**Model 2 — Goal Difference Regressor:**
```python
predictor_reg = TabularPredictor(
    label='goal_diff',
    problem_type='regression',
    eval_metric='mean_absolute_error',
    path='models/regressor'
)
predictor_reg.fit(
    train_data,
    sample_weight='sample_weight',
    presets='best_quality',
    time_limit=600
)
```

**Validation**: Hold out all 2022 World Cup matches from training. Report accuracy, log loss, and MAE.

### 04_prediction.ipynb

- Load 2026 World Cup group draw (all 48 teams, 12 groups)
- Generate feature rows for every group-stage matchup
- Predict: P(win), P(draw), P(loss) per match + expected goal difference
- Convert goal_diff to approximate scorelines
- Monte Carlo simulation (10,000 iterations) of full bracket
- Output: tournament winner probabilities, group stage predictions, expected top scorer teams

## Technical Constraints

- **Python 3.12.3** on Linux (WSL2)
- **15GB RAM** — sufficient for AutoGluon tabular (dataset is small, ~5k rows)
- **No sudo** — install packages via pip to user site
- **API-Football free tier**: 100 requests/day — sufficient for one-time squad injury pull
- **Transfermarkt**: Rate limit to 1 request per 2 seconds, respect robots.txt

## Dependencies

```
autogluon[tabular]
scrapling
cloakbrowser
pandas
numpy
requests
jupyter
matplotlib
seaborn
```

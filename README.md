# 2026 FIFA World Cup Match Prediction

Predicts match scores and win probabilities for the 2026 FIFA World Cup using AutoGluon.

## Quick Start

```bash
uv sync

# Collect data
uv run python scrapers/run_all.py

# Run notebooks in order
uv run jupyter notebook notebooks/
```

## Data Sources

| Source | Script | Method |
|--------|--------|--------|
| FIFA Rankings | `scrapers/fifa_ranking.py` | Direct API (api.fifa.com) |
| ELO Ratings | `scrapers/elo_ratings.py` | CloakBrowser + Scrapling |
| Match History | `scrapers/match_history.py` | CSV download (GitHub) |
| Market Values | `scrapers/market_values.py` | CloakBrowser + Scrapling (Transfermarkt) |
| Injuries | `scrapers/injuries.py` | API-Football / ESPN fallback |

## Models

Two AutoGluon models trained on 4 years of international match data (2022–2026), with recent matches overweighted:

1. **W/D/L Classifier** — predicts win/draw/loss probabilities (log_loss metric)
2. **Goal Difference Regressor** — predicts expected score margin (MAE metric)

## Notebooks

1. `01_data_exploration.ipynb` — Data quality checks and visualization
2. `02_feature_engineering.ipynb` — Feature computation and training set creation
3. `03_training.ipynb` — AutoGluon model training and validation (2022 WC hold-out)
4. `04_prediction.ipynb` — 2026 World Cup predictions and Monte Carlo simulation

## Features

| Feature | Description |
|---------|-------------|
| elo_diff | Home ELO - Away ELO |
| fifa_rank_diff | Away rank - Home rank |
| fifa_points_diff | Home points - Away points |
| squad_value_ratio | Home squad value / Away squad value |
| form_home/away | Win % in last 10 matches |
| goals_scored/conceded_avg | Rolling averages (last 10) |
| h2h_wins | Head-to-head win ratio (last 5 meetings) |
| match_importance | Tournament weight (Friendly=0.3 → WC=1.0) |
| confederation | Team confederation (categorical) |
| neutral_venue | Boolean |

## Tech Stack

- **AutoGluon** — automated ML with ensemble models
- **Scrapling** — web scraping framework with adaptive parsing
- **CloakBrowser** — stealth Chromium for anti-bot sites
- **pandas** — data manipulation

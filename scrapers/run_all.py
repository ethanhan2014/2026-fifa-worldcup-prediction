"""Run all scrapers sequentially."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers import fifa_ranking, dynamic_elo, match_history, market_values, injuries


def main():
    print("=" * 60)
    print("2026 FIFA World Cup Prediction — Data Collection")
    print("=" * 60)

    scrapers = [
        ("Match History", match_history.scrape),
        ("FIFA Rankings", fifa_ranking.scrape),
        ("ELO Ratings (dynamic)", dynamic_elo.scrape),
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
            results[name] = "OK" if result is not None else "EMPTY"
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

"""Train AutoGluon classifier and regressor on the processed training set.

Uses sample_weight column for importance weighting.
Validates on 2026 matches with scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, classification_report

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

NON_FEATURE_COLS = {"date", "home_team", "away_team", "tournament", "result", "goal_diff", "sample_weight"}


def main():
    df = pd.read_csv(PROCESSED_DIR / "training_set.csv", parse_dates=["date"])

    # Drop rows without scores (future WC matches)
    df = df.dropna(subset=["result", "goal_diff"]).copy()

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    # Fill NaN in rolling features
    for col in ["goals_scored_avg_home", "goals_scored_avg_away", "goals_conceded_avg_home", "goals_conceded_avg_away"]:
        df[col] = df[col].fillna(1.0)

    # Split: train on all data, validate on 2026 matches
    val_mask = df["date"].dt.year == 2026
    train_df = df[~val_mask].copy()
    val_df = df[val_mask].copy()

    print(f"Train: {len(train_df):,} rows (2023-2025)")
    print(f"Val:   {len(val_df):,} rows (2026)")
    print(f"\nTrain weight distribution:\n{train_df['sample_weight'].value_counts().sort_index()}")
    print(f"\nVal weight distribution:\n{val_df['sample_weight'].value_counts().sort_index()}")

    # === Classifier ===
    print("\n" + "=" * 60)
    print("Training W/D/L Classifier...")
    print("=" * 60)

    clf_train = train_df[feature_cols + ["result", "sample_weight"]].copy()
    clf_val = val_df[feature_cols + ["result"]].copy()

    predictor_clf = TabularPredictor(
        label="result",
        eval_metric="log_loss",
        sample_weight="sample_weight",
        path=str(MODELS_DIR / "classifier"),
    )
    predictor_clf.fit(
        clf_train,
        presets="best_quality",
        time_limit=600,
    )

    # Evaluate
    clf_val_features = clf_val.drop(columns=["result"])
    y_true = clf_val["result"]
    y_pred = predictor_clf.predict(clf_val_features)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nClassifier accuracy on 2026 validation: {acc:.3f}")
    print(classification_report(y_true, y_pred))

    # === Regressor ===
    print("\n" + "=" * 60)
    print("Training Goal Difference Regressor...")
    print("=" * 60)

    reg_train = train_df[feature_cols + ["goal_diff", "sample_weight"]].copy()
    reg_val = val_df[feature_cols + ["goal_diff"]].copy()

    predictor_reg = TabularPredictor(
        label="goal_diff",
        problem_type="regression",
        eval_metric="mean_absolute_error",
        sample_weight="sample_weight",
        path=str(MODELS_DIR / "regressor"),
    )
    predictor_reg.fit(
        reg_train,
        presets="best_quality",
        time_limit=600,
    )

    # Evaluate
    reg_val_features = reg_val.drop(columns=["goal_diff"])
    y_true_reg = reg_val["goal_diff"]
    y_pred_reg = predictor_reg.predict(reg_val_features)

    mae = np.mean(np.abs(y_true_reg - y_pred_reg))
    print(f"\nRegressor MAE on 2026 validation: {mae:.3f}")

    # Show WC predictions
    wc_matches = val_df[val_df["tournament"] == "FIFA World Cup"].copy()
    if not wc_matches.empty:
        wc_features = wc_matches[feature_cols]
        wc_matches = wc_matches.copy()
        wc_matches["pred_result"] = predictor_clf.predict(wc_features).values
        wc_matches["pred_goal_diff"] = predictor_reg.predict(wc_features).values
        print("\n\nWorld Cup match predictions vs actuals:")
        for _, m in wc_matches.iterrows():
            print(f"  {m['home_team']:20s} vs {m['away_team']:20s} | "
                  f"Actual: {m['result']} ({m['goal_diff']:+.0f}) | "
                  f"Pred: {m['pred_result']} ({m['pred_goal_diff']:+.1f})")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Classifier: {MODELS_DIR / 'classifier'}")
    print(f"Regressor:  {MODELS_DIR / 'regressor'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

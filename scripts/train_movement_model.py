"""Train line movement prediction model with XGBoost.

Supports both random split and walk-forward validation strategies.
Walk-forward validation provides more realistic performance estimates
for time-series predictions.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzers.features import FeatureEngineer
from src.analyzers.movement_predictor import LineMovementPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

if not all([POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB]):
    raise ValueError("Database configuration incomplete. Check POSTGRES_* environment variables.")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Model save path
MODEL_PATH = "models/line_movement_predictor.pkl"


def walk_forward_validation(
    df: pd.DataFrame,
    n_splits: int = 5,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> dict:
    """
    Perform walk-forward (expanding window) validation.

    This is more realistic for time-series data as it always trains
    on past data and tests on future data.

    Args:
        df: Full dataset with features and targets
        n_splits: Number of validation folds
        n_estimators: XGBoost n_estimators
        max_depth: XGBoost max_depth
        learning_rate: XGBoost learning_rate

    Returns:
        Dictionary with aggregated metrics across folds
    """
    logger.info(f"Starting walk-forward validation with {n_splits} splits...")

    # Prepare feature columns
    feature_cols = [col for col in df.columns if col not in [
        "price_movement", "point_movement", "directional_movement"
    ]]

    X = df[feature_cols]
    y_regression = df[["price_movement", "point_movement"]]
    y_classification = df["directional_movement"]

    # Create time-series splits
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = {
        "regression_mae": [],
        "regression_rmse": [],
        "classification_accuracy": [],
        "baseline_mae": [],
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        logger.info(f"\n--- Fold {fold}/{n_splits} ---")
        logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_reg_train = y_regression.iloc[train_idx]
        y_reg_test = y_regression.iloc[test_idx]
        y_class_train = y_classification.iloc[train_idx]
        y_class_test = y_classification.iloc[test_idx]

        # Train model
        predictor = LineMovementPredictor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )
        predictor.train(X_train, y_reg_train, y_class_train)

        # Evaluate
        reg_metrics = predictor.evaluate_regression(X_test, y_reg_test)
        class_metrics = predictor.evaluate_classification(X_test, y_class_test)

        baseline_mae = np.abs(y_reg_test["price_movement"]).mean()

        fold_metrics["regression_mae"].append(reg_metrics["ensemble_mae"])
        fold_metrics["regression_rmse"].append(reg_metrics["ensemble_rmse"])
        fold_metrics["classification_accuracy"].append(class_metrics["ensemble_accuracy"])
        fold_metrics["baseline_mae"].append(baseline_mae)

        logger.info(f"  MAE: {reg_metrics['ensemble_mae']:.4f} (baseline: {baseline_mae:.4f})")
        logger.info(f"  Accuracy: {class_metrics['ensemble_accuracy']:.4f}")

    # Aggregate results
    results = {
        "avg_mae": np.mean(fold_metrics["regression_mae"]),
        "std_mae": np.std(fold_metrics["regression_mae"]),
        "avg_rmse": np.mean(fold_metrics["regression_rmse"]),
        "std_rmse": np.std(fold_metrics["regression_rmse"]),
        "avg_accuracy": np.mean(fold_metrics["classification_accuracy"]),
        "std_accuracy": np.std(fold_metrics["classification_accuracy"]),
        "avg_baseline_mae": np.mean(fold_metrics["baseline_mae"]),
        "avg_improvement": (
            (np.mean(fold_metrics["baseline_mae"]) - np.mean(fold_metrics["regression_mae"]))
            / np.mean(fold_metrics["baseline_mae"]) * 100
        ),
        "fold_metrics": fold_metrics,
    }

    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD VALIDATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Average MAE: {results['avg_mae']:.4f} (+/- {results['std_mae']:.4f})")
    logger.info(f"Average RMSE: {results['avg_rmse']:.4f} (+/- {results['std_rmse']:.4f})")
    logger.info(f"Average Accuracy: {results['avg_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")
    logger.info(f"Average Improvement vs Baseline: {results['avg_improvement']:.1f}%")

    return results


def main(use_walk_forward: bool = False, n_splits: int = 5):
    """Train and evaluate line movement prediction model.

    Args:
        use_walk_forward: If True, use walk-forward validation instead of random split
        n_splits: Number of splits for walk-forward validation
    """
    logger.info("=" * 80)
    logger.info("LINE MOVEMENT PREDICTION MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Validation strategy: {'Walk-Forward' if use_walk_forward else 'Random Split'}")

    # Create database session
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    try:
        # Step 1: Prepare training data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: PREPARING TRAINING DATA")
        logger.info("=" * 80)

        engineer = FeatureEngineer()
        df = engineer.prepare_training_data(session)

        if len(df) == 0:
            logger.error("No training data available. Please collect odds data first.")
            return

        # Step 2: Analyze movement statistics
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: MOVEMENT STATISTICS")
        logger.info("=" * 80)

        total_records = len(df)
        moved_up = len(df[df["directional_movement"] == "UP"])
        moved_down = len(df[df["directional_movement"] == "DOWN"])
        stayed_same = len(df[df["directional_movement"] == "STAY"])

        significant_movement = len(df[np.abs(df["price_movement"]) > 0.02])
        avg_movement = df["price_movement"].abs().mean()
        median_movement = df["price_movement"].abs().median()
        max_movement = df["price_movement"].abs().max()

        logger.info(f"Total records: {total_records}")
        logger.info(f"\nDirectional Distribution:")
        logger.info(f"  UP:   {moved_up:>6} ({moved_up/total_records*100:>5.1f}%)")
        logger.info(f"  DOWN: {moved_down:>6} ({moved_down/total_records*100:>5.1f}%)")
        logger.info(f"  STAY: {stayed_same:>6} ({stayed_same/total_records*100:>5.1f}%)")
        logger.info(f"\nMovement Magnitude:")
        logger.info(f"  Significant movements (>0.02): {significant_movement} ({significant_movement/total_records*100:.1f}%)")
        logger.info(f"  Average absolute movement: {avg_movement:.4f}")
        logger.info(f"  Median absolute movement: {median_movement:.4f}")
        logger.info(f"  Maximum absolute movement: {max_movement:.4f}")

        # Breakdown by market type
        logger.info(f"\nMovement by Market Type:")
        for market in df["market_type"].unique():
            market_df = df[df["market_type"] == market]
            avg_move = market_df["price_movement"].abs().mean()
            logger.info(f"  {market:>8}: {len(market_df):>4} records, avg movement: {avg_move:.4f}")

        # Step 3: Walk-forward validation (optional)
        if use_walk_forward:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: WALK-FORWARD VALIDATION")
            logger.info("=" * 80)

            wf_results = walk_forward_validation(
                df,
                n_splits=n_splits,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
            )

        # Step 4: Split data for final model
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP {'4' if use_walk_forward else '3'}: SPLITTING DATA FOR FINAL MODEL")
        logger.info("=" * 80)

        X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = (
            engineer.train_test_split_data(df, test_size=0.2)
        )

        # Step 5: Train final models
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP {'5' if use_walk_forward else '4'}: TRAINING FINAL MODELS")
        logger.info("=" * 80)

        predictor = LineMovementPredictor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
        )
        predictor.train(X_train, y_reg_train, y_class_train)

        # Step 6: Evaluate regression model
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP {'6' if use_walk_forward else '5'}: REGRESSION MODEL EVALUATION")
        logger.info("=" * 80)

        regression_metrics = predictor.evaluate_regression(X_test, y_reg_test)

        # Calculate baseline (predict no movement)
        baseline_mae = np.abs(y_reg_test["price_movement"]).mean()
        improvement = ((baseline_mae - regression_metrics["ensemble_mae"]) / baseline_mae) * 100

        logger.info(f"\nBaseline Comparison:")
        logger.info(f"  Baseline MAE (predict no movement): {baseline_mae:.4f}")
        logger.info(f"  Ensemble MAE: {regression_metrics['ensemble_mae']:.4f}")
        logger.info(f"  Improvement vs baseline: {improvement:.1f}%")

        # Step 7: Evaluate classification model
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP {'7' if use_walk_forward else '6'}: CLASSIFICATION MODEL EVALUATION")
        logger.info("=" * 80)

        classification_metrics = predictor.evaluate_classification(X_test, y_class_test)

        # Calculate baseline (predict most frequent class)
        baseline_accuracy = y_class_train.value_counts().max() / len(y_class_train)
        logger.info(f"\nBaseline Comparison:")
        logger.info(f"  Baseline accuracy (predict most frequent): {baseline_accuracy:.4f}")
        logger.info(f"  Ensemble accuracy: {classification_metrics['ensemble_accuracy']:.4f}")
        logger.info(f"  Improvement vs baseline: {(classification_metrics['ensemble_accuracy'] - baseline_accuracy):.4f}")

        # Step 8: Feature importance
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP {'8' if use_walk_forward else '7'}: FEATURE IMPORTANCE")
        logger.info("=" * 80)

        importance = predictor.get_feature_importance()
        logger.info("\nTop 15 Most Important Features:")
        for i, (feature, score) in enumerate(list(importance.items())[:15], 1):
            logger.info(f"  {i:>2}. {feature:<35} {score:.4f}")

        # Step 9: Save model
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP {'9' if use_walk_forward else '8'}: SAVING MODEL")
        logger.info("=" * 80)

        predictor.save_model(MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Training records: {len(X_train)}")
        logger.info(f"Test records: {len(X_test)}")
        logger.info(f"\nEnsemble Regression Performance:")
        logger.info(f"  MAE: {regression_metrics['ensemble_mae']:.4f}")
        logger.info(f"  RMSE: {regression_metrics['ensemble_rmse']:.4f}")
        logger.info(f"  R²: {regression_metrics['ensemble_r2']:.4f}")
        logger.info(f"\nEnsemble Classification Performance:")
        logger.info(f"  Accuracy: {classification_metrics['ensemble_accuracy']:.4f}")
        logger.info(f"  Precision: {classification_metrics['precision']:.4f}")
        logger.info(f"  Recall: {classification_metrics['recall']:.4f}")

        if use_walk_forward:
            logger.info(f"\nWalk-Forward Validation:")
            logger.info(f"  Avg MAE: {wf_results['avg_mae']:.4f} (+/- {wf_results['std_mae']:.4f})")
            logger.info(f"  Avg Accuracy: {wf_results['avg_accuracy']:.4f} (+/- {wf_results['std_accuracy']:.4f})")

        logger.info(f"\nModel saved to: {MODEL_PATH}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train line movement prediction model")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use walk-forward validation instead of random split"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of splits for walk-forward validation (default: 5)"
    )

    args = parser.parse_args()
    main(use_walk_forward=args.walk_forward, n_splits=args.n_splits)

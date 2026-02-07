"""Auto-retraining script for line movement prediction model.

This script checks model performance against recent predictions and
triggers retraining if accuracy has degraded beyond a threshold.

Designed to be run daily via scheduler.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzers.features import FeatureEngineer
from src.analyzers.movement_predictor import LineMovementPredictor
from src.models.database import Bookmaker, ClosingLine, Game, OddsSnapshot

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

# Model paths
MODEL_PATH = "models/line_movement_predictor.pkl"
METRICS_PATH = "models/training_metrics.json"
RETRAIN_LOG_PATH = "models/retrain_log.json"


def load_training_metrics() -> dict:
    """Load metrics from last training run."""
    if not Path(METRICS_PATH).exists():
        logger.warning("No training metrics file found")
        return {}

    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def save_training_metrics(metrics: dict) -> None:
    """Save training metrics for future comparison."""
    Path(METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def load_retrain_log() -> list:
    """Load retraining history log."""
    if not Path(RETRAIN_LOG_PATH).exists():
        return []

    with open(RETRAIN_LOG_PATH, "r") as f:
        return json.load(f)


def save_retrain_log(log: list) -> None:
    """Save retraining history log."""
    Path(RETRAIN_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)


def evaluate_recent_performance(
    session,
    model: LineMovementPredictor,
    engineer: FeatureEngineer,
    days: int = 7,
) -> dict:
    """
    Evaluate model performance on recent data.

    Args:
        session: Database session
        model: Trained model
        engineer: Feature engineer
        days: Number of days to evaluate

    Returns:
        Dictionary with recent performance metrics
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Get recent snapshots with closing lines
    stmt = (
        select(OddsSnapshot, ClosingLine, Game, Bookmaker)
        .join(
            ClosingLine,
            (ClosingLine.game_id == OddsSnapshot.game_id)
            & (ClosingLine.bookmaker_id == OddsSnapshot.bookmaker_id)
            & (ClosingLine.market_type == OddsSnapshot.market_type),
        )
        .join(Game, Game.id == OddsSnapshot.game_id)
        .join(Bookmaker, Bookmaker.id == OddsSnapshot.bookmaker_id)
        .where(OddsSnapshot.timestamp >= cutoff)
    )

    results = session.execute(stmt).all()

    if len(results) < 10:
        logger.warning(f"Insufficient recent data ({len(results)} records) for evaluation")
        return {"insufficient_data": True, "record_count": len(results)}

    logger.info(f"Evaluating model on {len(results)} recent records (last {days} days)")

    errors = []
    correct_directions = 0
    total_directions = 0

    for snapshot, closing_line, game, bookmaker in results:
        hours_to_game = (game.commence_time - snapshot.timestamp).total_seconds() / 3600
        day_of_week = game.commence_time.weekday()
        is_weekend = day_of_week >= 5

        for outcome in snapshot.outcomes:
            outcome_name = outcome.get("name")
            opening_price = outcome.get("price")
            opening_point = outcome.get("point", 0.0)

            if not outcome_name or opening_price is None:
                continue

            # Find closing price
            closing_price = None
            for cl_outcome in closing_line.outcomes:
                if cl_outcome.get("name") == outcome_name:
                    closing_price = cl_outcome.get("price")
                    break

            if closing_price is None:
                continue

            # Calculate actual movement
            actual_movement = float(closing_price) - float(opening_price)

            # Calculate consensus
            consensus_line = engineer.calculate_consensus_line(
                session, game.id, snapshot.market_type, outcome_name
            )
            line_spread = engineer.calculate_line_spread(
                session, game.id, snapshot.market_type, outcome_name
            )

            # Create features - using defaults for new features
            import pandas as pd
            features = pd.DataFrame([{
                "bookmaker_id": snapshot.bookmaker_id,
                "market_type": snapshot.market_type,
                "hours_to_game": hours_to_game,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "outcome_name": outcome_name,
                "opening_price": float(opening_price),
                "opening_point": float(opening_point) if opening_point else 0.0,
                "consensus_line": consensus_line if consensus_line else float(opening_price),
                "line_spread": line_spread if line_spread else 0.0,
                "distance_from_consensus": (
                    float(opening_price) - consensus_line
                    if consensus_line
                    else 0.0
                ),
                "is_outlier": (
                    abs(float(opening_price) - consensus_line) > 0.05
                    if consensus_line
                    else False
                ),
                # Default values for new features
                "time_since_last_update": 0.0,
                "movement_velocity": 0.0,
                "updates_count": 1,
                "price_volatility_24h": 0.0,
                "cumulative_movement": 0.0,
                "movement_direction_changes": 0,
                "bookmaker_is_sharp": 1.0 if bookmaker.key.lower() == "pinnacle" else 0.0,
                "relative_to_pinnacle": 0.0,
                "books_moved_count": 0,
                "steam_move_signal": 0.0,
            }])

            try:
                prediction = model.predict_movement(features)
                predicted_movement = float(prediction["predicted_delta"][0])
                predicted_direction = prediction["predicted_direction"][0]

                # Calculate error
                error = abs(predicted_movement - actual_movement)
                errors.append(error)

                # Check direction
                if actual_movement > 0.01:
                    actual_direction = "UP"
                elif actual_movement < -0.01:
                    actual_direction = "DOWN"
                else:
                    actual_direction = "STAY"

                total_directions += 1
                if predicted_direction == actual_direction:
                    correct_directions += 1

            except Exception as e:
                logger.debug(f"Prediction error: {e}")
                continue

    if not errors:
        return {"insufficient_data": True, "record_count": 0}

    recent_mae = np.mean(errors)
    recent_accuracy = correct_directions / total_directions if total_directions > 0 else 0

    return {
        "insufficient_data": False,
        "record_count": len(errors),
        "recent_mae": recent_mae,
        "recent_accuracy": recent_accuracy,
        "days_evaluated": days,
    }


def retrain_model(session, engineer: FeatureEngineer) -> dict:
    """
    Retrain the model on all available data.

    Args:
        session: Database session
        engineer: Feature engineer

    Returns:
        Dictionary with new training metrics
    """
    logger.info("Starting model retraining...")

    # Prepare training data
    df = engineer.prepare_training_data(session)

    if len(df) == 0:
        logger.error("No training data available")
        return {"error": "No training data"}

    # Split data
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = (
        engineer.train_test_split_data(df, test_size=0.2)
    )

    # Train new model
    predictor = LineMovementPredictor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
    )
    predictor.train(X_train, y_reg_train, y_class_train)

    # Evaluate
    regression_metrics = predictor.evaluate_regression(X_test, y_reg_test)
    classification_metrics = predictor.evaluate_classification(X_test, y_class_test)

    # Save model
    predictor.save_model(MODEL_PATH)

    metrics = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "training_records": len(X_train),
        "test_records": len(X_test),
        "mae": regression_metrics["ensemble_mae"],
        "rmse": regression_metrics["ensemble_rmse"],
        "r2": regression_metrics["ensemble_r2"],
        "accuracy": classification_metrics["ensemble_accuracy"],
        "precision": classification_metrics["precision"],
        "recall": classification_metrics["recall"],
    }

    # Save metrics
    save_training_metrics(metrics)

    logger.info(f"Model retrained successfully. MAE: {metrics['mae']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    return metrics


def main(
    degradation_threshold: float = 10.0,
    force_retrain: bool = False,
    days_to_evaluate: int = 7,
):
    """
    Check model performance and retrain if needed.

    Args:
        degradation_threshold: Percentage degradation to trigger retraining
        force_retrain: If True, skip checks and retrain immediately
        days_to_evaluate: Number of recent days to evaluate
    """
    logger.info("=" * 70)
    logger.info("AUTO-RETRAINING CHECK")
    logger.info("=" * 70)
    logger.info(f"Degradation threshold: {degradation_threshold}%")
    logger.info(f"Days to evaluate: {days_to_evaluate}")

    # Create database session
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    try:
        engineer = FeatureEngineer()

        # Check if model exists
        if not Path(MODEL_PATH).exists():
            logger.info("No existing model found. Training new model...")
            metrics = retrain_model(session, engineer)
            log_entry = {
                "date": datetime.now(timezone.utc).isoformat(),
                "reason": "initial_training",
                "metrics": metrics,
            }
            log = load_retrain_log()
            log.append(log_entry)
            save_retrain_log(log)
            return

        # Load existing model
        model = LineMovementPredictor()
        model.load_model(MODEL_PATH)

        # Load training metrics
        training_metrics = load_training_metrics()

        if force_retrain:
            logger.info("Force retrain requested. Retraining model...")
            metrics = retrain_model(session, engineer)
            log_entry = {
                "date": datetime.now(timezone.utc).isoformat(),
                "reason": "force_retrain",
                "metrics": metrics,
            }
            log = load_retrain_log()
            log.append(log_entry)
            save_retrain_log(log)
            return

        # Evaluate recent performance
        logger.info("\nEvaluating recent performance...")
        recent_perf = evaluate_recent_performance(
            session, model, engineer, days=days_to_evaluate
        )

        if recent_perf.get("insufficient_data"):
            logger.warning("Insufficient recent data for evaluation. Skipping retraining check.")
            return

        logger.info(f"Recent MAE: {recent_perf['recent_mae']:.4f}")
        logger.info(f"Recent Accuracy: {recent_perf['recent_accuracy']:.4f}")

        # Compare with training metrics
        if training_metrics:
            training_mae = training_metrics.get("mae", 0)
            training_accuracy = training_metrics.get("accuracy", 0)

            if training_mae > 0:
                mae_degradation = ((recent_perf["recent_mae"] - training_mae) / training_mae) * 100
                logger.info(f"\nMAE Degradation: {mae_degradation:.1f}%")

                if mae_degradation > degradation_threshold:
                    logger.warning(f"MAE degradation ({mae_degradation:.1f}%) exceeds threshold ({degradation_threshold}%)")
                    logger.info("Triggering model retraining...")

                    metrics = retrain_model(session, engineer)

                    log_entry = {
                        "date": datetime.now(timezone.utc).isoformat(),
                        "reason": "mae_degradation",
                        "degradation_pct": mae_degradation,
                        "recent_mae": recent_perf["recent_mae"],
                        "training_mae": training_mae,
                        "new_metrics": metrics,
                    }
                    log = load_retrain_log()
                    log.append(log_entry)
                    save_retrain_log(log)
                    return

            if training_accuracy > 0:
                accuracy_drop = ((training_accuracy - recent_perf["recent_accuracy"]) / training_accuracy) * 100
                logger.info(f"Accuracy Drop: {accuracy_drop:.1f}%")

                if accuracy_drop > degradation_threshold:
                    logger.warning(f"Accuracy drop ({accuracy_drop:.1f}%) exceeds threshold ({degradation_threshold}%)")
                    logger.info("Triggering model retraining...")

                    metrics = retrain_model(session, engineer)

                    log_entry = {
                        "date": datetime.now(timezone.utc).isoformat(),
                        "reason": "accuracy_degradation",
                        "degradation_pct": accuracy_drop,
                        "recent_accuracy": recent_perf["recent_accuracy"],
                        "training_accuracy": training_accuracy,
                        "new_metrics": metrics,
                    }
                    log = load_retrain_log()
                    log.append(log_entry)
                    save_retrain_log(log)
                    return

        logger.info("\nModel performance is within acceptable range. No retraining needed.")

    except Exception as e:
        logger.error(f"Error during auto-retrain check: {e}", exc_info=True)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-retrain line movement model")
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Degradation percentage threshold to trigger retraining (default: 10.0)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining regardless of performance"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to evaluate for recent performance (default: 7)"
    )

    args = parser.parse_args()
    main(
        degradation_threshold=args.threshold,
        force_retrain=args.force,
        days_to_evaluate=args.days,
    )

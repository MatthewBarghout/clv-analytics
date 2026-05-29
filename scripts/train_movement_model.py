"""Train per-sport line movement prediction models.

Trains one XGBoost+RF ensemble per sport (basketball_nba, baseball_mlb, etc.)
and saves to models/line_movement_predictor_{sport_key}.pkl.

With --sport: train only that sport.
Without --sport: discover all sports in DB and train each.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzers.features import FeatureEngineer
from src.analyzers.movement_predictor import LineMovementPredictor
from src.models.database import Sport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

if not all([POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB]):
    raise ValueError("Database configuration incomplete. Check POSTGRES_* environment variables.")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

MIN_TRAINING_ROWS = 200


def model_path(sport_key: str) -> str:
    return f"models/line_movement_predictor_{sport_key}.pkl"


def walk_forward_validation(df, n_splits=5, n_estimators=100, max_depth=6, learning_rate=0.1):
    logger.info(f"Walk-forward validation — {n_splits} folds")

    feature_cols = [col for col in df.columns if col not in [
        "price_movement", "point_movement", "directional_movement", "snapshot_timestamp"
    ]]
    X = df[feature_cols]
    y_regression = df[["price_movement", "point_movement"]]
    y_classification = df["directional_movement"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = {"regression_mae": [], "regression_rmse": [], "classification_accuracy": [], "baseline_mae": []}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        logger.info(f"  Fold {fold}/{n_splits} — train={len(train_idx)}, test={len(test_idx)}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_reg_train = y_regression.iloc[train_idx]
        y_reg_test = y_regression.iloc[test_idx]
        y_class_train = y_classification.iloc[train_idx]
        y_class_test = y_classification.iloc[test_idx]

        predictor = LineMovementPredictor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
        predictor.train(X_train, y_reg_train, y_class_train)

        reg_metrics = predictor.evaluate_regression(X_test, y_reg_test)
        class_metrics = predictor.evaluate_classification(X_test, y_class_test)
        baseline_mae = np.abs(y_reg_test["price_movement"]).mean()

        fold_metrics["regression_mae"].append(reg_metrics["ensemble_mae"])
        fold_metrics["regression_rmse"].append(reg_metrics["ensemble_rmse"])
        fold_metrics["classification_accuracy"].append(class_metrics["ensemble_accuracy"])
        fold_metrics["baseline_mae"].append(baseline_mae)

        logger.info(f"    MAE={reg_metrics['ensemble_mae']:.4f} (baseline={baseline_mae:.4f})  Accuracy={class_metrics['ensemble_accuracy']:.4f}")

    results = {
        "avg_mae": np.mean(fold_metrics["regression_mae"]),
        "std_mae": np.std(fold_metrics["regression_mae"]),
        "avg_accuracy": np.mean(fold_metrics["classification_accuracy"]),
        "std_accuracy": np.std(fold_metrics["classification_accuracy"]),
    }
    logger.info(f"Walk-forward: MAE={results['avg_mae']:.4f} ±{results['std_mae']:.4f}  Accuracy={results['avg_accuracy']:.4f} ±{results['std_accuracy']:.4f}")
    return results


def train_sport(session: Session, sport_key: str, use_walk_forward: bool, n_splits: int) -> bool:
    """Train and save a model for one sport. Returns True if successful."""
    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING: {sport_key.upper()}")
    logger.info("=" * 80)

    engineer = FeatureEngineer()
    df = engineer.prepare_training_data(session, sport_key=sport_key)

    if len(df) == 0:
        logger.warning(f"No training data for {sport_key} — skipping")
        return False

    if len(df) < MIN_TRAINING_ROWS:
        logger.warning(f"Only {len(df)} rows for {sport_key} (min={MIN_TRAINING_ROWS}) — skipping")
        return False

    total = len(df)
    up = len(df[df["directional_movement"] == "UP"])
    down = len(df[df["directional_movement"] == "DOWN"])
    stay = len(df[df["directional_movement"] == "STAY"])
    logger.info(f"Records: {total}  (UP={up} {up/total*100:.1f}%  DOWN={down} {down/total*100:.1f}%  STAY={stay} {stay/total*100:.1f}%)")

    for market in df["market_type"].unique():
        mdf = df[df["market_type"] == market]
        logger.info(f"  {market}: {len(mdf)} records, avg movement={mdf['price_movement'].abs().mean():.4f}")

    wf_results = None
    if use_walk_forward:
        logger.info("\n--- Walk-Forward Validation ---")
        wf_results = walk_forward_validation(df, n_splits=n_splits)

    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = engineer.train_test_split_data(df)

    predictor = LineMovementPredictor(n_estimators=100, max_depth=6, learning_rate=0.1)
    predictor.train(X_train, y_reg_train, y_class_train)

    reg_metrics = predictor.evaluate_regression(X_test, y_reg_test)
    class_metrics = predictor.evaluate_classification(X_test, y_class_test)

    baseline_mae = np.abs(y_reg_test["price_movement"]).mean()
    baseline_acc = y_class_train.value_counts().max() / len(y_class_train)
    logger.info(f"\nFinal model vs baseline:")
    logger.info(f"  Accuracy: {class_metrics['ensemble_accuracy']:.4f} vs {baseline_acc:.4f} baseline")
    logger.info(f"  MAE:      {reg_metrics['ensemble_mae']:.4f} vs {baseline_mae:.4f} baseline")

    importance = predictor.get_feature_importance()
    logger.info("\nTop 10 features:")
    for i, (feat, score) in enumerate(list(importance.items())[:10], 1):
        logger.info(f"  {i:>2}. {feat:<35} {score:.4f}")

    out_path = model_path(sport_key)
    predictor.save_model(out_path)
    logger.info(f"Saved: {out_path}")

    if wf_results:
        logger.info(f"Walk-forward: Accuracy={wf_results['avg_accuracy']:.4f} ±{wf_results['std_accuracy']:.4f}")

    return True


def main(use_walk_forward: bool = False, n_splits: int = 5, sport_key: str = None):
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    try:
        if sport_key:
            sports_to_train = [sport_key]
        else:
            sports = session.execute(select(Sport)).scalars().all()
            sports_to_train = [s.key for s in sports]
            if not sports_to_train:
                logger.error("No sports found in database")
                return
            logger.info(f"Found {len(sports_to_train)} sports: {sports_to_train}")

        results = {}
        for sk in sports_to_train:
            results[sk] = train_sport(session, sk, use_walk_forward, n_splits)

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        for sk, success in results.items():
            status = "OK" if success else "SKIPPED"
            logger.info(f"  {sk}: {status}  →  {model_path(sk) if success else 'no model saved'}")

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train per-sport line movement prediction models")
    parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward validation")
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-forward folds (default: 5)")
    parser.add_argument("--sport", type=str, default=None, help="Train only this sport key (e.g. basketball_nba)")
    args = parser.parse_args()
    main(use_walk_forward=args.walk_forward, n_splits=args.n_splits, sport_key=args.sport)

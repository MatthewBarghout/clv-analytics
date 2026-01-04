"""Train line movement prediction model with XGBoost."""
import logging
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
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


def main():
    """Train and evaluate line movement prediction model."""
    logger.info("=" * 80)
    logger.info("LINE MOVEMENT PREDICTION MODEL TRAINING")
    logger.info("=" * 80)

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

        # Step 3: Split data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: SPLITTING DATA")
        logger.info("=" * 80)

        X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = (
            engineer.train_test_split_data(df, test_size=0.2)
        )

        # Step 4: Train models
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: TRAINING MODELS")
        logger.info("=" * 80)

        predictor = LineMovementPredictor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
        )
        predictor.train(X_train, y_reg_train, y_class_train)

        # Step 5: Evaluate regression model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: REGRESSION MODEL EVALUATION")
        logger.info("=" * 80)

        regression_metrics = predictor.evaluate_regression(X_test, y_reg_test)

        # Calculate baseline (predict no movement)
        baseline_mae = np.abs(y_reg_test["price_movement"]).mean()
        improvement = ((baseline_mae - regression_metrics["mae"]) / baseline_mae) * 100

        logger.info(f"\nBaseline Comparison:")
        logger.info(f"  Baseline MAE (predict no movement): {baseline_mae:.4f}")
        logger.info(f"  Model MAE: {regression_metrics['mae']:.4f}")
        logger.info(f"  Improvement: {improvement:.1f}%")

        # Step 6: Evaluate classification model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: CLASSIFICATION MODEL EVALUATION")
        logger.info("=" * 80)

        classification_metrics = predictor.evaluate_classification(X_test, y_class_test)

        # Calculate baseline (predict most frequent class)
        baseline_accuracy = y_class_train.value_counts().max() / len(y_class_train)
        logger.info(f"\nBaseline Comparison:")
        logger.info(f"  Baseline accuracy (predict most frequent): {baseline_accuracy:.4f}")
        logger.info(f"  Model accuracy: {classification_metrics['accuracy']:.4f}")
        logger.info(f"  Improvement: {(classification_metrics['accuracy'] - baseline_accuracy):.4f}")

        # Step 7: Feature importance
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: FEATURE IMPORTANCE")
        logger.info("=" * 80)

        importance = predictor.get_feature_importance()
        logger.info("\nTop 10 Most Important Features:")
        for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
            logger.info(f"  {i:>2}. {feature:<30} {score:.4f}")

        # Step 8: Save model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: SAVING MODEL")
        logger.info("=" * 80)

        predictor.save_model(MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Training records: {len(X_train)}")
        logger.info(f"Test records: {len(X_test)}")
        logger.info(f"\nRegression Performance:")
        logger.info(f"  MAE: {regression_metrics['mae']:.4f}")
        logger.info(f"  RMSE: {regression_metrics['rmse']:.4f}")
        logger.info(f"  R²: {regression_metrics['r2']:.4f}")
        logger.info(f"\nClassification Performance:")
        logger.info(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {classification_metrics['precision']:.4f}")
        logger.info(f"  Recall: {classification_metrics['recall']:.4f}")
        logger.info(f"\nModel saved to: {MODEL_PATH}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the closing line prediction ML model."""
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzers.features import FeatureEngineer
from src.analyzers.ml_predictor import ClosingLinePredictor

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Train and save the ML model."""
    logger.info("=" * 80)
    logger.info("Starting ML Model Training")
    logger.info("=" * 80)

    # Database setup
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        logger.error("DATABASE_URL not found in environment")
        sys.exit(1)

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    try:
        # Initialize feature engineer
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading and Preparing Data")
        logger.info("=" * 80)

        engineer = FeatureEngineer()

        # Prepare training data
        df = engineer.prepare_training_data(session)

        if len(df) == 0:
            logger.error("No training data available. Please collect odds data first.")
            sys.exit(1)

        # Print dataset statistics
        logger.info("\n" + "-" * 80)
        logger.info("Dataset Statistics:")
        logger.info("-" * 80)
        logger.info(f"Total records: {len(df)}")
        logger.info(f"\nFeature statistics:")
        logger.info(f"\n{df.describe()}")

        # Print bookmaker distribution
        logger.info(f"\nBookmaker distribution:")
        bookmaker_counts = df["bookmaker_id"].value_counts()
        for bookmaker_id, count in bookmaker_counts.items():
            logger.info(f"  Bookmaker {bookmaker_id}: {count} records")

        # Print market type distribution
        logger.info(f"\nMarket type distribution:")
        market_counts = df["market_type"].value_counts()
        for market_type, count in market_counts.items():
            logger.info(f"  {market_type}: {count} records")

        # Split data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Splitting Data")
        logger.info("=" * 80)

        X_train, X_test, y_train, y_test = engineer.train_test_split_data(df)

        # Get opening odds for directional accuracy calculation
        X_train_with_target = df.iloc[X_train.index]
        X_test_with_target = df.iloc[X_test.index]
        opening_odds_test = X_test_with_target["opening_odds"]

        # Train model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Training Model")
        logger.info("=" * 80)

        predictor = ClosingLinePredictor(n_estimators=100)
        predictor.train(X_train, y_train)

        # Evaluate model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Evaluating Model")
        logger.info("=" * 80)

        metrics = predictor.evaluate(X_test, y_test)

        logger.info("\nModel Performance Metrics:")
        logger.info("-" * 80)
        logger.info(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        logger.info(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        logger.info(f"R² Score: {metrics['r2_score']:.4f}")

        # Calculate directional accuracy
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Calculating Directional Accuracy")
        logger.info("=" * 80)

        predictions = predictor.predict(X_test)
        directional_accuracy = predictor.calculate_directional_accuracy(
            y_test, predictions, opening_odds_test
        )

        logger.info(f"\nDirectional Accuracy: {directional_accuracy:.2f}%")
        logger.info(
            "(Percentage of predictions that correctly predicted direction of line movement)"
        )

        # Get feature importance
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Feature Importance Analysis")
        logger.info("=" * 80)

        importance = predictor.get_feature_importance()

        logger.info("\nFeature Importance Ranking:")
        logger.info("-" * 80)
        for i, (feature, score) in enumerate(importance.items(), 1):
            logger.info(f"{i:2d}. {feature:30s} {score:.4f}")

        # Save model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Saving Model")
        logger.info("=" * 80)

        model_path = "models/closing_line_predictor.pkl"
        predictor.save_model(model_path)

        logger.info(f"\nModel successfully saved to: {model_path}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Training records: {len(X_train)}")
        logger.info(f"Test records: {len(X_test)}")
        logger.info(f"R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"Directional Accuracy: {directional_accuracy:.2f}%")
        logger.info(f"Model saved to: {model_path}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)

    finally:
        session.close()


if __name__ == "__main__":
    main()

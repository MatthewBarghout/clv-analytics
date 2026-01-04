"""Feature engineering for closing line prediction ML model."""
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.database import Bookmaker, ClosingLine, Game, OddsSnapshot

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature extraction and data preparation for ML model training."""

    def extract_features_from_snapshot(
        self, snapshot: OddsSnapshot, game: Game
    ) -> Optional[dict]:
        """
        Extract features from an odds snapshot.

        Args:
            snapshot: OddsSnapshot instance
            game: Associated Game instance

        Returns:
            Dictionary of features or None if extraction fails
        """
        try:
            # Extract opening odds (average price across outcomes)
            opening_odds = self._extract_average_odds(snapshot.outcomes)
            if opening_odds is None:
                return None

            # Calculate hours to game
            hours_to_game = (game.commence_time - snapshot.timestamp).total_seconds() / 3600

            # Extract day of week (0 = Monday, 6 = Sunday)
            day_of_week = game.commence_time.weekday()

            # Check if weekend (Saturday = 5, Sunday = 6)
            is_weekend = day_of_week >= 5

            return {
                "opening_odds": opening_odds,
                "bookmaker_id": snapshot.bookmaker_id,
                "market_type": snapshot.market_type,
                "hours_to_game": hours_to_game,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
            }

        except Exception as e:
            logger.error(f"Error extracting features from snapshot {snapshot.id}: {e}")
            return None

    def _extract_average_odds(self, outcomes: list) -> Optional[float]:
        """
        Extract average American odds from outcomes.

        Args:
            outcomes: List of outcome dictionaries

        Returns:
            Average odds value or None if extraction fails
        """
        try:
            odds_values = []
            for outcome in outcomes:
                price = outcome.get("price")
                if price is not None:
                    odds_values.append(float(price))

            if not odds_values:
                return None

            return sum(odds_values) / len(odds_values)

        except Exception as e:
            logger.error(f"Error extracting average odds: {e}")
            return None

    def _extract_closing_odds(self, closing_line: ClosingLine) -> Optional[float]:
        """
        Extract average closing odds from closing line.

        Args:
            closing_line: ClosingLine instance

        Returns:
            Average closing odds or None if extraction fails
        """
        return self._extract_average_odds(closing_line.outcomes)

    def prepare_training_data(self, session: Session) -> pd.DataFrame:
        """
        Prepare training data from database.

        Queries all odds snapshots that have corresponding closing lines,
        extracts features and target values.

        Args:
            session: SQLAlchemy database session

        Returns:
            DataFrame with features and target variable
        """
        logger.info("Preparing training data...")

        # Query snapshots with corresponding closing lines
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
        )

        results = session.execute(stmt).all()
        logger.info(f"Found {len(results)} snapshot-closing line pairs")

        # Extract features and targets
        data_rows = []
        for snapshot, closing_line, game, bookmaker in results:
            # Extract features
            features = self.extract_features_from_snapshot(snapshot, game)
            if features is None:
                continue

            # Extract target (closing odds)
            closing_odds = self._extract_closing_odds(closing_line)
            if closing_odds is None:
                continue

            # Combine into row
            row = {**features, "closing_odds": closing_odds}
            data_rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        # Filter to only complete records (no nulls)
        df = df.dropna()

        logger.info(f"Prepared {len(df)} complete training records")
        logger.info(f"Columns: {list(df.columns)}")

        return df

    def train_test_split_data(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.

        Args:
            df: DataFrame with features and target
            test_size: Proportion of data to use for testing (default 0.2)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split

        # Separate features and target
        X = df.drop("closing_odds", axis=1)
        y = df["closing_odds"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        return X_train, X_test, y_train, y_test

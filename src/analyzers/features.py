"""Feature engineering for line movement prediction ML model."""
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models.database import Bookmaker, ClosingLine, Game, OddsSnapshot

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature extraction and data preparation for line movement prediction."""

    def calculate_consensus_line(
        self, session: Session, game_id: int, market_type: str, outcome_name: str
    ) -> Optional[float]:
        """
        Calculate consensus (average) opening line across all bookmakers.

        Args:
            session: Database session
            game_id: Game ID
            market_type: Market type (spreads/totals/h2h)
            outcome_name: Outcome name (team name, Over/Under)

        Returns:
            Average opening price across bookmakers, or None if insufficient data
        """
        try:
            # Query all opening snapshots for this game/market
            stmt = (
                select(OddsSnapshot)
                .where(OddsSnapshot.game_id == game_id)
                .where(OddsSnapshot.market_type == market_type)
            )
            snapshots = session.execute(stmt).scalars().all()

            if len(snapshots) < 2:
                return None  # Need at least 2 bookmakers for consensus

            # Extract prices for this outcome
            prices = []
            for snapshot in snapshots:
                for outcome in snapshot.outcomes:
                    if outcome.get("name") == outcome_name:
                        price = outcome.get("price")
                        if price is not None:
                            prices.append(float(price))
                        break

            if len(prices) < 2:
                return None

            return float(np.mean(prices))

        except Exception as e:
            logger.error(f"Error calculating consensus line: {e}")
            return None

    def calculate_line_spread(
        self, session: Session, game_id: int, market_type: str, outcome_name: str
    ) -> Optional[float]:
        """
        Calculate line spread (max - min) across all bookmakers.

        Indicates market disagreement - higher spread = more uncertainty.

        Args:
            session: Database session
            game_id: Game ID
            market_type: Market type
            outcome_name: Outcome name

        Returns:
            Price spread (max - min), or None if insufficient data
        """
        try:
            # Query all opening snapshots
            stmt = (
                select(OddsSnapshot)
                .where(OddsSnapshot.game_id == game_id)
                .where(OddsSnapshot.market_type == market_type)
            )
            snapshots = session.execute(stmt).scalars().all()

            if len(snapshots) < 2:
                return None

            # Extract prices for this outcome
            prices = []
            for snapshot in snapshots:
                for outcome in snapshot.outcomes:
                    if outcome.get("name") == outcome_name:
                        price = outcome.get("price")
                        if price is not None:
                            prices.append(float(price))
                        break

            if len(prices) < 2:
                return None

            return float(np.max(prices) - np.min(prices))

        except Exception as e:
            logger.error(f"Error calculating line spread: {e}")
            return None

    def prepare_training_data(self, session: Session) -> pd.DataFrame:
        """
        Prepare training data with movement deltas as targets.

        Queries all odds snapshots that have corresponding closing lines,
        extracts features including consensus/spread, and calculates movement.

        Args:
            session: SQLAlchemy database session

        Returns:
            DataFrame with features, movement_delta, and directional_movement
        """
        logger.info("Preparing training data for movement prediction...")

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

        # Extract features and targets PER OUTCOME
        data_rows = []
        for snapshot, closing_line, game, bookmaker in results:
            # Calculate base features
            hours_to_game = (game.commence_time - snapshot.timestamp).total_seconds() / 3600
            day_of_week = game.commence_time.weekday()
            is_weekend = day_of_week >= 5

            # Process each outcome separately
            for opening_outcome in snapshot.outcomes:
                outcome_name = opening_outcome.get("name")
                opening_price = opening_outcome.get("price")
                opening_point = opening_outcome.get("point")

                if not outcome_name or opening_price is None:
                    continue

                # Find matching closing outcome
                closing_outcome = None
                for cl_outcome in closing_line.outcomes:
                    if cl_outcome.get("name") == outcome_name:
                        closing_outcome = cl_outcome
                        break

                if not closing_outcome:
                    continue

                closing_price = closing_outcome.get("price")
                closing_point = closing_outcome.get("point")

                if closing_price is None:
                    continue

                # Calculate consensus and spread
                consensus_line = self.calculate_consensus_line(
                    session, game.id, snapshot.market_type, outcome_name
                )
                line_spread = self.calculate_line_spread(
                    session, game.id, snapshot.market_type, outcome_name
                )

                # Calculate movement targets
                price_movement = float(closing_price) - float(opening_price)
                point_movement = (
                    float(closing_point) - float(opening_point)
                    if opening_point is not None and closing_point is not None
                    else 0.0
                )

                # Determine directional movement for price
                if price_movement > 0.01:  # Moved up
                    direction = "UP"
                elif price_movement < -0.01:  # Moved down
                    direction = "DOWN"
                else:  # Stayed same
                    direction = "STAY"

                # Create row
                row = {
                    "bookmaker_id": snapshot.bookmaker_id,
                    "market_type": snapshot.market_type,
                    "hours_to_game": hours_to_game,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "outcome_name": outcome_name,
                    "opening_price": float(opening_price),
                    "opening_point": float(opening_point) if opening_point is not None else 0.0,
                    "consensus_line": consensus_line if consensus_line is not None else float(opening_price),
                    "line_spread": line_spread if line_spread is not None else 0.0,
                    "distance_from_consensus": (
                        float(opening_price) - consensus_line
                        if consensus_line is not None
                        else 0.0
                    ),
                    "is_outlier": (
                        abs(float(opening_price) - consensus_line) > 0.05
                        if consensus_line is not None
                        else False
                    ),
                    # Targets
                    "price_movement": price_movement,
                    "point_movement": point_movement,
                    "directional_movement": direction,
                }
                data_rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        # Filter to only complete records
        df = df.dropna(subset=["price_movement", "directional_movement"])

        logger.info(f"Prepared {len(df)} training records")
        logger.info(f"Movement distribution:")
        logger.info(f"  UP: {len(df[df['directional_movement'] == 'UP'])}")
        logger.info(f"  DOWN: {len(df[df['directional_movement'] == 'DOWN'])}")
        logger.info(f"  STAY: {len(df[df['directional_movement'] == 'STAY'])}")
        logger.info(f"  Average price movement: {df['price_movement'].mean():.4f}")
        logger.info(f"  Average point movement: {df['point_movement'].mean():.4f}")

        return df

    def train_test_split_data(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Split data into training and test sets for movement prediction.

        Args:
            df: DataFrame with features and targets
            test_size: Proportion of data to use for testing (default 0.2)

        Returns:
            Tuple of (X_train, X_test, y_train_regression, y_train_classification)
            where y_train_regression has price/point movement columns
            and y_train_classification is directional_movement
        """
        from sklearn.model_selection import train_test_split

        # Separate features and targets
        feature_cols = [
            "bookmaker_id",
            "market_type",
            "hours_to_game",
            "day_of_week",
            "is_weekend",
            "outcome_name",
            "opening_price",
            "opening_point",
            "consensus_line",
            "line_spread",
            "distance_from_consensus",
            "is_outlier",
        ]

        X = df[feature_cols]
        y_regression = df[["price_movement", "point_movement"]]
        y_classification = df["directional_movement"]

        # Split data
        X_train, X_test, y_reg_train, y_reg_test = train_test_split(
            X, y_regression, test_size=test_size, random_state=42
        )

        # Also split classification target with same indices
        y_class_train = y_classification.iloc[X_train.index]
        y_class_test = y_classification.iloc[X_test.index]

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        return X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test

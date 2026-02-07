"""Feature engineering for line movement prediction ML model."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from src.models.database import Bookmaker, ClosingLine, Game, OddsSnapshot

logger = logging.getLogger(__name__)

# Market-specific direction thresholds for classifying movement
DIRECTION_THRESHOLDS = {
    'h2h': 0.02,      # Moneyline moves more freely
    'spreads': 0.01,  # Spreads are tighter
    'totals': 0.01,   # Totals are tighter
}

# Sharp bookmakers that set the market
SHARP_BOOKMAKERS = {'pinnacle', 'pinnaclesports', 'betcris', 'bookmaker'}


def get_direction_threshold(market_type: str) -> float:
    """Get the direction classification threshold for a market type."""
    return DIRECTION_THRESHOLDS.get(market_type, 0.01)


class FeatureEngineer:
    """Handles feature extraction and data preparation for line movement prediction."""

    def __init__(self):
        """Initialize feature engineer with cached bookmaker info."""
        self._bookmaker_cache: Dict[int, str] = {}
        self._pinnacle_id: Optional[int] = None

    def _get_bookmaker_name(self, session: Session, bookmaker_id: int) -> str:
        """Get bookmaker name with caching."""
        if bookmaker_id not in self._bookmaker_cache:
            bookmaker = session.query(Bookmaker).filter(Bookmaker.id == bookmaker_id).first()
            self._bookmaker_cache[bookmaker_id] = bookmaker.key.lower() if bookmaker else ""
        return self._bookmaker_cache[bookmaker_id]

    def _get_pinnacle_id(self, session: Session) -> Optional[int]:
        """Get Pinnacle bookmaker ID with caching."""
        if self._pinnacle_id is None:
            for sharp in SHARP_BOOKMAKERS:
                bookmaker = session.query(Bookmaker).filter(Bookmaker.key.ilike(f"%{sharp}%")).first()
                if bookmaker:
                    self._pinnacle_id = bookmaker.id
                    break
        return self._pinnacle_id

    def is_sharp_bookmaker(self, session: Session, bookmaker_id: int) -> bool:
        """Check if a bookmaker is considered 'sharp' (market-setting)."""
        name = self._get_bookmaker_name(session, bookmaker_id)
        return name in SHARP_BOOKMAKERS

    def calculate_temporal_features(
        self,
        session: Session,
        game_id: int,
        bookmaker_id: int,
        market_type: str,
        outcome_name: str,
        current_timestamp: datetime,
    ) -> Dict[str, float]:
        """
        Calculate temporal features based on historical snapshots.

        Returns:
            Dict with temporal features:
            - time_since_last_update: Hours since previous snapshot
            - movement_velocity: Rate of change (delta/hours) from last update
            - updates_count: Number of prior snapshots
            - price_volatility_24h: Std dev of prices in last 24 hours
            - cumulative_movement: Total movement from first snapshot
            - movement_direction_changes: Number of direction reversals
        """
        try:
            # Get all prior snapshots for this outcome
            stmt = (
                select(OddsSnapshot)
                .where(OddsSnapshot.game_id == game_id)
                .where(OddsSnapshot.bookmaker_id == bookmaker_id)
                .where(OddsSnapshot.market_type == market_type)
                .where(OddsSnapshot.timestamp < current_timestamp)
                .order_by(OddsSnapshot.timestamp.asc())
            )
            prior_snapshots = session.execute(stmt).scalars().all()

            if not prior_snapshots:
                return {
                    "time_since_last_update": 0.0,
                    "movement_velocity": 0.0,
                    "updates_count": 0,
                    "price_volatility_24h": 0.0,
                    "cumulative_movement": 0.0,
                    "movement_direction_changes": 0,
                }

            # Extract prices for this outcome
            prices_with_times = []
            for snapshot in prior_snapshots:
                for outcome in snapshot.outcomes:
                    if outcome.get("name") == outcome_name:
                        price = outcome.get("price")
                        if price is not None:
                            prices_with_times.append((snapshot.timestamp, float(price)))
                        break

            if not prices_with_times:
                return {
                    "time_since_last_update": 0.0,
                    "movement_velocity": 0.0,
                    "updates_count": 0,
                    "price_volatility_24h": 0.0,
                    "cumulative_movement": 0.0,
                    "movement_direction_changes": 0,
                }

            # Time since last update
            last_time, last_price = prices_with_times[-1]
            time_since_last = (current_timestamp - last_time).total_seconds() / 3600

            # Movement velocity
            if len(prices_with_times) >= 2:
                prev_time, prev_price = prices_with_times[-2]
                time_delta = (last_time - prev_time).total_seconds() / 3600
                price_delta = last_price - prev_price
                velocity = price_delta / time_delta if time_delta > 0 else 0.0
            else:
                velocity = 0.0

            # Cumulative movement from first snapshot
            first_price = prices_with_times[0][1]
            cumulative_movement = last_price - first_price

            # Price volatility in last 24 hours
            cutoff_24h = current_timestamp - timedelta(hours=24)
            recent_prices = [p for t, p in prices_with_times if t >= cutoff_24h]
            volatility = float(np.std(recent_prices)) if len(recent_prices) > 1 else 0.0

            # Direction changes
            direction_changes = 0
            if len(prices_with_times) >= 3:
                prices_only = [p for _, p in prices_with_times]
                for i in range(1, len(prices_only) - 1):
                    prev_direction = prices_only[i] - prices_only[i - 1]
                    next_direction = prices_only[i + 1] - prices_only[i]
                    if (prev_direction > 0 and next_direction < 0) or (prev_direction < 0 and next_direction > 0):
                        direction_changes += 1

            return {
                "time_since_last_update": time_since_last,
                "movement_velocity": velocity,
                "updates_count": len(prices_with_times),
                "price_volatility_24h": volatility,
                "cumulative_movement": cumulative_movement,
                "movement_direction_changes": direction_changes,
            }

        except Exception as e:
            logger.error(f"Error calculating temporal features: {e}")
            return {
                "time_since_last_update": 0.0,
                "movement_velocity": 0.0,
                "updates_count": 0,
                "price_volatility_24h": 0.0,
                "cumulative_movement": 0.0,
                "movement_direction_changes": 0,
            }

    def calculate_bookmaker_features(
        self,
        session: Session,
        game_id: int,
        bookmaker_id: int,
        market_type: str,
        outcome_name: str,
        current_price: float,
    ) -> Dict[str, float]:
        """
        Calculate bookmaker-specific features.

        Returns:
            Dict with bookmaker features:
            - bookmaker_is_sharp: 1 if Pinnacle/sharp book, 0 otherwise
            - relative_to_pinnacle: Distance from Pinnacle line
            - books_moved_count: Number of books that have moved in same direction
            - steam_move_signal: Signal indicating coordinated movement across books
        """
        try:
            is_sharp = 1.0 if self.is_sharp_bookmaker(session, bookmaker_id) else 0.0

            # Get Pinnacle line if available
            pinnacle_id = self._get_pinnacle_id(session)
            relative_to_pinnacle = 0.0

            if pinnacle_id and pinnacle_id != bookmaker_id:
                stmt = (
                    select(OddsSnapshot)
                    .where(OddsSnapshot.game_id == game_id)
                    .where(OddsSnapshot.bookmaker_id == pinnacle_id)
                    .where(OddsSnapshot.market_type == market_type)
                    .order_by(OddsSnapshot.timestamp.desc())
                    .limit(1)
                )
                pinnacle_snapshot = session.execute(stmt).scalar_one_or_none()

                if pinnacle_snapshot:
                    for outcome in pinnacle_snapshot.outcomes:
                        if outcome.get("name") == outcome_name:
                            pinnacle_price = outcome.get("price")
                            if pinnacle_price:
                                relative_to_pinnacle = current_price - float(pinnacle_price)
                            break

            # Count books that have moved - get latest snapshot from each bookmaker
            stmt = (
                select(OddsSnapshot)
                .where(OddsSnapshot.game_id == game_id)
                .where(OddsSnapshot.market_type == market_type)
                .order_by(OddsSnapshot.bookmaker_id, OddsSnapshot.timestamp.desc())
            )
            all_snapshots = session.execute(stmt).scalars().all()

            # Group by bookmaker and track latest prices
            latest_by_book = {}
            first_by_book = {}
            for snapshot in all_snapshots:
                bid = snapshot.bookmaker_id
                for outcome in snapshot.outcomes:
                    if outcome.get("name") == outcome_name:
                        price = outcome.get("price")
                        if price:
                            if bid not in latest_by_book:
                                latest_by_book[bid] = float(price)
                            first_by_book[bid] = float(price)  # Will be overwritten to get first
                        break

            # Count direction of movements
            up_moves = 0
            down_moves = 0
            for bid, latest_price in latest_by_book.items():
                first_price = first_by_book.get(bid, latest_price)
                if latest_price > first_price + 0.01:
                    up_moves += 1
                elif latest_price < first_price - 0.01:
                    down_moves += 1

            total_books = len(latest_by_book)
            steam_signal = 0.0
            if total_books >= 3:
                # Steam move if 60%+ of books moved in same direction
                max_moves = max(up_moves, down_moves)
                if max_moves / total_books >= 0.6:
                    steam_signal = 1.0 if up_moves > down_moves else -1.0

            return {
                "bookmaker_is_sharp": is_sharp,
                "relative_to_pinnacle": relative_to_pinnacle,
                "books_moved_count": up_moves + down_moves,
                "steam_move_signal": steam_signal,
            }

        except Exception as e:
            logger.error(f"Error calculating bookmaker features: {e}")
            return {
                "bookmaker_is_sharp": 0.0,
                "relative_to_pinnacle": 0.0,
                "books_moved_count": 0,
                "steam_move_signal": 0.0,
            }

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

                # Calculate temporal features
                temporal_features = self.calculate_temporal_features(
                    session, game.id, snapshot.bookmaker_id,
                    snapshot.market_type, outcome_name, snapshot.timestamp
                )

                # Calculate bookmaker features
                bookmaker_features = self.calculate_bookmaker_features(
                    session, game.id, snapshot.bookmaker_id,
                    snapshot.market_type, outcome_name, float(opening_price)
                )

                # Calculate movement targets
                price_movement = float(closing_price) - float(opening_price)
                point_movement = (
                    float(closing_point) - float(opening_point)
                    if opening_point is not None and closing_point is not None
                    else 0.0
                )

                # Determine directional movement using market-specific threshold
                threshold = get_direction_threshold(snapshot.market_type)
                if price_movement > threshold:
                    direction = "UP"
                elif price_movement < -threshold:
                    direction = "DOWN"
                else:
                    direction = "STAY"

                # Create row with all features
                row = {
                    # Base features
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
                    # Temporal features
                    "time_since_last_update": temporal_features["time_since_last_update"],
                    "movement_velocity": temporal_features["movement_velocity"],
                    "updates_count": temporal_features["updates_count"],
                    "price_volatility_24h": temporal_features["price_volatility_24h"],
                    "cumulative_movement": temporal_features["cumulative_movement"],
                    "movement_direction_changes": temporal_features["movement_direction_changes"],
                    # Bookmaker features
                    "bookmaker_is_sharp": bookmaker_features["bookmaker_is_sharp"],
                    "relative_to_pinnacle": bookmaker_features["relative_to_pinnacle"],
                    "books_moved_count": bookmaker_features["books_moved_count"],
                    "steam_move_signal": bookmaker_features["steam_move_signal"],
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
            # Base features
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
            # Temporal features
            "time_since_last_update",
            "movement_velocity",
            "updates_count",
            "price_volatility_24h",
            "cumulative_movement",
            "movement_direction_changes",
            # Bookmaker features
            "bookmaker_is_sharp",
            "relative_to_pinnacle",
            "books_moved_count",
            "steam_move_signal",
        ]

        # Filter to only columns that exist in the dataframe
        feature_cols = [col for col in feature_cols if col in df.columns]

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

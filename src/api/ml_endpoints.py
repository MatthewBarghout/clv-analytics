"""API endpoints for machine learning line movement prediction."""
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from src.analyzers.features import FeatureEngineer
from src.analyzers.movement_predictor import LineMovementPredictor
from src.models.database import BestEVPick, BettingOutcome, Bookmaker, ClosingLine, DailyCLVReport, Game, OddsSnapshot, OpportunityPerformance, Sport, Team

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Setup router
router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])

# Database setup
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

if not all([POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB]):
    raise ValueError("Database configuration incomplete. Check POSTGRES_* environment variables.")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Model path
MODEL_PATH = "models/line_movement_predictor.pkl"

# Global model instance
_model: Optional[LineMovementPredictor] = None


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass


def get_model() -> LineMovementPredictor:
    """Get or load the trained movement prediction model."""
    global _model

    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(
                status_code=404,
                detail="Model not trained yet. Please run scripts/train_movement_model.py first.",
            )

        _model = LineMovementPredictor()
        _model.load_model(MODEL_PATH)
        logger.info("Movement prediction model loaded successfully")

    return _model


# Response models
class MovementModelStats(BaseModel):
    """Movement prediction model statistics."""

    is_trained: bool
    movement_mae: Optional[float] = None
    movement_rmse: Optional[float] = None
    movement_r2: Optional[float] = None
    directional_accuracy: Optional[float] = None
    directional_precision: Optional[float] = None
    directional_recall: Optional[float] = None
    training_records: Optional[int] = None
    last_trained: Optional[str] = None
    baseline_mae: Optional[float] = None
    improvement_vs_baseline: Optional[float] = None


class FeatureImportance(BaseModel):
    """Feature importance data."""

    feature_name: str
    importance: float


class MovementPrediction(BaseModel):
    """Movement prediction for a single outcome."""

    outcome_name: str
    opening_price: float
    opening_point: float
    predicted_movement: float
    predicted_direction: str
    direction_confidence: float
    predicted_closing_price: float
    predicted_closing_point: float
    actual_closing_price: Optional[float] = None
    actual_closing_point: Optional[float] = None
    movement_error: Optional[float] = None
    was_constrained: bool = False
    unconstrained_movement: Optional[float] = None


class SnapshotMovementPrediction(BaseModel):
    """Movement predictions for a single snapshot."""

    snapshot_id: int
    bookmaker_name: str
    market_type: str
    timestamp: datetime
    outcomes: List[MovementPrediction]


class GameMovementPredictions(BaseModel):
    """Movement predictions for a game."""

    game_id: int
    predictions: List[SnapshotMovementPrediction]


class EVOpportunity(BaseModel):
    """Expected value betting opportunity based on predicted movement."""

    game_id: int
    home_team: str
    away_team: str
    commence_time: datetime
    bookmaker_name: str
    market_type: str
    outcome_name: str
    current_line: str
    predicted_movement: float
    predicted_direction: str
    confidence: float
    ev_score: float
    was_constrained: bool = False
    sport_key: Optional[str] = None


class RetrainingStatus(BaseModel):
    """Status of model retraining."""

    status: str
    message: str


def decimal_to_american(decimal_odds: float) -> str:
    """Convert decimal odds to American format string."""
    if decimal_odds >= 2.0:
        return f"+{int((decimal_odds - 1) * 100)}"
    else:
        return f"{int(-100 / (decimal_odds - 1))}"


@router.get("/stats", response_model=MovementModelStats)
async def get_model_stats():
    """
    Get movement prediction model performance metrics.

    Returns model statistics including movement MAE, directional accuracy, and baseline comparison.
    """
    try:
        # Check if model exists
        if not Path(MODEL_PATH).exists():
            return MovementModelStats(is_trained=False)

        # Load model
        model = get_model()

        # Get file modification time
        model_file = Path(MODEL_PATH)
        last_trained = datetime.fromtimestamp(
            model_file.stat().st_mtime, tz=timezone.utc
        ).isoformat()

        # Load training data to compute metrics
        db = get_db()
        try:
            engineer = FeatureEngineer()
            df = engineer.prepare_training_data(db)

            if len(df) == 0:
                return MovementModelStats(
                    is_trained=True, last_trained=last_trained, training_records=0
                )

            # Split data
            X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = (
                engineer.train_test_split_data(df)
            )

            # Evaluate regression
            regression_metrics = model.evaluate_regression(X_test, y_reg_test)

            # Evaluate classification
            classification_metrics = model.evaluate_classification(X_test, y_class_test)

            # Calculate baseline
            baseline_mae = y_reg_test["price_movement"].abs().mean()
            improvement = ((baseline_mae - regression_metrics["ensemble_mae"]) / baseline_mae) * 100

            return MovementModelStats(
                is_trained=True,
                movement_mae=regression_metrics["ensemble_mae"],
                movement_rmse=regression_metrics["ensemble_rmse"],
                movement_r2=regression_metrics["ensemble_r2"],
                directional_accuracy=classification_metrics["ensemble_accuracy"],
                directional_precision=classification_metrics["precision"],
                directional_recall=classification_metrics["recall"],
                training_records=len(df),
                last_trained=last_trained,
                baseline_mae=baseline_mae,
                improvement_vs_baseline=improvement,
            )

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance", response_model=List[FeatureImportance])
async def get_feature_importance():
    """
    Get feature importance rankings for movement prediction.

    Returns features sorted by their importance in the movement model.
    """
    try:
        model = get_model()
        importance_dict = model.get_feature_importance()

        importance_list = [
            FeatureImportance(feature_name=name, importance=score)
            for name, score in importance_dict.items()
        ]

        return importance_list

    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/{game_id}", response_model=GameMovementPredictions)
async def get_game_predictions(game_id: int):
    """
    Get predicted line movement for a game.

    Returns movement predictions for all odds snapshots.
    """
    db = get_db()

    try:
        model = get_model()
        engineer = FeatureEngineer()

        # Get all snapshots for this game, joining bookmaker so no per-row query is needed
        stmt = (
            select(OddsSnapshot, ClosingLine, Game, Bookmaker)
            .outerjoin(
                ClosingLine,
                (ClosingLine.game_id == OddsSnapshot.game_id)
                & (ClosingLine.bookmaker_id == OddsSnapshot.bookmaker_id)
                & (ClosingLine.market_type == OddsSnapshot.market_type),
            )
            .join(Game, Game.id == OddsSnapshot.game_id)
            .join(Bookmaker, Bookmaker.id == OddsSnapshot.bookmaker_id)
            .where(OddsSnapshot.game_id == game_id)
        )

        results = db.execute(stmt).all()

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No snapshots found for game {game_id}",
            )

        predictions = []

        for snapshot, closing_line, game, bookmaker in results:
            outcome_predictions = []

            # Calculate base features
            hours_to_game = (game.commence_time - snapshot.timestamp).total_seconds() / 3600
            day_of_week = game.commence_time.weekday()
            is_weekend = day_of_week >= 5

            for outcome in snapshot.outcomes:
                outcome_name = outcome.get("name")
                opening_price = outcome.get("price")
                opening_point = outcome.get("point", 0.0)

                if not outcome_name or opening_price is None:
                    continue

                # Calculate consensus features
                consensus_line = engineer.calculate_consensus_line(
                    db, game.id, snapshot.market_type, outcome_name
                )
                line_spread = engineer.calculate_line_spread(
                    db, game.id, snapshot.market_type, outcome_name
                )

                # Create feature row
                features = pd.DataFrame([{
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
                }])

                # Predict movement
                movement_pred = model.predict_movement(features)

                # Get actual closing if available
                actual_closing_price = None
                actual_closing_point = None
                movement_error = None

                if closing_line is not None:
                    for cl_outcome in closing_line.outcomes:
                        if cl_outcome.get("name") == outcome_name:
                            actual_closing_price = cl_outcome.get("price")
                            actual_closing_point = cl_outcome.get("point")
                            if actual_closing_price is not None:
                                actual_movement = float(actual_closing_price) - float(opening_price)
                                movement_error = abs(movement_pred["predicted_delta"][0] - actual_movement)
                            break

                outcome_predictions.append(MovementPrediction(
                    outcome_name=outcome_name,
                    opening_price=float(opening_price),
                    opening_point=float(opening_point) if opening_point is not None else 0.0,
                    predicted_movement=float(movement_pred["predicted_delta"][0]),
                    predicted_direction=movement_pred["predicted_direction"][0],
                    direction_confidence=float(movement_pred["confidence"][0]),
                    predicted_closing_price=float(movement_pred["predicted_closing"][0]),
                    predicted_closing_point=float(opening_point) if opening_point is not None else 0.0,
                    actual_closing_price=actual_closing_price,
                    actual_closing_point=actual_closing_point,
                    movement_error=movement_error,
                    was_constrained=bool(movement_pred["was_constrained"][0]),
                    unconstrained_movement=float(movement_pred["unconstrained_delta"][0]),
                ))

            predictions.append(SnapshotMovementPrediction(
                snapshot_id=snapshot.id,
                bookmaker_name=bookmaker.name if bookmaker else "Unknown",
                market_type=snapshot.market_type,
                timestamp=snapshot.timestamp,
                outcomes=outcome_predictions,
            ))

        return GameMovementPredictions(game_id=game_id, predictions=predictions)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting predictions for game {game_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/best-opportunities", response_model=List[EVOpportunity])
async def get_best_opportunities(
    limit: int = 50,
    min_confidence: float = 0.62,
    min_ev_score: float = 2.0,
    bookmaker_filter: str = None,
    market_filter: str = None,
    min_hours_to_game: float = 1.0,
    max_hours_to_game: float = None,
    today_only: bool = True,
):
    """
    Get best +EV betting opportunities based on predicted line movement.

    Returns opportunities where the model predicts unfavorable line movement
    with high confidence, meaning you should bet now before odds worsen.

    Args:
        limit: Maximum number of opportunities to return
        min_confidence: Minimum ML confidence threshold (default 0.62)
        min_ev_score: Minimum EV score to include (default 2.0)
        bookmaker_filter: Only include opportunities from this bookmaker
        market_filter: Only include opportunities from this market (h2h, spreads, totals)
        min_hours_to_game: Minimum hours until game starts (default 1.0)
        max_hours_to_game: Maximum hours until game starts
        today_only: If True, only return games commencing today (default True)
    """
    db = get_db()

    try:
        model = get_model()
        engineer = FeatureEngineer()

        # Get upcoming games
        now = datetime.now(timezone.utc)

        stmt = (
            select(OddsSnapshot, Game)
            .join(Game, Game.id == OddsSnapshot.game_id)
            .where(Game.commence_time > now)
            .order_by(Game.commence_time)
        )

        # Filter to today's games only when requested
        if today_only:
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = today_start + timedelta(days=1)
            stmt = stmt.where(Game.commence_time >= today_start).where(Game.commence_time < today_end)

        # Apply market filter
        if market_filter:
            stmt = stmt.where(OddsSnapshot.market_type == market_filter)

        results = db.execute(stmt).all()

        if not results:
            return []

        # Pre-fetch all bookmakers used in results to avoid N+1 queries
        best_opp_bk_ids = list({snapshot.bookmaker_id for snapshot, _ in results})
        best_opp_bk_map = {
            b.id: b
            for b in db.execute(select(Bookmaker).where(Bookmaker.id.in_(best_opp_bk_ids))).scalars().all()
        }

        # Pre-fetch home/away teams for all games in results
        best_opp_game_team_ids = list({
            tid for _, game in results
            for tid in (game.home_team_id, game.away_team_id)
        })
        best_opp_teams_map = {
            t.id: t
            for t in db.execute(select(Team).where(Team.id.in_(best_opp_game_team_ids))).scalars().all()
        }

        # Pre-fetch sports for all games
        best_opp_sport_ids = list({game.sport_id for _, game in results if game.sport_id})
        best_opp_sports_map = {
            s.id: s
            for s in db.execute(select(Sport).where(Sport.id.in_(best_opp_sport_ids))).scalars().all()
        }

        opportunities = []

        for snapshot, game in results:
            hours_to_game = (game.commence_time - snapshot.timestamp).total_seconds() / 3600

            # Apply hours filters
            if min_hours_to_game is not None and hours_to_game < min_hours_to_game:
                continue
            if max_hours_to_game is not None and hours_to_game > max_hours_to_game:
                continue

            # Apply bookmaker filter using pre-fetched map
            bookmaker = best_opp_bk_map.get(snapshot.bookmaker_id)
            if bookmaker_filter and (not bookmaker or bookmaker.name != bookmaker_filter):
                continue

            day_of_week = game.commence_time.weekday()
            is_weekend = day_of_week >= 5

            for outcome in snapshot.outcomes:
                outcome_name = outcome.get("name")
                opening_price = outcome.get("price")
                opening_point = outcome.get("point", 0.0)

                if not outcome_name or opening_price is None:
                    continue

                # Calculate consensus
                consensus_line = engineer.calculate_consensus_line(
                    db, game.id, snapshot.market_type, outcome_name
                )
                line_spread = engineer.calculate_line_spread(
                    db, game.id, snapshot.market_type, outcome_name
                )

                temporal = engineer.calculate_temporal_features(
                    db, game.id, snapshot.bookmaker_id,
                    snapshot.market_type, outcome_name, snapshot.timestamp,
                )
                bk_features = engineer.calculate_bookmaker_features(
                    db, game.id, snapshot.bookmaker_id,
                    snapshot.market_type, outcome_name, float(opening_price),
                )

                features = pd.DataFrame([{
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
                    "time_since_last_update": temporal["time_since_last_update"],
                    "movement_velocity": temporal["movement_velocity"],
                    "updates_count": temporal["updates_count"],
                    "price_volatility_24h": temporal["price_volatility_24h"],
                    "cumulative_movement": temporal["cumulative_movement"],
                    "movement_direction_changes": temporal["movement_direction_changes"],
                    "bookmaker_is_sharp": bk_features["bookmaker_is_sharp"],
                    "relative_to_pinnacle": bk_features["relative_to_pinnacle"],
                    "books_moved_count": bk_features["books_moved_count"],
                    "steam_move_signal": bk_features["steam_move_signal"],
                }])

                # Predict movement
                movement_pred = model.predict_movement(features)

                predicted_delta = float(movement_pred["predicted_delta"][0])
                confidence = float(movement_pred["confidence"][0])
                direction = movement_pred["predicted_direction"][0]
                was_constrained = bool(movement_pred["was_constrained"][0])

                # Strict thresholds: skip low-confidence and marginal signals
                MIN_MOVEMENT = 0.025  # Require meaningful line movement (was 0.01)
                if confidence < min_confidence:
                    continue

                # Calculate EV score (higher = better opportunity)
                # Unfavorable movement (odds getting worse) = good betting opportunity NOW
                if direction == "DOWN" and predicted_delta < -MIN_MOVEMENT:
                    # Price dropping (getting worse for bettor) - bet now!
                    ev_score = abs(predicted_delta) * confidence * 100
                elif direction == "UP" and predicted_delta > MIN_MOVEMENT:
                    # Price rising (getting worse for bettor on the other side)
                    ev_score = abs(predicted_delta) * confidence * 100
                else:
                    continue

                # Apply minimum EV score filter
                if ev_score < min_ev_score:
                    continue

                # Calculate edge estimate
                fair_prob = 1 / float(opening_price)
                edge_estimate = abs(predicted_delta) * fair_prob

                # Format current line
                if opening_point != 0:
                    current_line = f"{opening_point:+.1f} at {decimal_to_american(opening_price)}"
                else:
                    current_line = decimal_to_american(opening_price)

                home_t = best_opp_teams_map.get(game.home_team_id)
                away_t = best_opp_teams_map.get(game.away_team_id)
                sport = best_opp_sports_map.get(game.sport_id) if game.sport_id else None
                opportunities.append(EVOpportunity(
                    game_id=game.id,
                    home_team=home_t.name if home_t else "Unknown",
                    away_team=away_t.name if away_t else "Unknown",
                    commence_time=game.commence_time,
                    bookmaker_name=bookmaker.name if bookmaker else "Unknown",
                    market_type=snapshot.market_type,
                    outcome_name=outcome_name,
                    current_line=current_line,
                    predicted_movement=predicted_delta,
                    predicted_direction=direction,
                    confidence=confidence,
                    ev_score=ev_score,
                    was_constrained=was_constrained,
                    sport_key=sport.key if sport else None,
                ))

        # Sort by EV score
        opportunities.sort(key=lambda x: x.ev_score, reverse=True)

        return opportunities[:limit]

    except Exception as e:
        logger.error(f"Error getting best opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/retrain", response_model=RetrainingStatus)
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Trigger model retraining with latest data.

    Retrains the movement prediction model in the background.
    """
    try:

        def retrain():
            """Background task to retrain the model."""
            db = SessionLocal()
            try:
                logger.info("Starting background model retraining...")

                engineer = FeatureEngineer()
                df = engineer.prepare_training_data(db)

                if len(df) == 0:
                    logger.error("No training data available for retraining")
                    return

                X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = (
                    engineer.train_test_split_data(df)
                )

                predictor = LineMovementPredictor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                )
                predictor.train(X_train, y_reg_train, y_class_train)

                # Evaluate
                regression_metrics = predictor.evaluate_regression(X_test, y_reg_test)
                classification_metrics = predictor.evaluate_classification(X_test, y_class_test)
                logger.info(f"Regression metrics: {regression_metrics}")
                logger.info(f"Classification metrics: {classification_metrics}")

                # Save model
                predictor.save_model(MODEL_PATH)

                # Clear global model to force reload
                global _model
                _model = None

                logger.info("Model retraining completed successfully")

            except Exception as e:
                logger.error(f"Error during retraining: {e}", exc_info=True)
            finally:
                db.close()

        # Schedule retraining in background
        background_tasks.add_task(retrain)

        return RetrainingStatus(
            status="scheduled",
            message="Model retraining has been scheduled in the background",
        )

    except Exception as e:
        logger.error(f"Error scheduling retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/is-trained")
async def check_model_trained():
    """
    Check if movement prediction model is trained.

    Returns simple status indicating if model file exists.
    """
    is_trained = Path(MODEL_PATH).exists()
    return {"is_trained": is_trained, "model_path": MODEL_PATH}


# ============================================================================
# Comprehensive Opportunities Endpoints
# ============================================================================


class OpportunityDetail(BaseModel):
    """Detailed opportunity record."""
    id: int
    game_id: int
    home_team: str
    away_team: str
    commence_time: datetime
    bookmaker: str
    market_type: str
    outcome_name: str
    point_line: Optional[float]
    entry_odds: float
    closing_odds: float
    clv_percentage: float
    result: Optional[str]
    profit_loss: Optional[float]
    settled_at: Optional[datetime]
    status: str


@router.get("/opportunities")
async def get_opportunities(
    status: str = "all",
    min_confidence: float = None,
    min_clv: float = None,
    bookmaker: str = None,
    market_type: str = None,
    date_from: str = None,
    date_to: str = None,
    sort_by: str = "clv",
    limit: int = 100,
    offset: int = 0,
):
    """
    Get comprehensive list of opportunities with filtering and pagination.

    Args:
        status: Filter by status - all, pending, settled
        min_confidence: Minimum ML confidence (not currently tracked in DB)
        min_clv: Minimum CLV percentage to include
        bookmaker: Filter by bookmaker name
        market_type: Filter by market type (h2h, spreads, totals)
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        sort_by: Sort field - clv, confidence, ev_score, date
        limit: Maximum results to return
        offset: Skip first N results
    """
    db = get_db()

    try:
        # Build base query
        stmt = (
            select(OpportunityPerformance, Game)
            .join(Game, Game.id == OpportunityPerformance.game_id)
        )

        # Apply status filter
        if status == "pending":
            stmt = stmt.where(OpportunityPerformance.result == "pending")
        elif status == "settled":
            stmt = stmt.where(OpportunityPerformance.result.in_(["win", "loss", "push"]))

        # Apply CLV filter
        if min_clv is not None:
            stmt = stmt.where(OpportunityPerformance.clv_percentage >= min_clv)

        # Apply bookmaker filter
        if bookmaker:
            stmt = stmt.where(OpportunityPerformance.bookmaker == bookmaker)

        # Apply market filter
        if market_type:
            stmt = stmt.where(OpportunityPerformance.market_type == market_type)

        # Apply date filters
        if date_from:
            try:
                from_date = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                stmt = stmt.where(Game.commence_time >= from_date)
            except ValueError:
                pass

        if date_to:
            try:
                to_date = datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                stmt = stmt.where(Game.commence_time <= to_date)
            except ValueError:
                pass

        # Apply sorting
        if sort_by == "clv":
            stmt = stmt.order_by(OpportunityPerformance.clv_percentage.desc())
        elif sort_by == "date":
            stmt = stmt.order_by(Game.commence_time.desc())
        else:
            stmt = stmt.order_by(OpportunityPerformance.clv_percentage.desc())

        # Apply pagination
        stmt = stmt.offset(offset).limit(limit)

        results = db.execute(stmt).all()

        # Pre-fetch all teams to avoid N+1 queries
        if results:
            opp_team_ids = list({tid for _, game in results for tid in (game.home_team_id, game.away_team_id)})
            opp_teams_map = {
                t.id: t
                for t in db.execute(select(Team).where(Team.id.in_(opp_team_ids))).scalars().all()
            }
        else:
            opp_teams_map = {}

        opportunities = []
        for opp, game in results:
            home_team = opp_teams_map.get(game.home_team_id)
            away_team = opp_teams_map.get(game.away_team_id)

            opp_status = "pending" if opp.result == "pending" or opp.result is None else "settled"

            opportunities.append({
                "id": opp.id,
                "game_id": opp.game_id,
                "home_team": home_team.name if home_team else "Unknown",
                "away_team": away_team.name if away_team else "Unknown",
                "commence_time": game.commence_time.isoformat(),
                "bookmaker": opp.bookmaker,
                "market_type": opp.market_type,
                "outcome_name": opp.outcome_name,
                "point_line": opp.point_line,
                "entry_odds": opp.entry_odds,
                "closing_odds": opp.closing_odds,
                "clv_percentage": round(opp.clv_percentage, 2),
                "result": opp.result,
                "profit_loss": round(opp.profit_loss, 2) if opp.profit_loss else None,
                "settled_at": opp.settled_at.isoformat() if opp.settled_at else None,
                "status": opp_status,
            })

        # Get total count for pagination
        count_stmt = (
            select(OpportunityPerformance)
        )
        if status == "pending":
            count_stmt = count_stmt.where(OpportunityPerformance.result == "pending")
        elif status == "settled":
            count_stmt = count_stmt.where(OpportunityPerformance.result.in_(["win", "loss", "push"]))
        if min_clv is not None:
            count_stmt = count_stmt.where(OpportunityPerformance.clv_percentage >= min_clv)
        if bookmaker:
            count_stmt = count_stmt.where(OpportunityPerformance.bookmaker == bookmaker)
        if market_type:
            count_stmt = count_stmt.where(OpportunityPerformance.market_type == market_type)

        from sqlalchemy import func as sa_func
        count_result = db.execute(select(sa_func.count()).select_from(count_stmt.subquery())).scalar()
        total_count = count_result or 0

        return {
            "opportunities": opportunities,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(opportunities) < total_count,
        }

    except Exception as e:
        logger.error(f"Error getting opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/upcoming-opportunities")
async def get_upcoming_opportunities(
    hours_ahead: int = 24,
    min_ev_score: float = 0.0,
):
    """
    Get opportunities for upcoming games grouped by game.

    Args:
        hours_ahead: Number of hours to look ahead
        min_ev_score: Minimum EV score to include
    """
    db = get_db()

    try:
        model = get_model()
        engineer = FeatureEngineer()

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        # Get upcoming games
        stmt = (
            select(Game, Team)
            .join(Team, Team.id == Game.home_team_id)
            .where(Game.commence_time > now)
            .where(Game.commence_time <= cutoff)
            .order_by(Game.commence_time)
        )

        games = db.execute(stmt).all()

        if not games:
            return {"games": [], "total_opportunities": 0}

        # Pre-fetch away teams and all snapshots+bookmakers to avoid N+1 queries
        upcoming_game_ids = [game.id for game, _ in games]
        away_team_ids = list({game.away_team_id for game, _ in games})
        upcoming_away_map = {
            t.id: t
            for t in db.execute(select(Team).where(Team.id.in_(away_team_ids))).scalars().all()
        }
        # Load all snapshots for upcoming games at once
        all_snapshots = db.execute(
            select(OddsSnapshot).where(OddsSnapshot.game_id.in_(upcoming_game_ids))
        ).scalars().all()
        snapshots_by_game: dict = {}
        for snap in all_snapshots:
            snapshots_by_game.setdefault(snap.game_id, []).append(snap)

        # Pre-fetch all bookmakers used in those snapshots
        bk_ids = list({snap.bookmaker_id for snap in all_snapshots})
        bookmakers_map = {
            b.id: b
            for b in db.execute(select(Bookmaker).where(Bookmaker.id.in_(bk_ids))).scalars().all()
        }

        game_opportunities = []

        for game, home_team in games:
            away_team = upcoming_away_map.get(game.away_team_id)
            snapshots = snapshots_by_game.get(game.id, [])

            game_opps = []
            for snapshot in snapshots:
                bookmaker = bookmakers_map.get(snapshot.bookmaker_id)

                hours_to_game = (game.commence_time - now).total_seconds() / 3600
                day_of_week = game.commence_time.weekday()
                is_weekend = day_of_week >= 5

                for outcome in snapshot.outcomes:
                    outcome_name = outcome.get("name")
                    opening_price = outcome.get("price")
                    opening_point = outcome.get("point", 0.0)

                    if not outcome_name or opening_price is None:
                        continue

                    # Calculate consensus
                    consensus_line = engineer.calculate_consensus_line(
                        db, game.id, snapshot.market_type, outcome_name
                    )
                    line_spread = engineer.calculate_line_spread(
                        db, game.id, snapshot.market_type, outcome_name
                    )

                    temporal = engineer.calculate_temporal_features(
                        db, game.id, snapshot.bookmaker_id,
                        snapshot.market_type, outcome_name, snapshot.timestamp,
                    )
                    bk_features = engineer.calculate_bookmaker_features(
                        db, game.id, snapshot.bookmaker_id,
                        snapshot.market_type, outcome_name, float(opening_price),
                    )

                    features = pd.DataFrame([{
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
                        "time_since_last_update": temporal["time_since_last_update"],
                        "movement_velocity": temporal["movement_velocity"],
                        "updates_count": temporal["updates_count"],
                        "price_volatility_24h": temporal["price_volatility_24h"],
                        "cumulative_movement": temporal["cumulative_movement"],
                        "movement_direction_changes": temporal["movement_direction_changes"],
                        "bookmaker_is_sharp": bk_features["bookmaker_is_sharp"],
                        "relative_to_pinnacle": bk_features["relative_to_pinnacle"],
                        "books_moved_count": bk_features["books_moved_count"],
                        "steam_move_signal": bk_features["steam_move_signal"],
                    }])

                    try:
                        movement_pred = model.predict_movement(features)
                        predicted_delta = float(movement_pred["predicted_delta"][0])
                        confidence = float(movement_pred["confidence"][0])
                        direction = movement_pred["predicted_direction"][0]

                        # Calculate EV score
                        if direction == "DOWN" and predicted_delta < -0.01:
                            ev_score = abs(predicted_delta) * confidence * 100
                        elif direction == "UP" and predicted_delta > 0.01:
                            ev_score = abs(predicted_delta) * confidence * 100
                        else:
                            ev_score = 0.0

                        if ev_score < min_ev_score:
                            continue

                        # Format line
                        if opening_point != 0:
                            current_line = f"{opening_point:+.1f} at {decimal_to_american(opening_price)}"
                        else:
                            current_line = decimal_to_american(opening_price)

                        game_opps.append({
                            "bookmaker": bookmaker.name if bookmaker else "Unknown",
                            "market_type": snapshot.market_type,
                            "outcome_name": outcome_name,
                            "current_line": current_line,
                            "predicted_movement": round(predicted_delta, 4),
                            "predicted_direction": direction,
                            "confidence": round(confidence, 3),
                            "ev_score": round(ev_score, 2),
                        })
                    except Exception:
                        continue

            # Sort opportunities by EV score
            game_opps.sort(key=lambda x: x["ev_score"], reverse=True)

            if game_opps:
                game_opportunities.append({
                    "game_id": game.id,
                    "home_team": home_team.name,
                    "away_team": away_team.name if away_team else "Unknown",
                    "commence_time": game.commence_time.isoformat(),
                    "hours_to_game": round(hours_to_game, 1),
                    "opportunities": game_opps[:10],  # Top 10 per game
                    "total_opportunities": len(game_opps),
                })

        # Sort games by best opportunity
        game_opportunities.sort(
            key=lambda x: x["opportunities"][0]["ev_score"] if x["opportunities"] else 0,
            reverse=True
        )

        total_opps = sum(g["total_opportunities"] for g in game_opportunities)

        return {
            "games": game_opportunities,
            "total_opportunities": total_opps,
            "hours_ahead": hours_ahead,
        }

    except Exception as e:
        logger.error(f"Error getting upcoming opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ============================================================================
# Best EV Pick Tracking Endpoints
# ============================================================================


@router.get("/best-ev-history")
async def get_best_ev_history(days: int = 30):
    """
    Get historical Best EV picks with settled results.

    Returns picks from the BestEVPick table for the past N days,
    grouped by date with summary stats.
    """
    db = get_db()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        stmt = (
            select(BestEVPick, Game)
            .join(Game, Game.id == BestEVPick.game_id)
            .where(BestEVPick.created_at >= cutoff)
            .order_by(BestEVPick.report_date.desc(), BestEVPick.ev_score.desc())
        )

        results = db.execute(stmt).all()

        if not results:
            return {"picks": [], "summary": {"total_picks": 0}, "days": days}

        # Pre-fetch teams
        team_ids = list({tid for _, game in results for tid in (game.home_team_id, game.away_team_id)})
        teams_map = {
            t.id: t
            for t in db.execute(select(Team).where(Team.id.in_(team_ids))).scalars().all()
        }

        picks = []
        total_profit = 0.0
        wins = losses = pushes = pending = 0

        for pick, game in results:
            home = teams_map.get(game.home_team_id)
            away = teams_map.get(game.away_team_id)
            if pick.result == "win":
                wins += 1
                total_profit += float(pick.profit_loss or 0)
            elif pick.result == "loss":
                losses += 1
                total_profit += float(pick.profit_loss or 0)
            elif pick.result == "push":
                pushes += 1
            else:
                pending += 1

            picks.append({
                "id": pick.id,
                "report_date": pick.report_date.isoformat(),
                "game": f"{away.name if away else 'Unknown'} @ {home.name if home else 'Unknown'}",
                "commence_time": game.commence_time.isoformat(),
                "bookmaker": pick.bookmaker,
                "market_type": pick.market_type,
                "outcome_name": pick.outcome_name,
                "entry_odds": float(pick.entry_odds),
                "ev_score": float(pick.ev_score),
                "confidence": float(pick.confidence),
                "result": pick.result,
                "profit_loss": float(pick.profit_loss) if pick.profit_loss is not None else None,
                "settled_at": pick.settled_at.isoformat() if pick.settled_at else None,
            })

        settled = wins + losses + pushes
        return {
            "picks": picks,
            "summary": {
                "total_picks": len(picks),
                "pending": pending,
                "settled": settled,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
                "total_profit": round(total_profit, 2),
            },
            "days": days,
        }

    except Exception as e:
        logger.error(f"Error getting best EV history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


def _save_picks():
    """Snapshot today's top EV picks to BestEVPick. Safe to call multiple times — deduplicates on (game_id, market_type, outcome_name)."""
    db = SessionLocal()
    try:
        model = get_model()
        engineer = FeatureEngineer()
        now = datetime.now(timezone.utc)
        today = now.date()

        # Build set of already-saved (game_id, market_type, outcome_name) for today
        existing_keys = set(
            (row[0], row[1], row[2])
            for row in db.execute(
                select(BestEVPick.game_id, BestEVPick.market_type, BestEVPick.outcome_name)
                .where(BestEVPick.report_date == today)
            ).all()
        )

        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        stmt = (
            select(OddsSnapshot, Game)
            .join(Game, Game.id == OddsSnapshot.game_id)
            .where(Game.commence_time >= today_start)
            .where(Game.commence_time < today_end)
            .where(Game.commence_time > now)
        )
        results = db.execute(stmt).all()
        if not results:
            logger.info("No upcoming games today for Best EV picks")
            return

        bk_ids = list({snap.bookmaker_id for snap, _ in results})
        bk_map = {
            b.id: b
            for b in db.execute(select(Bookmaker).where(Bookmaker.id.in_(bk_ids))).scalars().all()
        }

        candidates = []
        for snapshot, game in results:
            hours_to_game = (game.commence_time - now).total_seconds() / 3600
            if hours_to_game < 1.0:
                continue
            bookmaker = bk_map.get(snapshot.bookmaker_id)
            day_of_week = game.commence_time.weekday()
            is_weekend = day_of_week >= 5

            for outcome in snapshot.outcomes:
                outcome_name = outcome.get("name")
                opening_price = outcome.get("price")
                opening_point = outcome.get("point", 0.0)
                if not outcome_name or opening_price is None:
                    continue

                # Skip already-tracked combos
                if (game.id, snapshot.market_type, outcome_name) in existing_keys:
                    continue

                consensus_line = engineer.calculate_consensus_line(
                    db, game.id, snapshot.market_type, outcome_name
                )
                line_spread = engineer.calculate_line_spread(
                    db, game.id, snapshot.market_type, outcome_name
                )

                temporal = engineer.calculate_temporal_features(
                    db, game.id, snapshot.bookmaker_id,
                    snapshot.market_type, outcome_name, snapshot.timestamp,
                )
                bk_features = engineer.calculate_bookmaker_features(
                    db, game.id, snapshot.bookmaker_id,
                    snapshot.market_type, outcome_name, float(opening_price),
                )

                features = pd.DataFrame([{
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
                    "distance_from_consensus": float(opening_price) - consensus_line if consensus_line else 0.0,
                    "is_outlier": abs(float(opening_price) - consensus_line) > 0.05 if consensus_line else False,
                    "time_since_last_update": temporal["time_since_last_update"],
                    "movement_velocity": temporal["movement_velocity"],
                    "updates_count": temporal["updates_count"],
                    "price_volatility_24h": temporal["price_volatility_24h"],
                    "cumulative_movement": temporal["cumulative_movement"],
                    "movement_direction_changes": temporal["movement_direction_changes"],
                    "bookmaker_is_sharp": bk_features["bookmaker_is_sharp"],
                    "relative_to_pinnacle": bk_features["relative_to_pinnacle"],
                    "books_moved_count": bk_features["books_moved_count"],
                    "steam_move_signal": bk_features["steam_move_signal"],
                }])

                try:
                    pred = model.predict_movement(features)
                    predicted_delta = float(pred["predicted_delta"][0])
                    confidence = float(pred["confidence"][0])
                    direction = pred["predicted_direction"][0]

                    if confidence < 0.62:
                        continue

                    MIN_MOVEMENT = 0.025
                    if direction == "DOWN" and predicted_delta < -MIN_MOVEMENT:
                        ev_score = abs(predicted_delta) * confidence * 100
                    elif direction == "UP" and predicted_delta > MIN_MOVEMENT:
                        ev_score = abs(predicted_delta) * confidence * 100
                    else:
                        continue

                    if ev_score < 2.0:
                        continue

                    candidates.append({
                        "game_id": game.id,
                        "bookmaker": bookmaker.name if bookmaker else "Unknown",
                        "market_type": snapshot.market_type,
                        "outcome_name": outcome_name,
                        "entry_odds": float(opening_price),
                        "point_line": float(opening_point) if opening_point is not None else None,
                        "ev_score": ev_score,
                        "confidence": confidence,
                        "predicted_delta": predicted_delta,
                    })
                except Exception:
                    continue

        # Save top new picks (up to 20 per run, no daily cap)
        candidates.sort(key=lambda x: x["ev_score"], reverse=True)
        saved = 0
        for c in candidates[:20]:
            pick = BestEVPick(
                game_id=c["game_id"],
                report_date=today,
                bookmaker=c["bookmaker"],
                market_type=c["market_type"],
                outcome_name=c["outcome_name"],
                entry_odds=c["entry_odds"],
                point_line=c["point_line"],
                ev_score=c["ev_score"],
                confidence=c["confidence"],
                predicted_delta=c["predicted_delta"],
                result="pending",
            )
            db.add(pick)
            saved += 1

        db.commit()
        logger.info(f"Saved {saved} new Best EV picks for {today} ({len(existing_keys)} already tracked)")

    except Exception as e:
        logger.error(f"Error saving daily picks: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


@router.post("/save-daily-picks")
async def save_daily_best_ev_picks(background_tasks: BackgroundTasks):
    """Snapshot today's best EV picks. Safe to call multiple times — deduplicates on (game_id, market_type, outcome_name)."""
    background_tasks.add_task(_save_picks)
    return {"status": "scheduled", "message": "Daily Best EV picks are being saved in the background"}


@router.post("/settle-picks")
async def settle_best_ev_picks(background_tasks: BackgroundTasks):
    """
    Settle pending BestEVPick records against known BettingOutcome results.

    Matches pending picks to completed game outcomes and records win/loss/push.
    """

    def _settle():
        db = SessionLocal()
        try:
            stmt = (
                select(BestEVPick, Game, BettingOutcome)
                .join(Game, Game.id == BestEVPick.game_id)
                .outerjoin(BettingOutcome, BettingOutcome.game_id == BestEVPick.game_id)
                .where(BestEVPick.result == "pending")
                .where(BettingOutcome.completed == True)  # noqa: E712
            )
            results = db.execute(stmt).all()

            # Pre-fetch all teams to avoid N+1 queries
            team_ids = list({tid for _, game, _ in results for tid in (game.home_team_id, game.away_team_id)})
            teams_map = {
                t.id: t
                for t in db.execute(select(Team).where(Team.id.in_(team_ids))).scalars().all()
            }

            settled_count = 0
            now = datetime.now(timezone.utc)

            for pick, game, outcome in results:
                if outcome is None:
                    continue

                result = "pending"
                profit_loss = None
                home_team = teams_map.get(game.home_team_id)
                picked_home = bool(home_team and pick.outcome_name.lower() in home_team.name.lower())

                if pick.market_type == "h2h":
                    # Win if the picked team won outright
                    if (picked_home and outcome.winner == "home") or (not picked_home and outcome.winner == "away"):
                        result = "win"
                        profit_loss = round(100.0 * (float(pick.entry_odds) - 1), 2)
                    elif outcome.winner == "push":
                        result = "push"
                        profit_loss = 0.0
                    else:
                        result = "loss"
                        profit_loss = -100.0

                elif pick.market_type == "spreads" and pick.point_line is not None:
                    # Cover margin: how much the picked team beat the spread by.
                    # point_differential = home_score - away_score, so away margin is its negative.
                    team_margin = outcome.point_differential if picked_home else -outcome.point_differential
                    cover_margin = team_margin + float(pick.point_line)
                    if cover_margin > 0:
                        result = "win"
                        profit_loss = round(100.0 * (float(pick.entry_odds) - 1), 2)
                    elif cover_margin == 0:
                        result = "push"
                        profit_loss = 0.0
                    else:
                        result = "loss"
                        profit_loss = -100.0

                elif pick.market_type == "totals" and pick.point_line is not None:
                    # Over/Under: compare actual total to the line
                    total = outcome.total_points
                    line = float(pick.point_line)
                    betting_over = pick.outcome_name.lower() == "over"
                    if total is None:
                        continue
                    if total == line:
                        result = "push"
                        profit_loss = 0.0
                    elif (betting_over and total > line) or (not betting_over and total < line):
                        result = "win"
                        profit_loss = round(100.0 * (float(pick.entry_odds) - 1), 2)
                    else:
                        result = "loss"
                        profit_loss = -100.0

                if result != "pending":
                    pick.result = result
                    pick.profit_loss = profit_loss
                    pick.settled_at = now
                    settled_count += 1

            db.commit()
            logger.info(f"Settled {settled_count} Best EV picks")

        except Exception as e:
            logger.error(f"Error settling picks: {e}", exc_info=True)
            db.rollback()
        finally:
            db.close()

    background_tasks.add_task(_settle)
    return {"status": "scheduled", "message": "Pick settlement running in background"}

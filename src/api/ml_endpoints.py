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
from src.models.database import Bookmaker, ClosingLine, DailyCLVReport, Game, OddsSnapshot, OpportunityPerformance, Team

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

        # Get all snapshots for this game
        stmt = (
            select(OddsSnapshot, ClosingLine, Game)
            .outerjoin(
                ClosingLine,
                (ClosingLine.game_id == OddsSnapshot.game_id)
                & (ClosingLine.bookmaker_id == OddsSnapshot.bookmaker_id)
                & (ClosingLine.market_type == OddsSnapshot.market_type),
            )
            .join(Game, Game.id == OddsSnapshot.game_id)
            .where(OddsSnapshot.game_id == game_id)
        )

        results = db.execute(stmt).all()

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No snapshots found for game {game_id}",
            )

        predictions = []

        for snapshot, closing_line, game in results:
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

            # Get bookmaker name
            bookmaker = db.query(Bookmaker).filter(
                Bookmaker.id == snapshot.bookmaker_id
            ).first()

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
    min_confidence: float = 0.5,
    bookmaker_filter: str = None,
    market_filter: str = None,
    min_hours_to_game: float = None,
    max_hours_to_game: float = None,
):
    """
    Get best +EV betting opportunities based on predicted line movement.

    Returns opportunities where the model predicts unfavorable line movement
    with high confidence, meaning you should bet now before odds worsen.

    Args:
        limit: Maximum number of opportunities to return
        min_confidence: Minimum ML confidence threshold
        bookmaker_filter: Only include opportunities from this bookmaker
        market_filter: Only include opportunities from this market (h2h, spreads, totals)
        min_hours_to_game: Minimum hours until game starts
        max_hours_to_game: Maximum hours until game starts
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

        # Apply market filter
        if market_filter:
            stmt = stmt.where(OddsSnapshot.market_type == market_filter)

        results = db.execute(stmt).all()

        if not results:
            return []

        opportunities = []

        for snapshot, game in results:
            hours_to_game = (game.commence_time - snapshot.timestamp).total_seconds() / 3600

            # Apply hours filters
            if min_hours_to_game is not None and hours_to_game < min_hours_to_game:
                continue
            if max_hours_to_game is not None and hours_to_game > max_hours_to_game:
                continue

            # Apply bookmaker filter
            bookmaker = db.query(Bookmaker).filter(
                Bookmaker.id == snapshot.bookmaker_id
            ).first()
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

                # Create features - with new temporal/bookmaker features set to defaults
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
                    # Temporal features - defaults for live prediction
                    "time_since_last_update": 0.0,
                    "movement_velocity": 0.0,
                    "updates_count": 1,
                    "price_volatility_24h": 0.0,
                    "cumulative_movement": 0.0,
                    "movement_direction_changes": 0,
                    # Bookmaker features
                    "bookmaker_is_sharp": 1.0 if bookmaker and bookmaker.key.lower() == "pinnacle" else 0.0,
                    "relative_to_pinnacle": 0.0,
                    "books_moved_count": 0,
                    "steam_move_signal": 0.0,
                }])

                # Predict movement
                movement_pred = model.predict_movement(features)

                predicted_delta = float(movement_pred["predicted_delta"][0])
                confidence = float(movement_pred["confidence"][0])
                direction = movement_pred["predicted_direction"][0]
                was_constrained = bool(movement_pred["was_constrained"][0])

                # Only include high-confidence predictions of unfavorable movement
                if confidence < min_confidence:
                    continue

                # Calculate EV score (higher = better opportunity)
                # Unfavorable movement (odds getting worse) = good betting opportunity NOW
                if direction == "DOWN" and predicted_delta < -0.01:
                    # Price dropping (getting worse for bettor) - bet now!
                    ev_score = abs(predicted_delta) * confidence * 100
                elif direction == "UP" and predicted_delta > 0.01:
                    # Price rising (getting worse for bettor on the other side)
                    ev_score = abs(predicted_delta) * confidence * 100
                else:
                    continue

                # Calculate edge estimate
                fair_prob = 1 / float(opening_price)
                edge_estimate = abs(predicted_delta) * fair_prob

                # Format current line
                if opening_point != 0:
                    current_line = f"{opening_point:+.1f} at {decimal_to_american(opening_price)}"
                else:
                    current_line = decimal_to_american(opening_price)

                opportunities.append(EVOpportunity(
                    game_id=game.id,
                    home_team=game.home_team.name,
                    away_team=game.away_team.name,
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

        opportunities = []
        for opp, game in results:
            # Get team names
            home_team = db.query(Team).filter(Team.id == game.home_team_id).first()
            away_team = db.query(Team).filter(Team.id == game.away_team_id).first()

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

        total_count = len(db.execute(count_stmt).all())

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

        game_opportunities = []

        for game, home_team in games:
            away_team = db.query(Team).filter(Team.id == game.away_team_id).first()

            # Get all snapshots for this game
            snapshot_stmt = (
                select(OddsSnapshot)
                .where(OddsSnapshot.game_id == game.id)
            )
            snapshots = db.execute(snapshot_stmt).scalars().all()

            game_opps = []
            for snapshot in snapshots:
                bookmaker = db.query(Bookmaker).filter(
                    Bookmaker.id == snapshot.bookmaker_id
                ).first()

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

                    # Create features with defaults for new features
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
                        # Default temporal features
                        "time_since_last_update": 0.0,
                        "movement_velocity": 0.0,
                        "updates_count": 1,
                        "price_volatility_24h": 0.0,
                        "cumulative_movement": 0.0,
                        "movement_direction_changes": 0,
                        # Default bookmaker features
                        "bookmaker_is_sharp": 1.0 if bookmaker and bookmaker.key.lower() == "pinnacle" else 0.0,
                        "relative_to_pinnacle": 0.0,
                        "books_moved_count": 0,
                        "steam_move_signal": 0.0,
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

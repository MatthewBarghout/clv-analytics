"""API endpoints for machine learning model."""
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from src.analyzers.features import FeatureEngineer
from src.analyzers.ml_predictor import ClosingLinePredictor
from src.models.database import ClosingLine, Game, OddsSnapshot

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Setup router
router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])

# Database setup - build URL from components
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
MODEL_PATH = "models/closing_line_predictor.pkl"

# Global model instance
_model: Optional[ClosingLinePredictor] = None


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass


def get_model() -> ClosingLinePredictor:
    """Get or load the trained model."""
    global _model

    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(
                status_code=404,
                detail="Model not trained yet. Please run scripts/train_model.py first.",
            )

        _model = ClosingLinePredictor()
        _model.load_model(MODEL_PATH)
        logger.info("ML model loaded successfully")

    return _model


# Response models
class ModelStats(BaseModel):
    """Model performance statistics."""

    is_trained: bool
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    directional_accuracy: Optional[float] = None
    training_records: Optional[int] = None
    last_trained: Optional[str] = None


class FeatureImportance(BaseModel):
    """Feature importance data."""

    feature_name: str
    importance: float


class PredictionResult(BaseModel):
    """Prediction result for a single snapshot."""

    snapshot_id: int
    bookmaker_name: str
    market_type: str
    opening_odds: float
    predicted_closing_odds: float
    actual_closing_odds: Optional[float] = None
    prediction_error: Optional[float] = None
    timestamp: datetime


class GamePredictions(BaseModel):
    """Predictions for a game."""

    game_id: int
    predictions: List[PredictionResult]
    avg_prediction_error: Optional[float] = None


class RetrainingStatus(BaseModel):
    """Status of model retraining."""

    status: str
    message: str


@router.get("/stats", response_model=ModelStats)
async def get_model_stats():
    """
    Get model performance metrics.

    Returns model statistics including MAE, RMSE, R² score, and directional accuracy.
    """
    try:
        # Check if model exists
        if not Path(MODEL_PATH).exists():
            return ModelStats(is_trained=False)

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
                return ModelStats(
                    is_trained=True, last_trained=last_trained, training_records=0
                )

            # Split data
            X_train, X_test, y_train, y_test = engineer.train_test_split_data(df)

            # Evaluate
            metrics = model.evaluate(X_test, y_test)

            # Calculate directional accuracy
            predictions = model.predict(X_test)
            X_test_with_target = df.iloc[X_test.index]
            opening_odds_test = X_test_with_target["opening_odds"]
            directional_accuracy = model.calculate_directional_accuracy(
                y_test, predictions, opening_odds_test
            )

            return ModelStats(
                is_trained=True,
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r2_score=metrics["r2_score"],
                directional_accuracy=directional_accuracy,
                training_records=len(df),
                last_trained=last_trained,
            )

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance", response_model=List[FeatureImportance])
async def get_feature_importance():
    """
    Get feature importance rankings.

    Returns features sorted by their importance in the model.
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


@router.get("/predictions/{game_id}", response_model=GamePredictions)
async def get_game_predictions(game_id: int):
    """
    Get predicted vs actual closing lines for a game.

    Returns predictions for all odds snapshots with corresponding closing lines.
    """
    db = get_db()

    try:
        model = get_model()

        # Get all snapshots with closing lines for this game
        stmt = (
            select(OddsSnapshot, ClosingLine, Game)
            .join(
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
                detail=f"No predictions available for game {game_id}",
            )

        predictions = []
        engineer = FeatureEngineer()

        for snapshot, closing_line, game in results:
            # Extract features
            features = engineer.extract_features_from_snapshot(snapshot, game)
            if features is None:
                continue

            opening_odds = features["opening_odds"]

            # Make prediction
            predicted_closing = model.predict_closing_line(snapshot, game)
            if predicted_closing is None:
                continue

            # Get actual closing odds
            actual_closing = engineer._extract_closing_odds(closing_line)

            # Calculate error if actual is available
            prediction_error = None
            if actual_closing is not None:
                prediction_error = abs(predicted_closing - actual_closing)

            # Get bookmaker name
            from src.models.database import Bookmaker

            bookmaker = db.query(Bookmaker).filter(
                Bookmaker.id == snapshot.bookmaker_id
            ).first()

            predictions.append(
                PredictionResult(
                    snapshot_id=snapshot.id,
                    bookmaker_name=bookmaker.name if bookmaker else "Unknown",
                    market_type=snapshot.market_type,
                    opening_odds=opening_odds,
                    predicted_closing_odds=predicted_closing,
                    actual_closing_odds=actual_closing,
                    prediction_error=prediction_error,
                    timestamp=snapshot.timestamp,
                )
            )

        # Calculate average prediction error
        errors = [p.prediction_error for p in predictions if p.prediction_error is not None]
        avg_error = sum(errors) / len(errors) if errors else None

        return GamePredictions(
            game_id=game_id, predictions=predictions, avg_prediction_error=avg_error
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting predictions for game {game_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/retrain", response_model=RetrainingStatus)
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Trigger model retraining with latest data.

    Retrains the model in the background using all available data.
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

                X_train, X_test, y_train, y_test = engineer.train_test_split_data(df)

                predictor = ClosingLinePredictor(n_estimators=100)
                predictor.train(X_train, y_train)

                # Evaluate
                metrics = predictor.evaluate(X_test, y_test)
                logger.info(f"Retraining metrics: {metrics}")

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
    Check if model is trained.

    Returns simple status indicating if model file exists.
    """
    is_trained = Path(MODEL_PATH).exists()
    return {"is_trained": is_trained, "model_path": MODEL_PATH}


@router.get("/prediction-accuracy")
async def get_prediction_accuracy():
    """
    Get latest prediction accuracy metrics.

    Returns accuracy metrics for recent predictions.
    """
    db = get_db()

    try:
        model = get_model()

        # Get recent snapshots with closing lines (last 30 days)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)

        stmt = (
            select(OddsSnapshot, ClosingLine, Game)
            .join(
                ClosingLine,
                (ClosingLine.game_id == OddsSnapshot.game_id)
                & (ClosingLine.bookmaker_id == OddsSnapshot.bookmaker_id)
                & (ClosingLine.market_type == OddsSnapshot.market_type),
            )
            .join(Game, Game.id == OddsSnapshot.game_id)
            .where(OddsSnapshot.timestamp >= cutoff_date)
            .limit(100)
        )

        results = db.execute(stmt).all()

        if not results:
            return {
                "period": "last_30_days",
                "sample_size": 0,
                "avg_prediction_error": None,
                "directional_accuracy": None,
            }

        engineer = FeatureEngineer()
        errors = []
        actual_movements = []
        predicted_movements = []

        for snapshot, closing_line, game in results:
            # Extract features
            features = engineer.extract_features_from_snapshot(snapshot, game)
            if features is None:
                continue

            opening_odds = features["opening_odds"]

            # Make prediction
            predicted_closing = model.predict_closing_line(snapshot, game)
            if predicted_closing is None:
                continue

            # Get actual closing odds
            actual_closing = engineer._extract_closing_odds(closing_line)
            if actual_closing is None:
                continue

            # Calculate error
            error = abs(predicted_closing - actual_closing)
            errors.append(error)

            # Track movements for directional accuracy
            import numpy as np

            actual_movements.append(np.sign(actual_closing - opening_odds))
            predicted_movements.append(np.sign(predicted_closing - opening_odds))

        # Calculate metrics
        avg_error = sum(errors) / len(errors) if errors else None

        directional_accuracy = None
        if actual_movements:
            import numpy as np

            correct = np.sum(
                np.array(actual_movements) == np.array(predicted_movements)
            )
            directional_accuracy = (correct / len(actual_movements)) * 100

        return {
            "period": "last_30_days",
            "sample_size": len(errors),
            "avg_prediction_error": avg_error,
            "directional_accuracy": directional_accuracy,
        }

    except Exception as e:
        logger.error(f"Error calculating prediction accuracy: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

"""Machine learning predictor for closing line odds."""
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

from src.models.database import Game, OddsSnapshot

logger = logging.getLogger(__name__)


class ClosingLinePredictor:
    """Random Forest model for predicting closing line odds."""

    def __init__(self, n_estimators: int = 100):
        """
        Initialize predictor with Random Forest model.

        Args:
            n_estimators: Number of trees in the forest (default 100)
        """
        self.n_estimators = n_estimators
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: Optional[list] = None
        self.categorical_features = ["market_type"]
        self.numerical_features = [
            "opening_odds",
            "bookmaker_id",
            "hours_to_game",
            "day_of_week",
            "is_weekend",
        ]

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline for categorical encoding.

        Returns:
            ColumnTransformer for preprocessing
        """
        return ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                    self.categorical_features,
                ),
            ],
            remainder="passthrough",
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the Random Forest model.

        Handles categorical encoding internally.

        Args:
            X_train: Training features
            y_train: Training target values
        """
        logger.info("Training Random Forest model...")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Create and fit preprocessor
        self.preprocessor = self._create_preprocessor()
        X_train_processed = self.preprocessor.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_processed, y_train)

        logger.info("Model training completed")
        logger.info(f"Number of features: {X_train_processed.shape[1]}")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test: Test features

        Returns:
            Array of predicted closing odds
        """
        if self.preprocessor is None:
            raise ValueError("Model must be trained before making predictions")

        X_test_processed = self.preprocessor.transform(X_test)
        predictions = self.model.predict(X_test_processed)

        return predictions

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: True target values

        Returns:
            Dictionary with MAE, RMSE, and R² score
        """
        predictions = self.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        logger.info(f"Model Evaluation:")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R² Score: {r2:.4f}")

        return {"mae": mae, "rmse": rmse, "r2_score": r2}

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.preprocessor is None or self.feature_names is None:
            raise ValueError("Model must be trained before getting feature importance")

        # Get feature names after preprocessing
        cat_features = []
        if self.preprocessor.named_transformers_["cat"]:
            cat_encoder = self.preprocessor.named_transformers_["cat"]
            cat_features = list(cat_encoder.get_feature_names_out(self.categorical_features))

        # Numerical features remain unchanged
        all_features = cat_features + self.numerical_features

        # Get importances
        importances = self.model.feature_importances_

        # Create dictionary
        importance_dict = dict(zip(all_features, importances))

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict

    def predict_closing_line(
        self, opening_snapshot: OddsSnapshot, game: Game
    ) -> Optional[float]:
        """
        Predict closing odds for a single opening snapshot.

        Args:
            opening_snapshot: OddsSnapshot with opening odds
            game: Associated Game instance

        Returns:
            Predicted closing odds or None if prediction fails
        """
        try:
            from src.analyzers.features import FeatureEngineer

            engineer = FeatureEngineer()
            features = engineer.extract_features_from_snapshot(opening_snapshot, game)

            if features is None:
                logger.error("Failed to extract features from snapshot")
                return None

            # Create DataFrame with single row
            X = pd.DataFrame([features])

            # Make prediction
            prediction = self.predict(X)

            return float(prediction[0])

        except Exception as e:
            logger.error(f"Error predicting closing line: {e}")
            return None

    def calculate_directional_accuracy(
        self, y_true: pd.Series, y_pred: np.ndarray, opening_odds: pd.Series
    ) -> float:
        """
        Calculate percentage of predictions with correct direction.

        Checks if prediction correctly predicts direction of movement
        (up/down/same) relative to opening odds.

        Args:
            y_true: True closing odds
            y_pred: Predicted closing odds
            opening_odds: Opening odds values

        Returns:
            Percentage of correct directional predictions (0-100)
        """
        # Calculate actual and predicted movements
        actual_movement = np.sign(y_true - opening_odds)
        predicted_movement = np.sign(y_pred - opening_odds)

        # Count correct predictions
        correct = np.sum(actual_movement == predicted_movement)
        total = len(y_true)

        accuracy = (correct / total) * 100 if total > 0 else 0

        logger.info(f"Directional Accuracy: {accuracy:.2f}%")

        return accuracy

    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.

        Args:
            filepath: Path to save model file
        """
        model_data = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "feature_names": self.feature_names,
            "n_estimators": self.n_estimators,
        }

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file.

        Args:
            filepath: Path to model file
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.preprocessor = model_data["preprocessor"]
        self.feature_names = model_data["feature_names"]
        self.n_estimators = model_data["n_estimators"]

        logger.info(f"Model loaded from {filepath}")

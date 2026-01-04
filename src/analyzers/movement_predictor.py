"""Line movement predictor using XGBoost models."""
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.models.database import Game, OddsSnapshot

logger = logging.getLogger(__name__)


class LineMovementPredictor:
    """Dual-model predictor for line movement (regression + classification)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_weight: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        """
        Initialize movement predictor with regularized XGBoost models.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (reduced to 4 for more conservative predictions)
            learning_rate: Learning rate for gradient boosting (reduced to 0.05)
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            min_child_weight: Minimum sum of instance weight needed in a child
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        # Regression model for movement magnitude
        self.regression_model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
        )

        # Classification model for movement direction
        self.classification_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
        )

        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: Optional[list] = None
        self.categorical_features = ["market_type", "outcome_name"]
        self.numerical_features = [
            "bookmaker_id",
            "hours_to_game",
            "day_of_week",
            "is_weekend",
            "opening_price",
            "opening_point",
            "consensus_line",
            "line_spread",
            "distance_from_consensus",
            "is_outlier",
        ]

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline with encoding and scaling.

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
                (
                    "num",
                    StandardScaler(),
                    self.numerical_features,
                ),
            ],
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_reg_train: pd.DataFrame,
        y_class_train: pd.Series,
    ) -> None:
        """
        Train both regression and classification models.

        Args:
            X_train: Training features
            y_reg_train: Regression targets (price_movement, point_movement)
            y_class_train: Classification target (UP/DOWN/STAY)
        """
        logger.info("Training line movement predictor...")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Create and fit preprocessor
        self.preprocessor = self._create_preprocessor()
        X_train_processed = self.preprocessor.fit_transform(X_train)

        # Train regression model (price movement only for now)
        logger.info("Training regression model for price movement...")
        self.regression_model.fit(X_train_processed, y_reg_train["price_movement"])

        # Encode classification labels
        self.label_encoder = LabelEncoder()
        y_class_encoded = self.label_encoder.fit_transform(y_class_train)

        # Train classification model
        logger.info("Training classification model for movement direction...")
        self.classification_model.fit(X_train_processed, y_class_encoded)

        logger.info("Model training completed")
        logger.info(f"Number of features after preprocessing: {X_train_processed.shape[1]}")
        logger.info(f"Direction classes: {self.label_encoder.classes_}")

    def predict_movement(self, X: pd.DataFrame) -> dict:
        """
        Predict line movement for given features.

        Args:
            X: Features DataFrame (single row or multiple rows)

        Returns:
            Dictionary with predicted_delta, predicted_direction, confidence, predicted_closing
        """
        if self.preprocessor is None or self.label_encoder is None:
            raise ValueError("Model must be trained before making predictions")

        X_processed = self.preprocessor.transform(X)

        # Predict movement magnitude
        predicted_delta = self.regression_model.predict(X_processed)

        # Predict movement direction with probability
        direction_encoded = self.classification_model.predict(X_processed)
        direction_proba = self.classification_model.predict_proba(X_processed)
        predicted_direction = self.label_encoder.inverse_transform(direction_encoded)

        # Get confidence (max probability)
        confidence = np.max(direction_proba, axis=1)

        # Calculate predicted closing price
        opening_price = X["opening_price"].values
        predicted_closing = opening_price + predicted_delta

        return {
            "predicted_delta": predicted_delta,
            "predicted_direction": predicted_direction,
            "confidence": confidence,
            "predicted_closing": predicted_closing,
        }

    def evaluate_regression(
        self, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> dict[str, float]:
        """
        Evaluate regression model performance.

        Args:
            X_test: Test features
            y_test: True movement values

        Returns:
            Dictionary with MAE, RMSE, R² metrics
        """
        X_test_processed = self.preprocessor.transform(X_test)
        predictions = self.regression_model.predict(X_test_processed)

        y_true = y_test["price_movement"].values

        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        r2 = r2_score(y_true, predictions)

        # Calculate metrics for significant movements only
        significant_mask = np.abs(y_true) > 0.02
        if significant_mask.any():
            mae_significant = mean_absolute_error(
                y_true[significant_mask], predictions[significant_mask]
            )
            rmse_significant = np.sqrt(
                mean_squared_error(y_true[significant_mask], predictions[significant_mask])
            )
        else:
            mae_significant = mae
            rmse_significant = rmse

        logger.info("Regression Model Evaluation:")
        logger.info(f"  MAE (all): {mae:.4f}")
        logger.info(f"  RMSE (all): {rmse:.4f}")
        logger.info(f"  R² (all): {r2:.4f}")
        logger.info(f"  MAE (significant movements >0.02): {mae_significant:.4f}")
        logger.info(f"  RMSE (significant movements >0.02): {rmse_significant:.4f}")

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mae_significant": mae_significant,
            "rmse_significant": rmse_significant,
        }

    def evaluate_classification(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, any]:
        """
        Evaluate classification model performance.

        Args:
            X_test: Test features
            y_test: True directional movement

        Returns:
            Dictionary with accuracy, precision, recall, confusion matrix
        """
        X_test_processed = self.preprocessor.transform(X_test)

        # Predict direction
        y_test_encoded = self.label_encoder.transform(y_test)
        predictions_encoded = self.classification_model.predict(X_test_processed)

        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, predictions_encoded)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_encoded, predictions_encoded, average="weighted", zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, predictions_encoded)

        logger.info("Classification Model Evaluation:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision (weighted): {precision:.4f}")
        logger.info(f"  Recall (weighted): {recall:.4f}")
        logger.info(f"  F1 Score (weighted): {f1:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    Classes: {self.label_encoder.classes_}")
        logger.info(f"{cm}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classes": self.label_encoder.classes_.tolist(),
        }

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance from regression model.

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

        all_features = cat_features + self.numerical_features

        # Get importances from regression model
        importances = self.regression_model.feature_importances_

        # Create dictionary
        importance_dict = dict(zip(all_features, importances))

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict

    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.

        Args:
            filepath: Path to save model file
        """
        model_data = {
            "regression_model": self.regression_model,
            "classification_model": self.classification_model,
            "preprocessor": self.preprocessor,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
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

        self.regression_model = model_data["regression_model"]
        self.classification_model = model_data["classification_model"]
        self.preprocessor = model_data["preprocessor"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_names = model_data["feature_names"]
        self.n_estimators = model_data["n_estimators"]
        self.max_depth = model_data["max_depth"]
        self.learning_rate = model_data["learning_rate"]

        logger.info(f"Model loaded from {filepath}")

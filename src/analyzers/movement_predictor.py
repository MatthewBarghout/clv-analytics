"""Line movement predictor using ensemble of XGBoost and Random Forest models."""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
    """Ensemble predictor for line movement using XGBoost + Random Forest."""

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
        xgb_weight: float = 0.5,
        rf_weight: float = 0.5,
    ):
        """
        Initialize ensemble movement predictor with XGBoost and Random Forest.

        Args:
            n_estimators: Number of estimators for both XGBoost and Random Forest
            max_depth: Maximum tree depth for both models
            learning_rate: Learning rate for XGBoost
            reg_alpha: L1 regularization for XGBoost
            reg_lambda: L2 regularization for XGBoost
            min_child_weight: Minimum child weight for XGBoost
            subsample: Subsample ratio for XGBoost
            colsample_bytree: Column subsample ratio for XGBoost
            xgb_weight: Weight for XGBoost predictions in ensemble (default 0.5)
            rf_weight: Weight for Random Forest predictions in ensemble (default 0.5)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.xgb_weight = xgb_weight
        self.rf_weight = rf_weight

        # Per-market weights (can be optimized via grid search)
        self.market_weights: Dict[str, Dict[str, float]] = {
            'h2h': {'xgb': 0.5, 'rf': 0.5},
            'spreads': {'xgb': 0.5, 'rf': 0.5},
            'totals': {'xgb': 0.5, 'rf': 0.5},
        }

        # XGBoost regression model for movement magnitude
        self.xgb_regression = XGBRegressor(
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

        # XGBoost classification model for movement direction
        self.xgb_classification = XGBClassifier(
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

        # Random Forest regression model for movement magnitude
        self.rf_regression = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
        )

        # Random Forest classification model for movement direction
        self.rf_classification = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
        )

        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: Optional[list] = None
        self.categorical_features = ["market_type", "outcome_name"]
        self.numerical_features = [
            # Base features
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

        # Load movement statistics for prediction capping
        self.movement_stats = self._load_movement_statistics()

    def _load_movement_statistics(self) -> Dict[str, float]:
        """
        Load historical movement statistics for prediction capping.

        Returns:
            Dictionary of 95th percentile caps by market type
        """
        stats_path = Path("models/movement_statistics.json")

        if not stats_path.exists():
            logger.warning(
                "Movement statistics not found. Using conservative default caps. "
                "Run scripts/analyze_movement_patterns.py to generate statistics."
            )
            return {
                'h2h': 0.10,
                'spreads': 0.05,
                'totals': 0.05,
            }

        try:
            with open(stats_path, 'r') as f:
                all_stats = json.load(f)

            # Extract 95th percentile caps
            caps = {}
            for market_type, stats in all_stats.items():
                caps[market_type] = stats['p95']

            logger.info(f"Loaded movement caps: {caps}")
            return caps

        except Exception as e:
            logger.error(f"Error loading movement statistics: {e}")
            return {
                'h2h': 0.10,
                'spreads': 0.05,
                'totals': 0.05,
            }

    def _constrain_prediction(
        self,
        predicted_movement: float,
        market_type: str
    ) -> tuple[float, bool]:
        """
        Constrain prediction to realistic range based on historical data.

        Args:
            predicted_movement: Raw model prediction
            market_type: Market type (h2h, spreads, totals)

        Returns:
            Tuple of (constrained_movement, was_constrained)
        """
        # Get cap for this market type
        cap = self.movement_stats.get(market_type, 0.10)

        # Check if capping is needed
        if abs(predicted_movement) > cap:
            constrained = np.sign(predicted_movement) * cap
            logger.debug(
                f"Constrained {market_type} prediction from {predicted_movement:.4f} "
                f"to {constrained:.4f} (cap: ±{cap:.4f})"
            )
            return constrained, True

        return predicted_movement, False

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
        Train ensemble models (XGBoost + Random Forest).

        Args:
            X_train: Training features
            y_reg_train: Regression targets (price_movement, point_movement)
            y_class_train: Classification target (UP/DOWN/STAY)
        """
        logger.info("Training ensemble movement predictor...")
        logger.info(f"Ensemble weights: XGBoost={self.xgb_weight}, RF={self.rf_weight}")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Create and fit preprocessor
        self.preprocessor = self._create_preprocessor()
        X_train_processed = self.preprocessor.fit_transform(X_train)

        # Train XGBoost regression model
        logger.info("Training XGBoost regression model...")
        self.xgb_regression.fit(X_train_processed, y_reg_train["price_movement"])

        # Train Random Forest regression model
        logger.info("Training Random Forest regression model...")
        self.rf_regression.fit(X_train_processed, y_reg_train["price_movement"])

        # Encode classification labels
        self.label_encoder = LabelEncoder()
        y_class_encoded = self.label_encoder.fit_transform(y_class_train)

        # Train XGBoost classification model
        logger.info("Training XGBoost classification model...")
        self.xgb_classification.fit(X_train_processed, y_class_encoded)

        # Train Random Forest classification model
        logger.info("Training Random Forest classification model...")
        self.rf_classification.fit(X_train_processed, y_class_encoded)

        logger.info("Ensemble training completed")
        logger.info(f"Number of features after preprocessing: {X_train_processed.shape[1]}")
        logger.info(f"Direction classes: {self.label_encoder.classes_}")

    def optimize_ensemble_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        weight_options: list = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize ensemble weights per market type using grid search on validation data.

        Args:
            X_val: Validation features
            y_val: Validation targets (price_movement column)
            weight_options: List of XGB weights to try (RF = 1 - XGB)

        Returns:
            Dictionary of optimal weights per market type
        """
        if weight_options is None:
            weight_options = [0.3, 0.4, 0.5, 0.6, 0.7]

        X_val_processed = self.preprocessor.transform(X_val)

        # Get individual model predictions
        xgb_pred = self.xgb_regression.predict(X_val_processed)
        rf_pred = self.rf_regression.predict(X_val_processed)

        y_true = y_val["price_movement"].values

        optimal_weights = {}

        # Get unique market types
        market_types = X_val["market_type"].unique()

        for market in market_types:
            market_mask = X_val["market_type"] == market
            market_indices = np.where(market_mask)[0]

            if len(market_indices) < 10:
                logger.info(f"Insufficient data for {market}, using default weights")
                optimal_weights[market] = {'xgb': 0.5, 'rf': 0.5}
                continue

            best_mae = float('inf')
            best_xgb_weight = 0.5

            for xgb_w in weight_options:
                rf_w = 1 - xgb_w
                ensemble_pred = (xgb_pred[market_indices] * xgb_w) + (rf_pred[market_indices] * rf_w)
                mae = mean_absolute_error(y_true[market_indices], ensemble_pred)

                if mae < best_mae:
                    best_mae = mae
                    best_xgb_weight = xgb_w

            optimal_weights[market] = {
                'xgb': best_xgb_weight,
                'rf': 1 - best_xgb_weight,
            }
            logger.info(
                f"Optimized weights for {market}: XGB={best_xgb_weight:.1f}, "
                f"RF={1-best_xgb_weight:.1f} (MAE={best_mae:.4f})"
            )

        self.market_weights = optimal_weights
        return optimal_weights

    def get_weights_for_market(self, market_type: str) -> tuple:
        """Get XGB and RF weights for a specific market type."""
        if market_type in self.market_weights:
            weights = self.market_weights[market_type]
            return weights['xgb'], weights['rf']
        return self.xgb_weight, self.rf_weight

    def predict_movement(self, X: pd.DataFrame) -> dict:
        """
        Predict line movement using ensemble (XGBoost + Random Forest average).

        Args:
            X: Features DataFrame (single row or multiple rows)

        Returns:
            Dictionary with predicted_delta, predicted_direction, confidence,
            predicted_closing, was_constrained, unconstrained_delta
        """
        if self.preprocessor is None or self.label_encoder is None:
            raise ValueError("Model must be trained before making predictions")

        X_processed = self.preprocessor.transform(X)

        # Get predictions from both regression models
        xgb_pred = self.xgb_regression.predict(X_processed)
        rf_pred = self.rf_regression.predict(X_processed)

        # Ensemble regression predictions with per-market weights
        unconstrained_delta = np.zeros(len(X))
        for i in range(len(X)):
            market_type = X.iloc[i]["market_type"]
            xgb_w, rf_w = self.get_weights_for_market(market_type)
            unconstrained_delta[i] = (xgb_pred[i] * xgb_w) + (rf_pred[i] * rf_w)

        # Constrain predictions to realistic ranges
        constrained_deltas = []
        was_constrained = []

        for i, raw_delta in enumerate(unconstrained_delta):
            market_type = X.iloc[i]["market_type"]
            constrained, is_capped = self._constrain_prediction(raw_delta, market_type)
            constrained_deltas.append(constrained)
            was_constrained.append(is_capped)

        predicted_delta = np.array(constrained_deltas)
        was_constrained = np.array(was_constrained)

        # Get predictions from both classification models
        xgb_direction_encoded = self.xgb_classification.predict(X_processed)
        xgb_direction_proba = self.xgb_classification.predict_proba(X_processed)

        rf_direction_encoded = self.rf_classification.predict(X_processed)
        rf_direction_proba = self.rf_classification.predict_proba(X_processed)

        # Ensemble classification predictions with per-market weights
        ensemble_proba = np.zeros_like(xgb_direction_proba)
        for i in range(len(X)):
            market_type = X.iloc[i]["market_type"]
            xgb_w, rf_w = self.get_weights_for_market(market_type)
            ensemble_proba[i] = (xgb_direction_proba[i] * xgb_w) + (rf_direction_proba[i] * rf_w)

        # Get final direction from ensemble probabilities
        direction_encoded = np.argmax(ensemble_proba, axis=1)
        predicted_direction = self.label_encoder.inverse_transform(direction_encoded)

        # Get confidence (max probability from ensemble)
        confidence = np.max(ensemble_proba, axis=1)

        # Calculate predicted closing price using constrained delta
        opening_price = X["opening_price"].values
        predicted_closing = opening_price + predicted_delta

        return {
            "predicted_delta": predicted_delta,
            "predicted_direction": predicted_direction,
            "confidence": confidence,
            "predicted_closing": predicted_closing,
            "was_constrained": was_constrained,
            "unconstrained_delta": unconstrained_delta,
        }

    def evaluate_regression(
        self, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> dict[str, float]:
        """
        Evaluate regression model performance for both models and ensemble.

        Args:
            X_test: Test features
            y_test: True movement values

        Returns:
            Dictionary with MAE, RMSE, R² metrics for XGBoost, RF, and Ensemble
        """
        X_test_processed = self.preprocessor.transform(X_test)
        y_true = y_test["price_movement"].values

        # Get individual model predictions
        xgb_pred = self.xgb_regression.predict(X_test_processed)
        rf_pred = self.rf_regression.predict(X_test_processed)

        # Calculate ensemble predictions
        ensemble_pred = (xgb_pred * self.xgb_weight) + (rf_pred * self.rf_weight)

        # Evaluate XGBoost
        xgb_mae = mean_absolute_error(y_true, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_true, xgb_pred))
        xgb_r2 = r2_score(y_true, xgb_pred)

        # Evaluate Random Forest
        rf_mae = mean_absolute_error(y_true, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_true, rf_pred))
        rf_r2 = r2_score(y_true, rf_pred)

        # Evaluate Ensemble
        ensemble_mae = mean_absolute_error(y_true, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
        ensemble_r2 = r2_score(y_true, ensemble_pred)

        # Calculate metrics for significant movements
        significant_mask = np.abs(y_true) > 0.02
        if significant_mask.any():
            ensemble_mae_sig = mean_absolute_error(
                y_true[significant_mask], ensemble_pred[significant_mask]
            )
            ensemble_rmse_sig = np.sqrt(
                mean_squared_error(y_true[significant_mask], ensemble_pred[significant_mask])
            )
        else:
            ensemble_mae_sig = ensemble_mae
            ensemble_rmse_sig = ensemble_rmse

        logger.info("=" * 70)
        logger.info("REGRESSION MODEL EVALUATION")
        logger.info("=" * 70)
        logger.info(f"XGBoost:      MAE={xgb_mae:.4f}  RMSE={xgb_rmse:.4f}  R²={xgb_r2:.4f}")
        logger.info(f"Random Forest: MAE={rf_mae:.4f}  RMSE={rf_rmse:.4f}  R²={rf_r2:.4f}")
        logger.info(f"Ensemble:     MAE={ensemble_mae:.4f}  RMSE={ensemble_rmse:.4f}  R²={ensemble_r2:.4f}")
        logger.info(f"Ensemble (significant movements >0.02): MAE={ensemble_mae_sig:.4f}  RMSE={ensemble_rmse_sig:.4f}")

        # Calculate improvement
        best_individual = min(xgb_mae, rf_mae)
        improvement = ((best_individual - ensemble_mae) / best_individual) * 100
        logger.info(f"Ensemble improvement over best individual: {improvement:+.1f}%")

        return {
            "xgb_mae": xgb_mae,
            "xgb_rmse": xgb_rmse,
            "xgb_r2": xgb_r2,
            "rf_mae": rf_mae,
            "rf_rmse": rf_rmse,
            "rf_r2": rf_r2,
            "ensemble_mae": ensemble_mae,
            "ensemble_rmse": ensemble_rmse,
            "ensemble_r2": ensemble_r2,
            "ensemble_mae_significant": ensemble_mae_sig,
            "ensemble_rmse_significant": ensemble_rmse_sig,
        }

    def evaluate_classification(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, any]:
        """
        Evaluate classification model performance for both models and ensemble.

        Args:
            X_test: Test features
            y_test: True directional movement

        Returns:
            Dictionary with accuracy, precision, recall for XGBoost, RF, and Ensemble
        """
        X_test_processed = self.preprocessor.transform(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)

        # XGBoost predictions
        xgb_pred = self.xgb_classification.predict(X_test_processed)
        xgb_proba = self.xgb_classification.predict_proba(X_test_processed)

        # Random Forest predictions
        rf_pred = self.rf_classification.predict(X_test_processed)
        rf_proba = self.rf_classification.predict_proba(X_test_processed)

        # Ensemble predictions (weighted average of probabilities)
        ensemble_proba = (xgb_proba * self.xgb_weight) + (rf_proba * self.rf_weight)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)

        # Calculate metrics for each model
        xgb_accuracy = accuracy_score(y_test_encoded, xgb_pred)
        rf_accuracy = accuracy_score(y_test_encoded, rf_pred)
        ensemble_accuracy = accuracy_score(y_test_encoded, ensemble_pred)

        # Ensemble confusion matrix
        cm = confusion_matrix(y_test_encoded, ensemble_pred)

        # Ensemble precision/recall
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_encoded, ensemble_pred, average="weighted", zero_division=0
        )

        logger.info("=" * 70)
        logger.info("CLASSIFICATION MODEL EVALUATION")
        logger.info("=" * 70)
        logger.info(f"XGBoost:       Accuracy={xgb_accuracy:.4f} ({xgb_accuracy*100:.1f}%)")
        logger.info(f"Random Forest: Accuracy={rf_accuracy:.4f} ({rf_accuracy*100:.1f}%)")
        logger.info(f"Ensemble:      Accuracy={ensemble_accuracy:.4f} ({ensemble_accuracy*100:.1f}%)")
        logger.info(f"Ensemble Precision: {precision:.4f}")
        logger.info(f"Ensemble Recall: {recall:.4f}")
        logger.info(f"Ensemble F1 Score: {f1:.4f}")
        logger.info(f"Confusion Matrix:")
        logger.info(f"  Classes: {self.label_encoder.classes_}")
        logger.info(f"{cm}")

        # Calculate improvement
        best_individual = max(xgb_accuracy, rf_accuracy)
        improvement = ((ensemble_accuracy - best_individual) / best_individual) * 100
        logger.info(f"Ensemble improvement over best individual: {improvement:+.1f}%")

        return {
            "xgb_accuracy": xgb_accuracy,
            "rf_accuracy": rf_accuracy,
            "ensemble_accuracy": ensemble_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classes": self.label_encoder.classes_.tolist(),
        }

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get average feature importance from both regression models.

        Returns:
            Dictionary mapping feature names to averaged importance scores
        """
        if self.preprocessor is None or self.feature_names is None:
            raise ValueError("Model must be trained before getting feature importance")

        # Get feature names after preprocessing
        cat_features = []
        if self.preprocessor.named_transformers_["cat"]:
            cat_encoder = self.preprocessor.named_transformers_["cat"]
            cat_features = list(cat_encoder.get_feature_names_out(self.categorical_features))

        all_features = cat_features + self.numerical_features

        # Get importances from both regression models
        xgb_importances = self.xgb_regression.feature_importances_
        rf_importances = self.rf_regression.feature_importances_

        # Average the importances (weighted by ensemble weights)
        avg_importances = (xgb_importances * self.xgb_weight) + (rf_importances * self.rf_weight)

        # Create dictionary
        importance_dict = dict(zip(all_features, avg_importances))

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict

    def save_model(self, filepath: str) -> None:
        """
        Save trained ensemble models to file.

        Args:
            filepath: Path to save model file
        """
        model_data = {
            "xgb_regression": self.xgb_regression,
            "xgb_classification": self.xgb_classification,
            "rf_regression": self.rf_regression,
            "rf_classification": self.rf_classification,
            "preprocessor": self.preprocessor,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "xgb_weight": self.xgb_weight,
            "rf_weight": self.rf_weight,
            "market_weights": self.market_weights,
        }

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Ensemble model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load trained ensemble models from file.

        Args:
            filepath: Path to model file
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.xgb_regression = model_data["xgb_regression"]
        self.xgb_classification = model_data["xgb_classification"]
        self.rf_regression = model_data["rf_regression"]
        self.rf_classification = model_data["rf_classification"]
        self.preprocessor = model_data["preprocessor"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_names = model_data["feature_names"]
        self.n_estimators = model_data["n_estimators"]
        self.max_depth = model_data["max_depth"]
        self.learning_rate = model_data["learning_rate"]
        self.xgb_weight = model_data.get("xgb_weight", 0.5)
        self.rf_weight = model_data.get("rf_weight", 0.5)
        self.market_weights = model_data.get("market_weights", {
            'h2h': {'xgb': 0.5, 'rf': 0.5},
            'spreads': {'xgb': 0.5, 'rf': 0.5},
            'totals': {'xgb': 0.5, 'rf': 0.5},
        })

        logger.info(f"Ensemble model loaded from {filepath}")
        logger.info(f"Market weights: {self.market_weights}")

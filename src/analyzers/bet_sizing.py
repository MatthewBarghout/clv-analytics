"""Bet sizing strategies for bankroll management."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class BetSizingStrategy(str, Enum):
    """Available bet sizing strategies."""
    FIXED = "fixed"
    FRACTIONAL = "fractional"
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    CONFIDENCE = "confidence"


@dataclass
class BetContext:
    """Context for calculating bet size."""
    bankroll: float
    odds: float  # Decimal odds (e.g., 2.0 for even money)
    win_probability: Optional[float] = None  # Estimated win probability
    clv_percentage: Optional[float] = None  # CLV edge
    ml_confidence: Optional[float] = None  # ML model confidence (0-1)

    # Optional constraints
    min_bet: float = 10.0
    max_bet: float = 1000.0
    max_fraction: float = 0.25  # Maximum fraction of bankroll per bet


@dataclass
class BetSizeResult:
    """Result of bet size calculation."""
    bet_size: float
    strategy_used: str
    raw_bet_size: float  # Before constraints
    was_capped: bool
    kelly_fraction: Optional[float] = None  # For Kelly-based strategies
    edge_estimate: Optional[float] = None


class BetSizer(ABC):
    """Abstract base class for bet sizing strategies."""

    @abstractmethod
    def calculate_bet_size(self, context: BetContext) -> BetSizeResult:
        """Calculate the recommended bet size."""
        pass

    def _apply_constraints(
        self,
        bet_size: float,
        context: BetContext
    ) -> tuple[float, bool]:
        """Apply min/max constraints to bet size."""
        original = bet_size

        # Apply max fraction constraint
        max_from_fraction = context.bankroll * context.max_fraction
        bet_size = min(bet_size, max_from_fraction)

        # Apply absolute constraints
        bet_size = max(bet_size, context.min_bet)
        bet_size = min(bet_size, context.max_bet)

        # Can't bet more than bankroll
        bet_size = min(bet_size, context.bankroll)

        was_capped = bet_size != original
        return bet_size, was_capped


class FixedBetSizer(BetSizer):
    """Fixed unit bet sizing - same amount per bet."""

    def __init__(self, unit_size: float = 100.0):
        self.unit_size = unit_size

    def calculate_bet_size(self, context: BetContext) -> BetSizeResult:
        """Return fixed bet amount regardless of edge or bankroll."""
        bet_size, was_capped = self._apply_constraints(self.unit_size, context)

        return BetSizeResult(
            bet_size=bet_size,
            strategy_used="fixed",
            raw_bet_size=self.unit_size,
            was_capped=was_capped,
        )


class FractionalBetSizer(BetSizer):
    """Fixed fractional bet sizing - percentage of current bankroll."""

    def __init__(self, fraction: float = 0.02):
        """
        Initialize fractional sizer.

        Args:
            fraction: Fraction of bankroll to bet (e.g., 0.02 = 2%)
        """
        if fraction <= 0 or fraction > 1:
            raise ValueError("Fraction must be between 0 and 1")
        self.fraction = fraction

    def calculate_bet_size(self, context: BetContext) -> BetSizeResult:
        """Calculate bet as fixed percentage of bankroll."""
        raw_bet_size = context.bankroll * self.fraction
        bet_size, was_capped = self._apply_constraints(raw_bet_size, context)

        return BetSizeResult(
            bet_size=bet_size,
            strategy_used="fractional",
            raw_bet_size=raw_bet_size,
            was_capped=was_capped,
        )


class KellyBetSizer(BetSizer):
    """
    Kelly Criterion bet sizing - optimal sizing based on edge.

    Kelly formula: f* = (bp - q) / b
    Where:
        b = decimal odds - 1 (net odds)
        p = probability of winning
        q = probability of losing (1 - p)
    """

    def __init__(self, fraction_of_kelly: float = 1.0, default_edge: float = 0.02):
        """
        Initialize Kelly sizer.

        Args:
            fraction_of_kelly: Multiplier for Kelly (e.g., 0.5 for half-Kelly)
            default_edge: Default edge to assume if none provided
        """
        self.fraction_of_kelly = fraction_of_kelly
        self.default_edge = default_edge

    def estimate_win_probability(self, context: BetContext) -> float:
        """
        Estimate win probability from CLV or use fair odds.

        If CLV is provided, we adjust the fair probability by the CLV edge.
        """
        # Fair probability from odds (no-vig estimate)
        fair_prob = 1 / context.odds

        if context.win_probability is not None:
            return context.win_probability

        if context.clv_percentage is not None:
            # CLV% represents edge over closing line
            # Positive CLV means we have better odds than closing
            # Adjust probability by edge estimate
            edge = context.clv_percentage / 100  # Convert to decimal
            adjusted_prob = fair_prob + (edge * fair_prob * 0.5)  # Conservative estimate
            return min(0.95, max(0.05, adjusted_prob))

        # Use small default edge
        return fair_prob + self.default_edge

    def calculate_kelly_fraction(self, context: BetContext) -> float:
        """Calculate optimal Kelly fraction."""
        win_prob = self.estimate_win_probability(context)
        lose_prob = 1 - win_prob

        # Net odds (decimal odds - 1)
        b = context.odds - 1

        # Kelly formula: f* = (bp - q) / b
        kelly_fraction = (b * win_prob - lose_prob) / b

        # Kelly should not recommend betting on negative edge
        kelly_fraction = max(0, kelly_fraction)

        return kelly_fraction

    def calculate_bet_size(self, context: BetContext) -> BetSizeResult:
        """Calculate optimal bet size using Kelly Criterion."""
        kelly_fraction = self.calculate_kelly_fraction(context)

        # Apply fractional Kelly
        adjusted_fraction = kelly_fraction * self.fraction_of_kelly

        raw_bet_size = context.bankroll * adjusted_fraction
        bet_size, was_capped = self._apply_constraints(raw_bet_size, context)

        # Calculate edge estimate
        win_prob = self.estimate_win_probability(context)
        expected_value = (win_prob * (context.odds - 1)) - (1 - win_prob)

        return BetSizeResult(
            bet_size=bet_size,
            strategy_used=f"kelly_{self.fraction_of_kelly}x",
            raw_bet_size=raw_bet_size,
            was_capped=was_capped,
            kelly_fraction=kelly_fraction,
            edge_estimate=expected_value,
        )


class ConfidenceWeightedBetSizer(BetSizer):
    """
    Confidence-weighted bet sizing - scale bet by ML model confidence.

    Combines fractional betting with confidence scaling:
    bet_size = bankroll * base_fraction * confidence_multiplier
    """

    def __init__(
        self,
        base_fraction: float = 0.02,
        min_confidence: float = 0.5,
        max_confidence: float = 0.9,
        min_multiplier: float = 0.5,
        max_multiplier: float = 2.0,
    ):
        """
        Initialize confidence-weighted sizer.

        Args:
            base_fraction: Base fraction of bankroll
            min_confidence: Confidence below this uses min_multiplier
            max_confidence: Confidence above this uses max_multiplier
            min_multiplier: Minimum bet size multiplier
            max_multiplier: Maximum bet size multiplier
        """
        self.base_fraction = base_fraction
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

    def calculate_confidence_multiplier(self, confidence: float) -> float:
        """Calculate bet size multiplier based on confidence."""
        if confidence <= self.min_confidence:
            return self.min_multiplier
        if confidence >= self.max_confidence:
            return self.max_multiplier

        # Linear interpolation between min and max
        confidence_range = self.max_confidence - self.min_confidence
        multiplier_range = self.max_multiplier - self.min_multiplier

        normalized_confidence = (confidence - self.min_confidence) / confidence_range
        multiplier = self.min_multiplier + (normalized_confidence * multiplier_range)

        return multiplier

    def calculate_bet_size(self, context: BetContext) -> BetSizeResult:
        """Calculate bet size weighted by ML confidence."""
        confidence = context.ml_confidence or 0.5
        multiplier = self.calculate_confidence_multiplier(confidence)

        raw_bet_size = context.bankroll * self.base_fraction * multiplier
        bet_size, was_capped = self._apply_constraints(raw_bet_size, context)

        return BetSizeResult(
            bet_size=bet_size,
            strategy_used="confidence",
            raw_bet_size=raw_bet_size,
            was_capped=was_capped,
        )


def get_bet_sizer(
    strategy: BetSizingStrategy,
    unit_size: float = 100.0,
    fraction: float = 0.02,
    kelly_fraction: float = 1.0,
) -> BetSizer:
    """
    Factory function to get the appropriate bet sizer.

    Args:
        strategy: The betting strategy to use
        unit_size: Fixed bet amount (for FIXED strategy)
        fraction: Fraction of bankroll (for FRACTIONAL/CONFIDENCE)
        kelly_fraction: Fraction of full Kelly (for KELLY/HALF_KELLY)

    Returns:
        Configured BetSizer instance
    """
    if strategy == BetSizingStrategy.FIXED:
        return FixedBetSizer(unit_size=unit_size)

    elif strategy == BetSizingStrategy.FRACTIONAL:
        return FractionalBetSizer(fraction=fraction)

    elif strategy == BetSizingStrategy.KELLY:
        return KellyBetSizer(fraction_of_kelly=kelly_fraction)

    elif strategy == BetSizingStrategy.HALF_KELLY:
        return KellyBetSizer(fraction_of_kelly=0.5)

    elif strategy == BetSizingStrategy.CONFIDENCE:
        return ConfidenceWeightedBetSizer(base_fraction=fraction)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Sharpe ratio for a series of bet returns.

    Args:
        returns: List of individual bet returns (profit/loss as decimal)
        risk_free_rate: Risk-free rate (default 0)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate

    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)

    if std_return == 0:
        return 0.0

    # Assume daily betting, annualize with sqrt(252)
    sharpe = (mean_return / std_return) * np.sqrt(252)

    return sharpe


def calculate_sortino_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Sortino ratio - like Sharpe but only considers downside volatility.

    Args:
        returns: List of individual bet returns
        risk_free_rate: Risk-free rate

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate

    mean_return = np.mean(excess_returns)

    # Only consider negative returns for downside deviation
    negative_returns = excess_returns[excess_returns < 0]

    if len(negative_returns) == 0:
        return float('inf') if mean_return > 0 else 0.0

    downside_std = np.std(negative_returns, ddof=1)

    if downside_std == 0:
        return 0.0

    sortino = (mean_return / downside_std) * np.sqrt(252)

    return sortino

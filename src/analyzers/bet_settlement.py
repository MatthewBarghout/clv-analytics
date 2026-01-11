"""
Bet Settlement Calculator

Determines bet outcomes and calculates profit/loss for betting opportunities.
"""
import logging
from typing import Dict, Optional, Tuple

from src.models.database import BettingOutcome

logger = logging.getLogger(__name__)


class BetSettlement:
    """Calculate bet results and profit/loss."""

    @staticmethod
    def american_odds_to_decimal(american_odds: float) -> float:
        """
        Convert American odds to decimal odds.

        Args:
            american_odds: American odds format (e.g., +150, -110)

        Returns:
            Decimal odds
        """
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    @staticmethod
    def calculate_profit(bet_amount: float, american_odds: float, result: str) -> float:
        """
        Calculate profit/loss for a bet.

        Args:
            bet_amount: Amount wagered (e.g., 100.0)
            american_odds: American odds (e.g., +150, -110)
            result: "win", "loss", or "push"

        Returns:
            Profit/loss amount (positive for profit, negative for loss)
        """
        if result == "push":
            return 0.0
        elif result == "loss":
            return -bet_amount
        elif result == "win":
            if american_odds > 0:
                # Underdog: profit = stake * (odds / 100)
                return bet_amount * (american_odds / 100)
            else:
                # Favorite: profit = stake * (100 / abs(odds))
                return bet_amount * (100 / abs(american_odds))
        else:
            return 0.0

    @staticmethod
    def determine_spread_result(
        outcome: BettingOutcome,
        bet_side: str,
        spread_line: float
    ) -> str:
        """
        Determine if a spread bet won/lost/pushed.

        Args:
            outcome: Game outcome with scores
            bet_side: "home" or "away"
            spread_line: Spread line (e.g., -5.5 for favorite, +5.5 for underdog)

        Returns:
            "win", "loss", or "push"
        """
        if not outcome.completed:
            return "pending"

        point_diff = outcome.point_differential  # home_score - away_score

        if bet_side == "home":
            # Home team covering: point_diff > spread_line
            # Example: Home -5.5, they win by 6+ → covers
            adjusted_diff = point_diff + spread_line
        else:
            # Away team covering: point_diff < -spread_line
            # Example: Away +5.5, they lose by 5 or less → covers
            adjusted_diff = -point_diff + spread_line

        if adjusted_diff > 0:
            return "win"
        elif adjusted_diff < 0:
            return "loss"
        else:
            return "push"

    @staticmethod
    def determine_total_result(
        outcome: BettingOutcome,
        bet_side: str,
        total_line: float
    ) -> str:
        """
        Determine if a totals bet won/lost/pushed.

        Args:
            outcome: Game outcome with scores
            bet_side: "Over" or "Under"
            total_line: Total points line (e.g., 220.5)

        Returns:
            "win", "loss", or "push"
        """
        if not outcome.completed:
            return "pending"

        total_points = outcome.total_points

        if bet_side == "Over":
            if total_points > total_line:
                return "win"
            elif total_points < total_line:
                return "loss"
            else:
                return "push"
        else:  # Under
            if total_points < total_line:
                return "win"
            elif total_points > total_line:
                return "loss"
            else:
                return "push"

    @staticmethod
    def determine_moneyline_result(
        outcome: BettingOutcome,
        bet_team: str
    ) -> str:
        """
        Determine if a moneyline bet won/lost.

        Args:
            outcome: Game outcome with scores
            bet_team: Team name that was bet on

        Returns:
            "win" or "loss" (no pushes in moneyline)
        """
        if not outcome.completed:
            return "pending"

        # This requires team name matching which is complex
        # For now, we'll use the winner field which should be "home" or "away"
        # Caller needs to determine if bet_team was home or away
        if outcome.winner == "push":
            # In case of tie (very rare in NBA)
            return "push"

        # This method needs team context - will be handled by caller
        # Return pending for now
        return "pending"

    @staticmethod
    def settle_opportunity(
        outcome: BettingOutcome,
        opportunity: Dict,
        bet_amount: float = 100.0
    ) -> Tuple[str, float]:
        """
        Settle a betting opportunity and calculate P/L.

        Args:
            outcome: Game outcome with final scores
            opportunity: Opportunity dict with keys:
                - market_type: "h2h", "spreads", or "totals"
                - outcome: Team name or "Over"/"Under"
                - entry_odds: American odds
            bet_amount: Amount wagered (default: $100)

        Returns:
            Tuple of (result, profit_loss)
            - result: "win", "loss", "push", or "pending"
            - profit_loss: Profit/loss amount in dollars
        """
        if not outcome or not outcome.completed:
            return ("pending", 0.0)

        market_type = opportunity.get("market_type")
        outcome_name = opportunity.get("outcome")
        entry_odds = opportunity.get("entry_odds")

        result = "pending"

        try:
            if market_type == "spreads":
                # Extract spread from outcome (stored in odds)
                # This is tricky - we need to parse the spread line from the odds data
                # For now, we'll mark as pending and handle in future enhancement
                logger.warning(f"Spread settlement not yet implemented for {outcome_name}")
                result = "pending"

            elif market_type == "totals":
                # Extract total line from the odds data
                # Similar issue - need the actual line
                logger.warning(f"Totals settlement not yet implemented")
                result = "pending"

            elif market_type == "h2h":
                # Moneyline - need to know if outcome_name was home or away
                logger.warning(f"Moneyline settlement not yet implemented")
                result = "pending"

        except Exception as e:
            logger.error(f"Error settling opportunity: {e}")
            result = "pending"

        # Calculate P/L
        profit_loss = BetSettlement.calculate_profit(bet_amount, entry_odds, result)

        return (result, profit_loss)

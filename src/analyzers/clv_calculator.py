from typing import Optional

from sqlalchemy.orm import Session

from src.models.database import OddsSnapshot


class CLVCalculator:
    """Calculate Closing Line Value for sports bets."""

    @staticmethod
    def decimal_to_implied_prob(odds: float) -> float:
        """
        Convert decimal odds to implied probability.

        Args:
            odds: Decimal odds (e.g., 2.1, 1.77)

        Returns:
            Implied probability as a decimal (0-1)

        Example:
            2.0 -> 0.5 (50%)
            1.5 -> 0.667 (66.7%)
        """
        if odds <= 1.0:
            raise ValueError("Decimal odds must be greater than 1.0")
        return 1.0 / odds

    @staticmethod
    def american_to_implied_prob(odds: int) -> float:
        """
        Convert American odds to implied probability.
        
        """
        if odds<0:
            absolute_value=abs(odds)
            denominator=absolute_value+100
            probability=absolute_value / denominator
            return probability 
        else:
            denominator= odds + 100
            probability= 100 / denominator
            return probability
        
    @staticmethod
    def calculate_clv(entry_odds: float, closing_odds: float) -> float:
        """
        Calculate CLV in percentage points using decimal odds.

        Args:
            entry_odds: Decimal odds when bet was placed (e.g., 2.1)
            closing_odds: Decimal odds at closing (e.g., 1.77)

        Returns:
            CLV as percentage points

        Example:
            Entry: 2.10 (Lakers) -> 47.6% implied
            Closing: 1.77 (Lakers) -> 56.5% implied
            CLV = 56.5 - 47.6 = +8.9% (you beat the closing line!)
        """
        entry_prob = CLVCalculator.decimal_to_implied_prob(entry_odds)
        closing_prob = CLVCalculator.decimal_to_implied_prob(closing_odds)
        clv = closing_prob - entry_prob
        clv = clv * 100
        return clv

    def _extract_odds_for_team(self, outcomes: list, team_name: str) -> Optional[float]:
        """
        Extract odds for a specific team from outcomes list.

        Args:
            outcomes: List of outcome dictionaries from JSONB field
            team_name: Name of the team to find odds for

        Returns:
            Odds (decimal format) for the team, or None if not found

        Example:
            outcomes = [
                {'name': 'Lakers', 'price': 2.1},
                {'name': 'Celtics', 'price': 1.77}
            ]
            _extract_odds_for_team(outcomes, 'Lakers') -> 2.1
        """
        for outcome in outcomes:
            if outcome.get('name') == team_name:
                return outcome.get('price')
        return None

    def calculate_clv_for_snapshot(
        self,
        session: Session,
        snapshot: OddsSnapshot,
        team_name: str
    ) -> Optional[float]:
        """
        Calculate CLV for a specific snapshot by comparing to closing line.

        Args:
            session: SQLAlchemy session
            snapshot: OddsSnapshot with the entry odds
            team_name: Name of the team to calculate CLV for

        Returns:
            CLV percentage or None if closing line not found

        Example:
            Entry odds: Lakers 2.1 (snapshot)
            Closing odds: Lakers 1.77 (closing line)
            CLV = 56.5% - 47.6% = +8.9% (you beat the closing line!)
        """
        from sqlalchemy import select
        from src.models.database import ClosingLine

        # Find closing line for same game/bookmaker/market
        stmt = select(ClosingLine).where(
            ClosingLine.game_id == snapshot.game_id,
            ClosingLine.bookmaker_id == snapshot.bookmaker_id,
            ClosingLine.market_type == snapshot.market_type
        )
        closing_line = session.execute(stmt).scalar_one_or_none()

        if not closing_line:
            return None

        # Extract odds for the specific team from snapshot outcomes
        entry_odds = self._extract_odds_for_team(snapshot.outcomes, team_name)
        if entry_odds is None:
            return None

        # Extract odds for the specific team from closing line outcomes
        closing_odds = self._extract_odds_for_team(closing_line.outcomes, team_name)
        if closing_odds is None:
            return None

        # Calculate and return CLV
        return self.calculate_clv(entry_odds, closing_odds)
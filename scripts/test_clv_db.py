"""
Test CLV calculation with real database data.
Run: poetry run python -m scripts.test_clv_db
"""
import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.analyzers.clv_calculator import CLVCalculator
from src.models.database import Bookmaker, Game, OddsSnapshot, Team

logging.basicConfig(level=logging.INFO)


def main():
    load_dotenv()

    # Setup database
    engine = create_engine(os.getenv("DATABASE_URL"))
    Session = sessionmaker(bind=engine)
    session = Session()

    # Initialize calculator
    calc = CLVCalculator()

    try:
        # Get a snapshot to test with
        stmt = (
            select(OddsSnapshot, Game, Team, Bookmaker)
            .join(Game, OddsSnapshot.game_id == Game.id)
            .join(Team, Game.home_team_id == Team.id)
            .join(Bookmaker, OddsSnapshot.bookmaker_id == Bookmaker.id)
            .where(OddsSnapshot.market_type == "h2h")
            .limit(1)
        )

        result = session.execute(stmt).first()
        if not result:
            print("No odds snapshots found")
            return

        snapshot, game, home_team, bookmaker = result

        print("\nTesting CLV Calculation")
        print("=" * 60)
        print(f"Game: {home_team.name} (home)")
        print(f"Bookmaker: {bookmaker.name}")
        print(f"Market: {snapshot.market_type}")
        print(f"Snapshot outcomes: {snapshot.outcomes}")

        # Try to calculate CLV (will be None since we don't have closing lines yet)
        clv = calc.calculate_clv_for_snapshot(
            session=session, snapshot=snapshot, team_name=home_team.name
        )

        if clv is None:
            print("\nNo closing line found yet (expected - we haven't collected closing lines)")
            print("CLV calculator is ready - will work once we have closing lines")
        else:
            print(f"\nCLV: {clv:+.2f}%")

        # Test odds extraction
        print("\n" + "=" * 60)
        print("Testing odds extraction:")
        for outcome in snapshot.outcomes:
            team_name = outcome.get("name")
            odds = calc._extract_odds_for_team(snapshot.outcomes, team_name)
            print(f"  {team_name}: {odds}")

        print("=" * 60)

    finally:
        session.close()


if __name__ == "__main__":
    main()

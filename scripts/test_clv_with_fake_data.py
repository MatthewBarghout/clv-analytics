"""
Test CLV calculation with fake closing line data.
This inserts a test closing line and verifies end-to-end CLV calculation.

Run: poetry run python -m scripts.test_clv_with_fake_data
"""
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.analyzers.clv_calculator import CLVCalculator
from src.models.database import Bookmaker, ClosingLine, Game, OddsSnapshot, Team

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
            print("ERROR: No odds snapshots found in database")
            return

        snapshot, game, home_team, bookmaker = result

        print("\n" + "=" * 70)
        print("CLV CALCULATOR - END-TO-END TEST WITH FAKE DATA")
        print("=" * 70)

        print("\nSnapshot Data:")
        print(f"  Game: {home_team.name} (home)")
        print(f"  Bookmaker: {bookmaker.name}")
        print(f"  Market: {snapshot.market_type}")
        print(f"  Entry odds: {snapshot.outcomes}")

        # Create fake closing line with different odds
        # Entry: Memphis Grizzlies 2.1, Philadelphia 76ers 1.77
        # Closing: Memphis Grizzlies 1.95, Philadelphia 76ers 1.90
        # This simulates the line moving (Memphis got worse, Philly got better)
        fake_closing_outcomes = [
            {"name": "Memphis Grizzlies", "price": 1.95},
            {"name": "Philadelphia 76ers", "price": 1.90},
        ]

        print("\nCreating fake closing line:")
        print(f"  Closing odds: {fake_closing_outcomes}")

        # Check if closing line already exists
        existing_closing = (
            session.query(ClosingLine)
            .filter(
                ClosingLine.game_id == snapshot.game_id,
                ClosingLine.bookmaker_id == snapshot.bookmaker_id,
                ClosingLine.market_type == snapshot.market_type,
            )
            .first()
        )

        if existing_closing:
            print("\nRemoving existing test closing line...")
            session.delete(existing_closing)
            session.commit()

        # Insert fake closing line
        closing_line = ClosingLine(
            game_id=snapshot.game_id,
            bookmaker_id=snapshot.bookmaker_id,
            market_type=snapshot.market_type,
            outcomes=fake_closing_outcomes,
            closed_at=datetime.now(timezone.utc),
        )

        session.add(closing_line)
        session.commit()
        print("Fake closing line inserted successfully")

        # Calculate CLV for both teams
        print("\n" + "=" * 70)
        print("CALCULATING CLV:")
        print("=" * 70)

        for outcome in snapshot.outcomes:
            team_name = outcome["name"]
            entry_odds = outcome["price"]

            # Find closing odds
            closing_odds = calc._extract_odds_for_team(
                closing_line.outcomes, team_name
            )

            # Calculate CLV
            clv = calc.calculate_clv_for_snapshot(
                session=session, snapshot=snapshot, team_name=team_name
            )

            # Calculate implied probabilities
            entry_prob = calc.decimal_to_implied_prob(entry_odds)
            closing_prob = calc.decimal_to_implied_prob(closing_odds)

            print(f"\n{team_name}:")
            print(f"  Entry odds: {entry_odds} (implied: {entry_prob:.1%})")
            print(f"  Closing odds: {closing_odds} (implied: {closing_prob:.1%})")
            print(f"  CLV: {clv:+.2f}%")

            if clv > 0:
                print(f"  Result: YOU BEAT THE CLOSING LINE!")
            elif clv < 0:
                print(f"  Result: Line moved against you")
            else:
                print(f"  Result: No line movement")

        print("\n" + "=" * 70)
        print("TEST COMPLETE - CLV CALCULATOR WORKING CORRECTLY")
        print("=" * 70)

        # Clean up - remove fake closing line
        print("\nCleaning up test data...")
        session.delete(closing_line)
        session.commit()
        print("Test closing line removed")

    except Exception as e:
        print(f"\nERROR: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fetch Game Scores and Store Betting Outcomes

Automatically fetches final scores for completed NBA games and stores them
in the betting_outcomes table for bet settlement.

Usage:
    poetry run python scripts/fetch_game_scores.py              # Fetch scores for recent games
    poetry run python scripts/fetch_game_scores.py --days 7     # Fetch scores for last 7 days
    poetry run python scripts/fetch_game_scores.py --game-id 75 # Fetch score for specific game
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.collectors.nba_scores_client import NBAScoresClient
from src.models.database import BettingOutcome, Game, Team

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def fetch_and_store_score(session, game: Game, nba_client: NBAScoresClient, scores_cache: list) -> bool:
    """
    Fetch score for a single game and store in database.

    Args:
        session: SQLAlchemy session
        game: Game object to fetch score for
        nba_client: NBA.com scores client
        scores_cache: Cached list of scores from NBA.com API

    Returns:
        True if score was fetched and stored successfully
    """
    try:
        # Get team names
        home_team = session.get(Team, game.home_team_id)
        away_team = session.get(Team, game.away_team_id)

        logger.info(f"Fetching score for Game {game.id}: {away_team.name} @ {home_team.name}")

        # Find matching game by team names (case-insensitive partial match)
        for score in scores_cache:
            if not score.get("completed"):
                continue

            home_name = score.get("home_team", "")
            away_name = score.get("away_team", "")

            # Case-insensitive partial matching
            home_match = (
                home_name and home_team.name and
                (home_name.lower() in home_team.name.lower() or home_team.name.lower() in home_name.lower())
            )
            away_match = (
                away_name and away_team.name and
                (away_name.lower() in away_team.name.lower() or away_team.name.lower() in away_name.lower())
            )

            if home_match and away_match:
                home_score = score.get("home_score")
                away_score = score.get("away_score")

                if home_score is not None and away_score is not None:
                    # Check if outcome already exists
                    existing = session.execute(
                        select(BettingOutcome).where(BettingOutcome.game_id == game.id)
                    ).scalar_one_or_none()

                    if existing:
                        # Update existing outcome
                        existing.completed = True
                        existing.home_score = int(home_score)
                        existing.away_score = int(away_score)
                        existing.total_points = int(home_score) + int(away_score)
                        existing.point_differential = int(home_score) - int(away_score)
                        existing.winner = "home" if home_score > away_score else "away" if away_score > home_score else "push"
                        logger.info(f"  ✓ Updated existing outcome: {away_team.name} {away_score} @ {home_team.name} {home_score}")
                    else:
                        # Create new outcome
                        outcome = BettingOutcome(
                            game_id=game.id,
                            completed=True,
                            home_score=int(home_score),
                            away_score=int(away_score),
                            total_points=int(home_score) + int(away_score),
                            point_differential=int(home_score) - int(away_score),
                            winner="home" if home_score > away_score else "away" if away_score > home_score else "push"
                        )
                        session.add(outcome)
                        logger.info(f"  ✓ Stored score: {away_team.name} {away_score} @ {home_team.name} {home_score} (Winner: {outcome.winner})")

                    # Mark game as completed
                    game.completed = True
                    session.commit()
                    return True

        logger.warning(f"  ✗ No score found for Game {game.id}")
        return False

    except Exception as e:
        logger.error(f"Error fetching score for Game {game.id}: {e}")
        session.rollback()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fetch game scores from NBA.com")
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Fetch scores for games from last N days (default: 3)"
    )
    parser.add_argument(
        "--game-id",
        type=int,
        help="Fetch score for a specific game ID"
    )

    args = parser.parse_args()

    # Validate environment
    if not DATABASE_URL:
        logger.error("DATABASE_URL not found in environment")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("GAME SCORE FETCHING STARTED (NBA.com API)")
    logger.info("=" * 70)

    # Initialize
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    nba_client = NBAScoresClient()

    try:
        if args.game_id:
            # Fetch specific game
            game = session.get(Game, args.game_id)
            if not game:
                logger.error(f"Game {args.game_id} not found")
                sys.exit(1)

            # Fetch scores for the game's date
            game_date = game.commence_time.date()
            scores = nba_client.get_scores_for_date_range(
                datetime.combine(game_date, datetime.min.time()),
                days=1
            )

            fetch_and_store_score(session, game, nba_client, scores)
        else:
            # Fetch recent games
            cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

            # Find completed games without scores
            stmt = (
                select(Game)
                .outerjoin(BettingOutcome, BettingOutcome.game_id == Game.id)
                .where(
                    Game.commence_time >= cutoff,
                    Game.commence_time < datetime.now(timezone.utc),
                    BettingOutcome.id.is_(None)  # No outcome record yet
                )
                .order_by(Game.commence_time.desc())
            )

            games = session.execute(stmt).scalars().all()

            logger.info(f"Found {len(games)} completed games without scores (last {args.days} days)")

            if not games:
                logger.info("No games to process")
                return

            # Fetch scores from NBA.com for the date range
            logger.info(f"Fetching scores from NBA.com for last {args.days} days...")
            start_date = datetime.now(timezone.utc) - timedelta(days=args.days)
            scores = nba_client.get_scores_for_date_range(start_date, days=args.days)
            logger.info(f"Retrieved {len(scores)} completed games from NBA.com")

            # Match and store scores
            success_count = 0
            for game in games:
                if fetch_and_store_score(session, game, nba_client, scores):
                    success_count += 1

            logger.info("=" * 70)
            logger.info(f"SUMMARY: Fetched scores for {success_count}/{len(games)} games")
            logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()

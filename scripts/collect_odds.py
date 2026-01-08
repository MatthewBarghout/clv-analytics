"""
Collect NBA odds and store in database.

Usage:
    poetry run python -m scripts.collect_odds
    poetry run python -m scripts.collect_odds --closing-only

Run: poetry run python -m scripts.collect_odds
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.collectors.odds_api_client import OddsAPIClient
from src.collectors.odds_proccessor import OddsDataProcessor
from src.utils.notifications import EmailNotifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def is_closing_line_time(game_commence_time: datetime, buffer_hours: int = 2) -> bool:
    """
    Check if current time is within buffer hours of game start.

    Args:
        game_commence_time: When the game starts
        buffer_hours: Hours before game to consider "closing line"

    Returns:
        True if within closing line window
    """
    now = datetime.now(timezone.utc)
    time_until_game = game_commence_time - now
    return timedelta(0) <= time_until_game <= timedelta(hours=buffer_hours)


def collect_nba_odds(closing_only: bool = False):
    """
    Collect NBA odds and store in database.

    Args:
        closing_only: If True, only store closing lines (skip regular snapshots)
    """
    load_dotenv()

    # Validate environment
    api_key = os.getenv("ODDS_API_KEY")
    db_url = os.getenv("DATABASE_URL")

    if not api_key:
        logger.error("ODDS_API_KEY not found in environment")
        sys.exit(1)

    if not db_url:
        logger.error("DATABASE_URL not found in environment")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("NBA ODDS COLLECTION STARTED")
    logger.info("=" * 70)
    logger.info(f"Mode: {'CLOSING LINES ONLY' if closing_only else 'REGULAR SNAPSHOTS'}")

    # Initialize API client
    client = OddsAPIClient(api_key=api_key)

    # Setup database
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Initialize processor
    processor = OddsDataProcessor(session=session)

    try:
        # Fetch NBA odds
        logger.info("Fetching NBA odds from API...")
        bookmakers = ["pinnacle", "fanduel", "draftkings", "espnbet"]

        response = client.get_odds(
            sport="basketball_nba",
            regions="us",
            markets="h2h,spreads,totals",
            bookmakers=bookmakers,
        )

        games_data = response.get("data", [])
        remaining_requests = response.get("remaining_requests")

        if not games_data:
            logger.warning("No games found")
            logger.info(f"API requests remaining: {remaining_requests}")
            return

        logger.info(f"Fetched {len(games_data)} games")

        # Process games
        snapshots_stored = 0
        closing_lines_stored = 0
        games_processed = 0

        for game_data in games_data:
            try:
                game_id = game_data.get("id")
                commence_time = datetime.fromisoformat(
                    game_data["commence_time"].replace("Z", "+00:00")
                )
                home_team = game_data.get("home_team")
                away_team = game_data.get("away_team")

                # Check if this is closing line time
                is_closing = is_closing_line_time(commence_time)

                logger.info(
                    f"Processing: {away_team} @ {home_team} "
                    f"({commence_time.strftime('%Y-%m-%d %H:%M')} UTC)"
                )

                if is_closing:
                    logger.info("  Within 2 hours of game time - storing closing lines")

                # Store regular snapshots unless closing_only mode
                if not closing_only:
                    count = processor.process_odds_response(
                        sport_key="basketball_nba", api_data={"data": [game_data]}
                    )
                    snapshots_stored += count
                    logger.info(f"  Stored {count} odds snapshots")

                # Store closing lines if within time window
                if is_closing:
                    # Get the game from database
                    from sqlalchemy import select

                    from src.models.database import Bookmaker, Game

                    # Find the game we just created/updated
                    stmt = (
                        select(Game)
                        .where(Game.commence_time == commence_time)
                        .order_by(Game.id.desc())
                        .limit(1)
                    )
                    game = session.execute(stmt).scalar_one_or_none()

                    if game:
                        for bookmaker_data in game_data.get("bookmakers", []):
                            # Get bookmaker
                            bookmaker_stmt = select(Bookmaker).where(
                                Bookmaker.key == bookmaker_data["key"]
                            )
                            bookmaker = session.execute(bookmaker_stmt).scalar_one_or_none()

                            if bookmaker:
                                for market in bookmaker_data.get("markets", []):
                                    closing_line = processor.store_closing_line(
                                        game_id=game.id,
                                        bookmaker_id=bookmaker.id,
                                        market_type=market["key"],
                                        outcomes=market["outcomes"],
                                    )
                                    if closing_line:
                                        closing_lines_stored += 1

                        logger.info(f"  Stored closing lines")

                games_processed += 1

            except Exception as e:
                logger.error(f"Error processing game {game_id}: {e}")
                session.rollback()
                continue

        # Summary
        logger.info("=" * 70)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Games processed: {games_processed}")

        if not closing_only:
            logger.info(f"Odds snapshots stored: {snapshots_stored}")

        if closing_lines_stored > 0:
            logger.info(f"Closing lines stored: {closing_lines_stored}")

        logger.info(f"API requests remaining: {remaining_requests}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Collection failed: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Collect NBA odds from The Odds API")
    parser.add_argument(
        "--closing-only",
        action="store_true",
        help="Only store closing lines (skip regular snapshots)",
    )

    args = parser.parse_args()
    notifier = EmailNotifier()

    try:
        collect_nba_odds(closing_only=args.closing_only)
    except Exception as e:
        logger.error(f"Fatal error: {e}")

        # Send failure notification if this was a closing-only batch
        if args.closing_only:
            batch_time = datetime.now().strftime('%I:%M %p')
            error_msg = str(e)

            # Try to estimate games affected by checking database
            try:
                load_dotenv()
                engine = create_engine(os.getenv("DATABASE_URL"))
                Session = sessionmaker(bind=engine)
                session = Session()

                from src.models.database import Game
                now = datetime.now(timezone.utc)
                games_count = session.query(Game).filter(
                    Game.commence_time >= now,
                    Game.commence_time <= now + timedelta(hours=2)
                ).count()
                session.close()
            except:
                games_count = 0

            notifier.send_collection_failure(
                error_message=error_msg,
                batch_time=batch_time,
                games_affected=games_count
            )

        sys.exit(1)


if __name__ == "__main__":
    main()

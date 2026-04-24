"""
Collect odds for all active sports and store in database.

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

from requests.exceptions import HTTPError

from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.collectors.odds_api_client import OddsAPIClient
from src.collectors.odds_proccessor import OddsDataProcessor
from src.models.database import Bookmaker, Game, Team
from src.utils.notifications import EmailNotifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BOOKMAKERS = ["pinnacle", "fanduel", "draftkings", "espnbet"]

SPORTS = [
    ("basketball_nba", "NBA"),
    ("baseball_mlb", "MLB"),
]


def is_closing_line_time(game_commence_time: datetime, buffer_hours: int = 2) -> bool:
    """Return True if current time is within buffer_hours of game start."""
    now = datetime.now(timezone.utc)
    time_until_game = game_commence_time - now
    return timedelta(0) <= time_until_game <= timedelta(hours=buffer_hours)


def collect_sport_odds(
    sport_key: str,
    sport_name: str,
    closing_only: bool,
    session,
    client: OddsAPIClient,
    processor: OddsDataProcessor,
) -> tuple[int, int]:
    """
    Collect odds for one sport. Returns (snapshots_stored, closing_lines_stored).
    """
    logger.info("=" * 70)
    logger.info(f"{sport_name} ODDS COLLECTION")
    logger.info("=" * 70)

    response = client.get_odds(
        sport=sport_key,
        regions="us",
        markets="h2h,spreads,totals",
        bookmakers=BOOKMAKERS,
    )

    games_data = response.get("data", [])
    remaining_requests = response.get("remaining_requests")

    if not games_data:
        logger.warning(f"No {sport_name} games found")
        logger.info(f"API requests remaining: {remaining_requests}")
        return 0, 0

    logger.info(f"Fetched {len(games_data)} games | API requests remaining: {remaining_requests}")

    snapshots_stored = 0
    closing_lines_stored = 0
    games_processed = 0

    for game_data in games_data:
        game_id = game_data.get("id")
        try:
            commence_time = datetime.fromisoformat(
                game_data["commence_time"].replace("Z", "+00:00")
            )
            home_team = game_data.get("home_team")
            away_team = game_data.get("away_team")

            is_closing = is_closing_line_time(commence_time)

            logger.info(
                f"Processing: {away_team} @ {home_team} "
                f"({commence_time.strftime('%Y-%m-%d %H:%M')} UTC)"
            )

            if is_closing:
                logger.info("  Within 2 hours of game time - storing closing lines")

            if not closing_only:
                count = processor.process_odds_response(
                    sport_key=sport_key, api_data={"data": [game_data]}
                )
                snapshots_stored += count
                logger.info(f"  Stored {count} odds snapshots")

            if is_closing:
                home_team_obj = session.execute(
                    select(Team).where(Team.name == home_team)
                ).scalar_one_or_none()
                away_team_obj = session.execute(
                    select(Team).where(Team.name == away_team)
                ).scalar_one_or_none()

                stmt = (
                    select(Game)
                    .where(
                        Game.commence_time == commence_time,
                        Game.home_team_id == home_team_obj.id if home_team_obj else None,
                        Game.away_team_id == away_team_obj.id if away_team_obj else None,
                    )
                    .order_by(Game.id.desc())
                    .limit(1)
                )
                game = session.execute(stmt).scalar_one_or_none()

                if game:
                    for bookmaker_data in game_data.get("bookmakers", []):
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

                    logger.info("  Stored closing lines")

            games_processed += 1

        except Exception as e:
            logger.error(f"Error processing game {game_id}: {e}")
            session.rollback()
            continue

    logger.info(f"{sport_name}: {games_processed} games | {snapshots_stored} snapshots | {closing_lines_stored} closing lines")
    return snapshots_stored, closing_lines_stored


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Collect odds from The Odds API")
    parser.add_argument(
        "--closing-only",
        action="store_true",
        help="Only store closing lines (skip regular snapshots)",
    )

    args = parser.parse_args()
    notifier = EmailNotifier()

    load_dotenv()
    api_key = os.getenv("ODDS_API_KEY")
    api_key_2 = os.getenv("ODDS_API_KEY_2")
    db_url = os.getenv("DATABASE_URL")

    if not api_key:
        logger.error("ODDS_API_KEY not found in environment")
        sys.exit(1)
    if not db_url:
        logger.error("DATABASE_URL not found in environment")
        sys.exit(1)

    api_keys = [k for k in [api_key, api_key_2] if k]

    logger.info("=" * 70)
    logger.info("MULTI-SPORT ODDS COLLECTION STARTED")
    logger.info(f"Mode: {'CLOSING LINES ONLY' if args.closing_only else 'REGULAR SNAPSHOTS'}")
    logger.info(f"API keys available: {len(api_keys)}")
    logger.info("=" * 70)

    client = OddsAPIClient(api_key=api_keys[0])
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    processor = OddsDataProcessor(session=session)

    total_snapshots = 0
    total_closing = 0
    failed_sports = []

    try:
        key_index = 0
        for sport_key, sport_name in SPORTS:
            try:
                snaps, closing = collect_sport_odds(
                    sport_key=sport_key,
                    sport_name=sport_name,
                    closing_only=args.closing_only,
                    session=session,
                    client=client,
                    processor=processor,
                )
                total_snapshots += snaps
                total_closing += closing
            except HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status in (401, 429) and key_index + 1 < len(api_keys):
                    key_index += 1
                    client.api_key = api_keys[key_index]
                    logger.warning(
                        f"{sport_name} key quota/auth error ({status}), "
                        f"rotating to key {key_index + 1} and retrying"
                    )
                    try:
                        snaps, closing = collect_sport_odds(
                            sport_key=sport_key,
                            sport_name=sport_name,
                            closing_only=args.closing_only,
                            session=session,
                            client=client,
                            processor=processor,
                        )
                        total_snapshots += snaps
                        total_closing += closing
                    except Exception as retry_err:
                        logger.error(f"{sport_name} collection failed after key rotation: {retry_err}")
                        failed_sports.append(sport_name)
                else:
                    logger.error(f"{sport_name} collection failed: {e}")
                    failed_sports.append(sport_name)
            except Exception as e:
                logger.error(f"{sport_name} collection failed: {e}")
                failed_sports.append(sport_name)
                continue

        logger.info("=" * 70)
        logger.info("ALL SPORTS COLLECTION COMPLETE")
        logger.info(f"Total snapshots stored: {total_snapshots}")
        logger.info(f"Total closing lines stored: {total_closing}")
        if failed_sports:
            logger.warning(f"Failed sports: {', '.join(failed_sports)}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Collection failed: {e}")
        session.rollback()

        if args.closing_only:
            batch_time = datetime.now().strftime('%I:%M %p')
            try:
                now = datetime.now(timezone.utc)
                games_count = session.query(Game).filter(
                    Game.commence_time >= now,
                    Game.commence_time <= now + timedelta(hours=2)
                ).count()
                session.close()
            except Exception:
                games_count = 0

            notifier.send_collection_failure(
                error_message=str(e),
                batch_time=batch_time,
                games_affected=games_count,
            )
        sys.exit(1)

    finally:
        session.close()


if __name__ == "__main__":
    main()

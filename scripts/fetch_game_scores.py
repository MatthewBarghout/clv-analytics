#!/usr/bin/env python3
"""
Fetch Game Scores and Store Betting Outcomes

Fetches final scores for completed games across all sports and stores them
in the betting_outcomes table for bet settlement.

- NBA: uses NBA.com API (free, no quota cost)
- MLB: uses MLB Stats API (free, no quota cost, statsapi.mlb.com)

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

from src.collectors.mlb_scores_client import MLBScoresClient
from src.collectors.nba_scores_client import NBAScoresClient
from src.models.database import BettingOutcome, Game, Sport, Team

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")



def fetch_and_store_score(session, game: Game, scores_cache: list) -> bool:
    """
    Match a game against a scores cache and store the BettingOutcome.

    Args:
        session: SQLAlchemy session
        game: Game object to fetch score for
        scores_cache: Normalized list of score dicts (home_team, away_team, home_score, away_score)

    Returns:
        True if score was matched and stored successfully
    """
    try:
        home_team = session.get(Team, game.home_team_id)
        away_team = session.get(Team, game.away_team_id)

        logger.info(f"Fetching score for Game {game.id}: {away_team.name} @ {home_team.name}")

        for score in scores_cache:
            if not score.get("completed"):
                continue

            home_name = score.get("home_team", "")
            away_name = score.get("away_team", "")

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
                    home_score = int(home_score)
                    away_score = int(away_score)

                    existing = session.execute(
                        select(BettingOutcome).where(BettingOutcome.game_id == game.id)
                    ).scalar_one_or_none()

                    if existing:
                        existing.completed = True
                        existing.home_score = home_score
                        existing.away_score = away_score
                        existing.total_points = home_score + away_score
                        existing.point_differential = home_score - away_score
                        existing.winner = "home" if home_score > away_score else "away" if away_score > home_score else "push"
                        logger.info(f"  Updated: {away_team.name} {away_score} @ {home_team.name} {home_score}")
                    else:
                        outcome = BettingOutcome(
                            game_id=game.id,
                            completed=True,
                            home_score=home_score,
                            away_score=away_score,
                            total_points=home_score + away_score,
                            point_differential=home_score - away_score,
                            winner="home" if home_score > away_score else "away" if away_score > home_score else "push"
                        )
                        session.add(outcome)
                        logger.info(f"  Stored: {away_team.name} {away_score} @ {home_team.name} {home_score} (Winner: {'home' if home_score > away_score else 'away' if away_score > home_score else 'push'})")

                    game.completed = True
                    session.commit()
                    return True

        logger.warning(f"  No score found for Game {game.id}")
        return False

    except Exception as e:
        logger.error(f"Error fetching score for Game {game.id}: {e}")
        session.rollback()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fetch game scores across all sports")
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

    if not DATABASE_URL:
        logger.error("DATABASE_URL not found in environment")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("GAME SCORE FETCHING STARTED")
    logger.info("=" * 70)

    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    nba_client = NBAScoresClient()
    mlb_client = MLBScoresClient()

    try:
        if args.game_id:
            game = session.get(Game, args.game_id)
            if not game:
                logger.error(f"Game {args.game_id} not found")
                sys.exit(1)

            sport = session.get(Sport, game.sport_id)
            sport_key = sport.key if sport else "basketball_nba"
            game_date = game.commence_time.date()
            start = datetime.combine(game_date, datetime.min.time())

            if sport_key == "baseball_mlb":
                scores = mlb_client.get_scores_for_date_range(start, days=1)
            else:
                scores = nba_client.get_scores_for_date_range(start, days=1)

            fetch_and_store_score(session, game, scores)

        else:
            cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

            stmt = (
                select(Game, Sport)
                .join(Sport, Sport.id == Game.sport_id)
                .outerjoin(BettingOutcome, BettingOutcome.game_id == Game.id)
                .where(
                    Game.commence_time >= cutoff,
                    Game.commence_time < datetime.now(timezone.utc),
                    BettingOutcome.id.is_(None)
                )
                .order_by(Game.commence_time.desc())
            )

            rows = session.execute(stmt).all()
            games_by_sport: dict[str, list] = {}
            for game, sport in rows:
                games_by_sport.setdefault(sport.key, []).append(game)

            total_games = sum(len(g) for g in games_by_sport.values())
            logger.info(f"Found {total_games} games without scores (last {args.days} days)")

            if not total_games:
                logger.info("No games to process")
                return

            start_date = datetime.now(timezone.utc) - timedelta(days=args.days)
            scores_caches: dict[str, list] = {}

            if "basketball_nba" in games_by_sport:
                n = len(games_by_sport["basketball_nba"])
                logger.info(f"Fetching NBA scores from NBA.com ({n} games)...")
                scores = nba_client.get_scores_for_date_range(start_date, days=args.days)
                scores_caches["basketball_nba"] = scores
                logger.info(f"Retrieved {len(scores)} completed NBA games")

            if "baseball_mlb" in games_by_sport:
                n = len(games_by_sport["baseball_mlb"])
                logger.info(f"Fetching MLB scores from MLB Stats API ({n} games)...")
                scores = mlb_client.get_scores_for_date_range(start_date, days=args.days)
                scores_caches["baseball_mlb"] = scores
                logger.info(f"Retrieved {len(scores)} completed MLB games")

            success_count = 0
            for sport_key, games in games_by_sport.items():
                cache = scores_caches.get(sport_key, [])
                if not cache:
                    logger.warning(f"No score data available for {sport_key} — skipping")
                    continue
                for game in games:
                    if fetch_and_store_score(session, game, cache):
                        success_count += 1

            logger.info("=" * 70)
            logger.info(f"SUMMARY: Fetched scores for {success_count}/{total_games} games")
            logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()

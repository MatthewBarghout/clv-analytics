#!/usr/bin/env python3
"""
Fetch Game Scores and Store Betting Outcomes

Fetches final scores for completed games across all sports and stores them
in the betting_outcomes table for bet settlement.

- NBA: uses ESPN public API (free, no quota cost)
- MLB: uses MLB Stats API (free, no quota cost, statsapi.mlb.com)

Usage:
    poetry run python scripts/fetch_game_scores.py              # Fetch scores for recent games
    poetry run python scripts/fetch_game_scores.py --days 7     # Fetch scores for last 7 days
    poetry run python scripts/fetch_game_scores.py --game-id 75 # Fetch score for specific game
    poetry run python scripts/fetch_game_scores.py --backfill   # Recover all past games missing scores
"""
import argparse
import logging
import os
import sys
import time
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

# Pause between per-date score fetches during backfill
BACKFILL_SLEEP = 0.5



def _team_matches(source_name: str, db_name: str) -> bool:
    """Substring match in either direction, tolerating abbreviated source names."""
    if not source_name or not db_name:
        return False
    a, b = source_name.lower(), db_name.lower()
    return a in b or b in a


def _rank_candidates(scores_cache: list, game: Game, home_team: Team, away_team: Team) -> list:
    """Team-matching entries, nearest game date first.

    Caches span +/- 1 day, so a three-game series puts the same matchup in the cache
    several times. Taking the first hit would assign another night's score. When the
    two nearest candidates are equidistant and disagree, nothing is returned — a
    wrong outcome is worse than a missing one.
    """
    target = game.commence_time.date()
    candidates = [
        s for s in scores_cache
        if _team_matches(s.get("home_team", ""), home_team.name)
        and _team_matches(s.get("away_team", ""), away_team.name)
    ]
    if len(candidates) <= 1:
        return candidates

    def distance(score: dict) -> int:
        raw = score.get("game_date")
        if not raw:
            return 99
        try:
            return abs((datetime.strptime(raw, "%Y-%m-%d").date() - target).days)
        except ValueError:
            return 99

    candidates.sort(key=distance)
    if distance(candidates[0]) == distance(candidates[1]) and (
        candidates[0].get("home_score"),
        candidates[0].get("away_score"),
    ) != (candidates[1].get("home_score"), candidates[1].get("away_score")):
        logger.warning(
            f"  Ambiguous score for Game {game.id}: {len(candidates)} equidistant "
            f"candidates with different scores — skipping rather than guessing"
        )
        return []
    return candidates


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

        # Score caches span neighbouring days, so the same two teams can appear more
        # than once (a series). Rank candidates by how close the source's game date is
        # to this game's date and take the nearest; refuse to guess on a tie.
        for score in _rank_candidates(scores_cache, game, home_team, away_team):
            if not score.get("completed"):
                continue

            home_score = score.get("home_score")
            away_score = score.get("away_score")
            if home_score is None or away_score is None:
                continue

            home_score = int(home_score)
            away_score = int(away_score)
            winner = "home" if home_score > away_score else "away" if away_score > home_score else "push"

            existing = session.execute(
                select(BettingOutcome).where(BettingOutcome.game_id == game.id)
            ).scalar_one_or_none()

            if existing:
                existing.completed = True
                existing.home_score = home_score
                existing.away_score = away_score
                existing.total_points = home_score + away_score
                existing.point_differential = home_score - away_score
                existing.winner = winner
                logger.info(f"  Updated: {away_team.name} {away_score} @ {home_team.name} {home_score}")
            else:
                session.add(BettingOutcome(
                    game_id=game.id,
                    completed=True,
                    home_score=home_score,
                    away_score=away_score,
                    total_points=home_score + away_score,
                    point_differential=home_score - away_score,
                    winner=winner,
                ))
                logger.info(
                    f"  Stored: {away_team.name} {away_score} @ {home_team.name} {home_score} (Winner: {winner})"
                )

            game.completed = True
            session.commit()
            return True

        logger.warning(f"  No score found for Game {game.id}")
        return False

    except Exception as e:
        logger.error(f"Error fetching score for Game {game.id}: {e}")
        session.rollback()
        return False


def run_backfill(session, nba_client, mlb_client) -> None:
    """Recover scores for every past game still missing a BettingOutcome.

    The nightly job only looks at a fixed recent window, so any game missed at the
    time — API hiccup, machine asleep — stays unscored forever, which leaves every
    BestEVPick on it stuck 'pending'. This walks the whole backlog.

    Fetches one score cache per distinct game date rather than per day in the span:
    the backlog is sparse, so a naive range fetch would be mostly empty calls.
    """
    stmt = (
        select(Game, Sport)
        .join(Sport, Sport.id == Game.sport_id)
        .outerjoin(BettingOutcome, BettingOutcome.game_id == Game.id)
        .where(
            Game.commence_time < datetime.now(timezone.utc) - timedelta(days=1),
            BettingOutcome.id.is_(None),
        )
        .order_by(Game.commence_time)
    )
    rows = session.execute(stmt).all()

    if not rows:
        logger.info("No games missing scores — nothing to backfill")
        return

    # (sport_key, date) -> [Game]
    buckets: dict[tuple, list] = {}
    for game, sport in rows:
        buckets.setdefault((sport.key, game.commence_time.date()), []).append(game)

    logger.info(
        f"Backfill: {len(rows)} games missing scores across {len(buckets)} sport-days "
        f"({rows[0][0].commence_time.date()} to {rows[-1][0].commence_time.date()})"
    )

    success_count = 0
    no_data: dict[str, int] = {}     # source returned nothing for that date
    unmatched: dict[str, int] = {}   # source had data but the game did not match

    for i, ((sport_key, game_date), games) in enumerate(sorted(buckets.items(), key=lambda kv: kv[0][1]), 1):
        start = datetime.combine(game_date, datetime.min.time())
        client = mlb_client if sport_key == "baseball_mlb" else nba_client

        try:
            cache = client.get_scores_for_date_range(start, days=1)
        except Exception as e:
            logger.warning(f"[{i}/{len(buckets)}] {sport_key} {game_date}: fetch failed — {e}")
            no_data[sport_key] = no_data.get(sport_key, 0) + len(games)
            time.sleep(BACKFILL_SLEEP)
            continue

        if not cache:
            logger.info(f"[{i}/{len(buckets)}] {sport_key} {game_date}: no score data available")
            no_data[sport_key] = no_data.get(sport_key, 0) + len(games)
            time.sleep(BACKFILL_SLEEP)
            continue

        matched = 0
        for game in games:
            if fetch_and_store_score(session, game, cache):
                matched += 1
                success_count += 1
        if matched < len(games):
            unmatched[sport_key] = unmatched.get(sport_key, 0) + (len(games) - matched)

        logger.info(f"[{i}/{len(buckets)}] {sport_key} {game_date}: {matched}/{len(games)} recovered")
        time.sleep(BACKFILL_SLEEP)

    logger.info("=" * 70)
    logger.info(f"BACKFILL SUMMARY: recovered {success_count}/{len(rows)} games")
    for sk, n in sorted(no_data.items()):
        logger.info(f"  {sk}: {n} games on dates the score source returned no data for")
    for sk, n in sorted(unmatched.items()):
        logger.info(
            f"  {sk}: {n} games the source had data for but could not be uniquely "
            f"identified (no team-name match, or several equidistant candidates)"
        )
    logger.info("=" * 70)


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
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Recover every past game still missing a score, regardless of age"
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

        elif args.backfill:
            run_backfill(session, nba_client, mlb_client)

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
                logger.info(f"Fetching NBA scores from ESPN ({n} games)...")
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

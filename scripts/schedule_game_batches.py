#!/usr/bin/env python3
"""
Dynamic Game Batch Scheduler

Runs daily at 8am to schedule odds collection batches based on actual game times.
Creates a batch 30 minutes before each game starts to capture closing lines.
"""
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.models.database import Game

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_todays_games():
    """Get all games scheduled for today."""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Get today's date range (midnight to midnight local time)
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        # Query games for today
        stmt = select(Game).where(
            Game.commence_time >= today_start,
            Game.commence_time < today_end
        ).order_by(Game.commence_time)

        games = session.execute(stmt).scalars().all()

        logger.info(f"Found {len(games)} games scheduled for today")
        return games

    finally:
        session.close()

def schedule_batch(game_time: datetime, game_id: int):
    """
    Schedule a batch collection 30 minutes before game time using 'at' command.

    Args:
        game_time: When the game starts
        game_id: Database ID of the game
    """
    # Calculate batch time (30 minutes before game)
    batch_time = game_time - timedelta(minutes=30)

    # Convert to local time for 'at' command
    batch_time_local = batch_time.astimezone()
    game_time_local = game_time.astimezone()

    # Check if batch time is in the past
    now = datetime.now(timezone.utc)
    if batch_time < now:
        logger.warning(f"Game {game_id} batch time {batch_time_local} is in the past, skipping")
        return False

    # Format time for 'at' command (HH:MM)
    at_time = batch_time_local.strftime("%H:%M")

    # Path to collection script wrapper
    project_dir = Path(__file__).parent.parent
    run_script = project_dir / "scripts" / "run_collection.sh"

    # Create the command to run using the Poetry wrapper
    command = f"cd {project_dir} && {run_script}"

    try:
        # Schedule with 'at' command
        # Use 'at' to run the command at the specified time
        process = subprocess.Popen(
            ['at', at_time],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=command)

        if process.returncode == 0:
            logger.info(f"✓ Scheduled batch at {at_time} for game {game_id} (starts {game_time_local.strftime('%H:%M')})")
            return True
        else:
            logger.error(f"Failed to schedule batch: {stderr}")
            return False

    except Exception as e:
        logger.error(f"Error scheduling batch: {e}")
        return False

def main():
    """Main scheduler function."""
    logger.info("=" * 60)
    logger.info("Starting dynamic game batch scheduler")
    logger.info("=" * 60)

    # Get today's games
    games = get_todays_games()

    if not games:
        logger.info("No games scheduled for today - no batches to schedule")
        return

    # Group games by batch time to avoid duplicates
    # If multiple games start at same time, only schedule one batch
    batch_times = {}
    for game in games:
        batch_time = game.commence_time - timedelta(minutes=30)
        batch_time_str = batch_time.strftime("%Y-%m-%d %H:%M")

        if batch_time_str not in batch_times:
            batch_times[batch_time_str] = []
        batch_times[batch_time_str].append(game)

    logger.info(f"Found {len(games)} games resulting in {len(batch_times)} unique batch times")

    # Schedule one batch per unique time
    scheduled_count = 0
    for batch_time_str, games_at_time in batch_times.items():
        # Log all games for this batch time
        game_ids = [g.id for g in games_at_time]
        logger.info(f"Batch time {batch_time_str}: {len(games_at_time)} games (IDs: {game_ids})")

        # Schedule using the first game's time (they're all the same batch time)
        first_game = games_at_time[0]
        if schedule_batch(first_game.commence_time, first_game.id):
            scheduled_count += 1

    logger.info("=" * 60)
    logger.info(f"Scheduled {scheduled_count} batches for {len(games)} games")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

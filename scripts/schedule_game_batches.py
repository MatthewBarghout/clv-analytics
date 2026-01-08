#!/usr/bin/env python3
"""
Dynamic Game Batch Scheduler

Runs daily at 8am to schedule odds collection batches based on actual game times.
Creates a batch 30 minutes before each game starts to capture closing lines.
Uses launchd to schedule individual batches.
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
        # Get games in the next 24 hours (from now)
        now = datetime.now(timezone.utc)
        tomorrow = now + timedelta(hours=24)

        # Query games starting in the next 24 hours
        stmt = select(Game).where(
            Game.commence_time >= now,
            Game.commence_time < tomorrow
        ).order_by(Game.commence_time)

        games = session.execute(stmt).scalars().all()

        logger.info(f"Found {len(games)} games scheduled for today")
        return games

    finally:
        session.close()

def schedule_batch(game_time: datetime, game_id: int, batch_index: int):
    """
    Schedule a batch collection 30 minutes before game time using launchd.

    Args:
        game_time: When the game starts
        game_id: Database ID of the game
        batch_index: Unique index for this batch (to create unique plist name)
    """
    # Calculate batch time (30 minutes before game)
    batch_time = game_time - timedelta(minutes=30)

    # Convert to local time
    batch_time_local = batch_time.astimezone()
    game_time_local = game_time.astimezone()

    # Check if batch time is in the past
    now = datetime.now(timezone.utc)
    if batch_time < now:
        logger.warning(f"Game {game_id} batch time {batch_time_local} is in the past, skipping")
        return False

    # Path to collection script wrapper
    project_dir = Path(__file__).parent.parent
    run_script = project_dir / "scripts" / "run_collection.sh"

    # Create unique label for this batch
    label = f"com.clvanalytics.batch.{batch_time_local.strftime('%Y%m%d_%H%M')}.{batch_index}"

    # Plist file path
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"

    # Create plist content
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>cd {project_dir} && {run_script} --closing-only</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{project_dir}</string>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Year</key>
        <integer>{batch_time_local.year}</integer>
        <key>Month</key>
        <integer>{batch_time_local.month}</integer>
        <key>Day</key>
        <integer>{batch_time_local.day}</integer>
        <key>Hour</key>
        <integer>{batch_time_local.hour}</integer>
        <key>Minute</key>
        <integer>{batch_time_local.minute}</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>/tmp/clv-batch-{label}.log</string>

    <key>StandardErrorPath</key>
    <string>/tmp/clv-batch-{label}-error.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
"""

    try:
        # Write plist file
        plist_path.write_text(plist_content)
        logger.info(f"Created plist: {plist_path}")

        # Load the plist
        result = subprocess.run(
            ['launchctl', 'load', str(plist_path)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info(f"✓ Scheduled batch at {batch_time_local.strftime('%H:%M')} for game {game_id} (starts {game_time_local.strftime('%H:%M')})")
            return True
        else:
            logger.error(f"Failed to load plist: {result.stderr}")
            plist_path.unlink(missing_ok=True)
            return False

    except Exception as e:
        logger.error(f"Error scheduling batch: {e}")
        plist_path.unlink(missing_ok=True)
        return False

def cleanup_old_batches():
    """Remove old batch plists that have already run."""
    launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
    cleaned_count = 0

    # Find all batch plists
    for plist_file in launch_agents_dir.glob("com.clvanalytics.batch.*.plist"):
        try:
            # Unload the plist
            subprocess.run(
                ['launchctl', 'unload', str(plist_file)],
                capture_output=True,
                text=True,
                check=False  # Don't raise if already unloaded
            )
            # Remove the file
            plist_file.unlink()
            cleaned_count += 1
            logger.info(f"Cleaned up old batch: {plist_file.name}")
        except Exception as e:
            logger.warning(f"Failed to clean up {plist_file.name}: {e}")

    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} old batch plists")

def main():
    """Main scheduler function."""
    logger.info("=" * 60)
    logger.info("Starting dynamic game batch scheduler")
    logger.info("=" * 60)

    # Clean up old batch plists from previous runs
    cleanup_old_batches()

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

    # Schedule PRIMARY and BACKUP batch per unique time
    # Backup runs 5 minutes later as failsafe for network issues
    scheduled_count = 0
    for batch_index, (batch_time_str, games_at_time) in enumerate(batch_times.items()):
        # Log all games for this batch time
        game_ids = [g.id for g in games_at_time]
        logger.info(f"Batch time {batch_time_str}: {len(games_at_time)} games (IDs: {game_ids})")

        # Schedule PRIMARY batch (30 min before game)
        first_game = games_at_time[0]
        if schedule_batch(first_game.commence_time, first_game.id, batch_index * 2):
            scheduled_count += 1

        # Schedule BACKUP batch (25 min before game, 5 min after primary)
        # Offset commence_time by 5 minutes to create backup batch
        backup_commence_time = first_game.commence_time - timedelta(minutes=5)
        if schedule_batch(backup_commence_time, first_game.id, batch_index * 2 + 1):
            scheduled_count += 1
            logger.info(f"  ↳ Backup batch scheduled 5 minutes after primary")

    logger.info("=" * 60)
    logger.info(f"Scheduled {scheduled_count} batches ({scheduled_count // 2} primary + {scheduled_count // 2} backup) for {len(games)} games")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

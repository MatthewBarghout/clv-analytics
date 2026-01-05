#!/usr/bin/env python3
"""
Health check script to verify scheduler ran and batches are properly queued.

Run this after 8am to verify dynamic scheduling worked correctly.
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def check_scheduler_logs():
    """Check if scheduler ran successfully today."""
    logger.info("=" * 80)
    logger.info("CHECKING SCHEDULER LOGS")
    logger.info("=" * 80)

    log_path = Path("/tmp/clv-scheduler.log")
    error_log_path = Path("/tmp/clv-scheduler-error.log")

    # Check if scheduler ran
    if not log_path.exists():
        logger.error("❌ Scheduler log not found - scheduler may not have run!")
        return False

    # Check for errors
    if error_log_path.exists() and error_log_path.stat().st_size > 0:
        logger.warning("⚠️  Scheduler error log has content:")
        with open(error_log_path, 'r') as f:
            errors = f.read()
            # Only show last 500 chars to avoid spam
            logger.warning(errors[-500:])
        return False

    # Check last run time
    with open(log_path, 'r') as f:
        lines = f.readlines()
        if lines:
            last_line = lines[-1]
            logger.info(f"✓ Last scheduler log entry: {last_line.strip()}")

            # Check if it ran today
            today = datetime.now().strftime("%Y-%m-%d")
            if today in last_line:
                logger.info(f"✓ Scheduler ran today ({today})")
                return True
            else:
                logger.warning(f"⚠️  Scheduler last ran on a different day")
                return False

    logger.error("❌ Scheduler log is empty")
    return False


def check_queued_batches():
    """Check what batches are queued with 'at' command."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECKING QUEUED BATCHES")
    logger.info("=" * 80)

    try:
        # Run atq to see queued jobs
        result = subprocess.run(['atq'], capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"❌ Failed to run 'atq': {result.stderr}")
            return False

        output = result.stdout.strip()

        if not output:
            logger.warning("⚠️  No batches queued! Scheduler may have found no upcoming games.")
            return True  # Not necessarily an error if no games today

        # Parse and display queued jobs
        lines = output.split('\n')
        logger.info(f"✓ Found {len(lines)} queued batch(es):")
        for line in lines:
            logger.info(f"  {line}")

        return True

    except FileNotFoundError:
        logger.error("❌ 'at' command not found. Dynamic batching requires 'at' to be installed.")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking queued batches: {e}")
        return False


def check_todays_games():
    """Check what games are scheduled for today and their batch times."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECKING TODAY'S GAMES & BATCH SCHEDULE")
    logger.info("=" * 80)

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    try:
        # Get today's games
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        stmt = select(Game).where(
            Game.commence_time >= today_start,
            Game.commence_time < today_end
        ).order_by(Game.commence_time)

        games = session.execute(stmt).scalars().all()

        if not games:
            logger.info("ℹ️  No games scheduled for today")
            return True

        logger.info(f"✓ Found {len(games)} game(s) today:")

        # Group by batch time
        batch_groups = {}
        for game in games:
            batch_time = game.commence_time - timedelta(minutes=30)
            batch_time_str = batch_time.strftime("%Y-%m-%d %H:%M")

            if batch_time_str not in batch_groups:
                batch_groups[batch_time_str] = []
            batch_groups[batch_time_str].append(game)

        logger.info(f"\n✓ Games grouped into {len(batch_groups)} batch(es):")
        logger.info("  (Multiple games at same time = SINGLE batch = 1 API call)")

        for batch_time_str, games_in_batch in batch_groups.items():
            batch_dt = datetime.strptime(batch_time_str, "%Y-%m-%d %H:%M")
            batch_dt = batch_dt.replace(tzinfo=timezone.utc).astimezone()

            logger.info(f"\n  Batch at {batch_dt.strftime('%I:%M %p %Z')}:")
            for game in games_in_batch:
                game_time_local = game.commence_time.astimezone()
                logger.info(
                    f"    - {game.away_team.name} @ {game.home_team.name} "
                    f"(starts {game_time_local.strftime('%I:%M %p')})"
                )

        # Calculate API calls saved
        total_games = len(games)
        total_batches = len(batch_groups)
        calls_saved = total_games - total_batches

        logger.info(f"\n✓ API Call Optimization:")
        logger.info(f"  Total games: {total_games}")
        logger.info(f"  Batches needed: {total_batches}")
        logger.info(f"  API calls saved: {calls_saved} ({calls_saved/total_games*100:.0f}% reduction)")

        return True

    except Exception as e:
        logger.error(f"❌ Error checking games: {e}")
        return False
    finally:
        session.close()


def check_scheduler_service():
    """Check if scheduler service is loaded and enabled."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECKING LAUNCHD SERVICE")
    logger.info("=" * 80)

    try:
        result = subprocess.run(
            ['launchctl', 'list', 'com.clvanalytics.scheduler'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info("✓ Scheduler service is loaded")
            # Parse output
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
            return True
        else:
            logger.error("❌ Scheduler service NOT loaded!")
            logger.error("  Run: launchctl load ~/Library/LaunchAgents/com.clvanalytics.scheduler.plist")
            return False

    except Exception as e:
        logger.error(f"❌ Error checking service: {e}")
        return False


def main():
    """Run all health checks."""
    logger.info("🏥 SCHEDULER HEALTH CHECK")
    logger.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")

    checks = {
        "Scheduler Service Loaded": check_scheduler_service(),
        "Scheduler Logs": check_scheduler_logs(),
        "Queued Batches": check_queued_batches(),
        "Today's Games": check_todays_games(),
    }

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("HEALTH CHECK SUMMARY")
    logger.info("=" * 80)

    all_passed = True
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        logger.info(f"{status:8s} - {check_name}")
        if not passed:
            all_passed = False

    logger.info("=" * 80)

    if all_passed:
        logger.info("🎉 ALL CHECKS PASSED - System is healthy!")
        return 0
    else:
        logger.error("⚠️  SOME CHECKS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Update Daily Reports with Profit Statistics

Calculates profit/loss stats from OpportunityPerformance records and updates
the daily_clv_reports table with settlement results.

Usage:
    poetry run python scripts/update_report_profit_stats.py              # Update all reports
    poetry run python scripts/update_report_profit_stats.py --report-id 4  # Update specific report
"""
import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.models.database import DailyCLVReport, OpportunityPerformance

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def update_report_stats(session, report: DailyCLVReport) -> bool:
    """
    Update a daily report with profit/loss statistics.

    Args:
        session: SQLAlchemy session
        report: DailyCLVReport to update

    Returns:
        True if updated successfully
    """
    # Get all OpportunityPerformance records for this report
    opps = session.execute(
        select(OpportunityPerformance).where(
            OpportunityPerformance.report_id == report.id
        )
    ).scalars().all()

    if not opps:
        logger.info(f"Report {report.id} has no tracked opportunities")
        return False

    # Calculate stats
    settled = [o for o in opps if o.result != "pending"]
    wins = [o for o in settled if o.result == "win"]
    losses = [o for o in settled if o.result == "loss"]
    pushes = [o for o in settled if o.result == "push"]

    total_profit = sum(o.profit_loss or 0.0 for o in settled)
    total_wagered = len(settled) * 100.0  # $100 per bet

    # Update report
    report.settled_count = len(settled)
    report.win_count = len(wins)
    report.loss_count = len(losses)
    report.push_count = len(pushes)
    report.hypothetical_profit = total_profit if settled else None
    report.win_rate = (len(wins) / len(settled) * 100) if settled else None
    report.roi = (total_profit / total_wagered * 100) if total_wagered > 0 else None

    session.commit()

    logger.info(f"Updated Report {report.id}:")
    logger.info(f"  Settled: {report.settled_count}/{len(opps)}")
    if settled:
        logger.info(f"  Win Rate: {report.win_rate:.1f}% ({report.win_count}W-{report.loss_count}L)")
        logger.info(f"  Profit: ${report.hypothetical_profit:.2f}")
        logger.info(f"  ROI: {report.roi:.2f}%")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Update daily reports with profit statistics")
    parser.add_argument(
        "--report-id",
        type=int,
        help="Update specific report ID"
    )

    args = parser.parse_args()

    if not DATABASE_URL:
        logger.error("DATABASE_URL not found in environment")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("UPDATING DAILY REPORT PROFIT STATISTICS")
    logger.info("=" * 70)

    # Initialize
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        if args.report_id:
            # Update specific report
            report = session.get(DailyCLVReport, args.report_id)
            if not report:
                logger.error(f"Report {args.report_id} not found")
                sys.exit(1)
            update_report_stats(session, report)
        else:
            # Update all reports
            reports = session.execute(
                select(DailyCLVReport).order_by(DailyCLVReport.report_date.desc())
            ).scalars().all()

            logger.info(f"Found {len(reports)} reports to update")
            print()

            updated_count = 0
            for report in reports:
                if update_report_stats(session, report):
                    updated_count += 1
                print()

            logger.info("=" * 70)
            logger.info(f"Updated {updated_count}/{len(reports)} reports with profit stats")
            logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()

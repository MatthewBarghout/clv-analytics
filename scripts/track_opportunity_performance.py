#!/usr/bin/env python3
"""
Track Opportunity Performance

Creates OpportunityPerformance records from daily reports and settles them
when game outcomes are available.

Usage:
    poetry run python scripts/track_opportunity_performance.py              # Track all unsettled opportunities
    poetry run python scripts/track_opportunity_performance.py --report-id 4  # Track specific report
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.analyzers.bet_settlement import BetSettlement
from src.models.database import BettingOutcome, DailyCLVReport, Game, OpportunityPerformance, Team

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def create_opportunity_records(session, report: DailyCLVReport) -> int:
    """
    Create OpportunityPerformance records for opportunities in a daily report.

    Args:
        session: SQLAlchemy session
        report: DailyCLVReport to process

    Returns:
        Number of opportunities created
    """
    if not report.best_opportunities:
        logger.info(f"Report {report.id} has no opportunities to track")
        return 0

    created_count = 0

    for opp in report.best_opportunities:
        game_id = opp.get("game_id")

        # Check if already tracked
        existing = session.execute(
            select(OpportunityPerformance).where(
                OpportunityPerformance.report_id == report.id,
                OpportunityPerformance.game_id == game_id,
                OpportunityPerformance.bookmaker == opp.get("bookmaker"),
                OpportunityPerformance.market_type == opp.get("market_type"),
                OpportunityPerformance.outcome_name == opp.get("outcome")
            )
        ).scalar_one_or_none()

        if existing:
            continue

        # Create new opportunity performance record
        performance = OpportunityPerformance(
            report_id=report.id,
            game_id=game_id,
            bookmaker=opp.get("bookmaker"),
            market_type=opp.get("market_type"),
            outcome_name=opp.get("outcome"),
            entry_odds=opp.get("entry_odds"),
            closing_odds=opp.get("closing_odds"),
            clv_percentage=opp.get("clv"),
            bet_amount=100.0,  # Standard $100 unit
            result="pending"
        )

        session.add(performance)
        created_count += 1

    session.commit()
    logger.info(f"Created {created_count} opportunity tracking records for Report {report.id}")
    return created_count


def settle_moneyline_bet(
    outcome: BettingOutcome,
    performance: OpportunityPerformance,
    session
) -> bool:
    """
    Settle a moneyline bet.

    Returns True if settled successfully.
    """
    # Get team names to determine which side was bet
    game = session.get(Game, performance.game_id)
    home_team = session.get(Team, game.home_team_id)
    away_team = session.get(Team, game.away_team_id)

    # Determine if bet was on home or away team
    bet_team_name = performance.outcome_name

    bet_on_home = bet_team_name.lower() in home_team.name.lower() or home_team.name.lower() in bet_team_name.lower()
    bet_on_away = bet_team_name.lower() in away_team.name.lower() or away_team.name.lower() in bet_team_name.lower()

    if not (bet_on_home or bet_on_away):
        logger.warning(f"Could not match team name '{bet_team_name}' to home ({home_team.name}) or away ({away_team.name})")
        return False

    # Determine winner
    if outcome.winner == "push":
        result = "push"
    elif (bet_on_home and outcome.winner == "home") or (bet_on_away and outcome.winner == "away"):
        result = "win"
    else:
        result = "loss"

    # Calculate P/L
    profit_loss = BetSettlement.calculate_profit(
        performance.bet_amount,
        performance.entry_odds,
        result
    )

    # Update performance record
    performance.result = result
    performance.profit_loss = profit_loss
    performance.settled_at = datetime.now(timezone.utc)

    return True


def settle_opportunities(session, report_id: int = None) -> dict:
    """
    Settle pending opportunities that have completed games.

    Args:
        session: SQLAlchemy session
        report_id: Optional - settle only opportunities from specific report

    Returns:
        Dict with settlement stats
    """
    # Find pending opportunities with completed games
    stmt = (
        select(OpportunityPerformance, BettingOutcome)
        .join(BettingOutcome, BettingOutcome.game_id == OpportunityPerformance.game_id)
        .where(
            OpportunityPerformance.result == "pending",
            BettingOutcome.completed == True
        )
    )

    if report_id:
        stmt = stmt.where(OpportunityPerformance.report_id == report_id)

    results = session.execute(stmt).all()

    logger.info(f"Found {len(results)} opportunities to settle")

    stats = {
        "total": len(results),
        "settled": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "total_profit": 0.0
    }

    for performance, outcome in results:
        try:
            if performance.market_type == "h2h":
                # Settle moneyline bet
                if settle_moneyline_bet(outcome, performance, session):
                    stats["settled"] += 1

                    if performance.result == "win":
                        stats["wins"] += 1
                    elif performance.result == "loss":
                        stats["losses"] += 1
                    elif performance.result == "push":
                        stats["pushes"] += 1

                    stats["total_profit"] += performance.profit_loss or 0.0

            elif performance.market_type == "spreads":
                # Spreads require the actual line value - mark as needing enhancement
                logger.debug(f"Skipping spread bet - need line value to settle")

            elif performance.market_type == "totals":
                # Totals require the actual line value - mark as needing enhancement
                logger.debug(f"Skipping totals bet - need line value to settle")

        except Exception as e:
            logger.error(f"Error settling opportunity {performance.id}: {e}")
            continue

    session.commit()

    logger.info(f"Settlement complete: {stats['settled']} settled, {stats['wins']} wins, {stats['losses']} losses")
    logger.info(f"Total P/L: ${stats['total_profit']:.2f}")

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Track and settle betting opportunities")
    parser.add_argument(
        "--report-id",
        type=int,
        help="Process specific report ID"
    )
    parser.add_argument(
        "--settle-only",
        action="store_true",
        help="Only settle existing opportunities, don't create new ones"
    )

    args = parser.parse_args()

    if not DATABASE_URL:
        logger.error("DATABASE_URL not found in environment")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("OPPORTUNITY PERFORMANCE TRACKING")
    logger.info("=" * 70)

    # Initialize
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Create opportunity records
        if not args.settle_only:
            if args.report_id:
                report = session.get(DailyCLVReport, args.report_id)
                if not report:
                    logger.error(f"Report {args.report_id} not found")
                    sys.exit(1)
                create_opportunity_records(session, report)
            else:
                # Process all reports
                reports = session.execute(
                    select(DailyCLVReport).order_by(DailyCLVReport.report_date.desc())
                ).scalars().all()

                total_created = 0
                for report in reports:
                    total_created += create_opportunity_records(session, report)

                logger.info(f"Total opportunities created: {total_created}")

        # Settle opportunities
        stats = settle_opportunities(session, report_id=args.report_id)

        # Summary
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Opportunities settled: {stats['settled']}")
        logger.info(f"Win rate: {stats['wins']}/{stats['settled']} ({stats['wins']/stats['settled']*100:.1f}%)" if stats['settled'] > 0 else "Win rate: N/A")
        logger.info(f"Total profit/loss: ${stats['total_profit']:.2f}")

        if stats['settled'] > 0:
            roi = (stats['total_profit'] / (stats['settled'] * 100)) * 100
            logger.info(f"ROI: {roi:.2f}%")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()

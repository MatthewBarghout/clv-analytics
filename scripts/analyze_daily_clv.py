#!/usr/bin/env python3
"""
Daily Post-Game CLV Analysis

Analyzes completed games to calculate CLV for all betting opportunities.
Generates reports showing which bets would have had the best closing line value.

Usage:
    poetry run python scripts/analyze_daily_clv.py                # Analyze yesterday's games
    poetry run python scripts/analyze_daily_clv.py --date 2026-01-05  # Analyze specific date
    poetry run python scripts/analyze_daily_clv.py --days 7       # Analyze last 7 days
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.analyzers.clv_calculator import CLVCalculator
from src.models.database import (
    Bookmaker,
    ClosingLine,
    DailyCLVReport,
    Game,
    OddsSnapshot,
    Team,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


class DailyCLVAnalyzer:
    """Analyzes CLV for completed games on a given date."""

    def __init__(self, session):
        self.session = session
        self.calc = CLVCalculator()

    def get_completed_games(self, start_date: datetime, end_date: datetime) -> List[Game]:
        """Get all games that started between start_date and end_date."""
        stmt = (
            select(Game)
            .where(
                Game.commence_time >= start_date,
                Game.commence_time < end_date
            )
            .order_by(Game.commence_time)
        )

        games = self.session.execute(stmt).scalars().all()
        logger.info(f"Found {len(games)} games between {start_date.date()} and {end_date.date()}")
        return games

    def analyze_game(self, game: Game) -> Dict:
        """Analyze CLV opportunities for a single game."""
        # Get all snapshots with closing lines for this game
        stmt = (
            select(OddsSnapshot, ClosingLine, Bookmaker)
            .join(
                ClosingLine,
                (ClosingLine.game_id == OddsSnapshot.game_id)
                & (ClosingLine.bookmaker_id == OddsSnapshot.bookmaker_id)
                & (ClosingLine.market_type == OddsSnapshot.market_type),
            )
            .join(Bookmaker, Bookmaker.id == OddsSnapshot.bookmaker_id)
            .where(OddsSnapshot.game_id == game.id)
        )

        results = self.session.execute(stmt).all()

        if not results:
            logger.debug(f"Game {game.id} has no snapshots with closing lines")
            return None

        opportunities = []

        for snapshot, closing_line, bookmaker in results:
            # Calculate CLV for each outcome
            for outcome in snapshot.outcomes:
                team_name = outcome.get("name")
                entry_odds = self.calc._extract_odds_for_team(snapshot.outcomes, team_name)
                closing_odds = self.calc._extract_odds_for_team(closing_line.outcomes, team_name)

                if entry_odds and closing_odds:
                    clv = self.calc.calculate_clv(entry_odds, closing_odds)

                    opportunities.append({
                        "game_id": game.id,
                        "bookmaker": bookmaker.name,
                        "market_type": snapshot.market_type,
                        "outcome": team_name,
                        "entry_odds": entry_odds,
                        "closing_odds": closing_odds,
                        "clv": clv,
                        "timestamp": snapshot.timestamp.isoformat(),
                    })

        # Get team names
        home_team = self.session.get(Team, game.home_team_id)
        away_team = self.session.get(Team, game.away_team_id)

        return {
            "game_id": game.id,
            "home_team": home_team.name if home_team else "Unknown",
            "away_team": away_team.name if away_team else "Unknown",
            "commence_time": game.commence_time,
            "opportunities": opportunities,
        }

    def analyze_date(self, report_date: datetime) -> DailyCLVReport:
        """Analyze all games for a specific date and create a report."""
        # Define date range (games that commenced on this day)
        start_date = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)

        logger.info("=" * 80)
        logger.info(f"ANALYZING CLV FOR {start_date.date()}")
        logger.info("=" * 80)

        # Get completed games
        games = self.get_completed_games(start_date, end_date)

        if not games:
            logger.info("No games found for this date")
            return None

        # Analyze each game
        all_opportunities = []
        game_summaries = []

        for game in games:
            game_analysis = self.analyze_game(game)

            if game_analysis and game_analysis["opportunities"]:
                all_opportunities.extend(game_analysis["opportunities"])

                # Create game summary
                game_clv_values = [opp["clv"] for opp in game_analysis["opportunities"]]
                game_summaries.append({
                    "game_id": game_analysis["game_id"],
                    "home_team": game_analysis["home_team"],
                    "away_team": game_analysis["away_team"],
                    "commence_time": game_analysis["commence_time"].isoformat(),
                    "opportunities_count": len(game_analysis["opportunities"]),
                    "avg_clv": sum(game_clv_values) / len(game_clv_values),
                    "max_clv": max(game_clv_values),
                    "min_clv": min(game_clv_values),
                })

        if not all_opportunities:
            logger.warning("No CLV opportunities found (no closing lines available)")
            return None

        # Calculate overall statistics
        clv_values = [opp["clv"] for opp in all_opportunities]
        total = len(clv_values)
        positive_count = sum(1 for clv in clv_values if clv > 0)

        # Aggregate by bookmaker
        by_bookmaker = {}
        for opp in all_opportunities:
            bookmaker = opp["bookmaker"]
            if bookmaker not in by_bookmaker:
                by_bookmaker[bookmaker] = []
            by_bookmaker[bookmaker].append(opp["clv"])

        by_bookmaker_stats = {
            name: {
                "avg_clv": sum(values) / len(values),
                "count": len(values),
                "positive_count": sum(1 for v in values if v > 0),
            }
            for name, values in by_bookmaker.items()
        }

        # Aggregate by market type
        by_market = {}
        for opp in all_opportunities:
            market = opp["market_type"]
            if market not in by_market:
                by_market[market] = []
            by_market[market].append(opp["clv"])

        by_market_stats = {
            market: {
                "avg_clv": sum(values) / len(values),
                "count": len(values),
                "positive_count": sum(1 for v in values if v > 0),
            }
            for market, values in by_market.items()
        }

        # Get top 10 best opportunities
        best_opportunities = sorted(all_opportunities, key=lambda x: x["clv"], reverse=True)[:10]

        # Create report
        report = DailyCLVReport(
            report_date=start_date,
            games_analyzed=len(games),
            total_opportunities=total,
            avg_clv=sum(clv_values) / total,
            median_clv=sorted(clv_values)[total // 2],
            positive_clv_count=positive_count,
            positive_clv_percentage=(positive_count / total * 100),
            best_opportunities=best_opportunities,
            by_bookmaker=by_bookmaker_stats,
            by_market=by_market_stats,
            game_summaries=game_summaries,
        )

        return report

    def save_report(self, report: DailyCLVReport) -> bool:
        """Save report to database."""
        try:
            # Check if report already exists for this date
            existing = self.session.execute(
                select(DailyCLVReport).where(
                    DailyCLVReport.report_date == report.report_date
                )
            ).scalar_one_or_none()

            if existing:
                logger.info(f"Updating existing report for {report.report_date.date()}")
                # Update existing report
                existing.games_analyzed = report.games_analyzed
                existing.total_opportunities = report.total_opportunities
                existing.avg_clv = report.avg_clv
                existing.median_clv = report.median_clv
                existing.positive_clv_count = report.positive_clv_count
                existing.positive_clv_percentage = report.positive_clv_percentage
                existing.best_opportunities = report.best_opportunities
                existing.by_bookmaker = report.by_bookmaker
                existing.by_market = report.by_market
                existing.game_summaries = report.game_summaries
            else:
                logger.info(f"Creating new report for {report.report_date.date()}")
                self.session.add(report)

            self.session.commit()
            return True

        except Exception as e:
            logger.error(f"Error saving report: {e}")
            self.session.rollback()
            return False

    def print_report(self, report: DailyCLVReport):
        """Print a formatted text report."""
        logger.info("\n" + "=" * 80)
        logger.info("DAILY CLV ANALYSIS REPORT")
        logger.info(f"Date: {report.report_date.date()}")
        logger.info("=" * 80)

        logger.info(f"\nOVERALL STATISTICS:")
        logger.info(f"  Games Analyzed: {report.games_analyzed}")
        logger.info(f"  Total Opportunities: {report.total_opportunities}")
        logger.info(f"  Average CLV: {report.avg_clv:.2f}%")
        logger.info(f"  Median CLV: {report.median_clv:.2f}%")
        logger.info(f"  Positive CLV: {report.positive_clv_count} ({report.positive_clv_percentage:.1f}%)")

        logger.info(f"\nBY BOOKMAKER:")
        for bookmaker, stats in sorted(
            report.by_bookmaker.items(),
            key=lambda x: x[1]["avg_clv"],
            reverse=True
        ):
            logger.info(f"  {bookmaker:15s} - Avg CLV: {stats['avg_clv']:6.2f}% | Count: {stats['count']:3d} | Positive: {stats['positive_count']:3d}")

        logger.info(f"\nBY MARKET TYPE:")
        for market, stats in sorted(
            report.by_market.items(),
            key=lambda x: x[1]["avg_clv"],
            reverse=True
        ):
            logger.info(f"  {market:10s} - Avg CLV: {stats['avg_clv']:6.2f}% | Count: {stats['count']:3d} | Positive: {stats['positive_count']:3d}")

        logger.info(f"\nTOP 10 BEST OPPORTUNITIES:")
        for i, opp in enumerate(report.best_opportunities[:10], 1):
            # Find game summary for this opportunity
            game_summary = next(
                (g for g in report.game_summaries if g["game_id"] == opp["game_id"]),
                None
            )

            if game_summary:
                matchup = f"{game_summary['away_team']} @ {game_summary['home_team']}"
            else:
                matchup = f"Game {opp['game_id']}"

            logger.info(
                f"  {i:2d}. CLV: {opp['clv']:6.2f}% | {opp['bookmaker']:12s} | "
                f"{opp['market_type']:8s} | {opp['outcome']:20s} | {matchup}"
            )

        logger.info("\n" + "=" * 80)


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Analyze daily CLV for completed games")
    parser.add_argument(
        "--date",
        type=str,
        help="Specific date to analyze (YYYY-MM-DD). Defaults to yesterday."
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Number of days to analyze (going backwards from yesterday)"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Don't print report to console (only save to database)"
    )

    args = parser.parse_args()

    # Setup database
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    analyzer = DailyCLVAnalyzer(session)

    try:
        # Determine dates to analyze
        if args.days:
            # Analyze multiple days
            dates_to_analyze = []
            for i in range(args.days):
                date = datetime.now(timezone.utc) - timedelta(days=i+1)
                dates_to_analyze.append(date.replace(hour=0, minute=0, second=0, microsecond=0))
        elif args.date:
            # Analyze specific date
            date = datetime.strptime(args.date, "%Y-%m-%d")
            dates_to_analyze = [date.replace(tzinfo=timezone.utc)]
        else:
            # Default: analyze yesterday
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            dates_to_analyze = [yesterday.replace(hour=0, minute=0, second=0, microsecond=0)]

        # Analyze each date
        reports_created = 0
        for report_date in dates_to_analyze:
            report = analyzer.analyze_date(report_date)

            if report:
                if not args.silent:
                    analyzer.print_report(report)

                if analyzer.save_report(report):
                    reports_created += 1
                    logger.info(f"✓ Report saved for {report_date.date()}")
            else:
                logger.warning(f"No report generated for {report_date.date()}")

        logger.info(f"\n{'=' * 80}")
        logger.info(f"SUMMARY: Created/updated {reports_created} report(s)")
        logger.info(f"{'=' * 80}")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()

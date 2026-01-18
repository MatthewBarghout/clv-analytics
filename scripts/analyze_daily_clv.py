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
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.analyzers.clv_calculator import CLVCalculator
from src.analyzers.features import FeatureEngineer
from src.analyzers.movement_predictor import LineMovementPredictor
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
        self.model = None
        self.engineer = None

        # Try to load ML model if available
        model_path = "models/line_movement_predictor.pkl"
        if Path(model_path).exists():
            try:
                self.model = LineMovementPredictor()
                self.model.load_model(model_path)
                self.engineer = FeatureEngineer()
                logger.info("ML model loaded successfully for EV opportunities")
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}")
                self.model = None
                self.engineer = None
        else:
            logger.info("ML model not found - EV opportunities will not be included")

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
                point_line = outcome.get("point")  # Spread or total line (e.g., -5.5, 220.5)
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
                        "point_line": point_line,  # For spreads/totals settlement
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

    def get_ev_opportunities(self, limit: int = 20, min_confidence: float = 0.5) -> List[Dict]:
        """
        Get top +EV betting opportunities from upcoming games using ML predictions.

        Args:
            limit: Maximum number of opportunities to return
            min_confidence: Minimum confidence threshold for predictions

        Returns:
            List of EV opportunity dictionaries
        """
        if not self.model or not self.engineer:
            logger.info("ML model not available - skipping EV opportunities")
            return []

        try:
            # Get upcoming games (within next 7 days)
            now = datetime.now(timezone.utc)
            future_cutoff = now + timedelta(days=7)

            stmt = (
                select(OddsSnapshot, Game, Bookmaker)
                .join(Game, Game.id == OddsSnapshot.game_id)
                .join(Bookmaker, Bookmaker.id == OddsSnapshot.bookmaker_id)
                .where(
                    Game.commence_time > now,
                    Game.commence_time < future_cutoff
                )
                .order_by(Game.commence_time)
            )

            results = self.session.execute(stmt).all()

            if not results:
                logger.info("No upcoming games found for EV opportunities")
                return []

            opportunities = []

            for snapshot, game, bookmaker in results:
                hours_to_game = (game.commence_time - snapshot.timestamp).total_seconds() / 3600
                day_of_week = game.commence_time.weekday()
                is_weekend = day_of_week >= 5

                # Get team names
                home_team = self.session.get(Team, game.home_team_id)
                away_team = self.session.get(Team, game.away_team_id)

                for outcome in snapshot.outcomes:
                    outcome_name = outcome.get("name")
                    opening_price = outcome.get("price")
                    opening_point = outcome.get("point", 0.0)

                    if not outcome_name or opening_price is None:
                        continue

                    # Calculate consensus
                    consensus_line = self.engineer.calculate_consensus_line(
                        self.session, game.id, snapshot.market_type, outcome_name
                    )
                    line_spread = self.engineer.calculate_line_spread(
                        self.session, game.id, snapshot.market_type, outcome_name
                    )

                    # Create features
                    features = pd.DataFrame([{
                        "bookmaker_id": snapshot.bookmaker_id,
                        "market_type": snapshot.market_type,
                        "hours_to_game": hours_to_game,
                        "day_of_week": day_of_week,
                        "is_weekend": is_weekend,
                        "outcome_name": outcome_name,
                        "opening_price": float(opening_price),
                        "opening_point": float(opening_point) if opening_point is not None else 0.0,
                        "consensus_line": consensus_line if consensus_line is not None else float(opening_price),
                        "line_spread": line_spread if line_spread is not None else 0.0,
                        "distance_from_consensus": (
                            float(opening_price) - consensus_line
                            if consensus_line is not None
                            else 0.0
                        ),
                        "is_outlier": (
                            abs(float(opening_price) - consensus_line) > 0.05
                            if consensus_line is not None
                            else False
                        ),
                    }])

                    # Predict movement
                    movement_pred = self.model.predict_movement(features)

                    predicted_delta = float(movement_pred["predicted_delta"][0])
                    confidence = float(movement_pred["confidence"][0])
                    direction = movement_pred["predicted_direction"][0]

                    # Only include high-confidence predictions of unfavorable movement
                    if confidence < min_confidence:
                        continue

                    # Calculate EV score
                    if (direction == "DOWN" and predicted_delta < -0.01) or (direction == "UP" and predicted_delta > 0.01):
                        ev_score = abs(predicted_delta) * confidence * 100
                    else:
                        continue

                    # Format current line
                    if opening_point != 0:
                        current_line = f"{opening_point:+.1f} at {self._decimal_to_american(opening_price)}"
                    else:
                        current_line = self._decimal_to_american(opening_price)

                    opportunities.append({
                        "game_id": game.id,
                        "home_team": home_team.name if home_team else "Unknown",
                        "away_team": away_team.name if away_team else "Unknown",
                        "commence_time": game.commence_time.isoformat(),
                        "bookmaker": bookmaker.name,
                        "market_type": snapshot.market_type,
                        "outcome": outcome_name,
                        "current_line": current_line,
                        "predicted_movement": predicted_delta,
                        "predicted_direction": direction,
                        "confidence": confidence,
                        "ev_score": ev_score,
                    })

            # Sort by EV score
            opportunities.sort(key=lambda x: x["ev_score"], reverse=True)

            logger.info(f"Found {len(opportunities)} +EV opportunities")
            return opportunities[:limit]

        except Exception as e:
            logger.error(f"Error getting EV opportunities: {e}")
            return []

    def _decimal_to_american(self, decimal_odds: float) -> str:
        """Convert decimal odds to American format string."""
        if decimal_odds >= 2.0:
            return f"+{int((decimal_odds - 1) * 100)}"
        else:
            return f"{int(-100 / (decimal_odds - 1))}"

    def analyze_date(self, report_date: datetime) -> DailyCLVReport:
        """Analyze all games for a specific date and create a report."""
        # Define date range using EST (games that commenced on this day in Eastern time)
        # NBA games are primarily in EST/PST, so use EST as the canonical "game date"
        eastern = ZoneInfo("America/New_York")

        # Get midnight EST for this date
        start_date_local = datetime.combine(report_date.date(), datetime.min.time())
        start_date_local = start_date_local.replace(tzinfo=eastern)

        # Convert to UTC for database query
        start_date = start_date_local.astimezone(timezone.utc)
        end_date = (start_date_local + timedelta(days=1)).astimezone(timezone.utc)

        logger.info("=" * 80)
        logger.info(f"ANALYZING CLV FOR {start_date_local.date()} (Eastern Time)")
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

        # Get top 10 best CLV opportunities
        best_opportunities = sorted(all_opportunities, key=lambda x: x["clv"], reverse=True)[:10]

        # Get top EV opportunities for upcoming games
        ev_opportunities = self.get_ev_opportunities(limit=20, min_confidence=0.5)

        # Create report
        # Store report_date as midnight EST converted to UTC for consistency
        report = DailyCLVReport(
            report_date=start_date_local.replace(hour=0, minute=0, second=0, microsecond=0),
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
            ev_opportunities=ev_opportunities if ev_opportunities else None,
        )

        return report

    def save_report(self, report: DailyCLVReport) -> DailyCLVReport:
        """
        Save report to database and return refreshed copy.

        Returns:
            The report object refreshed from database with all fields (including ROI if calculated)
        """
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
                existing.ev_opportunities = report.ev_opportunities
                report_to_return = existing
            else:
                logger.info(f"Creating new report for {report.report_date.date()}")
                self.session.add(report)
                report_to_return = report

            self.session.commit()

            # Refresh to get ROI fields that may have been calculated by other scripts
            self.session.refresh(report_to_return)

            logger.info(f"✓ Report saved for {report.report_date.date()}")
            return report_to_return

        except Exception as e:
            logger.error(f"Error saving report: {e}")
            self.session.rollback()
            return report

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

        # Print EV opportunities if available
        if report.ev_opportunities and len(report.ev_opportunities) > 0:
            logger.info(f"\nTOP {min(10, len(report.ev_opportunities))} +EV OPPORTUNITIES (Upcoming Games):")
            for i, opp in enumerate(report.ev_opportunities[:10], 1):
                matchup = f"{opp['away_team']} @ {opp['home_team']}"
                logger.info(
                    f"  {i:2d}. EV: {opp['ev_score']:5.2f} | {opp['bookmaker']:12s} | "
                    f"{opp['market_type']:8s} | {opp['outcome']:20s} | {opp['current_line']:12s} | "
                    f"Pred: {opp['predicted_direction']:4s} {abs(opp['predicted_movement']):.3f} "
                    f"(Conf: {opp['confidence']:.1%}) | {matchup}"
                )

        # Print ROI tracking stats if available
        if report.settled_count and report.settled_count > 0:
            logger.info(f"\nBETTING PERFORMANCE (Tracked Opportunities):")
            logger.info(f"  Tracked: {report.settled_count} bets settled")
            logger.info(f"  Record: {report.win_count}W - {report.loss_count}L - {report.push_count}P")
            logger.info(f"  Win Rate: {report.win_rate:.1f}%")
            logger.info(f"  Total Profit: ${report.hypothetical_profit:.2f}")
            logger.info(f"  ROI: {report.roi:.2f}%")

            # Show detailed bet breakdown
            logger.info(f"\n  TRACKED BET DETAILS:")
            from src.models.database import OpportunityPerformance

            # Get all tracked opportunities for this report
            tracked_opps = self.session.execute(
                select(OpportunityPerformance)
                .where(OpportunityPerformance.report_id == report.id)
                .order_by(OpportunityPerformance.clv_percentage.desc())
            ).scalars().all()

            for i, opp in enumerate(tracked_opps, 1):
                # Format result indicator
                if opp.result == "win":
                    result_icon = "✓"
                    result_color = "WIN"
                elif opp.result == "loss":
                    result_icon = "✗"
                    result_color = "LOSS"
                elif opp.result == "push":
                    result_icon = "="
                    result_color = "PUSH"
                else:
                    result_icon = "?"
                    result_color = "PENDING"

                # Get game info for matchup
                game = self.session.get(Game, opp.game_id)
                if game:
                    home_team = self.session.get(Team, game.home_team_id)
                    away_team = self.session.get(Team, game.away_team_id)
                    matchup = f"{away_team.name} @ {home_team.name}" if home_team and away_team else f"Game {opp.game_id}"
                else:
                    matchup = f"Game {opp.game_id}"

                logger.info(
                    f"  {i:2d}. {result_icon} CLV: {opp.clv_percentage:6.2f}% | {opp.bookmaker:12s} | "
                    f"{opp.market_type:8s} | {opp.outcome_name:20s} | "
                    f"{result_color:7s} ${opp.profit_loss:+7.2f} | {matchup}"
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
                # Save report first (returns refreshed copy with ROI if available)
                saved_report = analyzer.save_report(report)

                if saved_report:
                    reports_created += 1

                    # Print the refreshed report (includes ROI if calculated)
                    if not args.silent:
                        analyzer.print_report(saved_report)
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

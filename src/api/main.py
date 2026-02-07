"""FastAPI backend for CLV analytics dashboard."""
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from src.analyzers.clv_calculator import CLVCalculator
from src.analyzers.bet_sizing import (
    BetContext,
    BetSizingStrategy,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    get_bet_sizer,
)
from src.collectors.odds_api_client import OddsAPIClient
from src.api.schemas import (
    BookmakerStats,
    CLVHistoryPoint,
    CLVStats,
    ClosingLineResponse,
    DailyCLVReportResponse,
    GameWithCLV,
    HealthResponse,
    OddsSnapshotResponse,
)
from src.api.ml_endpoints import router as ml_router
from src.models.database import BettingOutcome, Bookmaker, ClosingLine, DailyCLVReport, Game, OddsSnapshot, OpportunityPerformance, Sport, Team, UserBet

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CLV Analytics API",
    description="API for Closing Line Value analytics",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ml_router)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", timestamp=datetime.now(timezone.utc))


@app.get("/api/stats", response_model=CLVStats)
async def get_clv_stats():
    """Get overall CLV statistics."""
    db = get_db()
    calc = CLVCalculator()

    try:
        # Get all snapshots with closing lines
        stmt = (
            select(OddsSnapshot, ClosingLine, Bookmaker)
            .join(
                ClosingLine,
                (ClosingLine.game_id == OddsSnapshot.game_id)
                & (ClosingLine.bookmaker_id == OddsSnapshot.bookmaker_id)
                & (ClosingLine.market_type == OddsSnapshot.market_type),
            )
            .join(Bookmaker, Bookmaker.id == OddsSnapshot.bookmaker_id)
        )

        results = db.execute(stmt).all()

        clv_values = []
        by_bookmaker = {}
        by_market_type = {}

        for snapshot, closing_line, bookmaker in results:
            # Calculate CLV for each outcome
            for outcome in snapshot.outcomes:
                team_name = outcome.get("name")
                entry_odds = calc._extract_odds_for_team(snapshot.outcomes, team_name)
                closing_odds = calc._extract_odds_for_team(
                    closing_line.outcomes, team_name
                )

                if entry_odds and closing_odds:
                    clv = calc.calculate_clv(entry_odds, closing_odds)
                    clv_values.append(clv)

                    # Track by bookmaker
                    if bookmaker.name not in by_bookmaker:
                        by_bookmaker[bookmaker.name] = []
                    by_bookmaker[bookmaker.name].append(clv)

                    # Track by market type
                    if snapshot.market_type not in by_market_type:
                        by_market_type[snapshot.market_type] = []
                    by_market_type[snapshot.market_type].append(clv)

        # Calculate statistics
        total = len(clv_values)
        positive_count = sum(1 for clv in clv_values if clv > 0)

        # Aggregate by bookmaker
        bookmaker_stats = {
            name: {
                "avg_clv": sum(values) / len(values) if values else 0,
                "count": len(values),
            }
            for name, values in by_bookmaker.items()
        }

        # Aggregate by market type
        market_stats = {
            market: {
                "avg_clv": sum(values) / len(values) if values else 0,
                "count": len(values),
            }
            for market, values in by_market_type.items()
        }

        return CLVStats(
            mean_clv=sum(clv_values) / total if total > 0 else None,
            median_clv=(
                sorted(clv_values)[total // 2] if total > 0 else None
            ),
            total_analyzed=total,
            positive_clv_count=positive_count,
            positive_clv_percentage=(positive_count / total * 100) if total > 0 else 0,
            by_bookmaker=bookmaker_stats,
            by_market_type=market_stats,
        )

    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/games", response_model=List[GameWithCLV])
async def get_games(limit: int = 50):
    """Get list of games with CLV data."""
    db = get_db()
    calc = CLVCalculator()

    try:
        # Get games with snapshot and closing line counts
        stmt = (
            select(
                Game,
                Team.name.label("home_team_name"),
                func.count(OddsSnapshot.id.distinct()).label("snapshots_count"),
                func.count(ClosingLine.id.distinct()).label("closing_lines_count"),
            )
            .join(Team, Team.id == Game.home_team_id)
            .outerjoin(OddsSnapshot, OddsSnapshot.game_id == Game.id)
            .outerjoin(
                ClosingLine,
                (ClosingLine.game_id == Game.id),
            )
            .group_by(Game.id, Team.name)
            .order_by(Game.commence_time.desc())
            .limit(limit)
        )

        results = db.execute(stmt).all()

        games_with_clv = []
        for game, home_team_name, snapshots_count, closing_lines_count in results:
            # Get away team name
            away_team = db.query(Team).filter(Team.id == game.away_team_id).first()

            # Get betting outcome (score) if available
            betting_outcome = db.query(BettingOutcome).filter(BettingOutcome.game_id == game.id).first()

            # Calculate CLV for this game
            avg_game_clv = None
            if closing_lines_count > 0:
                # Get all snapshots with closing lines for this game
                clv_stmt = (
                    select(OddsSnapshot, ClosingLine)
                    .join(
                        ClosingLine,
                        (ClosingLine.game_id == OddsSnapshot.game_id)
                        & (ClosingLine.bookmaker_id == OddsSnapshot.bookmaker_id)
                        & (ClosingLine.market_type == OddsSnapshot.market_type),
                    )
                    .where(OddsSnapshot.game_id == game.id)
                )

                clv_results = db.execute(clv_stmt).all()
                game_clv_values = []

                for snapshot, closing_line in clv_results:
                    for outcome in snapshot.outcomes:
                        team_name = outcome.get("name")
                        entry_odds = calc._extract_odds_for_team(snapshot.outcomes, team_name)
                        closing_odds = calc._extract_odds_for_team(
                            closing_line.outcomes, team_name
                        )

                        if entry_odds and closing_odds:
                            clv = calc.calculate_clv(entry_odds, closing_odds)
                            game_clv_values.append(clv)

                avg_game_clv = (
                    sum(game_clv_values) / len(game_clv_values)
                    if game_clv_values
                    else None
                )

            # Determine if game is completed based on commence time
            now = datetime.now(timezone.utc)
            is_completed = game.commence_time < now

            games_with_clv.append(
                GameWithCLV(
                    game_id=game.id,
                    home_team=home_team_name,
                    away_team=away_team.name if away_team else "Unknown",
                    commence_time=game.commence_time,
                    completed=is_completed,
                    snapshots_count=snapshots_count,
                    closing_lines_count=closing_lines_count,
                    avg_clv=avg_game_clv,
                    home_score=betting_outcome.home_score if betting_outcome else None,
                    away_score=betting_outcome.away_score if betting_outcome else None,
                    winner=betting_outcome.winner if betting_outcome else None,
                )
            )

        return games_with_clv

    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/clv-history", response_model=List[CLVHistoryPoint])
async def get_clv_history(time_range: str = "30d"):
    """Get CLV trend over time.

    Args:
        time_range: Time range filter - "7d", "30d", "90d", or "all"
    """
    db = get_db()
    calc = CLVCalculator()

    try:
        # Map time_range to days
        range_map = {
            "7d": 7,
            "30d": 30,
            "90d": 90,
            "all": None,
        }

        days_limit = range_map.get(time_range, 30)

        # Build base query
        stmt = (
            select(
                func.date(OddsSnapshot.timestamp).label("date"),
                OddsSnapshot,
                ClosingLine,
            )
            .join(
                ClosingLine,
                (ClosingLine.game_id == OddsSnapshot.game_id)
                & (ClosingLine.bookmaker_id == OddsSnapshot.bookmaker_id)
                & (ClosingLine.market_type == OddsSnapshot.market_type),
            )
        )

        # Add date filter if not "all"
        if days_limit:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_limit)
            stmt = stmt.where(OddsSnapshot.timestamp >= cutoff_date)

        stmt = stmt.order_by(func.date(OddsSnapshot.timestamp).desc()).limit(1000)

        results = db.execute(stmt).all()

        # Group by date
        by_date = {}
        for date, snapshot, closing_line in results:
            date_str = str(date)
            if date_str not in by_date:
                by_date[date_str] = []

            # Calculate CLV for each outcome
            for outcome in snapshot.outcomes:
                team_name = outcome.get("name")
                entry_odds = calc._extract_odds_for_team(snapshot.outcomes, team_name)
                closing_odds = calc._extract_odds_for_team(
                    closing_line.outcomes, team_name
                )

                if entry_odds and closing_odds:
                    clv = calc.calculate_clv(entry_odds, closing_odds)
                    by_date[date_str].append(clv)

        # Create history points
        history = []
        for date_str in sorted(by_date.keys()):
            clv_values = by_date[date_str]
            history.append(
                CLVHistoryPoint(
                    date=date_str,
                    avg_clv=sum(clv_values) / len(clv_values) if clv_values else None,
                    count=len(clv_values),
                )
            )

        return sorted(history, key=lambda x: x.date)

    except Exception as e:
        logger.error(f"Error fetching CLV history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/bookmakers", response_model=List[BookmakerStats])
async def get_bookmaker_stats():
    """Get statistics by bookmaker."""
    db = get_db()
    calc = CLVCalculator()

    try:
        # Get all bookmakers
        bookmakers = db.query(Bookmaker).all()
        stats_list = []

        for bookmaker in bookmakers:
            # Get snapshots with closing lines for this bookmaker
            stmt = (
                select(OddsSnapshot, ClosingLine)
                .join(
                    ClosingLine,
                    (ClosingLine.game_id == OddsSnapshot.game_id)
                    & (ClosingLine.bookmaker_id == OddsSnapshot.bookmaker_id)
                    & (ClosingLine.market_type == OddsSnapshot.market_type),
                )
                .where(OddsSnapshot.bookmaker_id == bookmaker.id)
            )

            results = db.execute(stmt).all()
            clv_values = []

            for snapshot, closing_line in results:
                for outcome in snapshot.outcomes:
                    team_name = outcome.get("name")
                    entry_odds = calc._extract_odds_for_team(
                        snapshot.outcomes, team_name
                    )
                    closing_odds = calc._extract_odds_for_team(
                        closing_line.outcomes, team_name
                    )

                    if entry_odds and closing_odds:
                        clv = calc.calculate_clv(entry_odds, closing_odds)
                        clv_values.append(clv)

            total = len(clv_values)
            positive_count = sum(1 for clv in clv_values if clv > 0)

            stats_list.append(
                BookmakerStats(
                    bookmaker_name=bookmaker.name,
                    total_snapshots=total,
                    avg_clv=sum(clv_values) / total if total > 0 else None,
                    positive_clv_percentage=(
                        (positive_count / total * 100) if total > 0 else 0
                    ),
                )
            )

        return stats_list

    except Exception as e:
        logger.error(f"Error fetching bookmaker stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/games/{game_id}/snapshots", response_model=List[OddsSnapshotResponse])
async def get_game_snapshots(game_id: int):
    """Get all odds snapshots for a specific game."""
    db = get_db()

    try:
        # Get all snapshots for this game with bookmaker info
        stmt = (
            select(OddsSnapshot, Bookmaker)
            .join(Bookmaker, Bookmaker.id == OddsSnapshot.bookmaker_id)
            .where(OddsSnapshot.game_id == game_id)
            .order_by(OddsSnapshot.timestamp.asc())
        )

        results = db.execute(stmt).all()

        snapshots = []
        for snapshot, bookmaker in results:
            snapshots.append(
                OddsSnapshotResponse(
                    id=snapshot.id,
                    timestamp=snapshot.timestamp,
                    bookmaker_name=bookmaker.name,
                    market_type=snapshot.market_type,
                    outcomes=snapshot.outcomes,
                )
            )

        return snapshots

    except Exception as e:
        logger.error(f"Error fetching game snapshots: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/games/{game_id}/closing-lines", response_model=List[ClosingLineResponse])
async def get_game_closing_lines(game_id: int):
    """Get all closing lines for a specific game."""
    db = get_db()

    try:
        # Get all closing lines for this game with bookmaker info
        stmt = (
            select(ClosingLine, Bookmaker)
            .join(Bookmaker, Bookmaker.id == ClosingLine.bookmaker_id)
            .where(ClosingLine.game_id == game_id)
        )

        results = db.execute(stmt).all()

        closing_lines = []
        for closing_line, bookmaker in results:
            closing_lines.append(
                ClosingLineResponse(
                    id=closing_line.id,
                    bookmaker_name=bookmaker.name,
                    market_type=closing_line.market_type,
                    outcomes=closing_line.outcomes,
                )
            )

        return closing_lines

    except Exception as e:
        logger.error(f"Error fetching closing lines: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/games/{game_id}/analysis")
async def get_game_analysis(game_id: int):
    """Get detailed CLV analysis for a completed game."""
    db = get_db()
    calc = CLVCalculator()

    try:
        # Get all snapshots with closing lines for this game
        stmt = (
            select(OddsSnapshot, ClosingLine, Bookmaker.name)
            .join(
                ClosingLine,
                (ClosingLine.game_id == OddsSnapshot.game_id)
                & (ClosingLine.bookmaker_id == OddsSnapshot.bookmaker_id)
                & (ClosingLine.market_type == OddsSnapshot.market_type),
            )
            .join(Bookmaker, Bookmaker.id == OddsSnapshot.bookmaker_id)
            .where(OddsSnapshot.game_id == game_id)
        )

        results = db.execute(stmt).all()

        if not results:
            return {
                "game_id": game_id,
                "total_opportunities": 0,
                "by_market": {},
                "by_bookmaker": {},
                "best_opportunities": [],
            }

        # Calculate CLV for each snapshot/outcome combination
        opportunities = []
        by_market = {}
        by_bookmaker = {}

        for snapshot, closing_line, bookmaker_name in results:
            market_type = snapshot.market_type

            for outcome in snapshot.outcomes:
                team_name = outcome.get("name")
                entry_odds = calc._extract_odds_for_team(snapshot.outcomes, team_name)
                closing_odds = calc._extract_odds_for_team(
                    closing_line.outcomes, team_name
                )

                if entry_odds and closing_odds:
                    clv = calc.calculate_clv(entry_odds, closing_odds)

                    # Store opportunity
                    opportunities.append(
                        {
                            "bookmaker": bookmaker_name,
                            "market_type": market_type,
                            "outcome": team_name,
                            "entry_odds": entry_odds,
                            "closing_odds": closing_odds,
                            "clv": clv,
                            "timestamp": snapshot.timestamp.isoformat(),
                        }
                    )

                    # Aggregate by market type
                    if market_type not in by_market:
                        by_market[market_type] = {"clv_values": [], "count": 0}
                    by_market[market_type]["clv_values"].append(clv)
                    by_market[market_type]["count"] += 1

                    # Aggregate by bookmaker
                    if bookmaker_name not in by_bookmaker:
                        by_bookmaker[bookmaker_name] = {"clv_values": [], "count": 0}
                    by_bookmaker[bookmaker_name]["clv_values"].append(clv)
                    by_bookmaker[bookmaker_name]["count"] += 1

        # Calculate averages
        for market in by_market.values():
            market["avg_clv"] = sum(market["clv_values"]) / len(market["clv_values"])
            del market["clv_values"]

        for bookmaker in by_bookmaker.values():
            bookmaker["avg_clv"] = sum(bookmaker["clv_values"]) / len(
                bookmaker["clv_values"]
            )
            del bookmaker["clv_values"]

        # Get top 5 best opportunities (highest CLV)
        best_opportunities = sorted(opportunities, key=lambda x: x["clv"], reverse=True)[
            :5
        ]

        return {
            "game_id": game_id,
            "total_opportunities": len(opportunities),
            "avg_clv": sum(o["clv"] for o in opportunities) / len(opportunities),
            "by_market": by_market,
            "by_bookmaker": by_bookmaker,
            "best_opportunities": best_opportunities,
            "positive_clv_count": sum(1 for o in opportunities if o["clv"] > 0),
            "positive_clv_percentage": (
                sum(1 for o in opportunities if o["clv"] > 0) / len(opportunities) * 100
            ),
        }
    except Exception as e:
        logger.error(f"Error analyzing game {game_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/daily-reports", response_model=List[DailyCLVReportResponse])
async def get_daily_reports(limit: int = 30):
    """Get daily CLV reports, most recent first."""
    db = get_db()

    try:
        stmt = (
            select(DailyCLVReport)
            .order_by(DailyCLVReport.report_date.desc())
            .limit(limit)
        )

        reports = db.execute(stmt).scalars().all()
        return reports

    except Exception as e:
        logger.error(f"Error fetching daily reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/daily-reports/{report_date}", response_model=DailyCLVReportResponse)
async def get_daily_report(report_date: str):
    """Get daily CLV report for a specific date (YYYY-MM-DD)."""
    db = get_db()

    try:
        # Parse date
        date = datetime.strptime(report_date, "%Y-%m-%d")
        date = date.replace(tzinfo=timezone.utc)

        stmt = select(DailyCLVReport).where(DailyCLVReport.report_date == date)
        report = db.execute(stmt).scalar_one_or_none()

        if not report:
            raise HTTPException(status_code=404, detail=f"No report found for {report_date}")

        return report

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching daily report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/daily-reports/{report_id}/opportunities")
async def get_report_opportunities(report_id: int):
    """Get tracked opportunities with their results for a specific daily report."""
    db = get_db()

    try:
        # Get the report
        report = db.get(DailyCLVReport, report_id)
        if not report:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

        # Get all tracked opportunities for this report
        stmt = (
            select(OpportunityPerformance, Game)
            .join(Game, Game.id == OpportunityPerformance.game_id)
            .where(OpportunityPerformance.report_id == report_id)
            .order_by(OpportunityPerformance.clv_percentage.desc())
        )

        results = db.execute(stmt).all()

        opportunities = []
        for opp, game in results:
            # Get team names
            home_team = db.query(Team).filter(Team.id == game.home_team_id).first()
            away_team = db.query(Team).filter(Team.id == game.away_team_id).first()

            # Get betting outcome for final score
            outcome = db.query(BettingOutcome).filter(BettingOutcome.game_id == game.id).first()

            opportunities.append({
                "id": opp.id,
                "game_id": opp.game_id,
                "home_team": home_team.name if home_team else "Unknown",
                "away_team": away_team.name if away_team else "Unknown",
                "home_score": outcome.home_score if outcome else None,
                "away_score": outcome.away_score if outcome else None,
                "bookmaker": opp.bookmaker,
                "market_type": opp.market_type,
                "outcome_name": opp.outcome_name,
                "point_line": opp.point_line,  # Spread/total line
                "entry_odds": opp.entry_odds,
                "closing_odds": opp.closing_odds,
                "clv_percentage": opp.clv_percentage,
                "bet_amount": opp.bet_amount,
                "result": opp.result,
                "profit_loss": opp.profit_loss,
                "settled_at": opp.settled_at.isoformat() if opp.settled_at else None,
            })

        return {
            "report_id": report_id,
            "report_date": report.report_date.isoformat(),
            "total_opportunities": len(opportunities),
            "opportunities": opportunities,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching report opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/bankroll-simulation")
async def get_bankroll_simulation(
    bet_size: float = 100.0,
    starting_bankroll: float = 10000.0,
    strategy: str = "fixed",
    fraction_percent: float = 2.0,
    max_bet_percent: float = 25.0,
    bookmaker_filter: str = None,
    market_filter: str = None,
    clv_threshold: float = None,
):
    """Get bankroll simulation data with P&L curves and drawdown analysis.

    Args:
        bet_size: Amount wagered per bet for fixed strategy (default $100)
        starting_bankroll: Starting bankroll amount (default $10,000)
        strategy: Betting strategy - fixed, fractional, kelly, half_kelly, confidence
        fraction_percent: Percentage of bankroll for fractional/confidence strategies
        max_bet_percent: Maximum bet as percentage of bankroll (Kelly cap)
        bookmaker_filter: Only include bets from this bookmaker
        market_filter: Only include bets from this market type (h2h, spreads, totals)
        clv_threshold: Minimum CLV% to include bet in simulation
    """
    db = get_db()

    try:
        # Get all settled opportunities ordered by settlement date
        stmt = (
            select(OpportunityPerformance, Game, DailyCLVReport)
            .join(Game, Game.id == OpportunityPerformance.game_id)
            .join(DailyCLVReport, DailyCLVReport.id == OpportunityPerformance.report_id)
            .where(OpportunityPerformance.result.in_(["win", "loss", "push"]))
            .order_by(OpportunityPerformance.settled_at.asc())
        )

        # Apply filters
        if bookmaker_filter:
            stmt = stmt.where(OpportunityPerformance.bookmaker == bookmaker_filter)
        if market_filter:
            stmt = stmt.where(OpportunityPerformance.market_type == market_filter)
        if clv_threshold is not None:
            stmt = stmt.where(OpportunityPerformance.clv_percentage >= clv_threshold)

        results = db.execute(stmt).all()

        if not results:
            return {
                "has_data": False,
                "message": "No settled bets found for simulation",
                "data_points": [],
                "summary": {},
                "by_bookmaker": {},
                "by_market": {},
            }

        # Initialize bet sizer based on strategy
        try:
            strategy_enum = BetSizingStrategy(strategy)
        except ValueError:
            strategy_enum = BetSizingStrategy.FIXED

        bet_sizer = get_bet_sizer(
            strategy=strategy_enum,
            unit_size=bet_size,
            fraction=fraction_percent / 100,
            kelly_fraction=1.0 if strategy_enum == BetSizingStrategy.KELLY else 0.5,
        )

        # Build cumulative P&L curve
        data_points = []
        bet_returns = []  # For Sharpe calculation
        cumulative_pl = 0.0
        current_bankroll = starting_bankroll
        peak_bankroll = starting_bankroll
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        current_drawdown = 0.0
        win_count = 0
        loss_count = 0
        push_count = 0
        total_wagered = 0.0
        bet_sizes = []

        # Track by bookmaker and market
        by_bookmaker = {}
        by_market = {}

        for opp, game, report in results:
            # Calculate bet size using the chosen strategy
            context = BetContext(
                bankroll=current_bankroll,
                odds=opp.entry_odds,
                clv_percentage=opp.clv_percentage,
                ml_confidence=0.6,  # Default confidence
                max_fraction=max_bet_percent / 100,
            )

            sizing_result = bet_sizer.calculate_bet_size(context)
            actual_bet_size = sizing_result.bet_size
            bet_sizes.append(actual_bet_size)

            # Calculate profit/loss for this bet
            if opp.result == "win":
                profit = actual_bet_size * (opp.entry_odds - 1)
                win_count += 1
            elif opp.result == "loss":
                profit = -actual_bet_size
                loss_count += 1
            else:  # push
                profit = 0
                push_count += 1

            # Track return for Sharpe ratio
            if actual_bet_size > 0:
                bet_return = profit / actual_bet_size
                bet_returns.append(bet_return)

            cumulative_pl += profit
            total_wagered += actual_bet_size
            current_bankroll = starting_bankroll + cumulative_pl

            # Track peak and drawdown
            if current_bankroll > peak_bankroll:
                peak_bankroll = current_bankroll
                current_drawdown = 0.0
            else:
                current_drawdown = peak_bankroll - current_bankroll
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                    max_drawdown_pct = (current_drawdown / peak_bankroll) * 100

            # Track by bookmaker
            if opp.bookmaker not in by_bookmaker:
                by_bookmaker[opp.bookmaker] = {
                    "bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                    "profit": 0.0, "wagered": 0.0, "clv_sum": 0.0
                }
            by_bookmaker[opp.bookmaker]["bets"] += 1
            by_bookmaker[opp.bookmaker]["wagered"] += actual_bet_size
            by_bookmaker[opp.bookmaker]["profit"] += profit
            by_bookmaker[opp.bookmaker]["clv_sum"] += opp.clv_percentage
            if opp.result == "win":
                by_bookmaker[opp.bookmaker]["wins"] += 1
            elif opp.result == "loss":
                by_bookmaker[opp.bookmaker]["losses"] += 1
            else:
                by_bookmaker[opp.bookmaker]["pushes"] += 1

            # Track by market
            if opp.market_type not in by_market:
                by_market[opp.market_type] = {
                    "bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                    "profit": 0.0, "wagered": 0.0, "clv_sum": 0.0
                }
            by_market[opp.market_type]["bets"] += 1
            by_market[opp.market_type]["wagered"] += actual_bet_size
            by_market[opp.market_type]["profit"] += profit
            by_market[opp.market_type]["clv_sum"] += opp.clv_percentage
            if opp.result == "win":
                by_market[opp.market_type]["wins"] += 1
            elif opp.result == "loss":
                by_market[opp.market_type]["losses"] += 1
            else:
                by_market[opp.market_type]["pushes"] += 1

            # Get team names
            home_team = db.query(Team).filter(Team.id == game.home_team_id).first()
            away_team = db.query(Team).filter(Team.id == game.away_team_id).first()

            data_points.append({
                "date": opp.settled_at.isoformat() if opp.settled_at else report.report_date.isoformat(),
                "game_date": game.commence_time.strftime("%Y-%m-%d") if game.commence_time else None,
                "bet_number": len(data_points) + 1,
                "cumulative_pl": round(cumulative_pl, 2),
                "bankroll": round(current_bankroll, 2),
                "drawdown": round(current_drawdown, 2),
                "drawdown_pct": round((current_drawdown / peak_bankroll) * 100 if peak_bankroll > 0 else 0, 2),
                "result": opp.result,
                "profit": round(profit, 2),
                "bet_size": round(actual_bet_size, 2),
                "game": f"{away_team.name if away_team else 'Unknown'} @ {home_team.name if home_team else 'Unknown'}",
                "bookmaker": opp.bookmaker,
                "outcome": opp.outcome_name,
                "market": opp.market_type,
                "odds": opp.entry_odds,
                "closing_odds": opp.closing_odds,
                "clv": round(opp.clv_percentage, 2),
            })

        total_bets = win_count + loss_count + push_count

        # Calculate risk metrics
        sharpe_ratio = calculate_sharpe_ratio(bet_returns) if bet_returns else 0.0
        sortino_ratio = calculate_sortino_ratio(bet_returns) if bet_returns else 0.0

        # Calculate bet size statistics
        import numpy as np
        avg_bet_size = float(np.mean(bet_sizes)) if bet_sizes else 0.0
        bet_size_std = float(np.std(bet_sizes)) if len(bet_sizes) > 1 else 0.0

        # Finalize by_bookmaker stats
        for bk in by_bookmaker.values():
            bk["roi"] = round((bk["profit"] / bk["wagered"]) * 100, 2) if bk["wagered"] > 0 else 0
            bk["avg_clv"] = round(bk["clv_sum"] / bk["bets"], 2) if bk["bets"] > 0 else 0
            bk["win_rate"] = round((bk["wins"] / (bk["wins"] + bk["losses"])) * 100, 2) if (bk["wins"] + bk["losses"]) > 0 else 0
            bk["profit"] = round(bk["profit"], 2)
            bk["wagered"] = round(bk["wagered"], 2)
            del bk["clv_sum"]

        # Finalize by_market stats
        for mkt in by_market.values():
            mkt["roi"] = round((mkt["profit"] / mkt["wagered"]) * 100, 2) if mkt["wagered"] > 0 else 0
            mkt["avg_clv"] = round(mkt["clv_sum"] / mkt["bets"], 2) if mkt["bets"] > 0 else 0
            mkt["win_rate"] = round((mkt["wins"] / (mkt["wins"] + mkt["losses"])) * 100, 2) if (mkt["wins"] + mkt["losses"]) > 0 else 0
            mkt["profit"] = round(mkt["profit"], 2)
            mkt["wagered"] = round(mkt["wagered"], 2)
            del mkt["clv_sum"]

        return {
            "has_data": True,
            "data_points": data_points,
            "summary": {
                "starting_bankroll": starting_bankroll,
                "ending_bankroll": round(current_bankroll, 2),
                "total_profit_loss": round(cumulative_pl, 2),
                "roi_pct": round((cumulative_pl / total_wagered) * 100, 2) if total_wagered > 0 else 0,
                "total_bets": total_bets,
                "win_count": win_count,
                "loss_count": loss_count,
                "push_count": push_count,
                "win_rate": round((win_count / (win_count + loss_count)) * 100, 2) if (win_count + loss_count) > 0 else 0,
                "max_drawdown": round(max_drawdown, 2),
                "max_drawdown_pct": round(max_drawdown_pct, 2),
                "total_wagered": round(total_wagered, 2),
                "peak_bankroll": round(peak_bankroll, 2),
                "strategy": strategy,
                "sharpe_ratio": round(sharpe_ratio, 3),
                "sortino_ratio": round(sortino_ratio, 3),
                "avg_bet_size": round(avg_bet_size, 2),
                "bet_size_std": round(bet_size_std, 2),
            },
            "by_bookmaker": by_bookmaker,
            "by_market": by_market,
        }

    except Exception as e:
        logger.error(f"Error running bankroll simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/bankroll-simulation/breakdown")
async def get_bankroll_breakdown(
    group_by: str = "bookmaker",
    clv_bucket_size: float = 1.0,
):
    """Get performance breakdown grouped by different dimensions.

    Args:
        group_by: Dimension to group by - bookmaker, market, month, clv_bucket
        clv_bucket_size: Size of CLV buckets in percentage points (for clv_bucket grouping)
    """
    db = get_db()

    try:
        # Get all settled opportunities
        stmt = (
            select(OpportunityPerformance, Game)
            .join(Game, Game.id == OpportunityPerformance.game_id)
            .where(OpportunityPerformance.result.in_(["win", "loss", "push"]))
            .order_by(OpportunityPerformance.settled_at.asc())
        )

        results = db.execute(stmt).all()

        if not results:
            return {"has_data": False, "breakdown": []}

        # Group data
        groups = {}

        for opp, game in results:
            # Determine group key
            if group_by == "bookmaker":
                key = opp.bookmaker
            elif group_by == "market":
                key = opp.market_type
            elif group_by == "month":
                if opp.settled_at:
                    key = opp.settled_at.strftime("%Y-%m")
                else:
                    key = "Unknown"
            elif group_by == "clv_bucket":
                # Create CLV buckets (e.g., 0-1%, 1-2%, 2-3%, etc.)
                bucket_floor = int(opp.clv_percentage / clv_bucket_size) * clv_bucket_size
                bucket_ceil = bucket_floor + clv_bucket_size
                if opp.clv_percentage < 0:
                    key = f"{bucket_floor:.1f}% to {bucket_ceil:.1f}%"
                else:
                    key = f"{bucket_floor:.1f}% to {bucket_ceil:.1f}%"
            else:
                key = "all"

            if key not in groups:
                groups[key] = {
                    "bets": 0,
                    "wins": 0,
                    "losses": 0,
                    "pushes": 0,
                    "profit": 0.0,
                    "clv_sum": 0.0,
                }

            groups[key]["bets"] += 1
            groups[key]["clv_sum"] += opp.clv_percentage

            # Assume $100 unit bets for breakdown
            if opp.result == "win":
                profit = 100 * (opp.entry_odds - 1)
                groups[key]["wins"] += 1
            elif opp.result == "loss":
                profit = -100
                groups[key]["losses"] += 1
            else:
                profit = 0
                groups[key]["pushes"] += 1

            groups[key]["profit"] += profit

        # Build breakdown list
        breakdown = []
        for key, data in groups.items():
            settled = data["wins"] + data["losses"]
            breakdown.append({
                "group": key,
                "bets": data["bets"],
                "wins": data["wins"],
                "losses": data["losses"],
                "pushes": data["pushes"],
                "profit": round(data["profit"], 2),
                "roi": round((data["profit"] / (data["bets"] * 100)) * 100, 2) if data["bets"] > 0 else 0,
                "win_rate": round((data["wins"] / settled) * 100, 2) if settled > 0 else 0,
                "avg_clv": round(data["clv_sum"] / data["bets"], 2) if data["bets"] > 0 else 0,
            })

        # Sort by profit descending
        breakdown.sort(key=lambda x: x["profit"], reverse=True)

        return {
            "has_data": True,
            "group_by": group_by,
            "breakdown": breakdown,
        }

    except Exception as e:
        logger.error(f"Error getting breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/games/{game_id}/score")
async def get_game_score(game_id: int):
    """Get the final score for a completed game."""
    db = get_db()

    try:
        # Get game details
        stmt = (
            select(Game, Team, Sport)
            .join(Team, Team.id == Game.home_team_id)
            .join(Sport, Sport.id == Game.sport_id)
            .where(Game.id == game_id)
        )
        result = db.execute(stmt).first()

        if not result:
            raise HTTPException(status_code=404, detail="Game not found")

        game, home_team, sport = result

        # Get away team
        away_team = db.query(Team).filter(Team.id == game.away_team_id).first()

        # Fetch scores from Odds API
        api_key = os.getenv("ODDS_API_KEY")
        if not api_key:
            logger.warning("No ODDS_API_KEY found - cannot fetch scores")
            return {
                "game_id": game_id,
                "has_score": False,
                "message": "Score data unavailable - API key not configured",
            }

        client = OddsAPIClient(api_key)

        try:
            # Fetch recent scores (typically last 3 days)
            scores_data = client.get_scores(sport.key)
            scores = scores_data.get("data", [])

            logger.info(f"Looking for game: {away_team.name} @ {home_team.name}")
            logger.info(f"Game commenced at: {game.commence_time}")
            logger.info(f"Found {len(scores)} recent games with scores from API")

            # Find matching game by team names (case-insensitive partial match)
            for score in scores:
                home_name = score.get("home_team")
                away_name = score.get("away_team")

                logger.info(f"Checking: {away_name} @ {home_name} (completed: {score.get('completed')})")

                # Case-insensitive partial matching
                home_match = (
                    home_name and home_team.name and
                    (home_name.lower() in home_team.name.lower() or home_team.name.lower() in home_name.lower())
                )
                away_match = (
                    away_name and away_team.name and
                    (away_name.lower() in away_team.name.lower() or away_team.name.lower() in away_name.lower())
                )

                if home_match and away_match:
                    final_scores = score.get("scores")
                    # Check if game has actual scores (means it's completed)
                    if final_scores and len(final_scores) > 0:
                        logger.info(f"Found matching game with scores!")
                        return {
                            "game_id": game_id,
                            "has_score": True,
                            "home_team": home_name,
                            "away_team": away_name,
                            "home_score": next(
                                (s["score"] for s in final_scores if s["name"] == home_name),
                                None,
                            ),
                            "away_score": next(
                                (s["score"] for s in final_scores if s["name"] == away_name),
                                None,
                            ),
                            "completed": True,
                        }

            return {
                "game_id": game_id,
                "has_score": False,
                "message": "Score not available yet - game may not be completed",
            }

        except Exception as api_error:
            logger.error(f"Error fetching scores from API: {api_error}")
            return {
                "game_id": game_id,
                "has_score": False,
                "message": f"Error fetching score: {str(api_error)}",
            }

    except Exception as e:
        logger.error(f"Error getting score for game {game_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ============================================================================
# User Bet Tracking Endpoints
# ============================================================================

@app.get("/api/user-bets")
async def get_user_bets(status: str = None):
    """Get all user bets, optionally filtered by status."""
    db = get_db()

    try:
        stmt = select(UserBet).order_by(UserBet.game_date.desc())

        if status:
            stmt = stmt.where(UserBet.result == status)

        bets = db.execute(stmt).scalars().all()

        return [
            {
                "id": bet.id,
                "game_description": bet.game_description,
                "game_date": bet.game_date.isoformat(),
                "bookmaker": bet.bookmaker,
                "market_type": bet.market_type,
                "bet_description": bet.bet_description,
                "odds": bet.odds,
                "stake": bet.stake,
                "result": bet.result,
                "profit_loss": bet.profit_loss,
                "closing_odds": bet.closing_odds,
                "clv_percentage": bet.clv_percentage,
                "notes": bet.notes,
                "created_at": bet.created_at.isoformat(),
                "settled_at": bet.settled_at.isoformat() if bet.settled_at else None,
            }
            for bet in bets
        ]

    except Exception as e:
        logger.error(f"Error fetching user bets: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/api/user-bets")
async def create_user_bet(
    game_description: str,
    game_date: str,
    bookmaker: str,
    market_type: str,
    bet_description: str,
    odds: int,
    stake: float = 100.0,
    notes: str = None,
):
    """Create a new user bet."""
    db = get_db()

    try:
        # Parse date
        parsed_date = datetime.fromisoformat(game_date.replace('Z', '+00:00'))

        bet = UserBet(
            game_description=game_description,
            game_date=parsed_date,
            bookmaker=bookmaker,
            market_type=market_type,
            bet_description=bet_description,
            odds=odds,
            stake=stake,
            notes=notes,
            result="pending",
        )

        db.add(bet)
        db.commit()
        db.refresh(bet)

        return {
            "id": bet.id,
            "message": "Bet created successfully",
            "bet": {
                "game_description": bet.game_description,
                "bet_description": bet.bet_description,
                "odds": bet.odds,
                "stake": bet.stake,
            }
        }

    except Exception as e:
        logger.error(f"Error creating user bet: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.put("/api/user-bets/{bet_id}")
async def update_user_bet(
    bet_id: int,
    result: str = None,
    closing_odds: int = None,
    notes: str = None,
    stake: float = None,
):
    """Update a user bet (settle it or add notes)."""
    db = get_db()

    try:
        bet = db.get(UserBet, bet_id)
        if not bet:
            raise HTTPException(status_code=404, detail="Bet not found")

        if stake is not None:
            bet.stake = stake

        if result:
            bet.result = result
            if result == "pending":
                # Revert to pending
                bet.settled_at = None
                bet.profit_loss = None
            else:
                bet.settled_at = datetime.now(timezone.utc)
                # Calculate profit/loss
                if result == "win":
                    if bet.odds > 0:
                        bet.profit_loss = bet.stake * (bet.odds / 100)
                    else:
                        bet.profit_loss = bet.stake * (100 / abs(bet.odds))
                elif result == "loss":
                    bet.profit_loss = -bet.stake
                else:  # push
                    bet.profit_loss = 0

        if closing_odds is not None:
            bet.closing_odds = closing_odds
            # Calculate CLV if we have closing odds
            if bet.odds and closing_odds:
                # Convert American odds to implied probability
                if bet.odds > 0:
                    entry_prob = 100 / (bet.odds + 100)
                else:
                    entry_prob = abs(bet.odds) / (abs(bet.odds) + 100)

                if closing_odds > 0:
                    closing_prob = 100 / (closing_odds + 100)
                else:
                    closing_prob = abs(closing_odds) / (abs(closing_odds) + 100)

                bet.clv_percentage = round((closing_prob - entry_prob) / closing_prob * 100, 2)

        if notes is not None:
            bet.notes = notes

        db.commit()

        return {
            "id": bet.id,
            "message": "Bet updated successfully",
            "result": bet.result,
            "profit_loss": bet.profit_loss,
            "clv_percentage": bet.clv_percentage,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user bet: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.delete("/api/user-bets/{bet_id}")
async def delete_user_bet(bet_id: int):
    """Delete a user bet."""
    db = get_db()

    try:
        bet = db.get(UserBet, bet_id)
        if not bet:
            raise HTTPException(status_code=404, detail="Bet not found")

        db.delete(bet)
        db.commit()

        return {"message": "Bet deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user bet: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/user-bets/summary")
async def get_user_bets_summary():
    """Get summary stats for user bets."""
    db = get_db()

    try:
        bets = db.execute(select(UserBet)).scalars().all()

        total = len(bets)
        pending = sum(1 for b in bets if b.result == "pending")
        wins = sum(1 for b in bets if b.result == "win")
        losses = sum(1 for b in bets if b.result == "loss")
        pushes = sum(1 for b in bets if b.result == "push")

        settled = wins + losses + pushes
        total_profit = sum(b.profit_loss or 0 for b in bets if b.result in ("win", "loss", "push"))
        total_staked = sum(b.stake for b in bets if b.result in ("win", "loss"))

        return {
            "total_bets": total,
            "pending": pending,
            "settled": settled,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
            "total_profit": round(total_profit, 2),
            "total_staked": round(total_staked, 2),
            "roi": round(total_profit / total_staked * 100, 2) if total_staked > 0 else 0,
        }

    except Exception as e:
        logger.error(f"Error getting user bets summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

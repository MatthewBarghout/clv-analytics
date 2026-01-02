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
from src.api.schemas import (
    BookmakerStats,
    CLVHistoryPoint,
    CLVStats,
    ClosingLineResponse,
    GameWithCLV,
    HealthResponse,
    OddsSnapshotResponse,
)
from src.models.database import Bookmaker, ClosingLine, Game, OddsSnapshot, Team

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

            games_with_clv.append(
                GameWithCLV(
                    game_id=game.id,
                    home_team=home_team_name,
                    away_team=away_team.name if away_team else "Unknown",
                    commence_time=game.commence_time,
                    completed=game.completed,
                    snapshots_count=snapshots_count,
                    closing_lines_count=closing_lines_count,
                    avg_clv=avg_game_clv,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

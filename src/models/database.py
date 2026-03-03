from datetime import date, datetime
from typing import List

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class Sport(Base):
    """Represents a sport category (e.g., basketball, football, baseball)."""

    __tablename__ = "sports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    key: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    teams: Mapped[List["Team"]] = relationship(
        "Team", back_populates="sport", cascade="all, delete-orphan"
    )
    games: Mapped[List["Game"]] = relationship(
        "Game", back_populates="sport", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Sport(id={self.id}, key='{self.key}', name='{self.name}', active={self.active})>"


class Team(Base):
    """Represents a team within a sport."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sport_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sports.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    key: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    sport: Mapped["Sport"] = relationship("Sport", back_populates="teams")
    home_games: Mapped[List["Game"]] = relationship(
        "Game", back_populates="home_team", foreign_keys="Game.home_team_id"
    )
    away_games: Mapped[List["Game"]] = relationship(
        "Game", back_populates="away_team", foreign_keys="Game.away_team_id"
    )

    # Composite unique constraint: same team key can't exist twice in same sport
    __table_args__ = (
        Index("ix_teams_sport_key", "sport_id", "key", unique=True),
    )

    def __repr__(self) -> str:
        return f"<Team(id={self.id}, key='{self.key}', name='{self.name}', sport_id={self.sport_id})>"


class Game(Base):
    """Represents a scheduled or completed game between two teams."""

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sport_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sports.id", ondelete="CASCADE"), nullable=False, index=True
    )
    home_team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False, index=True
    )
    away_team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False, index=True
    )
    commence_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    sport: Mapped["Sport"] = relationship("Sport", back_populates="games")
    home_team: Mapped["Team"] = relationship(
        "Team", back_populates="home_games", foreign_keys=[home_team_id]
    )
    away_team: Mapped["Team"] = relationship(
        "Team", back_populates="away_games", foreign_keys=[away_team_id]
    )
    odds_snapshots: Mapped[List["OddsSnapshot"]] = relationship(
        "OddsSnapshot", back_populates="game", cascade="all, delete-orphan"
    )
    closing_lines: Mapped[List["ClosingLine"]] = relationship(
        "ClosingLine", back_populates="game", cascade="all, delete-orphan"
    )
    outcome: Mapped["BettingOutcome"] = relationship(
        "BettingOutcome", back_populates="game", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Index for querying games by commence time
        Index("ix_games_commence_time", "commence_time"),
        # Index for finding games by sport and time
        Index("ix_games_sport_commence", "sport_id", "commence_time"),
        # Composite index for active game filters (completed + time)
        Index("ix_games_completed_commence", "completed", "commence_time"),
    )

    def __repr__(self) -> str:
        return (
            f"<Game(id={self.id}, sport_id={self.sport_id}, "
            f"home={self.home_team_id}, away={self.away_team_id}, "
            f"commence_time={self.commence_time}, completed={self.completed})>"
        )


class Bookmaker(Base):
    """Represents a bookmaker/sportsbook."""

    __tablename__ = "bookmakers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    key: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    odds_snapshots: Mapped[List["OddsSnapshot"]] = relationship(
        "OddsSnapshot", back_populates="bookmaker", cascade="all, delete-orphan"
    )
    closing_lines: Mapped[List["ClosingLine"]] = relationship(
        "ClosingLine", back_populates="bookmaker", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Bookmaker(id={self.id}, key='{self.key}', name='{self.name}', active={self.active})>"


class OddsSnapshot(Base):
    """
    Represents a snapshot of odds at a specific point in time.

    The outcomes field stores JSONB data with flexible structure for different market types:
    - Moneyline: [{"name": "Team A", "price": 150}, {"name": "Team B", "price": -170}]
    - Spread: [{"name": "Team A", "point": -3.5, "price": -110}, {"name": "Team B", "point": 3.5, "price": -110}]
    - Totals: [{"name": "Over", "point": 215.5, "price": -110}, {"name": "Under", "point": 215.5, "price": -110}]
    """

    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bookmaker_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("bookmakers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    market_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # 'h2h', 'spreads', 'totals'
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    outcomes: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="odds_snapshots")
    bookmaker: Mapped["Bookmaker"] = relationship("Bookmaker", back_populates="odds_snapshots")

    __table_args__ = (
        # Unique constraint to prevent duplicate snapshots
        Index(
            "ix_odds_snapshots_unique_snapshot",
            "game_id",
            "bookmaker_id",
            "market_type",
            "timestamp",
            unique=True,
        ),
        # Composite index for time-series queries
        Index("ix_odds_snapshots_game_timestamp", "game_id", "timestamp"),
        # Index for querying by bookmaker and time
        Index("ix_odds_snapshots_bookmaker_timestamp", "bookmaker_id", "timestamp"),
        # Index for market type queries
        Index("ix_odds_snapshots_game_market", "game_id", "market_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<OddsSnapshot(id={self.id}, game_id={self.game_id}, "
            f"bookmaker_id={self.bookmaker_id}, market_type='{self.market_type}', "
            f"timestamp={self.timestamp})>"
        )


class ClosingLine(Base):
    """
    Represents the closing line for a game at a specific bookmaker.

    The outcomes field stores JSONB data in the same format as OddsSnapshot.
    This represents the final odds before the game commenced.
    """

    __tablename__ = "closing_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bookmaker_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("bookmakers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    market_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # 'h2h', 'spreads', 'totals'
    outcomes: Mapped[dict] = mapped_column(JSONB, nullable=False)
    closed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="closing_lines")
    bookmaker: Mapped["Bookmaker"] = relationship("Bookmaker", back_populates="closing_lines")

    __table_args__ = (
        # Unique constraint: one closing line per game/bookmaker/market combination
        Index(
            "ix_closing_lines_unique",
            "game_id",
            "bookmaker_id",
            "market_type",
            unique=True,
        ),
        # Index for querying closing lines by game
        Index("ix_closing_lines_game", "game_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<ClosingLine(id={self.id}, game_id={self.game_id}, "
            f"bookmaker_id={self.bookmaker_id}, market_type='{self.market_type}', "
            f"closed_at={self.closed_at})>"
        )


class DailyCLVReport(Base):
    """
    Stores daily CLV analysis results for completed games.

    This table aggregates CLV data for all games that completed on a given day,
    providing historical tracking of betting performance and opportunity identification.
    """

    __tablename__ = "daily_clv_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, unique=True, index=True
    )

    # Overall statistics
    games_analyzed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_opportunities: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_clv: Mapped[float] = mapped_column(nullable=True)
    median_clv: Mapped[float] = mapped_column(nullable=True)
    positive_clv_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    positive_clv_percentage: Mapped[float] = mapped_column(nullable=True)

    # Best opportunities (JSONB array of top CLV opportunities)
    # Format: [{"game_id": 1, "bookmaker": "Pinnacle", "market": "h2h", "outcome": "Lakers", "clv": 5.2, ...}]
    best_opportunities: Mapped[dict] = mapped_column(JSONB, nullable=True)

    # Breakdown by bookmaker (JSONB object)
    # Format: {"Pinnacle": {"avg_clv": 2.1, "count": 45}, "FanDuel": {...}}
    by_bookmaker: Mapped[dict] = mapped_column(JSONB, nullable=True)

    # Breakdown by market type (JSONB object)
    # Format: {"h2h": {"avg_clv": 1.8, "count": 30}, "spreads": {...}}
    by_market: Mapped[dict] = mapped_column(JSONB, nullable=True)

    # Individual game summaries (JSONB array)
    # Format: [{"game_id": 1, "home": "Lakers", "away": "Celtics", "avg_clv": 2.5, ...}]
    game_summaries: Mapped[dict] = mapped_column(JSONB, nullable=True)

    # Top EV opportunities (JSONB array of +EV betting opportunities from ML predictions)
    # Format: [{"game_id": 1, "bookmaker": "FanDuel", "market": "h2h", "outcome": "Lakers", "ev_score": 8.5, "predicted_movement": -0.05, ...}]
    ev_opportunities: Mapped[dict] = mapped_column(JSONB, nullable=True)

    # Performance tracking (added for profit/loss analysis)
    settled_count: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    win_count: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    loss_count: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    push_count: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    hypothetical_profit: Mapped[float] = mapped_column(nullable=True)  # Total P/L in dollars
    win_rate: Mapped[float] = mapped_column(nullable=True)  # Win percentage
    roi: Mapped[float] = mapped_column(nullable=True)  # Return on investment percentage

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_daily_clv_reports_date", "report_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<DailyCLVReport(id={self.id}, report_date={self.report_date}, "
            f"games_analyzed={self.games_analyzed}, avg_clv={self.avg_clv})>"
        )


class BettingOutcome(Base):
    """Stores final game results and scores for bet settlement."""

    __tablename__ = "betting_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    # Game completion status
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Final scores
    home_score: Mapped[int] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int] = mapped_column(Integer, nullable=True)

    # Derived results for betting
    winner: Mapped[str] = mapped_column(String(100), nullable=True)  # "home", "away", or "push"
    total_points: Mapped[int] = mapped_column(Integer, nullable=True)  # home_score + away_score
    point_differential: Mapped[int] = mapped_column(Integer, nullable=True)  # home_score - away_score

    # Metadata
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    game: Mapped["Game"] = relationship("Game", back_populates="outcome")

    __table_args__ = (
        Index("ix_betting_outcomes_game_id", "game_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<BettingOutcome(id={self.id}, game_id={self.game_id}, "
            f"completed={self.completed}, score={self.away_score}-{self.home_score})>"
        )


class OpportunityPerformance(Base):
    """Tracks performance of betting opportunities from daily reports."""

    __tablename__ = "opportunity_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Link to the opportunity data (stored in daily_clv_reports.best_opportunities)
    report_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("daily_clv_reports.id", ondelete="CASCADE"), nullable=False
    )
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False
    )

    # Opportunity details (denormalized for easy querying)
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)
    market_type: Mapped[str] = mapped_column(String(20), nullable=False)  # h2h, spreads, totals
    outcome_name: Mapped[str] = mapped_column(String(100), nullable=False)  # Team name or "Over/Under"

    # Odds and CLV at time of opportunity
    entry_odds: Mapped[float] = mapped_column(nullable=False)  # American odds
    closing_odds: Mapped[float] = mapped_column(nullable=False)
    clv_percentage: Mapped[float] = mapped_column(nullable=False)

    # Point line for spreads/totals (e.g., -5.5 for spread, 220.5 for total)
    point_line: Mapped[float] = mapped_column(nullable=True)

    # Bet tracking (assuming $100 unit bets)
    bet_amount: Mapped[float] = mapped_column(nullable=False, default=100.0)

    # Result (calculated after game completion)
    result: Mapped[str] = mapped_column(String(10), nullable=True)  # "win", "loss", "push", "pending"
    profit_loss: Mapped[float] = mapped_column(nullable=True)  # Actual P/L in dollars

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    settled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    report: Mapped["DailyCLVReport"] = relationship("DailyCLVReport")
    game: Mapped["Game"] = relationship("Game")

    __table_args__ = (
        Index("ix_opportunity_performance_game_id", "game_id"),
        Index("ix_opportunity_performance_report_id", "report_id"),
        Index("ix_opportunity_performance_result", "result"),
        # Composite indexes for common query patterns
        Index("ix_opportunity_performance_report_game_result", "report_id", "game_id", "result"),
        Index("ix_opportunity_performance_settled_result", "settled_at", "result"),
    )

    def __repr__(self) -> str:
        return (
            f"<OpportunityPerformance(id={self.id}, game_id={self.game_id}, "
            f"market={self.market_type}, result={self.result}, p/l=${self.profit_loss})>"
        )


class UserBet(Base):
    """Tracks user's manually entered bets."""

    __tablename__ = "user_bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Game info (can be linked to game_id if exists, or manual entry)
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="SET NULL"), nullable=True
    )
    game_description: Mapped[str] = mapped_column(String(200), nullable=False)  # e.g., "Bulls vs Celtics"
    game_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Bet details
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)
    market_type: Mapped[str] = mapped_column(String(20), nullable=False)  # h2h, spreads, totals
    bet_description: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "Bulls -1.5"

    # Odds (stored as American odds)
    odds: Mapped[int] = mapped_column(Integer, nullable=False)  # e.g., +140, -110

    # Stake
    stake: Mapped[float] = mapped_column(nullable=False, default=100.0)

    # Result tracking
    result: Mapped[str] = mapped_column(String(10), nullable=True, default="pending")  # "win", "loss", "push", "pending"
    profit_loss: Mapped[float] = mapped_column(nullable=True)

    # CLV tracking (filled in after closing line is known)
    closing_odds: Mapped[int] = mapped_column(Integer, nullable=True)
    clv_percentage: Mapped[float] = mapped_column(nullable=True)

    # Metadata
    notes: Mapped[str] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    settled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationship
    game: Mapped["Game"] = relationship("Game")

    __table_args__ = (
        Index("ix_user_bets_game_date", "game_date"),
        Index("ix_user_bets_result", "result"),
        # Composite index for filtering by result and date
        Index("ix_user_bets_result_game_date", "result", "game_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<UserBet(id={self.id}, bet='{self.bet_description}', "
            f"odds={self.odds}, result={self.result})>"
        )


class PredictionMarketArb(Base):
    """Tracks arbitrage opportunities between sportsbooks and prediction markets (Kalshi/Polymarket)."""

    __tablename__ = "prediction_market_arb"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Optional link to a game if matched
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="SET NULL"), nullable=True
    )

    # Market identification
    event_title: Mapped[str] = mapped_column(String(300), nullable=False)
    market_source: Mapped[str] = mapped_column(String(20), nullable=False)  # "kalshi" or "polymarket"
    market_url: Mapped[str] = mapped_column(String(500), nullable=True)

    # Sportsbook side
    sportsbook_name: Mapped[str] = mapped_column(String(100), nullable=False)
    sportsbook_odds: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)  # decimal odds

    # Prediction market side
    pm_implied_prob: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)  # 0.0 to 1.0
    pm_implied_odds: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)  # 1 / pm_implied_prob

    # Arb metrics
    arb_spread: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)  # percentage, positive = arb exists

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    game: Mapped["Game"] = relationship("Game")

    __table_args__ = (
        Index("ix_pm_arb_timestamp_active", "timestamp", "is_active"),
        Index("ix_pm_arb_source_spread", "market_source", "arb_spread"),
        Index("ix_pm_arb_game_id", "game_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<PredictionMarketArb(id={self.id}, source='{self.market_source}', "
            f"event='{self.event_title[:40]}', spread={self.arb_spread:.2f}%)>"
        )


class BestEVPick(Base):
    """Tracks the daily best EV+ picks selected by the ML model for performance measurement."""

    __tablename__ = "best_ev_picks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Link to game
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False
    )

    # Pick metadata
    report_date: Mapped[date] = mapped_column(Date, nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)
    market_type: Mapped[str] = mapped_column(String(20), nullable=False)  # h2h, spreads, totals
    outcome_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Odds and ML scores at time of pick
    entry_odds: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)  # decimal odds
    ev_score: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    predicted_delta: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)

    # Settlement (filled after game completes)
    result: Mapped[str] = mapped_column(
        String(10), nullable=False, default="pending"
    )  # win, loss, push, pending
    profit_loss: Mapped[float] = mapped_column(Numeric(10, 2), nullable=True)
    settled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    game: Mapped["Game"] = relationship("Game")

    __table_args__ = (
        Index("ix_best_ev_picks_report_date", "report_date"),
        Index("ix_best_ev_picks_result_date", "result", "report_date"),
        Index("ix_best_ev_picks_game_id", "game_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<BestEVPick(id={self.id}, game_id={self.game_id}, "
            f"date={self.report_date}, ev_score={self.ev_score:.2f}, result={self.result})>"
        )

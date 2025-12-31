from datetime import datetime
from typing import List

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
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

    __table_args__ = (
        # Index for querying games by commence time
        Index("ix_games_commence_time", "commence_time"),
        # Index for finding games by sport and time
        Index("ix_games_sport_commence", "sport_id", "commence_time"),
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

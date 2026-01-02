"""Pydantic models for API responses."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime


class CLVStats(BaseModel):
    """Overall CLV statistics."""

    mean_clv: Optional[float]
    median_clv: Optional[float]
    total_analyzed: int
    positive_clv_count: int
    positive_clv_percentage: float
    by_bookmaker: dict
    by_market_type: dict


class BookmakerStats(BaseModel):
    """Statistics for a specific bookmaker."""

    bookmaker_name: str
    total_snapshots: int
    avg_clv: Optional[float]
    positive_clv_percentage: float


class TeamInfo(BaseModel):
    """Team information."""

    id: int
    name: str
    key: str

    model_config = ConfigDict(from_attributes=True)


class GameWithCLV(BaseModel):
    """Game with CLV data."""

    game_id: int
    home_team: str
    away_team: str
    commence_time: datetime
    completed: bool
    snapshots_count: int
    closing_lines_count: int
    avg_clv: Optional[float]


class CLVHistoryPoint(BaseModel):
    """CLV data point for trend chart."""

    date: str
    avg_clv: Optional[float]
    count: int


class OddsSnapshotResponse(BaseModel):
    """Odds snapshot response."""

    id: int
    timestamp: datetime
    bookmaker_name: str
    market_type: str
    outcomes: list

    model_config = ConfigDict(from_attributes=True)


class ClosingLineResponse(BaseModel):
    """Closing line response."""

    id: int
    bookmaker_name: str
    market_type: str
    outcomes: list

    model_config = ConfigDict(from_attributes=True)

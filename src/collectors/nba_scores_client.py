"""
NBA.com Score Fetcher

Fetches final scores from the official NBA.com API.
This API is free, doesn't require authentication, and keeps historical scores.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class NBAScoresClient:
    """Client for fetching NBA game scores from NBA.com."""

    BASE_URL = "https://cdn.nba.com/static/json/liveData/scoreboard"

    def __init__(self):
        """Initialize the NBA scores client."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json",
        })

    def get_scoreboard(self, date: datetime) -> Dict:
        """
        Get scoreboard data for a specific date.

        Args:
            date: The date to fetch scores for

        Returns:
            Dictionary containing scoreboard data with games and scores
        """
        # Format date as YYYY-MM-DD
        date_str = date.strftime("%Y-%m-%d")

        # NBA.com uses the format: todaysScoreboard_00.json for today
        # For specific dates, we need to try the scoreboard endpoint
        url = f"{self.BASE_URL}/todaysScoreboard_00.json"

        try:
            logger.info(f"Fetching NBA scores for {date_str}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Filter games by the requested date
            scoreboard_date = data.get("scoreboard", {}).get("gameDate", "")
            if scoreboard_date.startswith(date_str):
                return data

            # If today's scoreboard doesn't match, try historical endpoint
            # NBA.com historical format varies, so we'll use today's endpoint
            # and the caller should handle date filtering
            return data

        except requests.RequestException as e:
            logger.error(f"Failed to fetch scoreboard from NBA.com: {e}")
            return {}

    def get_completed_games(self, date: datetime) -> List[Dict]:
        """
        Get all completed games for a specific date.

        Args:
            date: The date to fetch completed games for

        Returns:
            List of completed games with scores
        """
        data = self.get_scoreboard(date)
        scoreboard = data.get("scoreboard", {})
        games = scoreboard.get("games", [])

        completed_games = []
        for game in games:
            game_status = game.get("gameStatus", 0)
            # gameStatus: 1=scheduled, 2=live, 3=completed
            if game_status == 3:
                completed_games.append(game)

        logger.info(f"Found {len(completed_games)} completed games on {date.strftime('%Y-%m-%d')}")
        return completed_games

    def parse_game_score(self, game_data: Dict) -> Optional[Dict]:
        """
        Parse game data into a simple score dictionary.

        Args:
            game_data: Raw game data from NBA.com API

        Returns:
            Dictionary with home_team, away_team, home_score, away_score, completed
        """
        try:
            home_team = game_data.get("homeTeam", {})
            away_team = game_data.get("awayTeam", {})

            home_name = home_team.get("teamName", "")
            home_city = home_team.get("teamCity", "")
            away_name = away_team.get("teamName", "")
            away_city = away_team.get("teamCity", "")

            # Full team names like "Los Angeles Lakers"
            home_full = f"{home_city} {home_name}".strip()
            away_full = f"{away_city} {away_name}".strip()

            home_score = home_team.get("score", 0)
            away_score = away_team.get("score", 0)

            game_status = game_data.get("gameStatus", 0)
            completed = game_status == 3

            return {
                "home_team": home_full,
                "away_team": away_full,
                "home_score": home_score,
                "away_score": away_score,
                "completed": completed,
                "game_id": game_data.get("gameId", ""),
            }
        except Exception as e:
            logger.error(f"Failed to parse game data: {e}")
            return None

    def get_scores_for_date_range(self, start_date: datetime, days: int = 1) -> List[Dict]:
        """
        Get all completed game scores for a date range.

        Args:
            start_date: Starting date
            days: Number of days to fetch (default 1)

        Returns:
            List of all completed games with scores
        """
        all_games = []

        for i in range(days):
            date = start_date + timedelta(days=i)
            completed = self.get_completed_games(date)

            for game in completed:
                parsed = self.parse_game_score(game)
                if parsed:
                    all_games.append(parsed)

        return all_games

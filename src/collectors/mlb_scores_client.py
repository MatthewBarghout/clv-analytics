"""
MLB Stats API Score Fetcher

Fetches final scores from the official MLB Stats API.
Free, no authentication required, keeps full historical scores.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class MLBScoresClient:
    """Client for fetching MLB game scores from the official MLB Stats API."""

    BASE_URL = "https://statsapi.mlb.com/api/v1"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        })

    def get_scores_for_date(self, date: datetime) -> List[Dict]:
        """
        Get all completed games for a specific date.

        Returns list of dicts: {home_team, away_team, home_score, away_score, completed}
        """
        date_str = date.strftime("%Y-%m-%d")
        url = f"{self.BASE_URL}/schedule"
        params = {
            "sportId": 1,       # MLB
            "date": date_str,
            "gameType": "R,F,D,L,W",  # Regular, Wild Card, Division, League, World Series
        }

        try:
            logger.info(f"Fetching MLB scores for {date_str}")
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch MLB scores for {date_str}: {e}")
            return []

        results = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                parsed = self._parse_game(game)
                if parsed:
                    results.append(parsed)

        logger.info(f"Found {len(results)} completed MLB games on {date_str}")
        return results

    def _parse_game(self, game: Dict) -> Optional[Dict]:
        """Parse a single game entry from the MLB Stats API schedule response."""
        try:
            status = game.get("status", {}).get("detailedState", "")
            # "Final" covers regular and extra-inning games ("Final: 10 Innings")
            if not status.startswith("Final"):
                return None

            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})

            home_name = home.get("team", {}).get("name", "")
            away_name = away.get("team", {}).get("name", "")
            home_score = home.get("score")
            away_score = away.get("score")

            if home_score is None or away_score is None:
                return None

            return {
                "home_team": home_name,
                "away_team": away_name,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "completed": True,
            }
        except Exception as e:
            logger.error(f"Failed to parse MLB game data: {e}")
            return None

    def get_scores_for_date_range(self, start_date: datetime, days: int = 1) -> List[Dict]:
        """Get all completed game scores for a date range."""
        all_games = []
        for i in range(days):
            date = start_date + timedelta(days=i)
            all_games.extend(self.get_scores_for_date(date))
        return all_games

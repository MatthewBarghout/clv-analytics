"""
NBA Score Fetcher

Fetches final scores from ESPN's public scoreboard API.
Free, no authentication required, accepts a real date parameter.

Replaced cdn.nba.com (2026-09-15): that endpoint only ever served *today's*
scoreboard — the previous client passed a date it then ignored, so historical
scores were never actually retrievable — and it now returns 403 outright.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class NBAScoresClient:
    """Client for fetching NBA game scores from ESPN's public scoreboard API."""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

    def __init__(self):
        # Deliberately no User-Agent override. ESPN 403s browser-spoofing and custom
        # agents on this endpoint and allows honest tool agents, so requests' own
        # default (python-requests/x.y.z) is what gets through. Do not "fix" this by
        # adding a Mozilla string.
        self.session = requests.Session()

    def _fetch_slate(self, date: datetime) -> List[Dict]:
        """Fetch one ESPN game-day slate. Returns raw event dicts."""
        try:
            response = self.session.get(
                self.BASE_URL, params={"dates": date.strftime("%Y%m%d")}, timeout=15
            )
            response.raise_for_status()
            return response.json().get("events", []) or []
        except requests.RequestException as e:
            logger.error(f"Failed to fetch NBA scores for {date.strftime('%Y-%m-%d')}: {e}")
            return []
        except ValueError as e:
            logger.error(f"Malformed NBA response for {date.strftime('%Y-%m-%d')}: {e}")
            return []

    def parse_game_score(self, event: Dict) -> Optional[Dict]:
        """Normalize one ESPN event to {home_team, away_team, home_score, away_score, completed}."""
        try:
            competition = (event.get("competitions") or [{}])[0]
            status = competition.get("status", {}).get("type", {})
            competitors = competition.get("competitors") or []

            sides = {c.get("homeAway"): c for c in competitors}
            home, away = sides.get("home"), sides.get("away")
            if not home or not away:
                return None

            home_score, away_score = home.get("score"), away.get("score")
            if home_score is None or away_score is None:
                return None

            # Emit the nickname ("Clippers"), not displayName. The caller matches by
            # substring against full DB names, and ESPN abbreviates some locations
            # ("LA Clippers" vs. our "Los Angeles Clippers"), which defeats that.
            # Nicknames are unique across all 30 teams and are always a substring.
            return {
                "home_team": home.get("team", {}).get("name")
                or home.get("team", {}).get("displayName", ""),
                "away_team": away.get("team", {}).get("name")
                or away.get("team", {}).get("displayName", ""),
                "home_score": int(home_score),
                "away_score": int(away_score),
                "completed": bool(status.get("completed")),
                # Tip-off date (UTC) — required to pick the right game when the same
                # two teams meet on nearby dates.
                "game_date": (event.get("date") or "")[:10],
            }
        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"Could not parse NBA event: {e}")
            return None

    def get_scores_for_date(self, date: datetime) -> List[Dict]:
        """
        Get all completed games for a specific date.

        ESPN slates are keyed by US game day, so an evening tip-off lands on the
        *following* UTC date. Callers bucket games by the UTC date of commence_time,
        which can fall either side of the slate, so the neighbouring days are fetched
        and merged too.

        Returns list of dicts: {home_team, away_team, home_score, away_score, completed}
        """
        events = (
            self._fetch_slate(date - timedelta(days=1))
            + self._fetch_slate(date)
            + self._fetch_slate(date + timedelta(days=1))
        )

        seen = set()
        games = []
        for event in events:
            event_id = event.get("id")
            if event_id in seen:
                continue
            seen.add(event_id)

            parsed = self.parse_game_score(event)
            if parsed and parsed["completed"]:
                games.append(parsed)

        logger.info(f"Found {len(games)} completed games on {date.strftime('%Y-%m-%d')}")
        return games

    def get_scores_for_date_range(self, start_date: datetime, days: int = 1) -> List[Dict]:
        """Get all completed game scores for a date range."""
        all_games = []
        for i in range(days):
            all_games.extend(self.get_scores_for_date(start_date + timedelta(days=i)))
        return all_games

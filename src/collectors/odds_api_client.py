import logging
import time
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import ConnectionError, Timeout, RequestException

logger = logging.getLogger(__name__)


class OddsAPIClient:
    """
    Client for The Odds API.

    Responsibilities:
    - Fetch sports and odds data
    - Rate limiting (500 req/month)
    - Error handling & retries
    - Transform API response to database models
    """

    def __init__(self, api_key: str, base_url: str = "https://api.the-odds-api.com"):
        """
        Initialize the Odds API client.

        Args:
            api_key: The Odds API key
            base_url: Base URL for The Odds API (default: https://api.the-odds-api.com)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = self._setup_session()

    def _setup_session(self) -> requests.Session:
        """
        Setup requests session with retry logic.

        Implements exponential backoff for transient failures:
        - 5 retries total (increased from 3)
        - Backoff factor of 1 (1s, 2s, 4s, 8s, 16s)
        - Retry on 500, 502, 503, 504 status codes
        - Retry on connection and DNS errors
        """
        session = requests.Session()

        # Configure retry strategy with connection error handling
        retry_strategy = Retry(
            total=5,  # More retries for network issues
            backoff_factor=1,  # Longer waits (1s, 2s, 4s, 8s, 16s)
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,  # Don't raise on HTTP errors immediately
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def get_sports(self) -> dict:
        """
        Fetch available sports from The Odds API.

        Returns:
            Dictionary with:
                - 'data': List of sport dictionaries
                - 'remaining_requests': Number of API requests remaining

        Example response:
            {
                "data": [
                    {
                        "key": "basketball_nba",
                        "group": "Basketball",
                        "title": "NBA",
                        "description": "US Basketball",
                        "active": true,
                        "has_outrights": false
                    }
                ],
                "remaining_requests": 450
            }
        """
        endpoint = "/v4/sports"
        params = {"apiKey": self.api_key}

        return self._make_request(endpoint, params)

    def get_odds(
        self,
        sport: str,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        bookmakers: Optional[List[str]] = None,
    ) -> dict:
        """
        Fetch odds for a specific sport.

        Args:
            sport: Sport key (e.g., 'basketball_nba', 'americanfootball_nfl')
            regions: Comma-separated regions (default: 'us')
            markets: Comma-separated markets (default: 'h2h,spreads,totals')
            bookmakers: Optional list of specific bookmakers to filter

        Returns:
            Dictionary with:
                - 'data': List of games with odds
                - 'remaining_requests': Number of API requests remaining

        Example response structure:
            {
                "data": [
                    {
                        "id": "game_id",
                        "sport_key": "basketball_nba",
                        "sport_title": "NBA",
                        "commence_time": "2024-01-15T00:00:00Z",
                        "home_team": "Los Angeles Lakers",
                        "away_team": "Boston Celtics",
                        "bookmakers": [
                            {
                                "key": "fanduel",
                                "title": "FanDuel",
                                "markets": [
                                    {
                                        "key": "h2h",
                                        "outcomes": [
                                            {"name": "Lakers", "price": -150},
                                            {"name": "Celtics", "price": 130}
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ],
                "remaining_requests": 450
            }
        """
        endpoint = f"/v4/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
        }

        # Add specific bookmakers if provided
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        return self._make_request(endpoint, params)

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """
        Make API request with error handling and rate limit checking.

        Includes manual retry logic for connection/DNS errors that urllib3 doesn't handle.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Dictionary with 'data' and 'remaining_requests' keys

        Raises:
            requests.exceptions.HTTPError: For 4xx/5xx responses
            requests.exceptions.RequestException: For network errors after all retries
        """
        url = f"{self.base_url}{endpoint}"
        max_retries = 5
        base_delay = 2  # Start with 2 seconds

        for attempt in range(max_retries):
            try:
                logger.info(f"Making request to {endpoint} (attempt {attempt + 1}/{max_retries})")
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()

                # Check rate limit from headers
                remaining = response.headers.get("x-requests-remaining")
                if remaining:
                    remaining_int = int(remaining)
                    self._check_rate_limit(remaining_int)
                    logger.info(f"API requests remaining: {remaining_int}")

                # Wrap response in consistent format
                # The Odds API returns JSON directly (list or object depending on endpoint)
                # We wrap it with metadata for consistent access to rate limit info
                return {
                    "data": response.json(),
                    "remaining_requests": int(remaining) if remaining else None,
                }

            except requests.exceptions.HTTPError as e:
                # Don't retry client errors (4xx)
                logger.error(f"HTTP error for {endpoint}: {e}")
                if e.response.status_code == 401:
                    logger.error("Invalid API key - check your ODDS_API_KEY in .env")
                elif e.response.status_code == 429:
                    logger.error("Rate limit exceeded - wait before making more requests")
                raise

            except (ConnectionError, Timeout) as e:
                # Retry on connection/DNS/timeout errors
                delay = base_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s, 16s, 32s

                if attempt < max_retries - 1:
                    logger.warning(
                        f"Connection error for {endpoint}: {e}. "
                        f"Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Connection error for {endpoint} after {max_retries} attempts: {e}"
                    )
                    raise

            except RequestException as e:
                # Other request errors - retry with backoff
                delay = base_delay * (2 ** attempt)

                if attempt < max_retries - 1:
                    logger.warning(
                        f"Request error for {endpoint}: {e}. "
                        f"Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Request failed for {endpoint} after {max_retries} attempts: {e}"
                    )
                    raise

        # Should never reach here, but just in case
        raise RequestException(f"Failed to make request to {endpoint} after {max_retries} attempts")

    def _check_rate_limit(self, remaining: int) -> None:
        """
        Log warning if approaching rate limit.

        Args:
            remaining: Number of requests remaining in current quota
        """
        if remaining < 100:
            logger.warning(
                f"Low API quota remaining: {remaining} requests left. "
                "Consider reducing polling frequency."
            )
        elif remaining < 50:
            logger.error(
                f"CRITICAL: Only {remaining} API requests remaining! "
                "Rate limit will reset at the start of next month."
            )

    def get_scores(self, sport_key: str) -> dict:
        """
        Get scores for completed games.

        Args:
            sport_key: Sport identifier (e.g., 'basketball_nba')

        Returns:
            Dictionary with 'data' (list of games with scores) and 'remaining_requests'

        Note:
            The scores endpoint returns recent completed games (typically last 3 days)
        """
        endpoint = f"/v4/sports/{sport_key}/scores/"
        params = {
            "apiKey": self.api_key,
        }
        return self._make_request(endpoint, params)

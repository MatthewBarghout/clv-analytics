"""Kalshi prediction market REST API client.

Fetches sports-related markets from Kalshi and returns implied odds
for arbitrage comparison against traditional sportsbooks.

Auth: API key via Authorization header (set KALSHI_API_KEY env var).
API: https://trading-api.kalshi.com/trade-api/v2
"""
import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

KALSHI_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
REQUEST_TIMEOUT = 10  # seconds


class KalshiClient:
    """Client for the Kalshi prediction market API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("KALSHI_API_KEY", "")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self.session.headers.update({"Content-Type": "application/json"})

    def _get(self, path: str, params: Dict[str, Any] = None) -> Dict:
        """Make a GET request to the Kalshi API."""
        url = f"{KALSHI_BASE_URL}{path}"
        try:
            resp = self.session.get(url, params=params or {}, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                logger.warning("Kalshi: unauthorized (check KALSHI_API_KEY). Returning empty.")
                return {}
            logger.error(f"Kalshi HTTP error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Kalshi request failed: {e}")
            raise

    def get_sports_markets(self, limit: int = 200) -> List[Dict]:
        """
        Fetch active sports markets from Kalshi.

        Returns a list of market dicts with:
          - ticker, title, yes_bid, yes_ask, no_bid, no_ask, close_time
        """
        try:
            data = self._get("/markets", params={
                "limit": limit,
                "status": "open",
                "series_ticker": None,
            })
            markets = data.get("markets", [])

            sports_markets = []
            for m in markets:
                # Filter for sports-related markets by checking category or title keywords
                category = (m.get("category") or "").lower()
                title = (m.get("title") or "").lower()
                if any(kw in category or kw in title for kw in
                       ["nba", "nfl", "mlb", "nhl", "soccer", "sport", "football",
                        "basketball", "baseball", "hockey", "tennis", "golf"]):
                    sports_markets.append(m)

            logger.info(f"Kalshi: fetched {len(sports_markets)} sports markets")
            return sports_markets

        except Exception as e:
            logger.error(f"Error fetching Kalshi sports markets: {e}")
            return []

    def parse_market_odds(self, market: Dict) -> Optional[Dict]:
        """
        Parse a Kalshi market dict into a standardized odds record.

        Returns dict with:
          - title, ticker, yes_implied_prob, no_implied_prob,
            yes_implied_odds, no_implied_odds, market_url, close_time
        Returns None if insufficient data.
        """
        try:
            yes_bid = market.get("yes_bid")  # cents (0-100)
            no_bid = market.get("no_bid")

            if yes_bid is None or no_bid is None:
                return None

            # Convert cents to probability (0.0 - 1.0)
            yes_prob = float(yes_bid) / 100.0
            no_prob = float(no_bid) / 100.0

            if yes_prob <= 0 or no_prob <= 0:
                return None

            return {
                "title": market.get("title", ""),
                "ticker": market.get("ticker", ""),
                "yes_implied_prob": yes_prob,
                "no_implied_prob": no_prob,
                "yes_implied_odds": round(1.0 / yes_prob, 4),
                "no_implied_odds": round(1.0 / no_prob, 4),
                "market_url": f"https://kalshi.com/markets/{market.get('ticker', '')}",
                "close_time": market.get("close_time"),
            }
        except Exception as e:
            logger.debug(f"Kalshi parse error for market {market.get('ticker')}: {e}")
            return None

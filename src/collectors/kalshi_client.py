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

# Known Kalshi sports series tickers — fetched directly rather than
# scanning all open markets for keyword matches.
SPORTS_SERIES = [
    "KXNBA",     # NBA game outcomes
    "KXNFL",     # NFL game outcomes
    "KXMLB",     # MLB game outcomes
    "KXNHL",     # NHL game outcomes
    "KXNCAAB",   # College basketball
    "KXNCAAF",   # College football
    "KXSOCCER",  # Soccer
    "KXMMA",     # MMA/UFC
    "KXNBAPTS",  # NBA player points props
    "KXNFLPTS",  # NFL player points props
]


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

        First attempts targeted fetch by known sports series tickers.
        Falls back to keyword-filtered scan of all open markets if series
        fetch yields nothing (e.g. off-season).

        Returns a list of market dicts.
        """
        markets = self._fetch_by_series()
        if markets:
            return markets
        return self._fetch_by_keyword(limit=limit)

    def _fetch_by_series(self) -> List[Dict]:
        """Fetch markets from known sports series tickers directly."""
        all_markets: List[Dict] = []
        for series_ticker in SPORTS_SERIES:
            try:
                data = self._get(f"/series/{series_ticker}/markets", params={
                    "status": "open",
                    "limit": 100,
                })
                markets = data.get("markets", [])
                all_markets.extend(markets)
                logger.debug(f"Kalshi series {series_ticker}: {len(markets)} markets")
            except Exception as e:
                logger.debug(f"Kalshi series {series_ticker} unavailable: {e}")
                continue

        if all_markets:
            logger.info(f"Kalshi: fetched {len(all_markets)} markets via series tickers")
        return all_markets

    def _fetch_by_keyword(self, limit: int = 200) -> List[Dict]:
        """Fallback: scan all open markets and filter by sports keywords."""
        try:
            data = self._get("/markets", params={
                "limit": limit,
                "status": "open",
            })
            markets = data.get("markets", [])

            sports_markets = []
            for m in markets:
                category = (m.get("category") or "").lower()
                title = (m.get("title") or "").lower()
                if any(kw in category or kw in title for kw in
                       ["nba", "nfl", "mlb", "nhl", "soccer", "sport", "football",
                        "basketball", "baseball", "hockey", "tennis", "golf", "mma", "ufc"]):
                    sports_markets.append(m)

            logger.info(f"Kalshi: fetched {len(sports_markets)} sports markets via keyword scan")
            return sports_markets

        except Exception as e:
            logger.error(f"Error fetching Kalshi markets by keyword: {e}")
            return []

    def get_market(self, ticker: str) -> Optional[Dict]:
        """Fetch a single market by its ticker. Returns the market dict or None."""
        try:
            data = self._get(f"/markets/{ticker}")
            return data.get("market")
        except Exception as e:
            logger.error(f"Kalshi: failed to fetch market {ticker}: {e}")
            return None

    def parse_market_odds(self, market: Dict) -> Optional[Dict]:
        """
        Parse a Kalshi market dict into a standardized odds record.

        Uses midpoint of bid/ask for a more accurate probability estimate
        than bid alone. Falls back to bid-only if ask is unavailable.

        Returns dict with:
          - title, ticker, yes_implied_prob, no_implied_prob,
            yes_implied_odds, no_implied_odds, market_url, close_time,
            volume, open_interest
        Returns None if insufficient data.
        """
        try:
            yes_bid = market.get("yes_bid")
            yes_ask = market.get("yes_ask")
            no_bid = market.get("no_bid")
            no_ask = market.get("no_ask")

            if yes_bid is None or no_bid is None:
                return None

            # Midpoint of bid/ask gives a cleaner probability estimate
            yes_cents = (float(yes_bid) + float(yes_ask)) / 2.0 if yes_ask is not None else float(yes_bid)
            no_cents = (float(no_bid) + float(no_ask)) / 2.0 if no_ask is not None else float(no_bid)

            yes_prob = yes_cents / 100.0
            no_prob = no_cents / 100.0

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
                "volume": market.get("volume", 0),
                "open_interest": market.get("open_interest", 0),
            }
        except Exception as e:
            logger.debug(f"Kalshi parse error for market {market.get('ticker')}: {e}")
            return None

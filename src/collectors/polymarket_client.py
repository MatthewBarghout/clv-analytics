"""Polymarket CLOB API client.

Fetches sports-related prediction markets from Polymarket and returns
implied odds for arbitrage comparison. Public read — no auth required.

API: https://clob.polymarket.com
"""
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

POLYMARKET_CLOB_URL = "https://clob.polymarket.com"
REQUEST_TIMEOUT = 10  # seconds

SPORTS_KEYWORDS = [
    "nba", "nfl", "mlb", "nhl", "soccer", "mls", "epl", "football",
    "basketball", "baseball", "hockey", "tennis", "golf", "ufc", "mma",
    "super bowl", "world series", "stanley cup", "nba finals", "championship",
]


class PolymarketClient:
    """Client for the Polymarket CLOB API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _get(self, path: str, params: Dict[str, Any] = None) -> Any:
        """Make a GET request to the Polymarket CLOB API."""
        url = f"{POLYMARKET_CLOB_URL}{path}"
        try:
            resp = self.session.get(url, params=params or {}, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Polymarket request failed: {e}")
            raise

    def get_sports_markets(self, next_cursor: str = None, max_pages: int = 5) -> List[Dict]:
        """
        Fetch active sports markets from Polymarket.

        Paginates up to `max_pages` pages and filters by sports keywords.
        Returns a flat list of market dicts.
        """
        all_markets: List[Dict] = []
        cursor = next_cursor

        for page in range(max_pages):
            try:
                params = {"next_cursor": cursor} if cursor else {}
                data = self._get("/markets", params=params)

                if isinstance(data, list):
                    markets = data
                    cursor = None
                else:
                    markets = data.get("data", [])
                    cursor = data.get("next_cursor")

                for m in markets:
                    question = (m.get("question") or "").lower()
                    if any(kw in question for kw in SPORTS_KEYWORDS):
                        if m.get("active") and not m.get("closed"):
                            all_markets.append(m)

                if not cursor:
                    break

            except Exception as e:
                logger.error(f"Polymarket page {page} fetch error: {e}")
                break

        logger.info(f"Polymarket: fetched {len(all_markets)} active sports markets")
        return all_markets

    def parse_market_odds(self, market: Dict) -> Optional[Dict]:
        """
        Parse a Polymarket market dict into standardized odds records.

        Each binary market has two outcomes (YES/NO). Returns the YES
        implied probability, treating the higher-priced outcome as the
        'favored' side.

        Returns dict or None if insufficient data.
        """
        try:
            tokens = market.get("tokens", [])
            if len(tokens) < 2:
                return None

            # Find YES token (outcome = 'Yes') and NO token
            yes_token = next((t for t in tokens if t.get("outcome", "").lower() == "yes"), tokens[0])
            no_token = next((t for t in tokens if t.get("outcome", "").lower() == "no"), tokens[1])

            best_ask_yes = yes_token.get("price")  # Price to buy YES (0.0-1.0)
            best_ask_no = no_token.get("price")

            if best_ask_yes is None or best_ask_no is None:
                return None

            yes_prob = float(best_ask_yes)
            no_prob = float(best_ask_no)

            if yes_prob <= 0 or no_prob <= 0:
                return None

            condition_id = market.get("condition_id", "")
            return {
                "title": market.get("question", ""),
                "condition_id": condition_id,
                "yes_implied_prob": yes_prob,
                "no_implied_prob": no_prob,
                "yes_implied_odds": round(1.0 / yes_prob, 4),
                "no_implied_odds": round(1.0 / no_prob, 4),
                "market_url": f"https://polymarket.com/event/{condition_id}",
                "end_date": market.get("end_date_iso"),
            }
        except Exception as e:
            logger.debug(f"Polymarket parse error for market {market.get('condition_id')}: {e}")
            return None

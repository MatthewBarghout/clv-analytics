"""Polymarket prediction market client.

Uses the public Gamma API — no auth required.
API: https://gamma-api.polymarket.com

Markets have binary YES/NO outcomes. outcomePrices[0] = YES price (0.0-1.0).
Focuses on sports futures and game-level markets.
"""
import json
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
REQUEST_TIMEOUT = 10

SPORTS_KEYWORDS = [
    "nba", "nfl", "mlb", "nhl", "mls", "ufc", "mma",
    "basketball", "football", "baseball", "hockey", "soccer",
    "playoffs", "championship", "finals", "stanley cup", "super bowl",
    "world series", "tennis", "golf",
]


class PolymarketClient:
    """Client for the Polymarket Gamma API (public, no auth)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _get(self, path: str, params: Dict[str, Any] = None) -> Any:
        url = f"{GAMMA_BASE_URL}{path}"
        try:
            resp = self.session.get(url, params=params or {}, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Polymarket request failed: {e}")
            raise

    def get_sports_markets(self, max_pages: int = 5, limit: int = 100) -> List[Dict]:
        """
        Fetch active sports markets from Polymarket.

        Paginates up to max_pages and filters by sports keywords.
        Returns list of raw market dicts.
        """
        all_markets: List[Dict] = []
        offset = 0

        for _ in range(max_pages):
            try:
                data = self._get("/markets", {
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "offset": offset,
                })
                if not isinstance(data, list):
                    break

                sports = [
                    m for m in data
                    if any(kw in (m.get("question") or "").lower() for kw in SPORTS_KEYWORDS)
                ]
                all_markets.extend(sports)

                if len(data) < limit:
                    break
                offset += limit

            except Exception as e:
                logger.error(f"Polymarket fetch error at offset {offset}: {e}")
                break

        logger.info(f"Polymarket: fetched {len(all_markets)} active sports markets")
        return all_markets

    def parse_market_odds(self, market: Dict) -> Optional[Dict]:
        """
        Parse a Polymarket market dict into a standardized odds record.

        outcomePrices[0] = YES price, outcomePrices[1] = NO price (strings, 0.0-1.0).
        Returns None if pricing is missing or market has no liquidity.
        """
        try:
            outcome_prices = market.get("outcomePrices")
            outcomes = market.get("outcomes")
            # API returns these as JSON strings — parse if needed
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            liquidity = float(market.get("liquidityNum") or market.get("liquidity") or 0)

            if not outcome_prices or len(outcome_prices) < 2:
                return None
            if liquidity < 100:  # skip dust markets
                return None

            # Find YES index
            yes_idx = 0
            if outcomes:
                yes_idx = next(
                    (i for i, o in enumerate(outcomes) if str(o).lower() == "yes"), 0
                )
            no_idx = 1 - yes_idx

            yes_prob = float(outcome_prices[yes_idx])
            no_prob = float(outcome_prices[no_idx])

            if yes_prob <= 0 or no_prob <= 0:
                return None

            condition_id = market.get("conditionId", "")
            slug = market.get("slug", "")

            return {
                "title": market.get("question", ""),
                "condition_id": condition_id,
                "yes_implied_prob": yes_prob,
                "no_implied_prob": no_prob,
                "yes_implied_odds": round(1.0 / yes_prob, 4),
                "no_implied_odds": round(1.0 / no_prob, 4),
                "market_url": f"https://polymarket.com/event/{slug}",
                "liquidity": liquidity,
                "volume": float(market.get("volume") or 0),
            }
        except Exception as e:
            logger.debug(f"Polymarket parse error for {market.get('question', '')[:50]}: {e}")
            return None

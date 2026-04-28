"""Kalshi prediction market REST API client.

Auth: RSA-PSS key pair. Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY env vars.
API: https://api.elections.kalshi.com/trade-api/v2

Fetch flow: series → events → markets (per event).
Sports series covered: NBA/MLB/NHL game winners, NBA points/rebounds props.
Non-sports series: crypto, politics, economics.
"""
import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as apadding
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
REQUEST_TIMEOUT = 10

# Series tickers → event-level markets (game outcomes and props)
SPORTS_SERIES = [
    "KXNBAGAME",  # NBA game winner (playoffs/regular season)
    "KXMLBGAME",  # MLB game winner
    "KXNHLGAME",  # NHL game winner
    "KXNFLGAME",  # NFL game winner
    "KXNBAPTS",   # NBA player points props
    "KXNBAREB",   # NBA player rebounds props
    "KXMMA",      # MMA/UFC
    "KXSOCCER",   # Soccer match winner
]

# Championship/futures series — these have Polymarket equivalents and are
# suitable for cross-platform signal generation
CHAMPIONSHIP_SERIES = [
    "KXNBA",   # NBA championship winner
    "KXMLB",   # MLB World Series winner
    "KXNHL",   # NHL Stanley Cup winner
    "KXNFL",   # NFL Super Bowl winner
]

NON_SPORTS_SERIES = [
    "KXBTC",
    "KXETH",
    "KXPRES",
    "KXFED",
    "KXHOUSE",
    "KXSENATE",
    "KXECON",
]

# Game-level series not suitable for cross-platform signal matching
_GAME_LEVEL_SERIES = {
    "KXNBAGAME", "KXMLBGAME", "KXNHLGAME", "KXNFLGAME",
    "KXNBAPTS", "KXNBAREB", "KXMMA", "KXSOCCER",
}


def _load_private_key(pem: str):
    pem = pem.replace("\\n", "\n").strip()
    if not pem.startswith("-----"):
        pem = f"-----BEGIN RSA PRIVATE KEY-----\n{pem}\n-----END RSA PRIVATE KEY-----"
    return serialization.load_pem_private_key(pem.encode(), password=None)


class KalshiClient:
    """Kalshi API client with RSA-PSS request signing."""

    def __init__(self, key_id: Optional[str] = None, private_key_pem: Optional[str] = None):
        self.key_id = key_id or os.getenv("KALSHI_KEY_ID", "")
        pem = private_key_pem or os.getenv("KALSHI_PRIVATE_KEY", "")
        self._private_key = _load_private_key(pem) if pem else None
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """RSA-PSS signed headers for Kalshi v2 auth."""
        if not self._private_key or not self.key_id:
            return {}
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + f"/trade-api/v2{path}").encode("utf-8")
        sig = self._private_key.sign(
            msg,
            apadding.PSS(
                mgf=apadding.MGF1(hashes.SHA256()),
                salt_length=apadding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-Access-Key": self.key_id,
            "KALSHI-Access-Timestamp": ts,
            "KALSHI-Access-Signature": base64.b64encode(sig).decode(),
        }

    def _get(self, path: str, params: Dict[str, Any] = None, _retries: int = 3) -> Dict:
        url = f"{KALSHI_BASE_URL}{path}"
        for attempt in range(_retries):
            try:
                resp = self.session.get(
                    url,
                    params=params or {},
                    headers=self._auth_headers("GET", path),
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 401:
                    logger.warning("Kalshi: unauthorized — check KALSHI_KEY_ID and KALSHI_PRIVATE_KEY.")
                    return {}
                if e.response is not None and e.response.status_code == 429:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"Kalshi 429 on {path}, retrying in {wait}s (attempt {attempt + 1}/{_retries})")
                    time.sleep(wait)
                    continue
                logger.error(f"Kalshi HTTP error on {path}: {e}")
                raise
            except requests.exceptions.RequestException as e:
                logger.error(f"Kalshi request failed on {path}: {e}")
                raise
        logger.error(f"Kalshi: gave up on {path} after {_retries} retries (429)")
        return {}

    def _get_events_for_series(self, series_ticker: str) -> List[Dict]:
        """Return open events for a series ticker."""
        try:
            data = self._get("/events", {"series_ticker": series_ticker, "status": "open", "limit": 50})
            return data.get("events", [])
        except Exception as e:
            logger.debug(f"Kalshi series {series_ticker} unavailable: {e}")
            return []

    def _get_markets_for_event(self, event_ticker: str) -> List[Dict]:
        """Return open markets within an event."""
        try:
            data = self._get("/markets", {"event_ticker": event_ticker, "status": "open", "limit": 50})
            return data.get("markets", [])
        except Exception as e:
            logger.debug(f"Kalshi event {event_ticker} markets unavailable: {e}")
            return []

    def get_all_markets(self, series_list: List[str] = None) -> List[Dict]:
        """Fetch active markets across all series (sports + non-sports).

        Defaults to SPORTS_SERIES + CHAMPIONSHIP_SERIES + NON_SPORTS_SERIES.
        Sleeps 0.5s between each series to avoid rate limiting.
        """
        if series_list is None:
            series_list = SPORTS_SERIES + CHAMPIONSHIP_SERIES + NON_SPORTS_SERIES
        all_markets: List[Dict] = []
        for series in series_list:
            events = self._get_events_for_series(series)
            count = 0
            for event in events:
                event_ticker = event.get("event_ticker", "")
                markets = self._get_markets_for_event(event_ticker)
                for m in markets:
                    m["_series"] = series
                    m["_event_title"] = event.get("title", "")
                all_markets.extend(markets)
                count += len(markets)
                time.sleep(0.25)
            logger.debug(f"Kalshi {series}: {count} markets")
            time.sleep(0.5)
        logger.info(f"Kalshi get_all_markets: {len(all_markets)} markets total")
        return all_markets

    def get_sports_markets(self, limit: int = 200) -> List[Dict]:
        """
        Fetch active sports markets from Kalshi.

        Iterates series → events → markets. Falls back to keyword scan
        of all open markets if no series events are found.
        """
        markets = self._fetch_by_series()
        if markets:
            return markets
        return self._fetch_by_keyword(limit=limit)

    def is_game_level(self, series: str) -> bool:
        """Return True if this series contains individual game/match markets."""
        return series in _GAME_LEVEL_SERIES

    def _fetch_by_series(self) -> List[Dict]:
        """Fetch markets via series → events → markets chain (game + championship series)."""
        all_markets: List[Dict] = []
        for series in SPORTS_SERIES + CHAMPIONSHIP_SERIES:
            events = self._get_events_for_series(series)
            for event in events:
                event_ticker = event.get("event_ticker", "")
                markets = self._get_markets_for_event(event_ticker)
                for m in markets:
                    m["_series"] = series
                    m["_event_title"] = event.get("title", "")
                all_markets.extend(markets)
            if events:
                logger.debug(f"Kalshi {series}: {len(events)} events")

        if all_markets:
            logger.info(f"Kalshi: fetched {len(all_markets)} markets via series/events")
        return all_markets

    def _fetch_by_keyword(self, limit: int = 200) -> List[Dict]:
        """Fallback: scan all open markets and filter by sports keywords."""
        try:
            data = self._get("/markets", {"limit": limit, "status": "open"})
            markets = data.get("markets", [])
            kws = ["nba", "nfl", "mlb", "nhl", "soccer", "basketball", "football",
                   "baseball", "hockey", "tennis", "golf", "mma", "ufc", "sport"]
            sports = [
                m for m in markets
                if any(kw in (m.get("title", "") + m.get("event_ticker", "")).lower() for kw in kws)
            ]
            logger.info(f"Kalshi: fetched {len(sports)} sports markets via keyword fallback")
            return sports
        except Exception as e:
            logger.error(f"Kalshi keyword fallback failed: {e}")
            return []

    def get_market(self, ticker: str) -> Optional[Dict]:
        """Fetch a single market by ticker."""
        try:
            data = self._get(f"/markets/{ticker}")
            return data.get("market")
        except Exception as e:
            logger.error(f"Kalshi: failed to fetch market {ticker}: {e}")
            return None

    def parse_market_odds(self, market: Dict) -> Optional[Dict]:
        """
        Parse a Kalshi market into a standardized odds record.

        Uses bid/ask midpoint for probability. Returns None if unpriced.
        Title prefers the event title (home vs away) over the market title
        so fuzzy matching against sportsbook event names works better.
        """
        try:
            yes_bid = market.get("yes_bid_dollars")
            yes_ask = market.get("yes_ask_dollars")
            no_bid = market.get("no_bid_dollars")
            no_ask = market.get("no_ask_dollars")
            last_price = market.get("last_price_dollars")

            if yes_bid is not None and no_bid is not None:
                yes_prob = (float(yes_bid) + float(yes_ask)) / 2.0 if yes_ask is not None else float(yes_bid)
                no_prob = (float(no_bid) + float(no_ask)) / 2.0 if no_ask is not None else float(no_bid)
            elif last_price is not None:
                yes_prob = float(last_price)
                no_prob = 1.0 - yes_prob
            else:
                return None

            if yes_prob <= 0 or no_prob <= 0:
                return None

            # Prefer event title for matching (e.g. "Lakers at Warriors")
            title = market.get("_event_title") or market.get("title", "")

            return {
                "title": title,
                "market_title": market.get("title", ""),
                "ticker": market.get("ticker", ""),
                "series": market.get("_series", ""),
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
            logger.debug(f"Kalshi parse error for {market.get('ticker')}: {e}")
            return None

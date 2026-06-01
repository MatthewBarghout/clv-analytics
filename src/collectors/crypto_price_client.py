"""Crypto price fetcher using CoinGecko public API (no auth required)."""
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_CACHE: dict = {}
_CACHE_TTL = 300  # 5 minutes


def get_crypto_price(coingecko_id: str) -> Optional[float]:
    """Return current USD price for a CoinGecko asset ID. Cached for 5 minutes."""
    now = time.time()
    if coingecko_id in _CACHE:
        price, ts = _CACHE[coingecko_id]
        if now - ts < _CACHE_TTL:
            return price

    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": coingecko_id, "vs_currencies": "usd"},
            timeout=5,
        )
        resp.raise_for_status()
        price = float(resp.json()[coingecko_id]["usd"])
        _CACHE[coingecko_id] = (price, now)
        logger.debug(f"CoinGecko {coingecko_id}: ${price:,.2f}")
        return price
    except Exception as e:
        logger.warning(f"CoinGecko price fetch failed for {coingecko_id}: {e}")
        return None

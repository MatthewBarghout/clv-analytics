"""Cross-platform prediction market signal generator.

Compares Kalshi prices against Polymarket forecasts to find
markets where the implied probability diverges significantly from consensus.
"""
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Minimum edge (probability points) to generate a signal
EDGE_THRESHOLD = 0.06

# Quarter-Kelly bankroll used for sizing
BANKROLL = 1000.0
KELLY_FRACTION = 0.25
MIN_SIZE_USD = 25.0
MAX_SIZE_USD = 200.0

# Minimum similarity score to accept a Polymarket match
SIMILARITY_THRESHOLD = 0.45

# Source weights for fair value calculation
_WEIGHTS = {
    "polymarket": 0.45,
    "metaculus": 0.35,
    "kalshi": 0.20,
}


class PMSignalGenerator:
    """Generates trading signals by comparing Kalshi prices to Polymarket forecasts.

    Fetches Polymarket sports markets once on first use and caches them for the
    lifetime of the instance. Call refresh_poly_cache() to force a reload.
    """

    def __init__(self):
        from src.collectors.polymarket_client import PolymarketClient
        from src.collectors.metaculus_client import MetaculusClient
        self._poly = PolymarketClient()
        self._meta = MetaculusClient()
        self._poly_cache: List[dict] = []

    def refresh_poly_cache(self) -> None:
        """Fetch and cache active Polymarket sports markets for local matching."""
        try:
            raw = self._poly.get_sports_markets(max_pages=5, limit=100)
            parsed = [self._poly.parse_market_odds(m) for m in raw]
            self._poly_cache = [p for p in parsed if p is not None]
            logger.info(f"Polymarket cache refreshed: {len(self._poly_cache)} sports markets")
        except Exception as e:
            logger.error(f"Polymarket cache refresh failed: {e}")
            self._poly_cache = []

    def _match_polymarket(self, question: str) -> Optional[float]:
        """Find the best-matching Polymarket market for a Kalshi question title.

        Returns the YES price of the best match if similarity >= SIMILARITY_THRESHOLD,
        else None.
        """
        if not self._poly_cache:
            self.refresh_poly_cache()
        if not self._poly_cache:
            return None

        best_sim = 0.0
        best_price = None
        for m in self._poly_cache:
            sim = _similarity(question, m.get("title", ""))
            if sim > best_sim:
                best_sim = sim
                best_price = m.get("yes_implied_prob")

        if best_sim >= SIMILARITY_THRESHOLD and best_price is not None:
            logger.debug(f"Polymarket match accepted (sim={best_sim:.2f}) for '{question[:50]}'")
            return float(best_price)

        logger.debug(f"Polymarket match rejected (best_sim={best_sim:.2f}) for '{question[:50]}'")
        return None

    def fair_value(
        self,
        kalshi_price: float,
        polymarket_price: Optional[float],
        metaculus_forecast: Optional[float],
    ) -> float:
        """Compute weighted fair value from available sources.

        Missing sources have their weight redistributed proportionally.
        Returns float in [0, 1].
        """
        sources: dict[str, float] = {"kalshi": kalshi_price}
        if polymarket_price is not None:
            sources["polymarket"] = polymarket_price
        if metaculus_forecast is not None:
            sources["metaculus"] = metaculus_forecast

        total_weight = sum(_WEIGHTS[k] for k in sources)
        fv = sum(sources[k] * (_WEIGHTS[k] / total_weight) for k in sources)
        return float(fv)

    def kelly_size(self, edge: float, price: float) -> float:
        """Quarter-Kelly position size in USD, clamped to [25, 200]."""
        if price >= 1.0 or price <= 0.0:
            return MIN_SIZE_USD
        raw = (edge / (1.0 - price)) * KELLY_FRACTION * BANKROLL
        return float(max(MIN_SIZE_USD, min(MAX_SIZE_USD, raw)))

    def generate_signal(self, market: dict) -> Optional[dict]:
        """Evaluate a Kalshi market for a cross-platform signal.

        market must contain: ticker, question (or title), yes_price, no_price.
        Returns a signal dict if edge > EDGE_THRESHOLD, else None.
        """
        ticker: str = market.get("ticker", "")
        question: str = market.get("question") or market.get("title") or market.get("market_title", "")
        yes_price: float = float(market.get("yes_price") or market.get("yes_implied_prob", 0))
        no_price: float = float(market.get("no_price") or market.get("no_implied_prob", 0))

        if yes_price <= 0 or no_price <= 0 or not ticker or not question:
            return None

        poly_price = self._match_polymarket(question)
        meta_forecast = self._meta.get_forecast(question)

        # If neither external source matched, no signal is possible
        if poly_price is None and meta_forecast is None:
            return None

        fv_yes = self.fair_value(yes_price, poly_price, meta_forecast)
        fv_no = self.fair_value(
            no_price,
            1.0 - poly_price if poly_price is not None else None,
            1.0 - meta_forecast if meta_forecast is not None else None,
        )

        edge_yes = fv_yes - yes_price
        edge_no = fv_no - no_price

        if max(edge_yes, edge_no) <= EDGE_THRESHOLD:
            return None

        if edge_yes >= edge_no:
            side = "YES"
            edge = edge_yes
            entry_price = yes_price
            fv = fv_yes
        else:
            side = "NO"
            edge = edge_no
            entry_price = no_price
            fv = fv_no

        return {
            "ticker": ticker,
            "question": question,
            "side": side,
            "entry_price": entry_price,
            "edge": round(edge, 4),
            "fair_value": round(fv, 4),
            "polymarket_price": poly_price,
            "metaculus_forecast": meta_forecast,
            "size_usd": self.kelly_size(edge, entry_price),
            "strategy_tag": "cross_platform_v1",
        }


def _similarity(a: str, b: str) -> float:
    """Word-overlap similarity, normalized to the shorter string."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / min(len(a_words), len(b_words))

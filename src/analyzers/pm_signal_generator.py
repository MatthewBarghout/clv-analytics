"""Cross-platform prediction market signal generator.

Compares Kalshi prices against Polymarket and Metaculus forecasts to find
markets where the implied probability diverges significantly from consensus.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum edge (probability points) to generate a signal
EDGE_THRESHOLD = 0.06

# Quarter-Kelly bankroll used for sizing
BANKROLL = 1000.0
KELLY_FRACTION = 0.25
MIN_SIZE_USD = 25.0
MAX_SIZE_USD = 200.0

# Source weights for fair value calculation
_WEIGHTS = {
    "polymarket": 0.45,
    "metaculus": 0.35,
    "kalshi": 0.20,
}


class PMSignalGenerator:
    """Generates trading signals by comparing Kalshi prices to external forecasts."""

    def __init__(self):
        from src.collectors.polymarket_client import PolymarketClient
        from src.collectors.metaculus_client import MetaculusClient
        self._poly = PolymarketClient()
        self._meta = MetaculusClient()

    def fair_value(
        self,
        kalshi_price: float,
        polymarket_price: Optional[float],
        metaculus_forecast: Optional[float],
    ) -> float:
        """
        Compute weighted fair value from available sources.

        If a source is None its weight is redistributed proportionally to present sources.
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
        """
        Evaluate a Kalshi market for a cross-platform signal.

        market must contain: ticker, question (or title), yes_price, no_price.
        Returns a signal dict if edge > EDGE_THRESHOLD, else None.
        """
        ticker: str = market.get("ticker", "")
        question: str = market.get("question") or market.get("title") or market.get("market_title", "")
        yes_price: float = float(market.get("yes_price") or market.get("yes_implied_prob", 0))
        no_price: float = float(market.get("no_price") or market.get("no_implied_prob", 0))

        if yes_price <= 0 or no_price <= 0 or not ticker or not question:
            return None

        # Extract a short keyword from the event title for external lookups
        keyword = _extract_keyword(question)

        # Fetch external forecasts
        poly_data = None
        poly_price = None
        try:
            poly_data = self._poly.get_market_price(keyword)
            if poly_data:
                poly_price = poly_data["yes_price"]
        except Exception as e:
            logger.debug(f"Polymarket lookup failed for '{keyword}': {e}")

        meta_forecast = None
        try:
            meta_forecast = self._meta.get_forecast(keyword)
        except Exception as e:
            logger.debug(f"Metaculus lookup failed for '{keyword}': {e}")

        # Compute fair values for YES and NO sides
        fv_yes = self.fair_value(yes_price, poly_price, meta_forecast)
        fv_no = self.fair_value(no_price, 1.0 - poly_price if poly_price is not None else None, None)

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


def _extract_keyword(title: str) -> str:
    """Extract a short search keyword from a market title."""
    stopwords = {"will", "the", "a", "an", "in", "to", "of", "at", "vs", "vs.", "win", "?", "-", "win?"}
    words = [w.strip("?.,-()") for w in title.split() if w.strip("?.,-()").lower() not in stopwords]
    return " ".join(words[:4])

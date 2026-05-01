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

# Module-level shared cache — persists across PMSignalGenerator instances
_POLY_CACHE: List[dict] = []


class PMSignalGenerator:
    """Generates trading signals by comparing Kalshi prices to Polymarket forecasts.

    Polymarket data is cached at module level so it persists across instances.
    The dedicated _refresh_poly_cache() scheduler job populates it every 30 min.
    """

    def __init__(self):
        from src.collectors.polymarket_client import PolymarketClient
        from src.collectors.metaculus_client import MetaculusClient
        self._poly = PolymarketClient()
        self._meta = MetaculusClient()

    @property
    def _poly_cache(self) -> List[dict]:
        return _POLY_CACHE

    def refresh_poly_cache(self) -> None:
        """Fetch and cache active Polymarket markets (all categories) for local matching."""
        global _POLY_CACHE
        try:
            raw = self._poly.get_all_markets_cached()
            parsed = [self._poly.parse_market_odds(m) for m in raw]
            _POLY_CACHE = [p for p in parsed if p is not None]
            logger.info(f"Polymarket cache refreshed: {len(_POLY_CACHE)} markets")
        except Exception as e:
            logger.error(f"Polymarket cache refresh failed: {e}")
            _POLY_CACHE = []

    def _match_polymarket(self, question: str) -> Optional[float]:
        """Find the best-matching Polymarket market for a Kalshi question title.

        First tries cached word-overlap similarity. On a cache miss, falls back
        to Polymarket's search API (?q=) which handles non-sports questions
        (crypto, politics, economics) where phrasing rarely overlaps enough.

        Returns the YES price of the best match, else None.
        """
        if not _POLY_CACHE:
            self.refresh_poly_cache()

        if _POLY_CACHE:
            best_sim = 0.0
            best_price = None
            for m in _POLY_CACHE:
                sim = _similarity(question, m.get("title", ""))
                if sim > best_sim:
                    best_sim = sim
                    best_price = m.get("yes_implied_prob")

            if best_sim >= SIMILARITY_THRESHOLD and best_price is not None:
                logger.debug(f"Polymarket cache match (sim={best_sim:.2f}) for '{question[:50]}'")
                return float(best_price)

            logger.debug(f"Polymarket cache miss (sim={best_sim:.2f}), trying search for '{question[:50]}'")

        # Search API fallback — handles non-sports/crypto/politics where phrasing diverges.
        # Guard with similarity check to reject irrelevant high-volume markets that
        # Polymarket's search ranks first regardless of query.
        try:
            result = self._poly.get_market_price(question[:200])
            if result:
                returned_question = result.get("question", "")
                relevance = _similarity(question, returned_question)
                if relevance >= SIMILARITY_THRESHOLD:
                    logger.debug(
                        f"Polymarket search match (sim={relevance:.2f}) for '{question[:50]}': "
                        f"{result['yes_price']:.3f} ('{returned_question[:50]}')"
                    )
                    return float(result["yes_price"])
                logger.debug(
                    f"Polymarket search rejected (sim={relevance:.2f}): "
                    f"'{returned_question[:50]}' for '{question[:50]}'"
                )
        except Exception as e:
            logger.debug(f"Polymarket search fallback failed for '{question[:50]}': {e}")

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

    def generate_signal(self, market: dict, category: str = "sports") -> Optional[dict]:
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
            "strategy_tag": f"cross_platform_{category}_v1",
        }


_STOPWORDS = {
    "will", "the", "a", "an", "of", "in", "to", "be", "is", "are", "was",
    "it", "for", "on", "at", "by", "or", "and", "before", "after", "than",
    "that", "this", "with", "have", "has", "any", "all", "not", "no", "do",
    "does", "did", "its", "their", "there", "than", "then", "when", "what",
    "which", "who", "how", "if", "as", "up", "out", "about", "into",
}


def _tokenize(s: str) -> set:
    import re
    # Normalize: lowercase, strip punctuation except $ and digits, split
    tokens = re.sub(r"[^\w\s$]", " ", s.lower()).split()
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _similarity(a: str, b: str) -> float:
    """Word-overlap similarity with stopword removal, normalized to the shorter string."""
    a_words = _tokenize(a)
    b_words = _tokenize(b)
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / min(len(a_words), len(b_words))

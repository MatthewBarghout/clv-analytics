"""Cross-platform prediction market signal generator.

Compares Kalshi prices against Polymarket forecasts to find
markets where the implied probability diverges significantly from consensus.

For KXBTC/KXETH markets, uses a log-normal options pricing model instead of
Polymarket text matching — hourly crypto bracket markets have no comparable
Polymarket equivalent so text similarity produces noise.
"""
import logging
import math
import re
from datetime import datetime, timezone
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

# ── Crypto pricing constants ────────────────────────────────────────────────

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Half-width of each Kalshi price bracket (observed from market structure)
_CRYPTO_BRACKET_HALF_WIDTH = {
    "KXETH": 10.0,    # $20 brackets (B1990, B2010, B2030...)
    "KXBTC": 125.0,   # $250 brackets (B73125, B73375...)
}

# CoinGecko asset IDs
_COINGECKO_ID = {
    "KXETH": "ethereum",
    "KXBTC": "bitcoin",
}

# Annualised historical volatility assumptions
_CRYPTO_ANNUAL_VOL = {
    "KXETH": 1.0,   # ~100% annualised
    "KXBTC": 0.70,  # ~70% annualised
}


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc — no scipy needed."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _parse_crypto_ticker(ticker: str) -> Optional[dict]:
    """Parse a KXETH/KXBTC ticker into pricing components.

    e.g. KXETH-26MAY3114-B2010  →  {asset_key, coingecko_id, expiry_dt, strike, half_width, annual_vol}
    """
    parts = ticker.split("-")
    if len(parts) != 3:
        return None

    asset_key = parts[0]
    if asset_key not in _COINGECKO_ID:
        return None

    # Date+hour segment: 26MAY3114 → year=2026, month=MAY, day=31, hour=14
    m = re.match(r"(\d{2})([A-Z]{3})(\d{2})(\d{2})$", parts[1])
    if not m:
        return None

    month = _MONTH_MAP.get(m.group(2))
    if not month:
        return None

    try:
        expiry_dt = datetime(
            2000 + int(m.group(1)), month, int(m.group(3)), int(m.group(4)),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None

    if not parts[2].startswith("B"):
        return None
    try:
        strike = float(parts[2][1:])
    except ValueError:
        return None

    return {
        "asset_key": asset_key,
        "coingecko_id": _COINGECKO_ID[asset_key],
        "expiry_dt": expiry_dt,
        "strike": strike,
        "half_width": _CRYPTO_BRACKET_HALF_WIDTH.get(asset_key, 10.0),
        "annual_vol": _CRYPTO_ANNUAL_VOL.get(asset_key, 0.8),
    }


def _lognormal_bracket_prob(
    current_price: float,
    strike: float,
    half_width: float,
    hours_to_expiry: float,
    annual_vol: float,
) -> float:
    """P(asset ends inside [strike - half_width, strike + half_width]) under log-normal.

    Uses zero-drift assumption (r=0) appropriate for short-horizon crypto brackets.
    """
    lower = max(strike - half_width, 0.01)
    upper = strike + half_width

    if hours_to_expiry <= 0:
        return 1.0 if lower <= current_price <= upper else 0.0

    T = hours_to_expiry / 8760.0
    sigma_sqrt_T = annual_vol * math.sqrt(T)
    drift = -0.5 * annual_vol ** 2 * T

    def prob_above(K: float) -> float:
        d2 = (math.log(current_price / K) + drift) / sigma_sqrt_T
        return _norm_cdf(d2)

    prob = prob_above(lower) - prob_above(upper)
    return max(0.001, min(0.999, prob))


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

    def _match_polymarket(self, question: str) -> tuple[Optional[float], float]:
        """Find the best-matching Polymarket market for a Kalshi question title.

        First tries cached word-overlap similarity. On a cache miss, falls back
        to Polymarket's search API (?q=) which handles non-sports questions
        (crypto, politics, economics) where phrasing rarely overlaps enough.

        Returns (yes_price, similarity_score). Price is None when no match accepted;
        similarity_score is always populated so callers can apply stricter checks
        on large price divergences.
        """
        if not _POLY_CACHE:
            self.refresh_poly_cache()

        best_sim = 0.0
        if _POLY_CACHE:
            best_price = None
            for m in _POLY_CACHE:
                sim = _similarity(question, m.get("title", ""))
                if sim > best_sim:
                    best_sim = sim
                    best_price = m.get("yes_implied_prob")

            if best_sim >= SIMILARITY_THRESHOLD and best_price is not None:
                logger.debug(f"Polymarket cache match (sim={best_sim:.2f}) for '{question[:50]}'")
                return float(best_price), best_sim

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
                    return float(result["yes_price"]), relevance
                logger.debug(
                    f"Polymarket search rejected (sim={relevance:.2f}): "
                    f"'{returned_question[:50]}' for '{question[:50]}'"
                )
                return None, relevance
        except Exception as e:
            logger.debug(f"Polymarket search fallback failed for '{question[:50]}': {e}")

        return None, best_sim

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
        """Quarter-Kelly position size in USD, clamped to [25, 200].

        Both tails are tightened to $50. At price>0.80 the win payout is tiny
        relative to loss exposure. At price<0.20 the (1-price) denominator is
        large, so Kelly runs straight to the clamp on a lottery-ticket contract.
        """
        if price >= 1.0 or price <= 0.0:
            return MIN_SIZE_USD
        max_size = 50.0 if (price > 0.80 or price < 0.20) else MAX_SIZE_USD
        raw = (edge / (1.0 - price)) * KELLY_FRACTION * BANKROLL
        return float(max(MIN_SIZE_USD, min(max_size, raw)))

    def _crypto_fair_value_yes(self, ticker: str) -> Optional[float]:
        """Compute theoretical P(YES) for a KXBTC/KXETH bracket market via log-normal pricing."""
        from src.collectors.crypto_price_client import get_crypto_price

        parsed = _parse_crypto_ticker(ticker)
        if parsed is None:
            return None

        current_price = get_crypto_price(parsed["coingecko_id"])
        if current_price is None:
            return None

        now = datetime.now(timezone.utc)
        hours_to_expiry = (parsed["expiry_dt"] - now).total_seconds() / 3600.0

        prob = _lognormal_bracket_prob(
            current_price=current_price,
            strike=parsed["strike"],
            half_width=parsed["half_width"],
            hours_to_expiry=hours_to_expiry,
            annual_vol=parsed["annual_vol"],
        )
        logger.debug(
            f"{ticker}: spot={current_price:.2f} strike={parsed['strike']:.2f} "
            f"T={hours_to_expiry:.2f}h P(YES)={prob:.4f}"
        )
        return prob

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

        # Crypto bracket markets: use log-normal pricing instead of Polymarket text matching
        is_crypto = ticker.startswith(("KXBTC", "KXETH"))
        if is_crypto:
            fv_yes_val = self._crypto_fair_value_yes(ticker)
            if fv_yes_val is None:
                return None
            fv_yes = fv_yes_val
            fv_no = 1.0 - fv_yes_val
            poly_price = None
            meta_forecast = None
        else:
            poly_price, poly_sim = self._match_polymarket(question)
            meta_forecast = self._meta.get_forecast(question)

            # High-divergence guard: large price gaps demand a stricter similarity
            # bar. Without this, championship futures collide with unrelated markets
            # that share team names (e.g. KXMLB-26-TEX vs. Texas Longhorns baseball).
            if poly_price is not None and abs(yes_price - poly_price) > 0.50 and poly_sim < 0.70:
                logger.debug(
                    f"Rejecting high-divergence Polymarket match for '{question[:50]}': "
                    f"|{yes_price:.3f}-{poly_price:.3f}|>0.50 with sim={poly_sim:.2f}<0.70"
                )
                poly_price = None

            if poly_price is None and meta_forecast is None:
                return None

            fv_yes = self.fair_value(yes_price, poly_price, meta_forecast)
            # Clamp inverted prices to (0, 1) exclusive
            poly_no = max(0.01, min(0.99, 1.0 - poly_price)) if poly_price is not None else None
            meta_no = max(0.01, min(0.99, 1.0 - meta_forecast)) if meta_forecast is not None else None
            fv_no = self.fair_value(no_price, poly_no, meta_no)

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

        # High-price edge bar: a 6% edge at 90¢ has unjustifiable risk/reward.
        if entry_price > 0.80 and edge < 0.10:
            return None

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

"""Arbitrage calculator between sportsbooks and prediction markets.

Converts prediction market implied probabilities to decimal odds,
compares them to sportsbook lines, and flags positive-spread arb
opportunities.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List

logger = logging.getLogger(__name__)


def american_to_implied(american_odds: float) -> float:
    """Convert American odds to implied probability (0.0-1.0)."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def decimal_to_implied(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def implied_to_decimal(implied_prob: float) -> float:
    """Convert implied probability to decimal odds."""
    if implied_prob <= 0:
        return 0.0
    return 1.0 / implied_prob


def calculate_arb_spread(sportsbook_decimal: float, pm_implied_prob: float) -> float:
    """
    Calculate arbitrage spread between sportsbook and prediction market.

    A positive spread means the PM implies a *higher* probability than
    the sportsbook's implied odds — i.e., the sportsbook is offering
    better value on that outcome than the market consensus.

    Args:
        sportsbook_decimal: Sportsbook decimal odds for the outcome
        pm_implied_prob: Prediction market implied probability (0.0-1.0)

    Returns:
        Arb spread in percentage points. Positive = arb exists.
    """
    if sportsbook_decimal <= 0 or pm_implied_prob <= 0:
        return 0.0
    sb_implied = decimal_to_implied(sportsbook_decimal)
    return round((pm_implied_prob - sb_implied) * 100, 4)


def find_arb_opportunities(
    sportsbook_odds: List[Dict],
    pm_markets: List[Dict],
    source: str,
    min_spread: float = 0.5,
) -> List[Dict]:
    """
    Find arbitrage opportunities between sportsbooks and a prediction market.

    Args:
        sportsbook_odds: List of dicts with keys:
            - event_title (str)
            - sportsbook_name (str)
            - decimal_odds (float)
        pm_markets: List of dicts from KalshiClient.parse_market_odds() or
            PolymarketClient.parse_market_odds()
        source: "kalshi" or "polymarket"
        min_spread: Minimum arb spread in percentage points to include

    Returns:
        List of arb opportunity dicts ready to be stored in PredictionMarketArb.
    """
    opportunities = []
    now = datetime.now(timezone.utc)

    for sb in sportsbook_odds:
        sb_event = (sb.get("event_title") or "").lower()
        sb_decimal = sb.get("decimal_odds", 0.0)
        sb_name = sb.get("sportsbook_name", "Unknown")

        if sb_decimal <= 0:
            continue

        for pm in pm_markets:
            pm_title = (pm.get("title") or "").lower()

            # Fuzzy match: at least 2 words in common between event titles
            sb_words = set(sb_event.split())
            pm_words = set(pm_title.split())
            common_words = sb_words & pm_words
            # Filter out common short/stop words
            meaningful = {w for w in common_words if len(w) > 3}
            if len(meaningful) < 2:
                continue

            # Check YES side
            yes_prob = pm.get("yes_implied_prob", 0.0)
            yes_spread = calculate_arb_spread(sb_decimal, yes_prob)

            if yes_spread >= min_spread:
                opportunities.append({
                    "event_title": sb.get("event_title", ""),
                    "market_source": source,
                    "sportsbook_name": sb_name,
                    "sportsbook_odds": sb_decimal,
                    "pm_implied_prob": yes_prob,
                    "pm_implied_odds": pm.get("yes_implied_odds", 0.0),
                    "arb_spread": yes_spread,
                    "market_url": pm.get("market_url"),
                    "timestamp": now,
                    "is_active": True,
                })

    # Sort by spread descending
    opportunities.sort(key=lambda x: x["arb_spread"], reverse=True)
    logger.info(f"Found {len(opportunities)} arb opportunities from {source}")
    return opportunities


def build_sportsbook_odds_from_snapshots(snapshots_data: List[Dict]) -> List[Dict]:
    """
    Build a list of sportsbook odds dicts from OddsSnapshot records.

    Args:
        snapshots_data: List of dicts with keys:
            - event_title, sportsbook_name, outcomes (list of outcome dicts)

    Returns:
        List of odds dicts with event_title, sportsbook_name, decimal_odds.
    """
    result = []
    for snap in snapshots_data:
        for outcome in snap.get("outcomes", []):
            price = outcome.get("price")
            name = outcome.get("name")
            if price and name:
                result.append({
                    "event_title": f"{snap['event_title']} - {name}",
                    "sportsbook_name": snap["sportsbook_name"],
                    "decimal_odds": float(price),
                })
    return result

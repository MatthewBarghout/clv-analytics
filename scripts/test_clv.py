"""
Test CLV calculator functions.
Run: poetry run python -m scripts.test_clv
"""
from src.analyzers.clv_calculator import CLVCalculator


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds < 0:
        return 1 + (100 / abs(american_odds))
    else:
        return 1 + (american_odds / 100)


def test_american_odds():
    """Test with American odds examples."""
    calc = CLVCalculator()

    print("=" * 60)
    print("Testing CLV Calculator with American Odds")
    print("=" * 60)

    # Test 1: Entry -150, Closing -170 (line moved against you, you got better price)
    print("\nTest 1: Favorite example")
    print("Entry: -150, Closing: -170")
    entry_decimal = american_to_decimal(-150)
    closing_decimal = american_to_decimal(-170)
    print(f"Entry decimal: {entry_decimal:.3f}")
    print(f"Closing decimal: {closing_decimal:.3f}")
    clv = calc.calculate_clv(entry_decimal, closing_decimal)
    print(f"CLV: {clv:+.2f}%")
    print(f"Expected: ~+2.96%")

    # Test 2: Entry -170, Closing -150 (line moved in your favor, you got worse price)
    print("\nTest 2: Favorite example (opposite)")
    print("Entry: -170, Closing: -150")
    entry_decimal = american_to_decimal(-170)
    closing_decimal = american_to_decimal(-150)
    print(f"Entry decimal: {entry_decimal:.3f}")
    print(f"Closing decimal: {closing_decimal:.3f}")
    clv = calc.calculate_clv(entry_decimal, closing_decimal)
    print(f"CLV: {clv:+.2f}%")
    print(f"Expected: ~-2.96%")

    # Test 3: Underdog example - Entry +200, Closing +180
    print("\nTest 3: Underdog example")
    print("Entry: +200, Closing: +180")
    entry_decimal = american_to_decimal(200)
    closing_decimal = american_to_decimal(180)
    print(f"Entry decimal: {entry_decimal:.3f}")
    print(f"Closing decimal: {closing_decimal:.3f}")
    clv = calc.calculate_clv(entry_decimal, closing_decimal)
    print(f"CLV: {clv:+.2f}%")

    print("\n" + "=" * 60)


def test_decimal_odds():
    """Test with decimal odds (what API returns)."""
    calc = CLVCalculator()

    print("\n" + "=" * 60)
    print("Testing CLV Calculator with Decimal Odds (API Format)")
    print("=" * 60)

    # Example from our database
    print("\nReal example from database:")
    print("Memphis Grizzlies - FanDuel")
    entry_odds = 2.10
    closing_odds = 1.95
    print(f"Entry: {entry_odds} (implied prob: {1/entry_odds:.1%})")
    print(f"Closing: {closing_odds} (implied prob: {1/closing_odds:.1%})")
    clv = calc.calculate_clv(entry_odds, closing_odds)
    print(f"CLV: {clv:+.2f}%")

    print("\n" + "=" * 60)


def test_implied_probabilities():
    """Test probability conversions."""
    calc = CLVCalculator()

    print("\n" + "=" * 60)
    print("Testing Probability Conversions")
    print("=" * 60)

    test_cases = [
        ("Even money", 2.0, -100),
        ("Heavy favorite", 1.5, -200),
        ("Slight favorite", 1.77, -130),
        ("Slight underdog", 2.10, 110),
        ("Heavy underdog", 3.0, 200),
    ]

    for label, decimal, american in test_cases:
        decimal_prob = calc.decimal_to_implied_prob(decimal)
        american_decimal = american_to_decimal(american)
        american_prob = calc.american_to_implied_prob(american)

        print(f"\n{label}:")
        print(f"  Decimal {decimal} -> {decimal_prob:.1%}")
        print(f"  American {american:+d} -> {american_decimal:.3f} -> {american_prob:.1%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_american_odds()
    test_decimal_odds()
    test_implied_probabilities()

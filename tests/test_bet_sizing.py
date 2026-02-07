"""Unit tests for bet sizing strategies."""
import pytest
from src.analyzers.bet_sizing import (
    BetContext,
    BetSizingStrategy,
    ConfidenceWeightedBetSizer,
    FixedBetSizer,
    FractionalBetSizer,
    KellyBetSizer,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    get_bet_sizer,
)


class TestBetContext:
    """Tests for BetContext dataclass."""

    def test_default_values(self):
        """Test default constraint values."""
        ctx = BetContext(bankroll=10000, odds=2.0)
        assert ctx.min_bet == 10.0
        assert ctx.max_bet == 1000.0
        assert ctx.max_fraction == 0.25

    def test_custom_constraints(self):
        """Test custom constraint values."""
        ctx = BetContext(
            bankroll=10000,
            odds=2.0,
            min_bet=50,
            max_bet=500,
            max_fraction=0.10,
        )
        assert ctx.min_bet == 50
        assert ctx.max_bet == 500
        assert ctx.max_fraction == 0.10


class TestFixedBetSizer:
    """Tests for fixed unit bet sizing."""

    def test_returns_fixed_amount(self):
        """Test that fixed sizer returns the same amount."""
        sizer = FixedBetSizer(unit_size=100)
        ctx = BetContext(bankroll=10000, odds=2.0)
        result = sizer.calculate_bet_size(ctx)

        assert result.bet_size == 100
        assert result.strategy_used == "fixed"
        assert not result.was_capped

    def test_respects_max_bet(self):
        """Test that max bet is respected."""
        sizer = FixedBetSizer(unit_size=500)
        ctx = BetContext(bankroll=10000, odds=2.0, max_bet=200)
        result = sizer.calculate_bet_size(ctx)

        assert result.bet_size == 200
        assert result.was_capped

    def test_respects_min_bet(self):
        """Test that min bet is respected."""
        sizer = FixedBetSizer(unit_size=5)
        ctx = BetContext(bankroll=10000, odds=2.0, min_bet=25)
        result = sizer.calculate_bet_size(ctx)

        assert result.bet_size == 25
        assert result.was_capped

    def test_cant_bet_more_than_bankroll(self):
        """Test that bet can't exceed bankroll."""
        sizer = FixedBetSizer(unit_size=500)
        ctx = BetContext(bankroll=100, odds=2.0, max_bet=1000)
        result = sizer.calculate_bet_size(ctx)

        assert result.bet_size == 100
        assert result.was_capped


class TestFractionalBetSizer:
    """Tests for fractional bet sizing."""

    def test_returns_fraction_of_bankroll(self):
        """Test that sizer returns correct fraction."""
        sizer = FractionalBetSizer(fraction=0.02)
        ctx = BetContext(bankroll=10000, odds=2.0)
        result = sizer.calculate_bet_size(ctx)

        assert result.bet_size == 200  # 2% of 10000
        assert result.strategy_used == "fractional"

    def test_scales_with_bankroll(self):
        """Test that bet size scales with bankroll."""
        sizer = FractionalBetSizer(fraction=0.05)

        ctx1 = BetContext(bankroll=10000, odds=2.0)
        ctx2 = BetContext(bankroll=5000, odds=2.0)

        result1 = sizer.calculate_bet_size(ctx1)
        result2 = sizer.calculate_bet_size(ctx2)

        assert result1.bet_size == 500
        assert result2.bet_size == 250

    def test_invalid_fraction_raises(self):
        """Test that invalid fraction raises error."""
        with pytest.raises(ValueError):
            FractionalBetSizer(fraction=0)

        with pytest.raises(ValueError):
            FractionalBetSizer(fraction=1.5)

    def test_respects_max_fraction(self):
        """Test that max fraction constraint is applied."""
        sizer = FractionalBetSizer(fraction=0.50)  # 50%
        ctx = BetContext(bankroll=10000, odds=2.0, max_fraction=0.10)
        result = sizer.calculate_bet_size(ctx)

        assert result.bet_size == 1000  # Capped at 10% = 1000
        assert result.was_capped


class TestKellyBetSizer:
    """Tests for Kelly Criterion bet sizing."""

    def test_full_kelly_positive_edge(self):
        """Test full Kelly with positive edge."""
        sizer = KellyBetSizer(fraction_of_kelly=1.0)
        ctx = BetContext(
            bankroll=10000,
            odds=2.0,  # Even money
            clv_percentage=5.0,  # 5% edge
        )
        result = sizer.calculate_bet_size(ctx)

        # Should recommend a positive bet
        assert result.bet_size > 0
        assert result.kelly_fraction is not None
        assert result.kelly_fraction > 0

    def test_half_kelly_reduces_bet(self):
        """Test that half Kelly returns half the bet."""
        full_kelly = KellyBetSizer(fraction_of_kelly=1.0)
        half_kelly = KellyBetSizer(fraction_of_kelly=0.5)

        ctx = BetContext(
            bankroll=10000,
            odds=2.0,
            clv_percentage=5.0,
        )

        full_result = full_kelly.calculate_bet_size(ctx)
        half_result = half_kelly.calculate_bet_size(ctx)

        # Half Kelly should be roughly half (before constraints)
        assert half_result.raw_bet_size == pytest.approx(full_result.raw_bet_size / 2, rel=0.01)

    def test_no_bet_on_negative_edge(self):
        """Test that no bet is recommended for negative edge."""
        sizer = KellyBetSizer(fraction_of_kelly=1.0)
        ctx = BetContext(
            bankroll=10000,
            odds=2.0,
            win_probability=0.4,  # 40% to win even money = negative EV
        )
        result = sizer.calculate_bet_size(ctx)

        # Kelly fraction should be 0 or near 0
        assert result.kelly_fraction == 0 or result.bet_size == ctx.min_bet

    def test_respects_max_fraction(self):
        """Test that Kelly is capped by max fraction."""
        sizer = KellyBetSizer(fraction_of_kelly=1.0)
        ctx = BetContext(
            bankroll=10000,
            odds=5.0,  # High odds
            win_probability=0.5,  # 50% chance at 5:1 = huge edge
            max_fraction=0.10,
        )
        result = sizer.calculate_bet_size(ctx)

        # Should be capped at 10% of bankroll
        assert result.bet_size <= 1000
        assert result.was_capped

    def test_uses_clv_to_estimate_probability(self):
        """Test that CLV is used when probability not given."""
        sizer = KellyBetSizer()
        ctx = BetContext(
            bankroll=10000,
            odds=2.0,
            clv_percentage=3.0,  # 3% CLV
        )
        result = sizer.calculate_bet_size(ctx)

        # Should have estimated probability and positive bet
        assert result.bet_size > 0
        assert result.edge_estimate is not None


class TestConfidenceWeightedBetSizer:
    """Tests for confidence-weighted bet sizing."""

    def test_scales_with_confidence(self):
        """Test that bet scales with ML confidence."""
        sizer = ConfidenceWeightedBetSizer(
            base_fraction=0.02,
            min_confidence=0.5,
            max_confidence=0.9,
            min_multiplier=0.5,
            max_multiplier=2.0,
        )

        low_conf = BetContext(bankroll=10000, odds=2.0, ml_confidence=0.5)
        mid_conf = BetContext(bankroll=10000, odds=2.0, ml_confidence=0.7)
        high_conf = BetContext(bankroll=10000, odds=2.0, ml_confidence=0.9)

        low_result = sizer.calculate_bet_size(low_conf)
        mid_result = sizer.calculate_bet_size(mid_conf)
        high_result = sizer.calculate_bet_size(high_conf)

        # Bet size should increase with confidence
        assert low_result.bet_size < mid_result.bet_size
        assert mid_result.bet_size < high_result.bet_size

    def test_min_confidence_uses_min_multiplier(self):
        """Test that low confidence uses minimum multiplier."""
        sizer = ConfidenceWeightedBetSizer(
            base_fraction=0.02,
            min_confidence=0.5,
            min_multiplier=0.5,
        )

        ctx = BetContext(bankroll=10000, odds=2.0, ml_confidence=0.3)
        result = sizer.calculate_bet_size(ctx)

        # 2% * 0.5 = 1% of 10000 = 100
        assert result.raw_bet_size == 100

    def test_max_confidence_uses_max_multiplier(self):
        """Test that high confidence uses maximum multiplier."""
        sizer = ConfidenceWeightedBetSizer(
            base_fraction=0.02,
            max_confidence=0.9,
            max_multiplier=2.0,
        )

        ctx = BetContext(bankroll=10000, odds=2.0, ml_confidence=0.95)
        result = sizer.calculate_bet_size(ctx)

        # 2% * 2.0 = 4% of 10000 = 400
        assert result.raw_bet_size == 400


class TestGetBetSizer:
    """Tests for the factory function."""

    def test_returns_fixed_sizer(self):
        """Test factory returns FixedBetSizer."""
        sizer = get_bet_sizer(BetSizingStrategy.FIXED, unit_size=100)
        assert isinstance(sizer, FixedBetSizer)

    def test_returns_fractional_sizer(self):
        """Test factory returns FractionalBetSizer."""
        sizer = get_bet_sizer(BetSizingStrategy.FRACTIONAL, fraction=0.02)
        assert isinstance(sizer, FractionalBetSizer)

    def test_returns_kelly_sizer(self):
        """Test factory returns KellyBetSizer."""
        sizer = get_bet_sizer(BetSizingStrategy.KELLY)
        assert isinstance(sizer, KellyBetSizer)

    def test_returns_half_kelly_sizer(self):
        """Test factory returns KellyBetSizer with 0.5 fraction."""
        sizer = get_bet_sizer(BetSizingStrategy.HALF_KELLY)
        assert isinstance(sizer, KellyBetSizer)

    def test_returns_confidence_sizer(self):
        """Test factory returns ConfidenceWeightedBetSizer."""
        sizer = get_bet_sizer(BetSizingStrategy.CONFIDENCE)
        assert isinstance(sizer, ConfidenceWeightedBetSizer)


class TestRiskMetrics:
    """Tests for risk metric calculations."""

    def test_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio with positive returns."""
        returns = [0.1, 0.05, 0.08, -0.02, 0.12, 0.03]
        sharpe = calculate_sharpe_ratio(returns)

        # Should be positive for net positive returns
        assert sharpe > 0

    def test_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation."""
        returns = [0.1, 0.1, 0.1]  # All same return
        sharpe = calculate_sharpe_ratio(returns)

        # Should return 0 to avoid division by zero
        assert sharpe == 0

    def test_sharpe_ratio_insufficient_data(self):
        """Test Sharpe ratio with insufficient data."""
        sharpe = calculate_sharpe_ratio([0.1])
        assert sharpe == 0

    def test_sortino_ratio_only_considers_downside(self):
        """Test that Sortino only considers negative returns."""
        # Mix of positive and negative returns
        returns = [0.1, 0.05, -0.02, 0.08, -0.03]
        sortino = calculate_sortino_ratio(returns)

        # Should return a value
        assert sortino != 0

    def test_sortino_ratio_no_negative_returns(self):
        """Test Sortino with no negative returns."""
        returns = [0.1, 0.05, 0.08]  # All positive
        sortino = calculate_sortino_ratio(returns)

        # Should be infinity for all positive returns
        assert sortino == float('inf')

    def test_sortino_ratio_insufficient_data(self):
        """Test Sortino ratio with insufficient data."""
        sortino = calculate_sortino_ratio([])
        assert sortino == 0

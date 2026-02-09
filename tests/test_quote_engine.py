"""Tests for QuoteEngine: Avellaneda-Stoikov + OBI overlay."""

import time
import pytest
from src.quote_engine import QuoteEngine
from src.state import MarketState, BotState


def make_config(**overrides):
    """Create a test config dict."""
    cfg = {
        "strategy": {
            "gamma": 0.3,
            "kappa": 1.5,
            "min_half_spread": 1.0,
            "base_size": 0.001,
            "max_position": 0.01,
            "vol_window": 60,
            "tick_size": 1.0,
        },
        "obi": {
            "enabled": True,
            "alpha": 0.3,
            "threshold": 0.3,
            "delta": 0.3,
            "depth": 5,
        },
    }
    for k, v in overrides.items():
        if '.' in k:
            section, key = k.split('.', 1)
            cfg[section][key] = v
        else:
            cfg[k] = v
    return cfg


def make_market(bid=97500.0, ask=97502.0):
    """Create a MarketState with given BBO."""
    ms = MarketState()
    ms.update_bbo(bid, ask)
    return ms


def make_bot(net_position=0.0):
    """Create a BotState with given position."""
    bs = BotState()
    bs.net_position = net_position
    return bs


def seed_prices(engine, prices):
    """Seed mid_prices with (timestamp, price) tuples."""
    t = time.time()
    for i, p in enumerate(prices):
        engine.mid_prices.append((t + i * 0.1, p))


class TestQuoteEngineBasics:
    """Basic quote generation tests."""

    def test_zero_inventory_symmetric_quotes(self):
        """With zero inventory, bid and ask should be symmetric around fair price."""
        engine = QuoteEngine(make_config())
        # Seed some mid prices for volatility
        seed_prices(engine, [97501.0] * 20)

        ms = make_market(97500.0, 97502.0)
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        assert result.bid_price > 0
        assert result.ask_price > 0
        assert result.bid_price < result.ask_price

        # Symmetric: fair should be near mid, bid/ask equidistant
        mid = (result.bid_price + result.ask_price) / 2
        assert abs(mid - result.fair_price) < 5.0  # Within $5 of fair

    def test_zero_inventory_equal_sizes(self):
        """With zero inventory, bid and ask sizes should be equal."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market()
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size == result.ask_size

    def test_invalid_market_data(self):
        """Should skip with invalid market data."""
        engine = QuoteEngine(make_config())
        ms = MarketState()  # No data
        bs = make_bot()
        result = engine.generate_quotes(ms, bs)

        assert result.skip_reason == "invalid_market_data"


class TestInventorySkewing:
    """Test inventory-based quote adjustments."""

    def test_long_inventory_ask_more_aggressive(self):
        """When long, ask should be closer to mid (more aggressive selling)."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market(97500.0, 97502.0)

        # Zero inventory
        bs_zero = make_bot(0.0)
        q_zero = engine.generate_quotes(ms, bs_zero)

        # Long inventory
        bs_long = make_bot(0.005)  # Half max
        q_long = engine.generate_quotes(ms, bs_long)

        # When long, fair_price moves down, so ask gets more aggressive (lower)
        assert q_long.fair_price < q_zero.fair_price or q_long.ask_price <= q_zero.ask_price

    def test_short_inventory_bid_more_aggressive(self):
        """When short, bid should be closer to mid (more aggressive buying)."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market(97500.0, 97502.0)

        # Zero inventory
        bs_zero = make_bot(0.0)
        q_zero = engine.generate_quotes(ms, bs_zero)

        # Short inventory
        bs_short = make_bot(-0.005)
        q_short = engine.generate_quotes(ms, bs_short)

        # When short, fair_price moves up, so bid gets more aggressive (higher)
        assert q_short.fair_price > q_zero.fair_price or q_short.bid_price >= q_zero.bid_price

    def test_long_position_reduces_bid_size(self):
        """When long, bid size should be reduced."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market()
        bs = make_bot(0.005)  # Half max position
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size < result.ask_size

    def test_short_position_reduces_ask_size(self):
        """When short, ask size should be reduced."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market()
        bs = make_bot(-0.005)
        result = engine.generate_quotes(ms, bs)

        assert result.ask_size < result.bid_size

    def test_max_long_position_zero_bid(self):
        """At max long position, bid size should be 0 (stop buying)."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market()
        bs = make_bot(0.01)  # At max position
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size == 0

    def test_max_short_position_zero_ask(self):
        """At max short position, ask size should be 0 (stop selling)."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market()
        bs = make_bot(-0.01)
        result = engine.generate_quotes(ms, bs)

        assert result.ask_size == 0


class TestOBIOverlay:
    """Test OBI (Order Book Imbalance) contrarian overlay."""

    def test_positive_obi_shifts_fair_down(self):
        """Positive OBI (buy pressure) should shift fair_price DOWN (contrarian)."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        # Strong buy pressure — call multiple times to build up EMA
        bids = [(97500 - i, 10.0) for i in range(5)]
        asks = [(97502 + i, 1.0) for i in range(5)]  # Much less ask volume
        for _ in range(10):
            engine.calc_obi(bids, asks)

        # OBI should be positive and above threshold
        assert engine.obi_smooth > 0.3

        ms = make_market()
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        # Fair price should be shifted down from mid
        assert result.fair_price < ms.mid_price

    def test_negative_obi_shifts_fair_up(self):
        """Negative OBI (sell pressure) should shift fair_price UP (contrarian)."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        # Strong sell pressure — call multiple times to build up EMA
        bids = [(97500 - i, 1.0) for i in range(5)]
        asks = [(97502 + i, 10.0) for i in range(5)]
        for _ in range(10):
            engine.calc_obi(bids, asks)

        assert engine.obi_smooth < -0.3

        ms = make_market()
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        assert result.fair_price > ms.mid_price

    def test_obi_below_threshold_no_shift(self):
        """OBI below threshold should not shift fair price."""
        engine = QuoteEngine(make_config(**{"obi.threshold": 0.3}))
        seed_prices(engine, [97501.0] * 20)

        # Balanced book
        bids = [(97500 - i, 5.0) for i in range(5)]
        asks = [(97502 + i, 5.0) for i in range(5)]
        engine.calc_obi(bids, asks)

        # OBI ~0, below threshold
        assert abs(engine.obi_smooth) < 0.3

        ms = make_market()
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        # Fair should be ~mid (no OBI shift)
        assert abs(result.fair_price - ms.mid_price) < 1.0

    def test_obi_disabled(self):
        """With OBI disabled, fair price should not be affected by order book."""
        engine = QuoteEngine(make_config(**{"obi.enabled": False}))
        seed_prices(engine, [97501.0] * 20)

        # Extreme imbalance
        bids = [(97500 - i, 100.0) for i in range(5)]
        asks = [(97502 + i, 1.0) for i in range(5)]
        engine.calc_obi(bids, asks)

        ms = make_market()
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        # Fair should be ~mid even with extreme OBI (disabled)
        assert abs(result.fair_price - ms.mid_price) < 1.0


class TestTickAlignment:
    """Test price alignment to tick size."""

    def test_bid_floor_to_tick(self):
        """Bid should be floored to tick size."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market()
        bs = make_bot()
        result = engine.generate_quotes(ms, bs)

        # Price should be integer (tick_size=1.0)
        assert result.bid_price == int(result.bid_price)

    def test_ask_ceil_to_tick(self):
        """Ask should be ceiled to tick size."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market()
        bs = make_bot()
        result = engine.generate_quotes(ms, bs)

        assert result.ask_price == int(result.ask_price)


class TestNoCrossingBBO:
    """Test that quotes don't cross the BBO."""

    def test_bid_not_above_best_bid(self):
        """Our bid should never exceed the best bid."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market(97500.0, 97502.0)
        bs = make_bot()
        result = engine.generate_quotes(ms, bs)

        assert result.bid_price <= ms.best_bid

    def test_ask_not_below_best_ask(self):
        """Our ask should never be below the best ask."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market(97500.0, 97502.0)
        bs = make_bot()
        result = engine.generate_quotes(ms, bs)

        assert result.ask_price >= ms.best_ask


class TestVolatility:
    """Test volatility calculation."""

    def test_low_volatility(self):
        """Stable prices should produce low volatility."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        sigma = engine.calc_volatility()
        assert sigma == 1.0  # Floor value

    def test_high_volatility(self):
        """Varying prices should produce higher volatility."""
        engine = QuoteEngine(make_config())
        # Oscillating prices
        seed_prices(engine, [97500 + (i % 2) * 10 for i in range(20)])

        sigma = engine.calc_volatility()
        assert sigma > 1.0

    def test_insufficient_data_default(self):
        """With < 10 samples, should return default volatility."""
        engine = QuoteEngine(make_config())
        t = time.time()
        engine.mid_prices.append((t, 97501.0))
        engine.mid_prices.append((t + 1, 97502.0))

        sigma = engine.calc_volatility()
        assert sigma == 5.0  # Default


class TestTightenMode:
    """Test tighten_mode behavior after inventory timeout."""

    def test_tighten_mode_long_ask_at_best_ask(self):
        """Tighten mode with long position should place ask at best_ask."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market(97500.0, 97502.0)
        bs = make_bot(0.0005)
        bs.tighten_mode = True

        result = engine.generate_quotes(ms, bs)
        assert result.ask_price == 97502.0  # At best_ask
        assert result.bid_size == 0  # No adding-direction

    def test_tighten_mode_short_bid_at_best_bid(self):
        """Tighten mode with short position should place bid at best_bid."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market(97500.0, 97502.0)
        bs = make_bot(-0.0005)
        bs.tighten_mode = True

        result = engine.generate_quotes(ms, bs)
        assert result.bid_price == 97500.0  # At best_bid
        assert result.ask_size == 0  # No adding-direction

    def test_tighten_mode_no_effect_when_flat(self):
        """Tighten mode should not affect flat position."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market(97500.0, 97502.0)
        bs = make_bot(0.0)
        bs.tighten_mode = True

        result = engine.generate_quotes(ms, bs)
        assert result.bid_size > 0
        assert result.ask_size > 0


class TestSpreadWidening:
    """Test spread widens with inventory."""

    def test_heavy_inventory_wider_spread(self):
        """Heavy inventory should produce wider spread than no inventory."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        ms = make_market()

        # No inventory
        bs_zero = make_bot(0.0)
        q_zero = engine.generate_quotes(ms, bs_zero)
        spread_zero = q_zero.ask_price - q_zero.bid_price

        # Heavy inventory
        bs_heavy = make_bot(0.008)  # 80% of max
        q_heavy = engine.generate_quotes(ms, bs_heavy)
        spread_heavy = q_heavy.ask_price - q_heavy.bid_price

        assert spread_heavy >= spread_zero

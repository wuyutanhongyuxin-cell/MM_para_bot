"""Tests for QuoteEngine: Avellaneda-Stoikov + OBI overlay."""

import math
import time
import pytest
from src.quote_engine import QuoteEngine, VolatilityEngine
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
            "threshold": 0.2,
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


class TestOBIProtectiveFilter:
    """Test OBI protective filter (blocks adverse-side entries)."""

    def test_positive_obi_blocks_ask(self):
        """Positive OBI (buy pressure) should block ask when flat (don't sell into rally)."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        # Strong buy pressure — call multiple times to build up EMA
        bids = [(97500 - i, 10.0) for i in range(5)]
        asks = [(97502 + i, 1.0) for i in range(5)]
        for _ in range(10):
            engine.calc_obi(bids, asks)

        assert engine.obi_smooth > 0.2  # Above protective threshold

        ms = make_market()
        bs = make_bot(0.0)  # Flat
        result = engine.generate_quotes(ms, bs)

        # Ask should be blocked (don't sell into buy pressure)
        assert result.ask_size == 0
        # Bid should still be active
        assert result.bid_size > 0

    def test_negative_obi_blocks_bid(self):
        """Negative OBI (sell pressure) should block bid when flat (don't buy into selloff)."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        # Strong sell pressure
        bids = [(97500 - i, 1.0) for i in range(5)]
        asks = [(97502 + i, 10.0) for i in range(5)]
        for _ in range(10):
            engine.calc_obi(bids, asks)

        assert engine.obi_smooth < -0.2

        ms = make_market()
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size == 0
        assert result.ask_size > 0

    def test_obi_allows_exit_when_long(self):
        """Buy pressure should NOT block ask when we're long (need to sell to exit)."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        bids = [(97500 - i, 10.0) for i in range(5)]
        asks = [(97502 + i, 1.0) for i in range(5)]
        for _ in range(10):
            engine.calc_obi(bids, asks)

        ms = make_market()
        bs = make_bot(0.003)  # Long position — need ask to exit
        result = engine.generate_quotes(ms, bs)

        # Ask should NOT be blocked (we're long, ask = exit direction)
        assert result.ask_size > 0

    def test_obi_below_threshold_no_filter(self):
        """OBI below threshold should not filter any side."""
        engine = QuoteEngine(make_config(**{"obi.threshold": 0.3}))
        seed_prices(engine, [97501.0] * 20)

        bids = [(97500 - i, 5.0) for i in range(5)]
        asks = [(97502 + i, 5.0) for i in range(5)]
        engine.calc_obi(bids, asks)

        assert abs(engine.obi_smooth) < 0.3

        ms = make_market()
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size > 0
        assert result.ask_size > 0

    def test_obi_disabled(self):
        """With OBI disabled, no filtering should occur."""
        engine = QuoteEngine(make_config(**{"obi.enabled": False}))
        seed_prices(engine, [97501.0] * 20)

        bids = [(97500 - i, 100.0) for i in range(5)]
        asks = [(97502 + i, 1.0) for i in range(5)]
        engine.calc_obi(bids, asks)

        ms = make_market()
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size > 0
        assert result.ask_size > 0


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
    """Test volatility calculation (EWMA × Rogers-Satchell)."""

    def test_no_data_returns_min_sigma(self):
        """Before any data, vol engine should return min_sigma."""
        engine = QuoteEngine(make_config())
        sigma = engine.calc_volatility()
        assert sigma == 8.0  # min_sigma floor (default)

    def test_stable_prices_returns_min_sigma(self):
        """Stable prices within 1-second candle should stay at min_sigma."""
        engine = QuoteEngine(make_config())
        # All prices identical → H==L → candle skipped → min_sigma
        t = time.time()
        for i in range(20):
            engine.vol_engine.update(97501.0, t + i * 0.05)
        sigma = engine.calc_volatility()
        assert sigma == 8.0

    def test_volatile_prices_above_min(self):
        """Oscillating prices across 1-second candles should produce sigma > min."""
        engine = QuoteEngine(make_config())
        t = 1000000.0  # Aligned integer timestamp
        # Create multiple 1-second candles with meaningful OHLC variation
        # O, H, L, C must all be different for RS to be non-trivial
        for sec in range(5):
            engine.vol_engine.update(97000.0, t + sec)          # Open
            engine.vol_engine.update(97080.0, t + sec + 0.2)    # High
            engine.vol_engine.update(96920.0, t + sec + 0.5)    # Low (below open)
            engine.vol_engine.update(97040.0, t + sec + 0.8)    # Close
        # Close last candle
        engine.vol_engine.update(97000.0, t + 6)
        sigma = engine.calc_volatility()
        assert sigma > 8.0  # Should be well above min_sigma

    def test_insufficient_candles_returns_min(self):
        """With only 1 candle (not yet closed), should return min_sigma."""
        engine = QuoteEngine(make_config())
        t = time.time()
        engine.vol_engine.update(97500.0, t)
        engine.vol_engine.update(97520.0, t + 0.5)
        # Still in first candle (same second), not closed yet
        sigma = engine.calc_volatility()
        assert sigma == 8.0


class TestVolatilityEngine:
    """Direct tests for VolatilityEngine (EWMA × Rogers-Satchell)."""

    def test_rs_variance_correctness(self):
        """Verify RS variance formula: v = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)."""
        ve = VolatilityEngine(lambda_=0.94, min_sigma=1.0)
        # Use aligned integer timestamps to avoid bucket misalignment
        t = 1000000.0
        # Create a 1-second candle: O=97000, H=97050, L=96950, C=97010
        ve.update(97000.0, t)       # Open
        ve.update(97050.0, t + 0.2) # High
        ve.update(96950.0, t + 0.4) # Low
        ve.update(97010.0, t + 0.6) # Close
        # Close candle by starting new second
        ve.update(97005.0, t + 1.0)

        # Manually compute expected RS variance
        O, H, L, C = 97000.0, 97050.0, 96950.0, 97010.0
        expected_rs = (
            math.log(H / C) * math.log(H / O)
            + math.log(L / C) * math.log(L / O)
        )
        assert expected_rs > 0
        assert abs(ve.ewma_variance - expected_rs) < 1e-12

    def test_ewma_decay(self):
        """After a volatile candle, sigma should decay toward min with calm candles."""
        ve = VolatilityEngine(lambda_=0.94, min_sigma=1.0)
        t = 1000000.0  # Aligned integer timestamp

        # 1 volatile candle ($200 range)
        ve.update(97000.0, t)
        ve.update(97100.0, t + 0.2)
        ve.update(96900.0, t + 0.5)
        ve.update(97050.0, t + 0.8)

        # Close and start calm candles ($4 range each)
        sigma_after_spike = None
        for sec in range(1, 20):
            ve.update(97000.0, t + sec)
            ve.update(97002.0, t + sec + 0.2)
            ve.update(96998.0, t + sec + 0.5)
            ve.update(97001.0, t + sec + 0.8)
            if sec == 1:
                sigma_after_spike = ve.get_sigma()

        sigma_decayed = ve.get_sigma()
        # Sigma should have decayed significantly
        if sigma_after_spike and sigma_after_spike > 1.0:
            assert sigma_decayed < sigma_after_spike

    def test_responds_to_spike(self):
        """Sigma should increase after a volatility spike."""
        ve = VolatilityEngine(lambda_=0.94, min_sigma=1.0)
        # Use aligned integer timestamps
        t = 1000000.0

        # Several calm candles first ($5 range)
        for sec in range(5):
            ve.update(97000.0, t + sec)
            ve.update(97005.0, t + sec + 0.3)
            ve.update(96995.0, t + sec + 0.6)
            ve.update(97002.0, t + sec + 0.9)

        sigma_calm = ve.get_sigma()

        # Multiple spike candles ($400 range) — λ=0.94 means each contributes 6%,
        # so need several spike candles for sigma to increase meaningfully
        for spike_offset in range(3):
            spike_sec = 6 + spike_offset
            ve.update(97000.0, t + spike_sec)
            ve.update(97200.0, t + spike_sec + 0.2)
            ve.update(96800.0, t + spike_sec + 0.5)
            ve.update(97050.0, t + spike_sec + 0.8)

        # Close last spike candle
        ve.update(97000.0, t + 10.0)
        sigma_spike = ve.get_sigma()

        # With 3 spike candles (6% contribution each ≈ 17% total weight),
        # sigma should increase noticeably
        assert sigma_spike > sigma_calm * 1.5

    def test_min_sigma_floor(self):
        """Sigma should never go below min_sigma."""
        ve = VolatilityEngine(lambda_=0.94, min_sigma=8.0)
        t = 1000000.0  # Aligned integer timestamp

        # Tiny range candles ($2)
        for sec in range(10):
            ve.update(97000.0, t + sec)
            ve.update(97001.0, t + sec + 0.2)
            ve.update(96999.0, t + sec + 0.5)
            ve.update(97000.0, t + sec + 0.8)

        # Close last candle
        ve.update(97000.0, t + 11)
        sigma = ve.get_sigma()
        assert sigma >= 8.0


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


class TestMomentumGuard:
    """Test momentum guard — blocks entries in trend direction."""

    def test_uptrend_blocks_ask(self):
        """Strong uptrend should block ask (don't sell into rally) when flat."""
        engine = QuoteEngine(make_config(**{
            "strategy.momentum_threshold": 30,
            "strategy.momentum_window": 300,
        }))
        # Price rising $50 over window
        t = time.time()
        for i in range(20):
            engine.mid_prices.append((t + i * 15, 97501.0 + i * 2.5))

        ms = make_market(97548.0, 97550.0)
        bs = make_bot(0.0)  # Flat
        result = engine.generate_quotes(ms, bs)

        assert result.ask_size == 0  # Blocked
        assert result.bid_size > 0   # Still active

    def test_downtrend_blocks_bid(self):
        """Strong downtrend should block bid (don't buy into selloff) when flat."""
        engine = QuoteEngine(make_config(**{
            "strategy.momentum_threshold": 30,
            "strategy.momentum_window": 300,
        }))
        t = time.time()
        for i in range(20):
            engine.mid_prices.append((t + i * 15, 97501.0 - i * 2.5))

        ms = make_market(97450.0, 97452.0)
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size == 0
        assert result.ask_size > 0

    def test_uptrend_allows_exit_when_long(self):
        """Uptrend should NOT block ask when long (ask = exit/profit-taking)."""
        engine = QuoteEngine(make_config(**{
            "strategy.momentum_threshold": 30,
            "strategy.momentum_window": 300,
        }))
        t = time.time()
        for i in range(20):
            engine.mid_prices.append((t + i * 15, 97501.0 + i * 2.5))

        ms = make_market(97548.0, 97550.0)
        bs = make_bot(0.003)  # Long — ask is exit direction
        result = engine.generate_quotes(ms, bs)

        assert result.ask_size > 0  # NOT blocked (exit direction)

    def test_downtrend_allows_exit_when_short(self):
        """Downtrend should NOT block bid when short (bid = exit/cover)."""
        engine = QuoteEngine(make_config(**{
            "strategy.momentum_threshold": 30,
            "strategy.momentum_window": 300,
        }))
        t = time.time()
        for i in range(20):
            engine.mid_prices.append((t + i * 15, 97501.0 - i * 2.5))

        ms = make_market(97450.0, 97452.0)
        bs = make_bot(-0.003)  # Short — bid is exit direction
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size > 0  # NOT blocked

    def test_no_momentum_no_filter(self):
        """Flat market should not trigger any filter."""
        engine = QuoteEngine(make_config(**{
            "strategy.momentum_threshold": 30,
        }))
        seed_prices(engine, [97501.0] * 20)

        ms = make_market()
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size > 0
        assert result.ask_size > 0

    def test_tighten_mode_overrides_momentum(self):
        """Tighten mode exit should NOT be blocked by momentum guard."""
        engine = QuoteEngine(make_config(**{
            "strategy.momentum_threshold": 30,
        }))
        # Strong uptrend
        t = time.time()
        for i in range(20):
            engine.mid_prices.append((t + i * 15, 97501.0 + i * 2.5))

        ms = make_market(97548.0, 97550.0)
        bs = make_bot(-0.003)  # Short position
        bs.tighten_mode = True  # Need to exit urgently

        result = engine.generate_quotes(ms, bs)
        # bid should be active (exit direction for short), NOT blocked by momentum
        assert result.bid_size > 0


class TestVolPause:
    """Test vol-adaptive pause — pauses quoting when 60s price range exceeds threshold."""

    def _seed_volatile(self, engine, range_dollars=15):
        """Seed prices with high 60s range (oscillating)."""
        t = time.time()
        for i in range(20):
            price = 97501.0 + (i % 2) * range_dollars
            engine.mid_prices.append((t + i * 3, price))  # 3s apart, all within 60s

    def _seed_calm(self, engine):
        """Seed prices with low 60s range (stable)."""
        t = time.time()
        for i in range(20):
            engine.mid_prices.append((t + i * 3, 97501.0 + (i % 2) * 2))  # $2 range

    def test_high_vol_pauses_both_when_flat(self):
        """When flat and 60s range > threshold, both sides should be paused."""
        engine = QuoteEngine(make_config(**{"strategy.vol_pause_threshold": 10}))
        self._seed_volatile(engine, 15)  # $15 range > $10 threshold

        ms = make_market()
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size == 0
        assert result.ask_size == 0

    def test_high_vol_keeps_exit_when_long(self):
        """When long and volatile, entry side paused but exit side kept."""
        engine = QuoteEngine(make_config(**{"strategy.vol_pause_threshold": 10}))
        self._seed_volatile(engine, 15)

        ms = make_market()
        bs = make_bot(0.003)  # Long — ask is exit
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size == 0   # Entry side paused
        assert result.ask_size > 0    # Exit side kept

    def test_high_vol_keeps_exit_when_short(self):
        """When short and volatile, entry side paused but exit side kept."""
        engine = QuoteEngine(make_config(**{"strategy.vol_pause_threshold": 10}))
        self._seed_volatile(engine, 15)

        ms = make_market()
        bs = make_bot(-0.003)  # Short — bid is exit
        result = engine.generate_quotes(ms, bs)

        assert result.ask_size == 0   # Entry side paused
        assert result.bid_size > 0    # Exit side kept

    def test_low_vol_no_pause(self):
        """When 60s range < threshold, no pause should occur."""
        engine = QuoteEngine(make_config(**{"strategy.vol_pause_threshold": 10}))
        self._seed_calm(engine)  # $2 range < $10 threshold

        ms = make_market()
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size > 0
        assert result.ask_size > 0

    def test_tighten_mode_overrides_vol_pause(self):
        """Tighten mode should override vol-pause (exit takes priority)."""
        engine = QuoteEngine(make_config(**{"strategy.vol_pause_threshold": 10}))
        self._seed_volatile(engine, 15)

        ms = make_market(97500.0, 97502.0)
        bs = make_bot(0.003)  # Long
        bs.tighten_mode = True  # Urgent exit

        result = engine.generate_quotes(ms, bs)
        # Ask (exit) should be active despite high volatility
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


class TestSpreadFactor:
    """Test configurable spread_factor parameter."""

    def test_lower_spread_factor_tighter_quotes(self):
        """Lower spread_factor should produce tighter half_spread."""
        # Higher spread_factor (default 0.4)
        engine_wide = QuoteEngine(make_config(**{"strategy.spread_factor": 0.4}))
        seed_prices(engine_wide, [97501.0 + (i % 2) * 5 for i in range(20)])

        # Lower spread_factor (0.25)
        engine_tight = QuoteEngine(make_config(**{"strategy.spread_factor": 0.25}))
        seed_prices(engine_tight, [97501.0 + (i % 2) * 5 for i in range(20)])

        ms = make_market(97498.0, 97504.0)  # $6 spread
        bs = make_bot(0.0)

        q_wide = engine_wide.generate_quotes(ms, bs)
        q_tight = engine_tight.generate_quotes(ms, bs)

        # Tighter spread_factor -> smaller half_spread -> narrower bid-ask
        assert q_tight.half_spread <= q_wide.half_spread

    def test_spread_factor_default_backward_compat(self):
        """Without spread_factor in config, should default to 0.4."""
        engine = QuoteEngine(make_config())  # No spread_factor key
        assert engine.spread_factor == 0.4

    def test_spread_factor_from_config(self):
        """spread_factor should be read from config."""
        engine = QuoteEngine(make_config(**{"strategy.spread_factor": 0.25}))
        assert engine.spread_factor == 0.25

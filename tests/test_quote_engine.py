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

    def test_positive_obi_blocks_both_when_flat(self):
        """Positive OBI + flat → flat guard blocks BOTH sides (one-sided = directional bet)."""
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

        # Flat guard: both sides blocked (one-sided entry when flat = directional bet)
        assert result.ask_size == 0
        assert result.bid_size == 0

    def test_negative_obi_blocks_both_when_flat(self):
        """Negative OBI + flat → flat guard blocks BOTH sides."""
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

        # Flat guard: both sides blocked
        assert result.bid_size == 0
        assert result.ask_size == 0

    def test_obi_one_sided_when_holding_position(self):
        """OBI should only block one side when there's a position to protect."""
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        # Sell pressure: blocks bid for long position (protects from adding)
        bids = [(97500 - i, 1.0) for i in range(5)]
        asks = [(97502 + i, 10.0) for i in range(5)]
        for _ in range(10):
            engine.calc_obi(bids, asks)

        ms = make_market()
        bs = make_bot(0.003)  # Long position — NOT flat
        result = engine.generate_quotes(ms, bs)

        # With position: bid blocked (protect), ask active (exit)
        # Flat guard does NOT fire because net_pos > 0.0001
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

    def test_obi_blocks_both_when_near_flat(self):
        """REGRESSION: pos=-0.0003 (< base_size) treated as flat, not 'short'.
        Bug: OBI=-0.478 + pos=-0.0003 → guards treated as 'short' → bid allowed
        → bot bought $68376 into a $170 selloff → -$0.4495 single loss.
        Fix: abs(net_pos) < base_size → treat as flat → block BOTH sides.
        """
        engine = QuoteEngine(make_config())
        seed_prices(engine, [97501.0] * 20)

        # Sell pressure OBI
        bids = [(97500 - i, 1.0) for i in range(5)]
        asks = [(97502 + i, 10.0) for i in range(5)]
        for _ in range(10):
            engine.calc_obi(bids, asks)

        ms = make_market()
        # Position is tiny (< base_size=0.001) — effectively flat
        bs = make_bot(-0.0003)
        result = engine.generate_quotes(ms, bs)

        # BOTH sides should be blocked (effectively flat + OBI signal)
        assert result.bid_size == 0
        assert result.ask_size == 0

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


class TestVolWindowGuard:
    """vol_window must be >= max(momentum_window, 60) to avoid truncation."""

    def test_vol_window_auto_corrects_when_too_small(self):
        """vol_window < momentum_window should be auto-corrected."""
        engine = QuoteEngine(make_config(**{
            "strategy.vol_window": 20,
            "strategy.momentum_window": 45,
        }))
        # Auto-corrected to max(45, 60) = 60
        assert engine.vol_window >= 60
        assert engine.vol_time_window >= 60

    def test_vol_window_ok_when_large_enough(self):
        """vol_window >= max(momentum_window, 60) should stay as-is."""
        engine = QuoteEngine(make_config(**{
            "strategy.vol_window": 90,
            "strategy.momentum_window": 45,
        }))
        assert engine.vol_window == 90

    def test_momentum_uses_full_window(self):
        """Momentum should use the full momentum_window of data, not truncated."""
        engine = QuoteEngine(make_config(**{
            "strategy.vol_window": 60,
            "strategy.momentum_window": 45,
            "strategy.momentum_threshold": 100,
        }))
        t = time.time()
        # Seed 50s of data: slow uptrend $1/s = $45 over 45s
        for i in range(50):
            engine.mid_prices.append((t + i, 97500.0 + i))
        # Momentum over 45s should be ~$45 (not truncated to 20s = $20)
        momentum = engine.calc_momentum()
        assert momentum > 40  # Full 45s window used


class TestVolatility:
    """Test volatility calculation (EWMA × Rogers-Satchell on 5s candles)."""

    def test_no_data_returns_min_sigma(self):
        """Before any data, vol engine should return min_sigma."""
        engine = QuoteEngine(make_config())
        sigma = engine.calc_volatility()
        assert sigma == 8.0  # min_sigma floor (default)

    def test_stable_prices_returns_min_sigma(self):
        """Stable prices within one 5s candle should stay at min_sigma."""
        engine = QuoteEngine(make_config())
        # All prices identical → H==L → close-to-close fallback needs prev_close
        t = 1000000.0  # Aligned to 5s boundary
        for i in range(20):
            engine.vol_engine.update(97501.0, t + i * 0.2)
        sigma = engine.calc_volatility()
        assert sigma == 8.0  # No candle closed yet or cc return = 0

    def test_volatile_prices_above_min(self):
        """Oscillating prices across 5-second candles should produce sigma > min."""
        engine = QuoteEngine(make_config())
        t = 1000000.0  # Aligned to 5s boundary
        # Create multiple 5-second candles with meaningful OHLC variation
        for candle_idx in range(5):
            base = t + candle_idx * 5
            engine.vol_engine.update(97000.0, base)          # Open
            engine.vol_engine.update(97080.0, base + 1.0)    # High
            engine.vol_engine.update(96920.0, base + 2.5)    # Low
            engine.vol_engine.update(97040.0, base + 4.0)    # Close
        # Close last candle by starting new one
        engine.vol_engine.update(97000.0, t + 30)
        sigma = engine.calc_volatility()
        assert sigma > 8.0  # Should be well above min_sigma

    def test_insufficient_candles_returns_min(self):
        """With only 1 candle (not yet closed), should return min_sigma."""
        engine = QuoteEngine(make_config())
        t = 1000000.0
        engine.vol_engine.update(97500.0, t)
        engine.vol_engine.update(97520.0, t + 2.0)
        # Still in first 5s candle, not closed yet
        sigma = engine.calc_volatility()
        assert sigma == 8.0

    def test_cc_fallback_when_h_equals_l(self):
        """When all candles have H==L (1 tick each), close-to-close fallback kicks in."""
        engine = QuoteEngine(make_config())
        t = 1000000.0
        # Each 5s candle has exactly 1 tick → H==L → RS skipped
        # But close-to-close (log(C/C_prev))² provides variance
        engine.vol_engine.update(97000.0, t)           # candle 0: single tick
        engine.vol_engine.update(97050.0, t + 5)       # candle 1: cc return $50
        engine.vol_engine.update(96950.0, t + 10)      # candle 2: cc return -$100
        engine.vol_engine.update(97100.0, t + 15)      # candle 3: cc return $150
        engine.vol_engine.update(96900.0, t + 20)      # candle 4: closes candle 3
        sigma = engine.calc_volatility()
        # With $50-150 moves between candles, sigma should exceed min_sigma
        assert sigma > 8.0


class TestVolatilityEngine:
    """Direct tests for VolatilityEngine (EWMA × Rogers-Satchell, 5s candles)."""

    def test_rs_variance_correctness(self):
        """Verify RS variance formula: v = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)."""
        ve = VolatilityEngine(lambda_=0.94, min_sigma=1.0)
        t = 1000000.0  # Aligned to 5s boundary
        # Create a 5-second candle: O=97000, H=97050, L=96950, C=97010
        ve.update(97000.0, t)       # Open
        ve.update(97050.0, t + 1.0) # High
        ve.update(96950.0, t + 2.5) # Low
        ve.update(97010.0, t + 4.0) # Close
        # Close candle by starting next 5s bucket
        ve.update(97005.0, t + 5.0)

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
        t = 1000000.0  # Aligned to 5s boundary

        # 1 volatile 5s candle ($200 range)
        ve.update(97000.0, t)
        ve.update(97100.0, t + 1.0)
        ve.update(96900.0, t + 2.5)
        ve.update(97050.0, t + 4.0)

        # Close and start calm 5s candles ($4 range each)
        sigma_after_spike = None
        for candle_idx in range(1, 10):
            base = t + candle_idx * 5
            ve.update(97000.0, base)
            ve.update(97002.0, base + 1.0)
            ve.update(96998.0, base + 2.5)
            ve.update(97001.0, base + 4.0)
            if candle_idx == 1:
                sigma_after_spike = ve.get_sigma()

        # Close last calm candle
        ve.update(97000.0, t + 55)
        sigma_decayed = ve.get_sigma()
        # Sigma should have decayed significantly
        if sigma_after_spike and sigma_after_spike > 1.0:
            assert sigma_decayed < sigma_after_spike

    def test_responds_to_spike(self):
        """Sigma should increase after a volatility spike."""
        ve = VolatilityEngine(lambda_=0.94, min_sigma=1.0)
        t = 1000000.0  # Aligned to 5s boundary

        # Several calm 5s candles first ($5 range)
        for candle_idx in range(3):
            base = t + candle_idx * 5
            ve.update(97000.0, base)
            ve.update(97005.0, base + 1.5)
            ve.update(96995.0, base + 3.0)
            ve.update(97002.0, base + 4.5)

        sigma_calm = ve.get_sigma()

        # Multiple spike 5s candles ($400 range)
        for spike_idx in range(3):
            base = t + (3 + spike_idx) * 5
            ve.update(97000.0, base)
            ve.update(97200.0, base + 1.0)
            ve.update(96800.0, base + 2.5)
            ve.update(97050.0, base + 4.0)

        # Close last spike candle
        ve.update(97000.0, t + 35)
        sigma_spike = ve.get_sigma()

        # With 3 spike candles, sigma should increase noticeably
        assert sigma_spike > sigma_calm * 1.5

    def test_min_sigma_floor(self):
        """Sigma should never go below min_sigma."""
        ve = VolatilityEngine(lambda_=0.94, min_sigma=8.0)
        t = 1000000.0

        # Tiny range 5s candles ($2)
        for candle_idx in range(5):
            base = t + candle_idx * 5
            ve.update(97000.0, base)
            ve.update(97001.0, base + 1.0)
            ve.update(96999.0, base + 2.5)
            ve.update(97000.0, base + 4.0)

        # Close last candle
        ve.update(97000.0, t + 30)
        sigma = ve.get_sigma()
        assert sigma >= 8.0

    def test_cc_fallback_initializes(self):
        """Close-to-close fallback should initialize EWMA when RS cannot."""
        ve = VolatilityEngine(lambda_=0.94, min_sigma=1.0)
        t = 1000000.0
        # Single-tick 5s candles (H==L) → RS fails, cc fallback used
        ve.update(97000.0, t)           # candle 0: 1 tick, prev_close=0
        ve.update(97100.0, t + 5)       # candle 1: closes candle 0, cc=$100 return
        ve.update(96900.0, t + 10)      # candle 2: closes candle 1, cc=-$200 return
        assert ve._initialized  # Should be initialized via cc fallback
        sigma = ve.get_sigma()
        assert sigma > 1.0  # cc returns should produce meaningful sigma

    def test_5s_bucketing(self):
        """Timestamps within same 5s window should belong to same candle."""
        ve = VolatilityEngine(lambda_=0.94, min_sigma=1.0)
        t = 1000000.0  # Exact 5s boundary
        ve.update(97000.0, t)        # t=0s
        ve.update(97010.0, t + 1.0)  # t=1s, same candle
        ve.update(97020.0, t + 2.5)  # t=2.5s, same candle
        ve.update(97005.0, t + 4.9)  # t=4.9s, same candle
        assert ve._candle_count == 4
        assert ve._candle_high == 97020.0
        assert ve._candle_low == 97000.0


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

    def test_uptrend_blocks_both_when_flat(self):
        """Strong uptrend + flat → flat guard blocks BOTH sides (one-sided = directional bet)."""
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

        # Flat guard: momentum blocked ask → both sides paused
        assert result.ask_size == 0
        assert result.bid_size == 0

    def test_downtrend_blocks_both_when_flat(self):
        """Strong downtrend + flat → flat guard blocks BOTH sides."""
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

        # Flat guard: momentum blocked bid → both sides paused
        assert result.bid_size == 0
        assert result.ask_size == 0

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

    def test_downtrend_blocks_both_when_near_flat(self):
        """REGRESSION: tiny short position (-0.0003) should be treated as flat.
        With old code: downtrend + pos=-0.0003 → bid allowed → adverse selection.
        """
        engine = QuoteEngine(make_config(**{
            "strategy.momentum_threshold": 30,
            "strategy.momentum_window": 300,
        }))
        t = time.time()
        for i in range(20):
            engine.mid_prices.append((t + i * 15, 97501.0 - i * 2.5))

        ms = make_market(97450.0, 97452.0)
        # Tiny short position (< base_size=0.001) — effectively flat
        bs = make_bot(-0.0003)
        result = engine.generate_quotes(ms, bs)

        # BOTH sides should be blocked (flat + momentum)
        assert result.bid_size == 0
        assert result.ask_size == 0

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

    def _seed_volatile(self, engine, range_dollars=80):
        """Seed prices with high 60s range (oscillating).
        Default $80 exceeds dynamic threshold max(base, 2.5×sigma×√12).
        With min_sigma=8: dynamic_threshold ≈ $69, so $80 triggers pause.
        """
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
        """When flat and 60s range > dynamic threshold, both sides should be paused."""
        engine = QuoteEngine(make_config(**{"strategy.vol_pause_threshold": 10}))
        self._seed_volatile(engine)  # $80 range > dynamic ~$69

        ms = make_market()
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size == 0
        assert result.ask_size == 0

    def test_high_vol_keeps_exit_when_long(self):
        """When long and volatile, entry side paused but exit side kept."""
        engine = QuoteEngine(make_config(**{"strategy.vol_pause_threshold": 10}))
        self._seed_volatile(engine)

        ms = make_market()
        bs = make_bot(0.003)  # Long — ask is exit
        result = engine.generate_quotes(ms, bs)

        assert result.bid_size == 0   # Entry side paused
        assert result.ask_size > 0    # Exit side kept

    def test_high_vol_keeps_exit_when_short(self):
        """When short and volatile, entry side paused but exit side kept."""
        engine = QuoteEngine(make_config(**{"strategy.vol_pause_threshold": 10}))
        self._seed_volatile(engine)

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
        self._seed_volatile(engine)

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


class TestRapidMomentumGuard:
    """Test rapid momentum micro-guard (10s window flash move detection).

    Ref: Cartea et al. (2015) Ch.10 — inventory risk in fast markets.
    Live data: $86 drop in 6s caused 6 cascading BUY fills.
    """

    def _seed_flash_drop(self, engine, drop_amount=50):
        """Seed prices that simulate a flash drop over ~10 seconds."""
        t = time.time()
        base = 97500.0
        # 60 prices over 10 seconds: first half stable, second half dropping
        for i in range(30):
            engine.mid_prices.append((t - 10 + i * 0.2, base))
        for i in range(30):
            price = base - (drop_amount * (i + 1) / 30)
            engine.mid_prices.append((t - 4 + i * 0.13, price))

    def _seed_flash_rally(self, engine, rally_amount=50):
        """Seed prices that simulate a flash rally over ~10 seconds."""
        t = time.time()
        base = 97500.0
        for i in range(30):
            engine.mid_prices.append((t - 10 + i * 0.2, base))
        for i in range(30):
            price = base + (rally_amount * (i + 1) / 30)
            engine.mid_prices.append((t - 4 + i * 0.13, price))

    def test_flash_drop_blocks_both_when_flat(self):
        """Flash drop should block both sides when flat (flat guard)."""
        engine = QuoteEngine(make_config(**{
            "strategy.momentum_threshold": 200,  # High so 45s guard doesn't fire
        }))
        self._seed_flash_drop(engine, drop_amount=60)

        ms = make_market(97450.0, 97452.0)
        bs = make_bot(0.0)
        result = engine.generate_quotes(ms, bs)

        # Flat + rapid momentum → both sides blocked
        assert result.bid_size == 0
        assert result.ask_size == 0

    def test_flash_drop_blocks_bid_when_long(self):
        """Flash drop with long position should block bid (don't add)."""
        engine = QuoteEngine(make_config(**{
            "strategy.momentum_threshold": 200,
        }))
        self._seed_flash_drop(engine, drop_amount=60)

        ms = make_market(97450.0, 97452.0)
        bs = make_bot(0.003)  # Long
        result = engine.generate_quotes(ms, bs)

        # Long + fast drop → bid blocked (don't buy more into drop)
        assert result.bid_size == 0
        assert result.ask_size > 0  # Exit side still allowed

    def test_flash_rally_blocks_ask_when_short(self):
        """Flash rally with short position should block ask (don't add)."""
        engine = QuoteEngine(make_config(**{
            "strategy.momentum_threshold": 200,
        }))
        self._seed_flash_rally(engine, rally_amount=60)

        ms = make_market(97550.0, 97552.0)
        bs = make_bot(-0.003)  # Short
        result = engine.generate_quotes(ms, bs)

        # Short + fast rally → ask blocked (don't sell more into rally)
        assert result.ask_size == 0
        assert result.bid_size > 0  # Exit side still allowed

    def test_small_rapid_move_no_block(self):
        """Small rapid move (< threshold) should not trigger guard."""
        engine = QuoteEngine(make_config(**{
            "strategy.momentum_threshold": 200,
        }))
        self._seed_flash_drop(engine, drop_amount=15)  # Only $15 < $30 threshold

        ms = make_market(97485.0, 97487.0)
        bs = make_bot(0.003)
        result = engine.generate_quotes(ms, bs)

        # Small move → no rapid momentum block
        assert result.bid_size > 0 or result.ask_size > 0

    def test_calc_rapid_momentum_empty(self):
        """calc_rapid_momentum should return 0 with insufficient data."""
        engine = QuoteEngine(make_config())
        assert engine.calc_rapid_momentum(10) == 0.0

    def test_calc_rapid_momentum_value(self):
        """calc_rapid_momentum should return price change over window."""
        engine = QuoteEngine(make_config())
        t = time.time()
        engine.mid_prices.append((t - 5, 97500.0))
        engine.mid_prices.append((t, 97460.0))
        rapid = engine.calc_rapid_momentum(10)
        assert rapid == pytest.approx(-40.0)

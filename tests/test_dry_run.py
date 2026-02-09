"""Tests for dry-run mode and state machine transitions."""

import pytest
import time
from src.state import MarketState, BotState, PnLTracker
from src.quote_engine import QuoteEngine
from src.risk_manager import RiskManager


def make_config():
    return {
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
        "risk": {
            "max_loss_per_trade": 0.30,
            "max_loss_per_hour": 0.50,
            "max_loss_per_day": 1.00,
            "inventory_timeout": 120,
            "emergency_timeout": 300,
            "max_unrealized_loss": 0.30,
        },
        "rate_limit": {
            "max_orders_per_second": 2,
            "max_orders_per_minute": 25,
            "max_orders_per_hour": 280,
            "max_orders_per_day": 950,
        },
        "schedule": {
            "active_hours_utc": list(range(24)),  # All hours active for testing
            "pause_hours_utc": [],
        },
    }


class TestPnLTracker:
    """Test PnL tracking across fills."""

    def test_buy_then_sell_profit(self):
        """Buy low, sell high should produce profit."""
        tracker = PnLTracker()

        # Buy 0.001 BTC at 97500
        s1 = tracker.on_fill("BUY", 97500.0, 0.001)
        assert s1["net_position"] == pytest.approx(0.001)
        assert s1["realized_pnl"] == 0.0

        # Sell 0.001 BTC at 97510
        s2 = tracker.on_fill("SELL", 97510.0, 0.001)
        assert s2["net_position"] == pytest.approx(0.0)
        assert s2["realized_pnl"] == pytest.approx(0.01)  # $10 * 0.001

    def test_buy_then_sell_loss(self):
        """Buy high, sell low should produce loss."""
        tracker = PnLTracker()
        tracker.on_fill("BUY", 97500.0, 0.001)
        s2 = tracker.on_fill("SELL", 97490.0, 0.001)

        assert s2["realized_pnl"] == pytest.approx(-0.01)  # -$10 * 0.001

    def test_short_profit(self):
        """Sell high, buy low should produce profit (short)."""
        tracker = PnLTracker()
        tracker.on_fill("SELL", 97510.0, 0.001)
        s2 = tracker.on_fill("BUY", 97500.0, 0.001)

        assert s2["realized_pnl"] == pytest.approx(0.01)

    def test_partial_close(self):
        """Partial close should realize proportional PnL."""
        tracker = PnLTracker()
        tracker.on_fill("BUY", 97500.0, 0.002)
        s2 = tracker.on_fill("SELL", 97510.0, 0.001)

        assert s2["net_position"] == pytest.approx(0.001)
        assert s2["realized_pnl"] == pytest.approx(0.01)  # $10 * 0.001

    def test_multiple_trades_cumulative(self):
        """Multiple round trips should accumulate PnL."""
        tracker = PnLTracker()

        # Round trip 1: +$0.01
        tracker.on_fill("BUY", 97500.0, 0.001)
        tracker.on_fill("SELL", 97510.0, 0.001)

        # Round trip 2: -$0.005
        tracker.on_fill("BUY", 97500.0, 0.001)
        tracker.on_fill("SELL", 97495.0, 0.001)

        stats = tracker.get_stats()
        assert stats["total_pnl"] == pytest.approx(0.005)
        assert stats["total_trades"] == 4
        assert stats["closing_trades"] == 2

    def test_stats_win_rate(self):
        """Win rate should reflect winning vs losing trades."""
        tracker = PnLTracker()

        # 3 wins, 1 loss
        for _ in range(3):
            tracker.on_fill("BUY", 97500.0, 0.001)
            tracker.on_fill("SELL", 97510.0, 0.001)

        tracker.on_fill("BUY", 97500.0, 0.001)
        tracker.on_fill("SELL", 97490.0, 0.001)

        stats = tracker.get_stats()
        assert stats["win_rate"] == pytest.approx(0.75)

    def test_empty_stats(self):
        """Empty tracker should return zero stats."""
        tracker = PnLTracker()
        stats = tracker.get_stats()
        assert stats["total_trades"] == 0
        assert stats["total_pnl"] == 0.0
        assert stats["win_rate"] == 0.0


class TestStateMachineTransitions:
    """Test bot state transitions that would occur in dry-run."""

    def test_idle_to_quoting(self):
        """From IDLE, generating quotes should produce bid+ask."""
        engine = QuoteEngine(make_config())
        for p in [97501.0] * 20:
            engine.mid_prices.append(p)

        ms = MarketState()
        ms.update_bbo(97500.0, 97502.0)

        bs = BotState()
        bs.net_position = 0.0

        result = engine.generate_quotes(ms, bs)
        assert result.bid_price > 0
        assert result.ask_price > 0
        assert result.bid_size > 0
        assert result.ask_size > 0

    def test_bid_fill_creates_long(self):
        """A bid fill should create a long position."""
        tracker = PnLTracker()
        summary = tracker.on_fill("BUY", 97500.0, 0.001)

        assert summary["net_position"] > 0
        assert summary["side"] == "BUY"

    def test_ask_fill_creates_short(self):
        """An ask fill should create a short position."""
        tracker = PnLTracker()
        summary = tracker.on_fill("SELL", 97502.0, 0.001)

        assert summary["net_position"] < 0

    def test_both_fills_return_to_idle(self):
        """Bid fill + ask fill should return to zero position (spread captured)."""
        tracker = PnLTracker()
        tracker.on_fill("BUY", 97500.0, 0.001)
        s2 = tracker.on_fill("SELL", 97502.0, 0.001)

        assert abs(s2["net_position"]) < 0.00001
        assert s2["realized_pnl"] > 0  # Captured spread

    def test_long_inventory_skewed_quotes(self):
        """Long position should produce skewed quotes for exit."""
        engine = QuoteEngine(make_config())
        for p in [97501.0] * 20:
            engine.mid_prices.append(p)

        ms = MarketState()
        ms.update_bbo(97500.0, 97502.0)

        bs = BotState()
        bs.net_position = 0.005

        result = engine.generate_quotes(ms, bs)
        # With long inventory, bid size should be smaller than ask size
        assert result.bid_size <= result.ask_size

    def test_market_state_validity(self):
        """MarketState should track validity."""
        ms = MarketState()
        assert not ms.is_valid

        ms.update_bbo(97500.0, 97502.0)
        assert ms.is_valid
        assert ms.mid_price == pytest.approx(97501.0)
        assert ms.spread == pytest.approx(2.0)


class TestDryRunSimulation:
    """Test simulated trading scenario."""

    def test_full_simulation_cycle(self):
        """Simulate a full quote -> fill -> exit cycle."""
        config = make_config()
        engine = QuoteEngine(config)
        rm = RiskManager(config)
        tracker = PnLTracker()

        # Seed volatility data
        for p in [97501.0] * 20:
            engine.mid_prices.append(p)

        ms = MarketState()
        ms.update_bbo(97500.0, 97502.0)

        bs = BotState()

        # Step 1: Generate quotes (IDLE -> QUOTING)
        quotes = engine.generate_quotes(ms, bs)
        assert quotes.bid_price > 0
        assert quotes.ask_price > 0

        # Step 2: Simulate bid fill (QUOTING -> INVENTORY_LONG)
        summary = tracker.on_fill("BUY", quotes.bid_price, quotes.bid_size)
        bs.net_position = summary["net_position"]
        bs.position_entry_time = time.time()
        assert bs.has_position

        # Step 3: Generate skewed quotes for exit
        quotes2 = engine.generate_quotes(ms, bs)
        # Ask should still be present for exit
        assert quotes2.ask_size > 0

        # Step 4: Simulate ask fill (INVENTORY_LONG -> IDLE)
        summary2 = tracker.on_fill("SELL", quotes2.ask_price, quotes2.ask_size)
        bs.net_position = summary2["net_position"]

        # Should be back to ~flat
        assert not bs.has_position or abs(bs.net_position) < 0.0001

    def test_rate_limit_respected_in_simulation(self):
        """Rate limits should be tracked even in simulation."""
        rm = RiskManager(make_config())

        # Simulate placing many orders
        for _ in range(950):
            rm.record_order()

        # Should be near daily limit
        assert rm.can_place_orders(1) is False or rm.get_usage()["day"] >= 950

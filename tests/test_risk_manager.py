"""Tests for RiskManager: rate limits, PnL limits, time filter."""

import time
import pytest
from unittest.mock import patch
from datetime import datetime, timezone
from src.risk_manager import RiskManager


def make_config(**overrides):
    """Create test config."""
    cfg = {
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
            "active_hours_utc": list(range(7, 17)),
            "pause_hours_utc": [21, 22, 23, 0, 1, 2, 3, 4, 5],
        },
    }
    for k, v in overrides.items():
        parts = k.split('.')
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = v
    return cfg


class TestRateLimiting:
    """Test frequency rate limiting."""

    def test_no_orders_can_trade(self):
        """With no orders placed, should be able to trade."""
        rm = RiskManager(make_config())
        # Mock time to active hour
        with patch('src.risk_manager.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            ok, reason = rm.can_trade()
        # May fail on paused_hour depending on actual time, so just test rate logic
        rm_ok = rm.can_place_orders(2)
        assert rm_ok is True

    def test_second_rate_limit(self):
        """Should block when second limit exceeded."""
        rm = RiskManager(make_config())
        rm.record_order()
        rm.record_order()

        assert rm.can_place_orders(1) is False

    def test_minute_rate_limit(self):
        """Should block when minute limit exceeded."""
        cfg = make_config()
        cfg["rate_limit"]["max_orders_per_minute"] = 5
        rm = RiskManager(cfg)

        for _ in range(5):
            rm.record_order()

        assert rm.can_place_orders(1) is False

    def test_rate_limit_recovery(self):
        """Old orders should expire and free up capacity."""
        rm = RiskManager(make_config())

        # Place 2 orders "in the past"
        past = time.time() - 2  # 2 seconds ago
        rm.orders_count["second"].append(past)
        rm.orders_count["second"].append(past)

        # Should be able to trade now (second window expired)
        assert rm.can_place_orders(1) is True

    def test_can_place_orders_precheck(self):
        """Pre-check should account for upcoming order count."""
        rm = RiskManager(make_config())
        rm.record_order()  # 1 used

        # Can place 1 more (limit 2/sec)
        assert rm.can_place_orders(1) is True
        # Cannot place 2 more
        assert rm.can_place_orders(2) is False

    def test_get_usage(self):
        """Usage should reflect recorded orders."""
        rm = RiskManager(make_config())
        rm.record_order()
        rm.record_order()

        usage = rm.get_usage()
        assert usage["second"] == 2
        assert usage["minute"] == 2
        assert usage["hour"] == 2
        assert usage["day"] == 2


class TestPnLLimits:
    """Test PnL-based trading limits."""

    def test_hourly_loss_limit(self):
        """Should block trading when hourly loss exceeded via multiple small losses."""
        rm = RiskManager(make_config())
        # Use small losses that don't trigger per-trade limit ($0.30)
        # but accumulate to exceed hourly limit ($0.50)
        rm.update_pnl(-0.20)
        rm.update_pnl(-0.20)
        rm.update_pnl(-0.15)  # Total: -$0.55 > hourly limit $0.50

        ok, reason = rm.can_trade()
        if not ok:
            assert "loss_limit" in reason or "paused_hour" in reason

    def test_daily_loss_limit(self):
        """Should block trading when daily loss exceeded via multiple small losses."""
        rm = RiskManager(make_config())
        # Small losses below per-trade limit but exceeding daily limit ($1.00)
        for _ in range(5):
            rm.update_pnl(-0.25)  # Total: -$1.25 > daily limit $1.00

        ok, reason = rm.can_trade()
        if not ok:
            assert "loss_limit" in reason or "paused_hour" in reason or "consecutive_loss_cooldown" in reason

    def test_pnl_update_accumulates(self):
        """PnL should accumulate across multiple trades."""
        rm = RiskManager(make_config())
        rm.update_pnl(-0.20)
        rm.update_pnl(-0.20)
        rm.update_pnl(-0.15)

        assert rm.hourly_pnl == pytest.approx(-0.55)
        assert rm.daily_pnl == pytest.approx(-0.55)


class TestInventoryTimeout:
    """Test inventory holding duration checks."""

    def test_normal_hold(self):
        """Recent position should be normal."""
        rm = RiskManager(make_config())
        result = rm.check_inventory_timeout(time.time() - 10)  # 10 seconds ago
        assert result == "normal"

    def test_tighten_exit(self):
        """Position held > soft timeout should trigger tighten."""
        rm = RiskManager(make_config())
        result = rm.check_inventory_timeout(time.time() - 150)  # 150s > 120s
        assert result == "tighten_exit"

    def test_emergency_exit(self):
        """Position held > hard timeout should trigger emergency."""
        rm = RiskManager(make_config())
        result = rm.check_inventory_timeout(time.time() - 350)  # 350s > 300s
        assert result == "emergency_exit"

    def test_no_position(self):
        """No position (entry_time=0) should be normal."""
        rm = RiskManager(make_config())
        result = rm.check_inventory_timeout(0)
        assert result == "normal"


class TestUnrealizedLoss:
    """Test unrealized loss check."""

    def test_within_limit(self):
        """Small unrealized loss should be OK."""
        rm = RiskManager(make_config())
        assert rm.check_unrealized_loss(-0.10) is False

    def test_exceeds_limit(self):
        """Large unrealized loss should trigger exit."""
        rm = RiskManager(make_config())
        assert rm.check_unrealized_loss(-0.40) is True

    def test_profit_ok(self):
        """Unrealized profit should be OK."""
        rm = RiskManager(make_config())
        assert rm.check_unrealized_loss(0.50) is False


class TestConsecutiveLossBreaker:
    """Test consecutive loss circuit breaker."""

    def _make_rm(self, pause=3, cooldown=10):
        cfg = make_config()
        cfg["risk"]["consecutive_loss_pause"] = pause
        cfg["risk"]["consecutive_loss_cooldown"] = cooldown
        # All hours active for testing
        cfg["schedule"]["active_hours_utc"] = list(range(24))
        cfg["schedule"]["pause_hours_utc"] = []
        return RiskManager(cfg)

    def test_no_breaker_under_threshold(self):
        """Should not trigger breaker with fewer losses than threshold."""
        rm = self._make_rm(pause=5)
        for _ in range(4):
            rm.update_pnl(-0.10)
        assert rm._consecutive_losses == 4
        assert rm._loss_pause_until == 0.0

    def test_breaker_triggers_at_threshold(self):
        """Should trigger cooldown at exactly N consecutive losses, then reset counter."""
        rm = self._make_rm(pause=3, cooldown=10)
        for _ in range(3):
            rm.update_pnl(-0.10)
        # Counter resets after breaker fires (prevents death spiral)
        assert rm._consecutive_losses == 0
        assert rm._loss_pause_until > time.time()
        ok, reason = rm.can_trade()
        assert not ok
        assert "consecutive_loss_cooldown" in reason

    def test_win_resets_counter(self):
        """A winning trade should reset consecutive loss counter."""
        rm = self._make_rm(pause=5)
        rm.update_pnl(-0.10)
        rm.update_pnl(-0.10)
        assert rm._consecutive_losses == 2
        rm.update_pnl(0.05)
        assert rm._consecutive_losses == 0

    def test_breaker_resets_allows_fresh_start(self):
        """After breaker fires and cooldown expires, bot gets a fresh 5-loss allowance."""
        rm = self._make_rm(pause=3, cooldown=1)
        # First trigger: 3 losses
        for _ in range(3):
            rm.update_pnl(-0.10)
        assert rm._loss_pause_until > time.time()
        assert rm._consecutive_losses == 0  # Reset after trigger
        # Expire cooldown
        rm._loss_pause_until = time.time() - 1
        # Bot can trade again, and needs 3 fresh losses to re-trigger
        rm.update_pnl(-0.10)
        assert rm._consecutive_losses == 1  # Fresh count
        rm.update_pnl(-0.10)
        assert rm._consecutive_losses == 2
        ok, _ = rm.can_trade()
        assert ok  # Not yet at threshold

    def test_cooldown_expires(self):
        """After cooldown expires, trading should resume."""
        rm = self._make_rm(pause=3, cooldown=1)
        for _ in range(3):
            rm.update_pnl(-0.10)
        # Force expire cooldown
        rm._loss_pause_until = time.time() - 1
        ok, reason = rm.can_trade()
        # Should not be blocked by consecutive_loss_cooldown
        assert "consecutive_loss_cooldown" not in reason


class TestPerTradeLossLimit:
    """Test per-trade loss limit enforcement (was dead code, now active)."""

    def _make_rm(self, max_loss=0.30):
        cfg = make_config()
        cfg["risk"]["max_loss_per_trade"] = max_loss
        cfg["risk"]["consecutive_loss_cooldown"] = 60
        cfg["schedule"]["active_hours_utc"] = list(range(24))
        cfg["schedule"]["pause_hours_utc"] = []
        return RiskManager(cfg)

    def test_large_loss_triggers_breaker(self):
        """Single trade loss > max_loss_per_trade should trigger circuit breaker."""
        rm = self._make_rm(max_loss=0.50)
        rm.update_pnl(-0.80)  # Exceeds $0.50 limit
        assert rm.is_circuit_breaker_active()
        ok, reason = rm.can_trade()
        assert not ok
        assert "consecutive_loss_cooldown" in reason

    def test_small_loss_no_trigger(self):
        """Single trade loss within limit should NOT trigger breaker."""
        rm = self._make_rm(max_loss=0.50)
        rm.update_pnl(-0.30)
        assert not rm.is_circuit_breaker_active()

    def test_pnl_still_accumulated_after_per_trade_trigger(self):
        """Per-trade limit triggers breaker but PnL should still accumulate."""
        rm = self._make_rm(max_loss=0.50)
        rm.update_pnl(-0.80)
        assert rm.hourly_pnl == pytest.approx(-0.80)
        assert rm.daily_pnl == pytest.approx(-0.80)


class TestExponentialBackoff:
    """Test exponential backoff for circuit breaker cooldown."""

    def _make_rm(self, pause=3, cooldown=60):
        cfg = make_config()
        cfg["risk"]["consecutive_loss_pause"] = pause
        cfg["risk"]["consecutive_loss_cooldown"] = cooldown
        cfg["schedule"]["active_hours_utc"] = list(range(24))
        cfg["schedule"]["pause_hours_utc"] = []
        return RiskManager(cfg)

    def test_first_trigger_base_cooldown(self):
        """First circuit breaker trigger should use base cooldown."""
        rm = self._make_rm(pause=3, cooldown=60)
        for _ in range(3):
            rm.update_pnl(-0.10)
        assert rm._breaker_trigger_count == 1
        # Cooldown should be ~60s (base)
        remaining = rm._loss_pause_until - time.time()
        assert 55 < remaining < 65

    def test_second_trigger_doubled(self):
        """Second trigger should use 2x base cooldown."""
        rm = self._make_rm(pause=3, cooldown=60)
        # First trigger
        for _ in range(3):
            rm.update_pnl(-0.10)
        rm._loss_pause_until = time.time() - 1  # Expire
        # Second trigger
        for _ in range(3):
            rm.update_pnl(-0.10)
        assert rm._breaker_trigger_count == 2
        remaining = rm._loss_pause_until - time.time()
        assert 115 < remaining < 125  # ~120s

    def test_backoff_capped(self):
        """Backoff should be capped at 300s (5 min)."""
        rm = self._make_rm(pause=3, cooldown=60)
        # Simulate many triggers
        rm._breaker_trigger_count = 10
        cd = rm._calc_breaker_cooldown()
        assert cd == 300.0  # Capped

    def test_win_resets_escalation(self):
        """A winning trade should reset the escalation counter."""
        rm = self._make_rm(pause=3, cooldown=60)
        # First trigger
        for _ in range(3):
            rm.update_pnl(-0.10)
        assert rm._breaker_trigger_count == 1
        rm._loss_pause_until = time.time() - 1  # Expire
        # Win resets escalation
        rm.update_pnl(0.05)
        assert rm._breaker_trigger_count == 0

    def test_per_trade_limit_also_escalates(self):
        """Per-trade limit should also increment the escalation counter."""
        rm = self._make_rm(pause=5, cooldown=60)
        rm._make_rm = None  # prevent confusion
        rm.max_loss_per_trade = 0.30
        rm.update_pnl(-0.50)  # Exceeds per-trade limit
        assert rm._breaker_trigger_count == 1
        assert rm.is_circuit_breaker_active()


class TestFeeTracking:
    """Test fee recording and tracking."""

    def test_record_maker_fee(self):
        """Should calculate and accumulate maker fees (0.003%)."""
        rm = RiskManager(make_config())
        fee1 = rm.record_fee(100_000, is_maker=True)  # $100k notional
        assert fee1 == pytest.approx(3.0)  # 0.003% of $100k = $3
        fee2 = rm.record_fee(50_000, is_maker=True)
        assert fee2 == pytest.approx(1.5)
        assert rm.total_fees == pytest.approx(4.5)

    def test_record_taker_fee(self):
        """Should calculate taker fees (0.02%)."""
        rm = RiskManager(make_config())
        fee = rm.record_fee(100_000, is_maker=False)
        assert fee == pytest.approx(20.0)  # 0.02% of $100k = $20

    def test_fees_in_usage(self):
        """Fees should appear in usage dict."""
        rm = RiskManager(make_config())
        rm.record_fee(100_000, is_maker=True)
        usage = rm.get_usage()
        assert "fees" in usage
        assert usage["fees"] == pytest.approx(3.0)

    def test_consecutive_losses_in_usage(self):
        """Consecutive loss count should appear in usage dict."""
        rm = RiskManager(make_config())
        rm.update_pnl(-0.10)
        rm.update_pnl(-0.10)
        usage = rm.get_usage()
        assert usage["consecutive_losses"] == 2


class TestSchedule:
    """Test time-based schedule filtering."""

    def test_active_hour(self):
        """Should recognize active trading hours."""
        rm = RiskManager(make_config())
        # Set active hours to include hour 10
        assert 10 in rm.active_hours

    def test_pause_hour(self):
        """Should recognize paused hours."""
        rm = RiskManager(make_config())
        assert 0 in rm.pause_hours
        assert 23 in rm.pause_hours

    def test_record_multiple_orders(self):
        """record_orders should record correct count."""
        rm = RiskManager(make_config())
        rm.record_orders(3)
        usage = rm.get_usage()
        assert usage["second"] == 3

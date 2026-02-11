"""Risk Manager: rate limiting, PnL limits, inventory timeout, time filter."""

import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Tuple

log = logging.getLogger("MM-BOT")


class RiskManager:
    """Comprehensive risk management for the market making bot."""

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})
        rate_cfg = config.get("rate_limit", {})
        sched_cfg = config.get("schedule", {})

        # PnL limits
        self.max_loss_per_trade = risk_cfg.get("max_loss_per_trade", 0.30)
        self.max_loss_per_hour = risk_cfg.get("max_loss_per_hour", 0.50)
        self.max_loss_per_day = risk_cfg.get("max_loss_per_day", 1.00)
        self.max_unrealized_loss = risk_cfg.get("max_unrealized_loss", 0.30)

        # Inventory timeouts
        self.inventory_timeout = risk_cfg.get("inventory_timeout", 120)
        self.emergency_timeout = risk_cfg.get("emergency_timeout", 300)

        # Consecutive loss circuit breaker with exponential backoff
        # Academic: Easley, López de Prado & O'Hara (2012) "Flow Toxicity and
        # Liquidity in a High-Frequency World" — market makers should increase
        # retreat duration when toxicity persists. Fixed cooldowns are suboptimal
        # because they re-enter the same adverse conditions.
        self.consecutive_loss_pause = risk_cfg.get("consecutive_loss_pause", 5)
        self.consecutive_loss_cooldown = risk_cfg.get("consecutive_loss_cooldown", 30)
        self._consecutive_losses: int = 0
        self._loss_pause_until: float = 0.0
        self._breaker_trigger_count: int = 0  # For exponential backoff
        self._max_breaker_cooldown: float = 300.0  # Cap at 5 minutes

        # Fee tracking (from config, default = Pro rates)
        self.total_fees: float = 0.0
        self.maker_fee_rate: float = risk_cfg.get("maker_fee_rate", 0.00003)
        self.taker_fee_rate: float = risk_cfg.get("taker_fee_rate", 0.0002)

        # Rate limits
        self.limits = {
            "second": rate_cfg.get("max_orders_per_second", 2),
            "minute": rate_cfg.get("max_orders_per_minute", 25),
            "hour": rate_cfg.get("max_orders_per_hour", 280),
            "day": rate_cfg.get("max_orders_per_day", 950),
        }

        # Windows in seconds
        self._windows = {
            "second": 1,
            "minute": 60,
            "hour": 3600,
            "day": 86400,
        }

        # Order timestamp deques
        self.orders_count = {
            "second": deque(),
            "minute": deque(),
            "hour": deque(),
            "day": deque(),
        }

        # Schedule
        self.active_hours = set(sched_cfg.get("active_hours_utc", list(range(7, 17))))
        self.pause_hours = set(sched_cfg.get("pause_hours_utc", [21, 22, 23, 0, 1, 2, 3, 4, 5]))

        # PnL tracking
        self.hourly_pnl = 0.0
        self.daily_pnl = 0.0
        self._last_hour_reset = time.time()
        self._last_day_reset = time.time()
        self._current_hour = datetime.now(timezone.utc).hour
        self._current_day = datetime.now(timezone.utc).date()

    def _prune_window(self, window: str):
        """Remove expired timestamps from a rate limit window."""
        cutoff = time.time() - self._windows[window]
        dq = self.orders_count[window]
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _count_in_window(self, window: str) -> int:
        """Count orders in a time window after pruning."""
        self._prune_window(window)
        return len(self.orders_count[window])

    def can_trade(self) -> Tuple[bool, str]:
        """Comprehensive check: time + rate + PnL. Returns (ok, reason)."""
        # 1. Time filter
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour

        if hour in self.pause_hours:
            return False, f"paused_hour (UTC {hour}:00)"

        # 2. Auto-reset PnL counters
        if now_utc.hour != self._current_hour:
            self._current_hour = now_utc.hour
            self.hourly_pnl = 0.0

        if now_utc.date() != self._current_day:
            self._current_day = now_utc.date()
            self.daily_pnl = 0.0

        # 3. Consecutive loss circuit breaker
        if time.time() < self._loss_pause_until:
            remaining = self._loss_pause_until - time.time()
            return False, f"consecutive_loss_cooldown ({remaining:.0f}s left)"

        # 4. PnL limits
        if self.hourly_pnl < -self.max_loss_per_hour:
            return False, f"hourly_loss_limit (${self.hourly_pnl:.2f})"

        if self.daily_pnl < -self.max_loss_per_day:
            return False, f"daily_loss_limit (${self.daily_pnl:.2f})"

        # 5. Rate limits (check all windows)
        for window in ["second", "minute", "hour", "day"]:
            count = self._count_in_window(window)
            if count >= self.limits[window]:
                return False, f"rate_limit_{window} ({count}/{self.limits[window]})"

        return True, "ok"

    def can_place_orders(self, count: int = 2) -> bool:
        """Pre-check: can we place `count` orders without exceeding any limit?

        Typical cycle: cancel 2 old + place 2 new = 4 operations counted.
        But only new orders count toward rate limits.
        """
        for window in ["second", "minute", "hour", "day"]:
            current = self._count_in_window(window)
            if current + count > self.limits[window]:
                log.debug(f"Rate limit {window}: {current}+{count} > {self.limits[window]}")
                return False
        return True

    def record_order(self):
        """Record that one order was placed/cancelled (counts toward rate limit)."""
        now = time.time()
        for window in self.orders_count:
            self.orders_count[window].append(now)

    def record_orders(self, count: int):
        """Record multiple orders at once."""
        for _ in range(count):
            self.record_order()

    def check_inventory_timeout(self, position_entry_time: float) -> str:
        """Check inventory holding duration.

        Returns:
            "normal" - within limits
            "tighten_exit" - soft timeout, tighten spread to encourage exit
            "emergency_exit" - hard timeout, must IOC force close
        """
        if position_entry_time <= 0:
            return "normal"

        elapsed = time.time() - position_entry_time

        if elapsed >= self.emergency_timeout:
            return "emergency_exit"
        elif elapsed >= self.inventory_timeout:
            return "tighten_exit"
        return "normal"

    def check_unrealized_loss(self, unrealized_pnl: float) -> bool:
        """Check if unrealized loss exceeds threshold. Returns True if must exit."""
        return unrealized_pnl < -self.max_unrealized_loss

    def update_pnl(self, realized_pnl: float):
        """Update PnL tracking with a realized trade PnL.

        Checks:
            1. Per-trade loss limit (Almgren & Chriss 2000: single-trade stop-loss
               is the most basic risk control for execution algorithms)
            2. Consecutive loss circuit breaker with exponential backoff
        """
        self.hourly_pnl += realized_pnl
        self.daily_pnl += realized_pnl

        # Per-trade loss limit: if a single closing trade exceeds max_loss,
        # immediately activate circuit breaker. This catches adverse selection
        # events where one fill leads to catastrophic loss (e.g., buying into
        # a $170 selloff). Previously this config value was loaded but never checked.
        if realized_pnl < -self.max_loss_per_trade:
            self._breaker_trigger_count += 1
            cooldown = self._calc_breaker_cooldown()
            self._loss_pause_until = time.time() + cooldown
            log.warning(
                f"[PER-TRADE LIMIT] Single trade loss ${realized_pnl:.4f} "
                f"exceeds -${self.max_loss_per_trade:.2f}, "
                f"circuit breaker ON ({cooldown:.0f}s)"
            )
            self._consecutive_losses = 0
            return

        # Consecutive loss tracking (only count closing trades with nonzero PnL)
        if realized_pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.consecutive_loss_pause:
                self._breaker_trigger_count += 1
                cooldown = self._calc_breaker_cooldown()
                self._loss_pause_until = time.time() + cooldown
                log.warning(
                    f"[CIRCUIT BREAKER] {self._consecutive_losses} consecutive losses, "
                    f"pausing {cooldown:.0f}s (trigger #{self._breaker_trigger_count})"
                )
                # Reset counter after triggering — give bot a fresh start after cooldown.
                # Without reset: counter stays at 5+, every subsequent loss immediately
                # re-triggers breaker, creating a death spiral where bot can barely trade.
                self._consecutive_losses = 0
        elif realized_pnl > 0:
            self._consecutive_losses = 0
            # Win resets escalation — market conditions improved
            self._breaker_trigger_count = 0

    def _calc_breaker_cooldown(self) -> float:
        """Calculate exponential backoff cooldown for circuit breaker.

        1st trigger: base cooldown (60s)
        2nd trigger: 2x (120s)
        3rd trigger: 4x (240s)
        Capped at _max_breaker_cooldown (300s = 5 min).

        Rationale: Easley, López de Prado & O'Hara (2012) show that flow
        toxicity tends to cluster. Repeated breaker triggers indicate
        persistent adverse market conditions — escalating retreat time
        avoids repeated re-entry into toxic flow.
        """
        multiplier = 2 ** (self._breaker_trigger_count - 1)
        return min(self.consecutive_loss_cooldown * multiplier,
                   self._max_breaker_cooldown)

    def is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker cooldown is currently active."""
        return time.time() < self._loss_pause_until

    def record_fee(self, notional: float, is_maker: bool = True):
        """Record fee for a fill (maker or taker rate)."""
        rate = self.maker_fee_rate if is_maker else self.taker_fee_rate
        fee = notional * rate
        self.total_fees += fee
        return fee

    def get_usage(self) -> dict:
        """Return current rate limit usage for status display."""
        return {
            "second": self._count_in_window("second"),
            "minute": self._count_in_window("minute"),
            "hour": self._count_in_window("hour"),
            "day": self._count_in_window("day"),
            "fees": self.total_fees,
            "consecutive_losses": self._consecutive_losses,
        }

    def is_active_hour(self) -> bool:
        """Check if current UTC hour is in active trading hours."""
        return datetime.now(timezone.utc).hour in self.active_hours

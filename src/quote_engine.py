"""Quote Engine: Avellaneda-Stoikov + Momentum Guard + OBI Protective Filter.

Core algorithm:
    1. Fair Price = mid - gamma * (q/q_max) * sigma^2  (normalized inventory)
    2. Half Spread = max(min_spread, spread*factor + kappa*sigma)
    3. Bid = fair - half_spread (floor to tick), Ask = fair + half_spread (ceil to tick)
    4. Enforce no-crossing: bid <= best_bid, ask >= best_ask
    5. Size skewing based on inventory
    6. Protective filters: vol-pause + momentum guard + OBI filter

Volatility: EWMA × Rogers-Satchell on 1-second OHLC candles.
    - Rogers & Satchell (1991): ~6x more efficient than close-to-close, handles drift
    - RiskMetrics (1996): EWMA λ=0.94, industry standard for financial vol estimation
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict

from .utils import floor_to_tick, ceil_to_tick

log = logging.getLogger("MM-BOT")


class VolatilityEngine:
    """EWMA × Rogers-Satchell volatility estimator on 1-second OHLC candles.

    Academic basis:
        - Rogers & Satchell (1991): RS variance handles non-zero drift (BTC trends),
          ~6x more efficient than close-to-close estimator.
        - RiskMetrics (1996): EWMA with λ=0.94 (~11s half-life) is the industry
          standard for financial time-series volatility smoothing.
        - Yang-Zhang (2000) rejected: its overnight-gap component adds noise
          for 24/7 BTC markets with no session boundaries.

    Output: dollar-denominated sigma per second (same units as spread formula).
    """

    def __init__(self, lambda_: float = 0.94, min_sigma: float = 8.0):
        self.lambda_ = lambda_
        self.min_sigma = min_sigma
        self.ewma_variance: float = 0.0
        self._initialized: bool = False
        # Current 1-second candle
        self._candle_start: int = 0
        self._candle_open: float = 0.0
        self._candle_high: float = 0.0
        self._candle_low: float = float('inf')
        self._candle_close: float = 0.0
        self._candle_count: int = 0

    def update(self, price: float, timestamp: float) -> float:
        """Feed a mid-price tick. Returns current dollar sigma estimate."""
        bucket = int(timestamp)

        if self._candle_start == 0:
            self._init_candle(bucket, price)
            return self.min_sigma

        if bucket == self._candle_start:
            self._update_candle(price)
        else:
            if self._candle_count >= 2:
                self._close_candle()
            self._init_candle(bucket, price)

        return self.get_sigma()

    def _init_candle(self, bucket: int, price: float):
        self._candle_start = bucket
        self._candle_open = price
        self._candle_high = price
        self._candle_low = price
        self._candle_close = price
        self._candle_count = 1

    def _update_candle(self, price: float):
        self._candle_high = max(self._candle_high, price)
        self._candle_low = min(self._candle_low, price)
        self._candle_close = price
        self._candle_count += 1

    def _close_candle(self):
        """Close candle → compute RS variance → update EWMA."""
        O = self._candle_open
        H = self._candle_high
        L = self._candle_low
        C = self._candle_close

        if O <= 0 or H == L:
            return

        # Rogers-Satchell: v = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)
        ln_hc = math.log(H / C)
        ln_ho = math.log(H / O)
        ln_lc = math.log(L / C)
        ln_lo = math.log(L / O)
        rs_var = ln_hc * ln_ho + ln_lc * ln_lo
        rs_var = max(0.0, rs_var)  # Noisy data can yield negative

        if not self._initialized:
            self.ewma_variance = rs_var
            self._initialized = True
        else:
            self.ewma_variance = (
                self.lambda_ * self.ewma_variance
                + (1 - self.lambda_) * rs_var
            )

    def get_sigma(self) -> float:
        """Return current dollar sigma (per-second volatility)."""
        if not self._initialized:
            return self.min_sigma
        price = self._candle_close if self._candle_close > 0 else 97000.0
        sigma = price * math.sqrt(self.ewma_variance)
        return max(self.min_sigma, sigma)


@dataclass
class QuoteResult:
    """Output of the quote engine."""
    bid_price: float = 0.0
    bid_size: float = 0.0
    ask_price: float = 0.0
    ask_size: float = 0.0
    fair_price: float = 0.0
    sigma: float = 0.0
    obi: float = 0.0
    half_spread: float = 0.0
    skip_reason: str = ""
    vol_paused: bool = False


class QuoteEngine:
    """Avellaneda-Stoikov market making quote generator with OBI overlay."""

    def __init__(self, config: dict):
        self.cfg = config
        strategy = config.get("strategy", {})
        obi_cfg = config.get("obi", {})

        # Strategy params
        self.gamma = strategy.get("gamma", 0.3)
        self.kappa = strategy.get("kappa", 1.5)
        self.min_half_spread = strategy.get("min_half_spread", 1.0)
        self.base_size = strategy.get("base_size", 0.0003)
        self.max_position = strategy.get("max_position", 0.001)
        self.vol_window = strategy.get("vol_window", 300)
        self.tick_size = strategy.get("tick_size", 1.0)
        self.min_sigma = strategy.get("min_sigma", 8.0)
        self.spread_factor = strategy.get("spread_factor", 0.4)
        self.momentum_window = strategy.get("momentum_window", 300)
        self.momentum_threshold = strategy.get("momentum_threshold", 40)
        self.vol_pause_threshold = strategy.get("vol_pause_threshold", 10)

        # OBI params
        self.obi_enabled = obi_cfg.get("enabled", True)
        self.obi_alpha = obi_cfg.get("alpha", 0.3)
        self.obi_threshold = obi_cfg.get("threshold", 0.2)
        self.obi_depth = obi_cfg.get("depth", 5)

        # Volatility engine: EWMA × Rogers-Satchell (replaces naive stddev)
        self.vol_engine = VolatilityEngine(lambda_=0.94, min_sigma=self.min_sigma)

        # Enforce: vol_window must hold enough data for momentum and vol_pause
        # calc_momentum() needs momentum_window seconds of data
        # calc_recent_range(60) needs 60 seconds of data
        min_required = max(self.momentum_window, 60)
        if self.vol_window < min_required:
            log.warning(
                f"vol_window={self.vol_window}s < required {min_required}s "
                f"(momentum_window={self.momentum_window}s, vol_pause=60s). "
                f"Auto-correcting to {min_required}s."
            )
            self.vol_window = min_required

        # State — mid_prices stores (timestamp, price) tuples
        self.mid_prices: deque = deque()
        self.vol_time_window: float = float(self.vol_window)  # seconds
        self.obi_smooth: float = 0.0
        self._last_sigma: float = self.min_sigma  # Default volatility = min_sigma

    def update(self, bbo: dict, orderbook: dict = None):
        """Update internal state from market data.

        Args:
            bbo: {"bid": float, "ask": float, ...}
            orderbook: {"bids": [(price,size),...], "asks": [(price,size),...]}
        """
        bid = bbo.get("bid", 0)
        ask = bbo.get("ask", 0)
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            now = time.time()
            self.mid_prices.append((now, mid))
            # Feed volatility engine with every tick
            self.vol_engine.update(mid, now)
            # Prune entries older than time window
            cutoff = now - self.vol_time_window
            while self.mid_prices and self.mid_prices[0][0] < cutoff:
                self.mid_prices.popleft()

        if orderbook and self.obi_enabled:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            if bids and asks:
                self.obi_smooth = self.calc_obi(bids, asks)

    def calc_volatility(self) -> float:
        """Return current dollar sigma from EWMA × Rogers-Satchell engine.

        The VolatilityEngine builds 1-second OHLC candles from BBO mid prices
        and computes Rogers-Satchell variance with EWMA smoothing (λ=0.94).
        This replaces the naive tick-diff stddev which was stuck at min_sigma.
        """
        sigma = self.vol_engine.get_sigma()
        self._last_sigma = sigma
        return sigma

    def calc_recent_range(self, window_sec: float = 60) -> float:
        """Price range (high - low) over recent window. Measures absolute volatility."""
        if len(self.mid_prices) < 2:
            return 0.0
        now_ts = self.mid_prices[-1][0]
        cutoff = now_ts - window_sec
        prices_in_window = [p for ts, p in self.mid_prices if ts >= cutoff]
        if len(prices_in_window) < 2:
            return 0.0
        return max(prices_in_window) - min(prices_in_window)

    def calc_momentum(self) -> float:
        """Price change over momentum window. Positive = rising, negative = falling."""
        if len(self.mid_prices) < 2:
            return 0.0
        now_ts = self.mid_prices[-1][0]
        cutoff = now_ts - self.momentum_window
        for ts, price in self.mid_prices:
            if ts >= cutoff:
                return self.mid_prices[-1][1] - price
        return self.mid_prices[-1][1] - self.mid_prices[0][1]

    def calc_obi(self, bids: list, asks: list) -> float:
        """Calculate EMA-smoothed Order Book Imbalance.

        OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        Positive = more buy pressure, Negative = more sell pressure.

        Used as protective filter: block entries in OBI direction.
        """
        depth = self.obi_depth
        bid_vol = sum(s for _, s in bids[:depth]) if bids else 0
        ask_vol = sum(s for _, s in asks[:depth]) if asks else 0
        total = bid_vol + ask_vol

        if total == 0:
            return self.obi_smooth

        raw_obi = (bid_vol - ask_vol) / total

        # EMA smoothing
        self.obi_smooth = self.obi_alpha * raw_obi + (1 - self.obi_alpha) * self.obi_smooth
        return self.obi_smooth

    def generate_quotes(self, market_state, bot_state) -> QuoteResult:
        """Generate bid/ask quotes based on current market and bot state.

        Args:
            market_state: MarketState with current BBO
            bot_state: BotState with current position

        Returns:
            QuoteResult with bid/ask prices and sizes
        """
        result = QuoteResult()

        if not market_state.is_valid:
            result.skip_reason = "invalid_market_data"
            return result

        mid = market_state.mid_price
        best_bid = market_state.best_bid
        best_ask = market_state.best_ask
        spread = market_state.spread
        net_pos = bot_state.net_position

        # Step 1: Volatility
        sigma = self.calc_volatility()
        result.sigma = sigma

        # Step 2: Fair Price with inventory adjustment (Avellaneda-Stoikov)
        # Normalize position to [-1, +1] range (GLFT 2013 recommendation)
        # Old: gamma * net_pos * σ² → $0.04 at typical params (zero ticks)
        # New: gamma * (q/q_max) * σ² → $1-12 depending on inventory & vol
        inv_ratio = net_pos / self.max_position if self.max_position > 0 else 0
        fair_price = mid - self.gamma * inv_ratio * (sigma ** 2)

        # Step 3: Record OBI (protective filter applied in Step 7)
        result.obi = self.obi_smooth

        result.fair_price = fair_price

        # Step 4: Half Spread calculation
        half_spread = max(self.min_half_spread, spread * self.spread_factor + self.kappa * sigma)

        # Widen spread when inventory is heavy
        inv_ratio = abs(net_pos) / self.max_position if self.max_position > 0 else 0
        if inv_ratio > 0.5:
            half_spread *= (1 + inv_ratio)

        result.half_spread = half_spread

        # Step 5: Calculate quote prices
        raw_bid = fair_price - half_spread
        raw_ask = fair_price + half_spread

        # Align to tick size
        bid_price = floor_to_tick(raw_bid, self.tick_size)
        ask_price = ceil_to_tick(raw_ask, self.tick_size)

        # Enforce no-crossing BBO: our bid must not exceed best_bid,
        # our ask must not be below best_ask (POST_ONLY would reject anyway)
        bid_price = min(bid_price, best_bid)
        ask_price = max(ask_price, best_ask)

        # Ensure bid < ask
        if bid_price >= ask_price:
            bid_price = floor_to_tick(mid - self.min_half_spread, self.tick_size)
            ask_price = ceil_to_tick(mid + self.min_half_spread, self.tick_size)
            bid_price = min(bid_price, best_bid)
            ask_price = max(ask_price, best_ask)

        if bid_price >= ask_price:
            # When holding inventory, still quote the exit side
            if net_pos > 0:
                # Long — only quote ask (sell to exit)
                result.ask_price = max(ask_price, best_ask)
                result.ask_size = self.base_size
                result.bid_price = 0
                result.bid_size = 0
                result.fair_price = fair_price
                return result
            elif net_pos < 0:
                # Short — only quote bid (buy to exit)
                result.bid_price = min(bid_price, best_bid)
                result.bid_size = self.base_size
                result.ask_price = 0
                result.ask_size = 0
                result.fair_price = fair_price
                return result
            else:
                # No position — skip (no edge to capture)
                result.skip_reason = "spread_too_tight"
                return result

        result.bid_price = bid_price
        result.ask_price = ask_price

        # Step 5.5: Tighten mode — after inventory timeout, place exit at BBO
        if bot_state.tighten_mode and net_pos != 0:
            if net_pos > 0:
                # Long timeout: ask at best_ask (most aggressive sell), no bid
                result.ask_price = best_ask
                result.bid_price = 0
            else:
                # Short timeout: bid at best_bid (most aggressive buy), no ask
                result.bid_price = best_bid
                result.ask_price = 0

        # Step 6: Size skewing based on inventory
        # When long: reduce bid size (less buying), keep ask size (sell to reduce)
        # When short: keep bid size (buy to reduce), reduce ask size (less selling)
        # When at max position: zero out the adding-to-position side
        if bot_state.tighten_mode and net_pos != 0:
            # Tighten mode: only exit direction at base_size
            if net_pos > 0:
                result.bid_size = 0
                result.ask_size = self.base_size
            else:
                result.bid_size = self.base_size
                result.ask_size = 0
        elif net_pos > 0:
            pos_ratio = net_pos / self.max_position if self.max_position > 0 else 1
            # Hard cutoff at 50%: stop adding when half max position reached
            # Data: all 4 major losses occurred at position >= 0.0045 BTC (45% max)
            result.bid_size = 0 if pos_ratio >= 0.5 else self.base_size * max(0, 1 - pos_ratio * 2)
            result.ask_size = self.base_size
        elif net_pos < 0:
            pos_ratio = abs(net_pos) / self.max_position if self.max_position > 0 else 1
            result.bid_size = self.base_size
            result.ask_size = 0 if pos_ratio >= 0.5 else self.base_size * max(0, 1 - pos_ratio * 2)
        else:
            result.bid_size = self.base_size
            result.ask_size = self.base_size

        # Step 7: Protective filters — block entries in adverse direction
        # Skip in tighten_mode: exit takes absolute priority over filters
        if not (bot_state.tighten_mode and net_pos != 0):
            # 7a: Vol-adaptive pause — don't make markets when too volatile
            # If 60s price range > threshold, stale quotes will be adversely selected
            recent_range = self.calc_recent_range(60)
            if recent_range > self.vol_pause_threshold:
                result.vol_paused = True
                if net_pos == 0:
                    result.bid_size = 0
                    result.ask_size = 0
                    log.info(f"[VOL-PAUSE] 60s range ${recent_range:.0f} > ${self.vol_pause_threshold}, both sides paused")
                elif net_pos > 0:
                    result.bid_size = 0
                    log.info(f"[VOL-PAUSE] 60s range ${recent_range:.0f}, bid paused (holding long)")
                else:
                    result.ask_size = 0
                    log.info(f"[VOL-PAUSE] 60s range ${recent_range:.0f}, ask paused (holding short)")

            # 7b: Momentum guard — don't enter trades in trend direction
            momentum = self.calc_momentum()
            if abs(momentum) > self.momentum_threshold:
                if momentum > 0 and net_pos <= 0:
                    # Uptrend + flat/short: block ask (don't sell into rally)
                    result.ask_size = 0
                    log.info(f"[MOMENTUM] +${momentum:.0f}/{self.momentum_window}s → ask blocked")
                elif momentum < 0 and net_pos >= 0:
                    # Downtrend + flat/long: block bid (don't buy into selloff)
                    result.bid_size = 0
                    log.info(f"[MOMENTUM] -${abs(momentum):.0f}/{self.momentum_window}s → bid blocked")

            # 7c: OBI protective filter — block entries in pressure direction
            if self.obi_enabled and abs(self.obi_smooth) > self.obi_threshold:
                if self.obi_smooth > 0 and net_pos <= 0:
                    # Buy pressure + flat/short: block ask (don't sell)
                    result.ask_size = 0
                    log.info(f"[OBI-GUARD] buy pressure {self.obi_smooth:+.2f} → ask blocked")
                elif self.obi_smooth < 0 and net_pos >= 0:
                    # Sell pressure + flat/long: block bid (don't buy)
                    result.bid_size = 0
                    log.info(f"[OBI-GUARD] sell pressure {self.obi_smooth:+.2f} → bid blocked")

            # 7d: Flat guard — one-sided entry when flat = directional bet, NOT market making
            # A flat market maker must quote BOTH sides (spread capture) or NEITHER.
            # Ref: Cartea et al. (2015) Ch.10 — protective filters should only be
            # one-sided when there's an existing position to protect.
            if abs(net_pos) < 0.0001:  # below min_size 0.0003 = effectively flat
                if (result.bid_size == 0) != (result.ask_size == 0):
                    result.bid_size = 0
                    result.ask_size = 0
                    log.info("[FLAT-GUARD] Filter blocked one side while flat → both sides paused")

        # Enforce minimum size (0.0003 BTC for Paradex)
        # Round UP to min_size (keep dual-sided quoting for spread capture).
        # Only truly zero sizes (from max position or tighten mode) stay at 0.
        # The hard position cap in bot.py blocks adding direction at max_position.
        min_size = 0.0003
        if 0 < result.bid_size < min_size:
            result.bid_size = min_size
        if 0 < result.ask_size < min_size:
            result.ask_size = min_size

        # Round to lot size precision (5 decimal places)
        # Fix: SDK rejects sizes not a multiple of 0.00001
        result.bid_size = round(result.bid_size, 5)
        result.ask_size = round(result.ask_size, 5)

        return result

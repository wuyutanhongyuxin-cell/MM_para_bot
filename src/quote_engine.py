"""Quote Engine: Simplified Avellaneda-Stoikov + OBI Contrarian Overlay.

Core algorithm:
    1. Fair Price = mid - gamma * net_position * sigma^2
    2. OBI Overlay: if |obi_smooth| > threshold, shift fair_price contra OBI
    3. Half Spread = max(min_spread, spread*0.4 + kappa*sigma)
    4. Bid = fair - half_spread (floor to tick), Ask = fair + half_spread (ceil to tick)
    5. Enforce no-crossing: bid <= best_bid, ask >= best_ask
    6. Size skewing based on inventory
"""

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict

from .utils import floor_to_tick, ceil_to_tick

log = logging.getLogger("MM-BOT")


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
        self.vol_window = strategy.get("vol_window", 60)
        self.tick_size = strategy.get("tick_size", 1.0)

        # OBI params
        self.obi_enabled = obi_cfg.get("enabled", True)
        self.obi_alpha = obi_cfg.get("alpha", 0.3)
        self.obi_threshold = obi_cfg.get("threshold", 0.3)
        self.obi_delta = obi_cfg.get("delta", 0.3)
        self.obi_depth = obi_cfg.get("depth", 5)

        # State
        self.mid_prices: deque = deque(maxlen=self.vol_window)
        self.obi_smooth: float = 0.0
        self._last_sigma: float = 5.0  # Default volatility

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
            self.mid_prices.append(mid)

        if orderbook and self.obi_enabled:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            if bids and asks:
                self.obi_smooth = self.calc_obi(bids, asks)

    def calc_volatility(self) -> float:
        """Estimate sigma from mid price differences (standard deviation).

        Returns annualized-ish volatility in $ terms, but for our purposes
        we just need a relative measure scaled to the quote interval.
        """
        if len(self.mid_prices) < 10:
            return self._last_sigma if self._last_sigma > 0 else 5.0

        prices = list(self.mid_prices)
        diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        if not diffs:
            return self._last_sigma

        mean = sum(diffs) / len(diffs)
        variance = sum((d - mean) ** 2 for d in diffs) / len(diffs)
        sigma = max(1.0, math.sqrt(variance))

        self._last_sigma = sigma
        return sigma

    def calc_obi(self, bids: list, asks: list) -> float:
        """Calculate EMA-smoothed Order Book Imbalance.

        OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        Positive = more buy pressure, Negative = more sell pressure.

        We use CONTRARIAN overlay: shift fair_price AGAINST OBI direction.
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
        # When long (net_pos > 0), lower fair_price to encourage selling
        # When short (net_pos < 0), raise fair_price to encourage buying
        fair_price = mid - self.gamma * net_pos * (sigma ** 2)

        # Step 3: OBI Contrarian Overlay
        # When OBI is positive (buy pressure), we shift fair_price DOWN (contrarian)
        # This makes our ask more aggressive (lower), willing to sell into buy pressure
        result.obi = self.obi_smooth
        if self.obi_enabled and abs(self.obi_smooth) > self.obi_threshold:
            obi_shift = -self.obi_delta * self.obi_smooth * spread
            fair_price += obi_shift
            log.debug(f"[OBI] obi={self.obi_smooth:.3f} shift=${obi_shift:.2f}")

        result.fair_price = fair_price

        # Step 4: Half Spread calculation
        half_spread = max(self.min_half_spread, spread * 0.4 + self.kappa * sigma)

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
            result.skip_reason = "spread_too_tight"
            return result

        result.bid_price = bid_price
        result.ask_price = ask_price

        # Step 6: Size skewing based on inventory
        # When long: reduce bid size (less buying), keep ask size (sell to reduce)
        # When short: keep bid size (buy to reduce), reduce ask size (less selling)
        # When at max position: zero out the adding-to-position side
        if net_pos > 0:
            pos_ratio = net_pos / self.max_position if self.max_position > 0 else 1
            result.bid_size = self.base_size * max(0, 1 - pos_ratio)
            result.ask_size = self.base_size
        elif net_pos < 0:
            pos_ratio = abs(net_pos) / self.max_position if self.max_position > 0 else 1
            result.bid_size = self.base_size
            result.ask_size = self.base_size * max(0, 1 - pos_ratio)
        else:
            result.bid_size = self.base_size
            result.ask_size = self.base_size

        # Enforce minimum size (0.0003 BTC for Paradex)
        min_size = 0.0003
        if result.bid_size < min_size:
            result.bid_size = 0
        if result.ask_size < min_size:
            result.ask_size = 0

        return result

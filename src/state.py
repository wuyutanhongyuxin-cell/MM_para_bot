"""State management: MarketState, BotState, PnLTracker."""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class MarketState:
    """Current market data snapshot."""
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid_price: float = 0.0
    spread: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    bids: list = field(default_factory=list)   # [(price, size), ...]
    asks: list = field(default_factory=list)
    last_update: float = 0.0

    def update_bbo(self, bid: float, ask: float,
                   bid_size: float = 0.0, ask_size: float = 0.0):
        """Update from BBO data."""
        self.best_bid = bid
        self.best_ask = ask
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.mid_price = (bid + ask) / 2.0
        self.spread = ask - bid
        self.last_update = time.time()

    def update_orderbook(self, bids: list, asks: list):
        """Update full orderbook depth. bids/asks: [(price, size), ...]"""
        self.bids = bids
        self.asks = asks

    @property
    def is_valid(self) -> bool:
        return self.best_bid > 0 and self.best_ask > 0 and self.best_ask > self.best_bid

    @property
    def age(self) -> float:
        """Seconds since last update."""
        return time.time() - self.last_update if self.last_update > 0 else float('inf')


@dataclass
class BotState:
    """Bot internal state tracking."""
    net_position: float = 0.0
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl_session: float = 0.0

    open_bid_id: Optional[str] = None
    open_ask_id: Optional[str] = None
    open_bid_price: float = 0.0
    open_ask_price: float = 0.0

    last_quote_time: float = 0.0
    position_entry_time: float = 0.0

    total_trades: int = 0
    maker_fills: int = 0
    taker_fills: int = 0

    # Current quote engine output (for logging)
    last_fair_price: float = 0.0
    last_sigma: float = 0.0
    last_obi: float = 0.0

    @property
    def has_position(self) -> bool:
        return abs(self.net_position) >= 0.00005

    @property
    def inventory_ratio(self) -> float:
        """Ratio of current position to max position. Must be set externally."""
        return 0.0  # Calculated by bot with max_position context

    def update_unrealized_pnl(self, mid_price: float):
        """Recalculate unrealized PnL from current market price."""
        if self.has_position and self.avg_entry_price > 0:
            self.unrealized_pnl = (mid_price - self.avg_entry_price) * self.net_position
        else:
            self.unrealized_pnl = 0.0

    def clear_orders(self):
        """Reset tracked order IDs."""
        self.open_bid_id = None
        self.open_ask_id = None
        self.open_bid_price = 0.0
        self.open_ask_price = 0.0


class PnLTracker:
    """Track per-trade PnL and compute aggregate statistics."""

    def __init__(self):
        self.trades: List[Dict] = []
        self.realized_pnl: float = 0.0
        self._position: float = 0.0
        self._avg_entry: float = 0.0
        self._cost_basis: float = 0.0

    def on_fill(self, side: str, price: float, size: float) -> Dict:
        """Process a fill and return trade summary.

        Uses FIFO cost basis to calculate realized PnL when reducing position.
        """
        signed_size = size if side.upper() == "BUY" else -size
        old_position = self._position
        new_position = old_position + signed_size

        realized = 0.0

        # Check if this fill reduces existing position (opposing direction)
        if old_position != 0 and (
            (old_position > 0 and signed_size < 0) or
            (old_position < 0 and signed_size > 0)
        ):
            # Closing (or partially closing) position
            close_size = min(abs(signed_size), abs(old_position))
            realized = (price - self._avg_entry) * close_size * (1 if old_position > 0 else -1)
            self.realized_pnl += realized

            # If flipping sides, set new avg_entry for the remaining
            if abs(new_position) > 0.00001 and (
                (new_position > 0 and old_position < 0) or
                (new_position < 0 and old_position > 0)
            ):
                self._avg_entry = price
        elif abs(new_position) > abs(old_position):
            # Adding to position - update average entry
            if abs(old_position) < 0.00001:
                self._avg_entry = price
            else:
                total_cost = self._avg_entry * abs(old_position) + price * size
                self._avg_entry = total_cost / abs(new_position)

        self._position = new_position

        summary = {
            "side": side,
            "price": price,
            "size": size,
            "realized_pnl": realized,
            "cumulative_pnl": self.realized_pnl,
            "net_position": new_position,
            "avg_entry": self._avg_entry,
            "is_win": realized > 0,
        }
        self.trades.append(summary)
        return summary

    def get_stats(self) -> Dict:
        """Return aggregate statistics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_pnl": 0.0,
                "max_drawdown": 0.0,
            }

        closing_trades = [t for t in self.trades if t["realized_pnl"] != 0]
        wins = [t for t in closing_trades if t["realized_pnl"] > 0]
        losses = [t for t in closing_trades if t["realized_pnl"] < 0]

        total_profit = sum(t["realized_pnl"] for t in wins)
        total_loss = abs(sum(t["realized_pnl"] for t in losses))

        # Max drawdown from cumulative PnL curve
        peak = 0.0
        max_dd = 0.0
        for t in self.trades:
            cum = t["cumulative_pnl"]
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd

        return {
            "total_trades": len(self.trades),
            "closing_trades": len(closing_trades),
            "total_pnl": self.realized_pnl,
            "win_rate": len(wins) / len(closing_trades) if closing_trades else 0.0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else float('inf'),
            "avg_pnl": self.realized_pnl / len(closing_trades) if closing_trades else 0.0,
            "max_drawdown": max_dd,
            "net_position": self._position,
            "avg_entry": self._avg_entry,
        }

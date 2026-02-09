"""Structured logging + CSV trade recording."""

import csv
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional


def setup_logger(level: str = "INFO") -> logging.Logger:
    """Configure and return the bot logger."""
    log = logging.getLogger("MM-BOT")
    if log.handlers:
        return log

    log.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    ))
    log.addHandler(handler)
    return log


class TradeCSVWriter:
    """Write every fill to a CSV file for post-trade analysis."""

    FIELDS = [
        "timestamp", "side", "price", "size", "fee",
        "maker_taker", "net_position", "unrealized_pnl",
        "realized_pnl", "obi", "sigma", "fair_price",
        "bid_quote", "ask_quote",
    ]

    def __init__(self, filepath: str = "trades.csv"):
        self.filepath = filepath
        self._ensure_header()

    def _ensure_header(self):
        """Write CSV header if file doesn't exist or is empty."""
        write_header = not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0
        if write_header:
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()

    def write_fill(
        self,
        side: str,
        price: float,
        size: float,
        fee: float = 0.0,
        maker_taker: str = "maker",
        net_position: float = 0.0,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        obi: float = 0.0,
        sigma: float = 0.0,
        fair_price: float = 0.0,
        bid_quote: float = 0.0,
        ask_quote: float = 0.0,
    ):
        """Append a fill record to the CSV."""
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "side": side,
            "price": f"{price:.1f}",
            "size": f"{size:.4f}",
            "fee": f"{fee:.6f}",
            "maker_taker": maker_taker,
            "net_position": f"{net_position:.4f}",
            "unrealized_pnl": f"{unrealized_pnl:.4f}",
            "realized_pnl": f"{realized_pnl:.4f}",
            "obi": f"{obi:.4f}",
            "sigma": f"{sigma:.2f}",
            "fair_price": f"{fair_price:.1f}",
            "bid_quote": f"{bid_quote:.1f}",
            "ask_quote": f"{ask_quote:.1f}",
        }
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(row)


class StatusPrinter:
    """Periodically print bot status to console."""

    def __init__(self, interval: float = 30.0):
        self.interval = interval
        self.last_print = 0.0
        self.log = logging.getLogger("MM-BOT")

    def should_print(self) -> bool:
        return time.time() - self.last_print >= self.interval

    def print_status(
        self,
        position: float,
        unrealized_pnl: float,
        realized_pnl: float,
        bid: Optional[float],
        ask: Optional[float],
        spread: float,
        obi: float,
        sigma: float,
        orders_used: dict,
        total_trades: int,
        maker_fills: int,
    ):
        """Print a compact status line."""
        self.last_print = time.time()
        pos_str = f"{position:+.4f}" if position != 0 else "0"
        maker_pct = (maker_fills / total_trades * 100) if total_trades > 0 else 0

        self.log.info(
            f"[STATUS] pos={pos_str} BTC | "
            f"uPnL=${unrealized_pnl:+.4f} rPnL=${realized_pnl:+.4f} | "
            f"bid={bid or 0:.0f} ask={ask or 0:.0f} spd=${spread:.1f} | "
            f"obi={obi:+.2f} vol={sigma:.1f} | "
            f"fills={total_trades} maker={maker_pct:.0f}% | "
            f"rate: {orders_used.get('hour', 0)}/hr {orders_used.get('day', 0)}/day"
        )

"""Utility functions: tick alignment, time helpers, formatting."""

import math
import time
from datetime import datetime, timezone


def floor_to_tick(price: float, tick_size: float) -> float:
    """Round price DOWN to nearest tick size.

    >>> floor_to_tick(97500.7, 1.0)
    97500.0
    >>> floor_to_tick(97501.0, 1.0)
    97501.0
    """
    return math.floor(price / tick_size) * tick_size


def ceil_to_tick(price: float, tick_size: float) -> float:
    """Round price UP to nearest tick size.

    >>> ceil_to_tick(97500.3, 1.0)
    97501.0
    >>> ceil_to_tick(97501.0, 1.0)
    97501.0
    """
    return math.ceil(price / tick_size) * tick_size


def round_to_tick(price: float, tick_size: float) -> float:
    """Round price to nearest tick size."""
    return round(price / tick_size) * tick_size


def format_price(price: float, tick_size: float = 1.0) -> str:
    """Format price string for order submission."""
    decimals = max(0, -int(math.log10(tick_size))) if tick_size < 1 else 0
    return f"{price:.{decimals}f}"


def format_size(size: float, min_size: float = 0.0003) -> str:
    """Format size string for order submission (4 decimal places for BTC)."""
    return f"{size:.4f}"


def utc_now() -> datetime:
    """Current UTC datetime."""
    return datetime.now(timezone.utc)


def utc_hour() -> int:
    """Current UTC hour (0-23)."""
    return datetime.now(timezone.utc).hour


def timestamp_ms() -> int:
    """Current timestamp in milliseconds."""
    return int(time.time() * 1000)


def elapsed_since(ts: float) -> float:
    """Seconds elapsed since a timestamp."""
    return time.time() - ts if ts > 0 else float('inf')


def safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

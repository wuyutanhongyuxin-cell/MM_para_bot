"""Vol-Pause Event Logger: records vol_pause events and tracks subsequent price movements.

Pure data collection — no trading logic, no order placement, no state mutation.
Reads market_state.mid_price (updated by existing WS BBO subscription) at fixed
intervals after each vol_pause trigger. Data is written to CSV and JSONL for
offline backtesting analysis.
"""

import asyncio
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("MM-BOT")

# Price tracking intervals (seconds after trigger)
TRACK_INTERVALS = [30, 60, 120, 180, 300]

# Minimum seconds between recorded events (deduplication)
COOLDOWN_SEC = 300

CSV_HEADER = [
    "timestamp", "trigger_price", "range_60s",
    "momentum_300s", "momentum_direction", "obi_value", "spread",
    "price_30s", "change_30s",
    "price_60s", "change_60s",
    "price_120s", "change_120s",
    "price_180s", "change_180s",
    "price_300s", "change_300s",
    "max_price_300s", "min_price_300s",
]


class VolPauseLogger:
    """Records vol_pause events and tracks price movements afterward."""

    def __init__(self, market_state, csv_path: str = "vol_pause_events.csv",
                 jsonl_path: str = "vol_pause_events.jsonl"):
        self.market_state = market_state
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self._last_event_time: float = 0.0
        self._tracking_tasks: list = []

        # Write CSV header if file doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)

    def record_event(self, trigger_price: float, range_60s: float,
                     momentum: float, obi: float, spread: float):
        """Record a vol_pause event and start price tracking.

        Deduplicates: only one event per 300s cooldown window.
        """
        now = time.time()
        if now - self._last_event_time < COOLDOWN_SEC:
            return
        self._last_event_time = now

        direction = "UP" if momentum >= 0 else "DOWN"
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        event = {
            "timestamp": ts_str,
            "trigger_price": round(trigger_price, 1),
            "range_60s": round(range_60s, 1),
            "momentum_300s": round(momentum, 1),
            "momentum_direction": direction,
            "obi_value": round(obi, 3),
            "spread": round(spread, 1),
        }

        log.info(
            f"[VOL-PAUSE-LOG] Recording event: mid=${trigger_price:.0f} "
            f"range=${range_60s:.0f} momentum={direction}{abs(momentum):.0f} obi={obi:+.2f}"
        )

        # Start async price tracking task
        task = asyncio.create_task(self._track_prices(event))
        self._tracking_tasks.append(task)
        task.add_done_callback(lambda t: self._tracking_tasks.remove(t) if t in self._tracking_tasks else None)

    async def _track_prices(self, event: dict):
        """Track mid_price at fixed intervals after trigger, then write results."""
        trigger_price = event["trigger_price"]
        max_price = trigger_price
        min_price = trigger_price
        start_time = time.time()

        for interval in TRACK_INTERVALS:
            # Sleep until next interval
            elapsed = time.time() - start_time
            sleep_time = interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            price = self.market_state.mid_price
            if price > 0:
                max_price = max(max_price, price)
                min_price = min(min_price, price)
                event[f"price_{interval}s"] = round(price, 1)
                event[f"change_{interval}s"] = round(price - trigger_price, 1)
            else:
                event[f"price_{interval}s"] = None
                event[f"change_{interval}s"] = None

        event["max_price_300s"] = round(max_price, 1)
        event["min_price_300s"] = round(min_price, 1)

        self._write_csv(event)
        self._write_jsonl(event)

        log.info(
            f"[VOL-PAUSE-LOG] Event complete: "
            f"trigger=${trigger_price:.0f} "
            f"5min_range=[${min_price:.0f}, ${max_price:.0f}] "
            f"change_300s={event.get('change_300s', 'N/A')}"
        )

    def _write_csv(self, event: dict):
        """Append one row to CSV."""
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                row = [event.get(col, "") for col in CSV_HEADER]
                writer.writerow(row)
        except Exception as e:
            log.error(f"[VOL-PAUSE-LOG] CSV write error: {e}")

    def _write_jsonl(self, event: dict):
        """Append one JSON line to JSONL file."""
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"[VOL-PAUSE-LOG] JSONL write error: {e}")

    async def flush_pending(self):
        """Wait for all in-progress tracking tasks to complete (for graceful shutdown)."""
        if self._tracking_tasks:
            log.info(f"[VOL-PAUSE-LOG] Waiting for {len(self._tracking_tasks)} tracking task(s)...")
            await asyncio.gather(*self._tracking_tasks, return_exceptions=True)

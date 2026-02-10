"""Tests for VolPauseLogger."""

import asyncio
import csv
import json
import os
import time
import pytest
from unittest.mock import MagicMock

from src.vol_pause_logger import VolPauseLogger, COOLDOWN_SEC, CSV_HEADER


@pytest.fixture
def tmp_dir(tmp_path):
    """Return temp output directory."""
    return str(tmp_path / "vollogs")


@pytest.fixture
def mock_market():
    """Create a mock MarketState with mid_price."""
    ms = MagicMock()
    ms.mid_price = 97500.0
    return ms


class TestVolPauseLogger:

    def test_csv_header_created(self, tmp_dir, mock_market):
        """CSV file should be created with header on init."""
        logger = VolPauseLogger(mock_market, output_dir=tmp_dir)

        csv_path = os.path.join(tmp_dir, "vol_pause_events.csv")
        assert os.path.exists(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == CSV_HEADER

    @pytest.mark.asyncio
    async def test_cooldown_dedup(self, tmp_dir, mock_market):
        """Events within 300s cooldown should be deduplicated."""
        logger = VolPauseLogger(mock_market, output_dir=tmp_dir)

        # First event — should be recorded
        logger.record_event(97500.0, 15.0, -30.0, -0.15, 2.0)

        # Second event immediately after — should be skipped (cooldown)
        logger.record_event(97510.0, 20.0, 40.0, 0.25, 3.0)

        # Only 1 task should be created
        assert len(logger._tracking_tasks) == 1

    @pytest.mark.asyncio
    async def test_cooldown_allows_after_expiry(self, tmp_dir, mock_market):
        """Events after cooldown should be recorded."""
        logger = VolPauseLogger(mock_market, output_dir=tmp_dir)

        # First event
        logger.record_event(97500.0, 15.0, -30.0, -0.15, 2.0)

        # Simulate cooldown expiry
        logger._last_event_time -= COOLDOWN_SEC + 1

        # Second event — should now be recorded
        logger.record_event(97510.0, 20.0, 40.0, 0.25, 3.0)

        assert len(logger._tracking_tasks) == 2

    @pytest.mark.asyncio
    async def test_price_tracking_writes_files(self, tmp_dir, mock_market):
        """After tracking completes, CSV and JSONL should contain event data."""
        logger = VolPauseLogger(mock_market, output_dir=tmp_dir)

        csv_path = logger.csv_path
        jsonl_path = logger.jsonl_path

        # Simulate price moving during tracking
        prices = iter([97510.0, 97520.0, 97490.0, 97505.0, 97515.0])

        original_sleep = asyncio.sleep

        async def fast_sleep(seconds):
            """Speed up tracking: skip real waits, update mock price."""
            try:
                mock_market.mid_price = next(prices)
            except StopIteration:
                pass
            await original_sleep(0)

        # Monkey-patch asyncio.sleep for fast test
        _orig = asyncio.sleep
        asyncio.sleep = fast_sleep

        try:
            logger.record_event(97500.0, 15.0, -30.0, -0.15, 2.0)
            await logger.flush_pending()
        finally:
            asyncio.sleep = _orig

        # Check CSV has 1 data row
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 2  # header + 1 event

        # Check JSONL has 1 line
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1

        event = json.loads(lines[0])
        assert event["trigger_price"] == 97500.0
        assert event["momentum_direction"] == "DOWN"
        assert event["range_60s"] == 15.0
        assert "price_30s" in event
        assert "max_price_300s" in event
        assert "min_price_300s" in event

    @pytest.mark.asyncio
    async def test_flush_pending_empty(self, tmp_dir, mock_market):
        """flush_pending should be safe with no tasks."""
        logger = VolPauseLogger(mock_market, output_dir=tmp_dir)
        await logger.flush_pending()  # Should not raise

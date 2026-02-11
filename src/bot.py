"""SpreadCaptureBot: Main market making bot with state machine.

State Machine:
    IDLE (no position)
      -> BBO update + risk OK + refresh_interval elapsed
    QUOTING (dual-sided quotes)
      -> bid filled: INVENTORY_LONG
      -> ask filled: INVENTORY_SHORT
      -> both filled: IDLE (perfect spread capture)
    INVENTORY_LONG / INVENTORY_SHORT
      -> skewed quote for exit
      -> exit filled -> IDLE
      -> 120s timeout -> tighten spread
      -> 300s timeout -> IOC emergency exit
      -> unrealized loss > threshold -> IOC emergency exit
"""

import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional

import yaml
from dotenv import load_dotenv

from .logger import TradeCSVWriter, StatusPrinter, setup_logger
from .paradex_client import ParadexClient
from .quote_engine import QuoteEngine
from .risk_manager import RiskManager
from .state import MarketState, BotState, PnLTracker
from .utils import format_price, format_size, elapsed_since
from .vol_pause_logger import VolPauseLogger

log = logging.getLogger("MM-BOT")


class SpreadCaptureBot:
    """Paradex BTC-USD-PERP spread capture market making bot."""

    def __init__(self, config_path: str = "config.yaml",
                 dry_run: bool = False, testnet: bool = False):
        # Load config
        self.config = self._load_config(config_path)
        self.dry_run = dry_run

        # Override env if --testnet flag
        if testnet:
            self.config.setdefault("paradex", {})["env"] = "testnet"

        # Initialize logger
        log_cfg = self.config.get("logging", {})
        setup_logger(log_cfg.get("level", "INFO"))

        # State
        self.market_state = MarketState()
        self.bot_state = BotState()
        self.pnl_tracker = PnLTracker()

        # Components
        self.quote_engine = QuoteEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        self.csv_writer = TradeCSVWriter(log_cfg.get("trade_csv", "trades.csv"))
        self.status_printer = StatusPrinter(log_cfg.get("console_interval", 30))
        self.vol_pause_logger = VolPauseLogger(self.market_state)

        # Client (initialized in run())
        self.client: Optional[ParadexClient] = None

        # Control
        self.running = False
        self._bbo_event = asyncio.Event()
        self._fill_event = asyncio.Event()  # Fix 3: immediate requote after fill
        self._last_bbo_time = 0.0

        # Market params from config
        strategy = self.config.get("strategy", {})
        self.refresh_interval = strategy.get("refresh_interval", 8.0)
        self.max_position = strategy.get("max_position", 0.001)
        self.base_size = strategy.get("base_size", 0.0003)
        self.market_name = self.config.get("paradex", {}).get("market", "BTC-USD-PERP")

    def _load_config(self, path: str) -> dict:
        """Load YAML config file."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        log.warning(f"Config file not found: {path}, using defaults")
        return {}

    # =========================================================================
    # Main Entry
    # =========================================================================

    async def run(self):
        """Main entry: connect -> subscribe -> loop -> shutdown."""
        self.running = True
        load_dotenv()

        # Get credentials
        l2_key = os.getenv("PARADEX_L2_PRIVATE_KEY", "")
        l2_addr = os.getenv("PARADEX_L2_ADDRESS", "")
        if not l2_key:
            log.error("PARADEX_L2_PRIVATE_KEY not set in .env")
            return

        paradex_cfg = self.config.get("paradex", {})

        # Initialize client
        self.client = ParadexClient(
            l2_private_key=l2_key,
            l2_address=l2_addr,
            env=paradex_cfg.get("env", "mainnet"),
            market=self.market_name,
            profile=paradex_cfg.get("profile", "retail"),
            dry_run=self.dry_run,
        )

        # Print banner
        self._print_banner()

        # Connect
        try:
            await self.client.connect()
        except Exception as e:
            log.error(f"Connection failed: {e}")
            return

        # Check system state
        state = await self.client.get_system_state()
        if state not in ("ok", "unknown"):
            log.error(f"System state: {state}, aborting")
            return

        # Get and display account info
        await self._display_account_info()

        # Cancel any stale orders from previous sessions
        await self.client.cancel_all(self.market_name)

        # Subscribe to WS channels
        await self._setup_subscriptions()

        # Main loop
        try:
            await self.main_loop()
        except asyncio.CancelledError:
            pass
        finally:
            await self.graceful_shutdown()

    # =========================================================================
    # Subscriptions & Callbacks
    # =========================================================================

    async def _setup_subscriptions(self):
        """Subscribe to WebSocket channels."""
        await self.client.subscribe_bbo(self._on_bbo)
        await self.client.subscribe_fills(self._on_fill_ws)
        await self.client.subscribe_orders(self._on_order_update)
        await self.client.subscribe_positions(self._on_position_update)
        # Fix 2: Subscribe WS orderbook for real-time OBI updates.
        # Cont, Kukanov & Stoikov (2014): OBI predictive power depends on
        # signal freshness. 5s stale REST OBI has R²≈25%, real-time ≈65%.
        await self.client.subscribe_orderbook(self._on_orderbook)
        # Fix 5: Register reconnect callback to clean stale state after WS drop.
        # Hasbrouck & Saar (2013): stale orders after disconnect are operational risk.
        self.client._reconnect_callback = self._on_ws_reconnect

    async def _on_bbo(self, channel, message: dict):
        """BBO update callback."""
        try:
            data = message.get("params", {}).get("data", message.get("data", message))
        except AttributeError:
            log.debug(f"BBO unexpected message type: {type(message)}")
            return
        bid = float(data.get("bid", 0))
        ask = float(data.get("ask", 0))
        bid_size = float(data.get("bid_size", 0))
        ask_size = float(data.get("ask_size", 0))

        log.debug(f"BBO update: bid={bid} ask={ask}")
        if bid > 0 and ask > 0 and ask > bid:
            self.market_state.update_bbo(bid, ask, bid_size, ask_size)
            self.quote_engine.update(
                {"bid": bid, "ask": ask},
            )
            self.bot_state.update_unrealized_pnl(self.market_state.mid_price)
            self._bbo_event.set()

    async def _on_fill_ws(self, channel, message: dict):
        """Fill callback from WebSocket."""
        data = message.get("params", {}).get("data", message.get("data", message))
        await self.on_fill(data)

    async def _on_order_update(self, channel, message: dict):
        """Order status update callback."""
        data = message.get("params", {}).get("data", message.get("data", message))
        await self._handle_order_update(data)

    async def _on_position_update(self, channel, message: dict):
        """Position update callback — authoritative source of position truth."""
        data = message.get("params", {}).get("data", message.get("data", message))

        # Handle list format: [{"market": ..., "side": ..., "size": ...}, ...]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    self._apply_position_update(item)
            return

        if isinstance(data, dict):
            self._apply_position_update(data)

    async def _on_orderbook(self, channel, message: dict):
        """WS orderbook update → real-time OBI calculation.

        Fix 2: Previously OBI was updated via REST polling every 5s (stale signal).
        Now OBI tracks real-time orderbook changes via WS subscription.
        Only updates OBI (via calc_obi), does NOT re-feed mid_prices to avoid
        double-counting with _on_bbo callback.
        """
        try:
            data = message.get("params", {}).get("data", message.get("data", message))
        except AttributeError:
            return
        if not isinstance(data, dict):
            return
        raw_bids = data.get("bids", [])
        raw_asks = data.get("asks", [])
        if not raw_bids or not raw_asks:
            return
        bids = [(float(b[0]), float(b[1])) for b in raw_bids]
        asks = [(float(a[0]), float(a[1])) for a in raw_asks]
        self.market_state.update_orderbook(bids, asks)
        # OBI-only update: don't call quote_engine.update() to avoid
        # double-counting mid prices (BBO callback handles that)
        if self.quote_engine.obi_enabled and bids and asks:
            self.quote_engine.calc_obi(bids, asks)

    async def _on_ws_reconnect(self):
        """Called after WS reconnects. Clean stale state.

        Fix 5: During WS disconnect, fills/cancels may have been missed.
        Must cancel any potentially stale orders and reconcile position
        with REST before resuming trading.
        """
        log.warning("[WS-RECONNECT] Cleaning up stale state...")
        try:
            await self.client.cancel_all(self.market_name)
            self.bot_state.clear_orders()
            await self._reconcile_position()
            log.info("[WS-RECONNECT] Stale orders cancelled, position reconciled")
        except Exception as e:
            log.error(f"[WS-RECONNECT] Cleanup failed: {e}")

    def _apply_position_update(self, pos: dict):
        """Apply a single position update from WS."""
        # Filter to our market
        market = pos.get("market", "")
        if market and market != self.market_name:
            return

        # CRITICAL: Paradex API returns signed size (negative for SHORT).
        # Always abs() before applying sign from side field to avoid double-negation.
        size = abs(float(pos.get("size", 0)))
        side = pos.get("side", "")
        old_pos = self.bot_state.net_position

        if side == "SHORT":
            self.bot_state.net_position = -size
        elif side == "LONG":
            self.bot_state.net_position = size
        elif size == 0 or side in ("", "NONE"):
            self.bot_state.net_position = 0.0

        new_pos = self.bot_state.net_position
        if abs(new_pos - old_pos) > 0.00005:
            log.info(f"[POS-WS] {side} {size:.4f} -> net={new_pos:+.4f} (was {old_pos:+.4f})")

        # Update avg_entry from exchange if available
        avg_entry = pos.get("average_entry_price")
        if avg_entry:
            self.bot_state.avg_entry_price = float(avg_entry)

        # Track position entry time
        if self.bot_state.has_position and self.bot_state.position_entry_time == 0:
            self.bot_state.position_entry_time = time.time()
        elif not self.bot_state.has_position:
            self.bot_state.position_entry_time = 0
            self.bot_state.tighten_mode = False
        # Reset entry time on position reversal (sign flip through zero)
        # Without this, a long→short flip keeps the old entry time,
        # making inventory timeout fire based on the stale long's age
        elif (old_pos > 0.0001 and new_pos < -0.0001) or (old_pos < -0.0001 and new_pos > 0.0001):
            self.bot_state.position_entry_time = time.time()
            self.bot_state.tighten_mode = False
            log.info("[POS-REVERSAL] Position sign flipped, reset entry time")

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def main_loop(self):
        """Core event loop: wait for data -> check risk -> quote -> manage."""
        log.info("Main loop started")
        last_quote = 0.0
        last_orderbook_fetch = 0.0
        orderbook_interval = 30.0  # REST fallback only (WS orderbook is primary)
        last_reconcile = 0.0
        reconcile_interval = 2.0  # REST position reconciliation every 2s
        # Data: 8 mismatches in 19min at 5s interval, 7 with sign flips.
        # Wrong-direction quoting for up to 5s per event → doubled losses.
        # 2s reduces max wrong-direction window by 60%.
        # Cost: 0.5 GET/s << Pro 120 GET/s limit.
        last_timeout_log = 0.0
        timeout_log_interval = 30.0  # Debounce timeout log to 30s

        while self.running:
            try:
                # Wait for BBO update from WS callback (SDK auto-dispatches)
                try:
                    await asyncio.wait_for(
                        self._bbo_event.wait(), timeout=5.0
                    )
                    self._bbo_event.clear()
                except asyncio.TimeoutError:
                    # No BBO in 5s — try REST fallback
                    log.debug("No BBO update in 5s, trying REST poll")
                    if self.market_state.age > 10:
                        await self._poll_bbo()
                    continue

                # Check market data staleness
                if self.market_state.age > 30:
                    log.warning("Market data stale, skipping quote")
                    continue

                # Periodically fetch orderbook for OBI
                if time.time() - last_orderbook_fetch > orderbook_interval:
                    await self._fetch_orderbook()
                    last_orderbook_fetch = time.time()

                # Periodic position reconciliation via REST
                if time.time() - last_reconcile > reconcile_interval:
                    await self._reconcile_position()
                    last_reconcile = time.time()

                # ===============================================================
                # EXIT CHECKS — always run, even during cooldown/pause.
                # Principle: risk filters block ENTRIES, never EXITS.
                # ===============================================================

                # Inventory timeout check
                if self.bot_state.has_position:
                    inv_status = self.risk_manager.check_inventory_timeout(
                        self.bot_state.position_entry_time
                    )
                    if inv_status == "emergency_exit":
                        # Guard: skip if in progress OR cooldown active (suppress log spam)
                        now_t = time.time()
                        if (not self.bot_state.emergency_exit_in_progress
                                and now_t - self.bot_state.emergency_exit_last_attempt >= 10):
                            log.warning("[EMERGENCY] Inventory timeout exceeded, force closing")
                            await self.emergency_exit()
                        continue
                    elif inv_status == "tighten_exit":
                        self.bot_state.tighten_mode = True
                        now_t = time.time()
                        if now_t - last_timeout_log > timeout_log_interval:
                            hold_time = now_t - self.bot_state.position_entry_time
                            log.info(f"[TIMEOUT] Tightening exit spread (held {hold_time:.0f}s)")
                            last_timeout_log = now_t

                # Unrealized loss check (only when holding position)
                if self.bot_state.has_position and self.risk_manager.check_unrealized_loss(
                    self.bot_state.unrealized_pnl
                ):
                    now_t = time.time()
                    if (not self.bot_state.emergency_exit_in_progress
                            and now_t - self.bot_state.emergency_exit_last_attempt >= 10):
                        log.warning(
                            f"[EMERGENCY] Unrealized loss ${self.bot_state.unrealized_pnl:.4f} "
                            f"exceeds limit, force closing"
                        )
                        await self.emergency_exit()
                    continue

                # Circuit breaker + open position: close immediately
                # 5 consecutive losses = market moving against us
                # Don't hold through cooldown — flatten and wait
                if self.bot_state.has_position and self.risk_manager.is_circuit_breaker_active():
                    now_t = time.time()
                    if (not self.bot_state.emergency_exit_in_progress
                            and now_t - self.bot_state.emergency_exit_last_attempt >= 10):
                        log.warning("[CIRCUIT BREAKER] Position still open during cooldown, force closing")
                        await self.emergency_exit()
                    continue

                # ===============================================================
                # ENTRY CHECKS — block new quotes when risk limits exceeded.
                # ===============================================================

                # Risk check (time filter + PnL limits + rate limits + circuit breaker)
                can_trade, reason = self.risk_manager.can_trade()
                if not can_trade:
                    if "paused_hour" in reason:
                        log.info(f"[SCHEDULE] {reason}, sleeping 60s")
                        await asyncio.sleep(60)
                    elif "rate_limit" in reason:
                        log.debug(f"Rate limit: {reason}")
                        await asyncio.sleep(1)
                    else:
                        # Debounce: only log every 60s for cooldown/loss limits
                        now_t = time.time()
                        if now_t - getattr(self, '_last_blocked_log', 0) > 60:
                            log.warning(f"Trading blocked: {reason}")
                            self._last_blocked_log = now_t
                        await asyncio.sleep(10)
                    continue

                # Check if it's time to refresh quotes
                # Fix 3: Guilbaud & Pham (2013) — optimal MM adjusts ALL quotes
                # immediately after any inventory change. _fill_event bypasses
                # the refresh_interval wait to close the "naked window" where
                # bot has no orders on book after a fill + cancel.
                now = time.time()
                fill_triggered = self._fill_event.is_set()
                if now - last_quote < self.refresh_interval and not fill_triggered:
                    await asyncio.sleep(0.05)  # 50ms poll (not 100ms)
                    continue
                if fill_triggered:
                    self._fill_event.clear()
                    log.debug("[FILL-REQUOTE] Immediate requote after fill")

                # Check rate limit headroom for new placements
                if not self.risk_manager.can_place_orders(2):
                    log.debug("Not enough rate headroom for requote")
                    await asyncio.sleep(1)
                    continue

                # Generate quotes
                quotes = self.quote_engine.generate_quotes(
                    self.market_state, self.bot_state
                )

                # Record vol_pause event for offline analysis
                if quotes.vol_paused:
                    self.vol_pause_logger.record_event(
                        trigger_price=self.market_state.mid_price,
                        range_60s=self.quote_engine.calc_recent_range(60),
                        momentum=self.quote_engine.calc_momentum(),
                        obi=quotes.obi,
                        spread=self.market_state.spread,
                    )

                if quotes.skip_reason:
                    log.info(f"Quote skipped: {quotes.skip_reason}")
                    await asyncio.sleep(1)
                    continue

                log.info(
                    f"[QUOTE] fair=${quotes.fair_price:.1f} "
                    f"bid=${quotes.bid_price:.0f}x{quotes.bid_size:.4f} "
                    f"ask=${quotes.ask_price:.0f}x{quotes.ask_size:.4f} "
                    f"sigma={quotes.sigma:.2f} obi={quotes.obi:+.3f}"
                )

                # Update state for logging
                self.bot_state.last_fair_price = quotes.fair_price
                self.bot_state.last_sigma = quotes.sigma
                self.bot_state.last_obi = quotes.obi

                # Cancel old orders and place new ones
                await self.cancel_and_requote(quotes)
                last_quote = now

                # Periodic status print
                if self.status_printer.should_print():
                    self._print_status()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Main loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    # =========================================================================
    # Order Management
    # =========================================================================

    async def cancel_and_requote(self, quotes):
        """Cancel existing orders and place new bid/ask.

        Pro mode: cancel_all + rapid place (no inter-op delay).
        At 800 req/s, 3 API calls (1 cancel_all + 2 place) is negligible.
        """
        # Hard position cap: block adding-direction orders regardless of quote engine
        current_pos = self.bot_state.net_position
        if current_pos >= self.max_position:
            if quotes.bid_size > 0:
                log.info(f"[POSITION CAP] Long {current_pos:+.4f} >= max {self.max_position}, bid blocked")
                quotes.bid_size = 0
        elif current_pos <= -self.max_position:
            if quotes.ask_size > 0:
                log.info(f"[POSITION CAP] Short {current_pos:+.4f} >= max {self.max_position}, ask blocked")
                quotes.ask_size = 0

        # Check if quotes actually changed — skip cancel+place cycle if identical
        bid_unchanged = (
            self.bot_state.open_bid_id
            and quotes.bid_size > 0
            and quotes.bid_price == self.bot_state.open_bid_price
        )
        ask_unchanged = (
            self.bot_state.open_ask_id
            and quotes.ask_size > 0
            and quotes.ask_price == self.bot_state.open_ask_price
        )
        if bid_unchanged and ask_unchanged:
            log.debug("[QUOTE] Prices unchanged, skipping requote")
            return

        # Cancel all existing orders in one call
        if self.bot_state.open_bid_id or self.bot_state.open_ask_id:
            await self.client.cancel_all(self.market_name)
            self.risk_manager.record_order()  # 1 API call
            self.bot_state.open_bid_id = None
            self.bot_state.open_bid_price = 0.0
            self.bot_state.open_ask_id = None
            self.bot_state.open_ask_price = 0.0

        # Place new bid (no delay — Pro mode has no speed throttle)
        if quotes.bid_size > 0:
            result = await self.client.place_order(
                side="BUY",
                price=quotes.bid_price,
                size=quotes.bid_size,
                instruction="POST_ONLY",
            )
            if result:
                self.bot_state.open_bid_id = result.get("id")
                self.bot_state.open_bid_price = quotes.bid_price
                self.risk_manager.record_order()
                log.info(
                    f"[BID] {quotes.bid_size:.4f} @ ${quotes.bid_price:.0f} "
                    f"(fair=${quotes.fair_price:.1f} obi={quotes.obi:+.2f})"
                )

        # Place new ask
        if quotes.ask_size > 0:
            result = await self.client.place_order(
                side="SELL",
                price=quotes.ask_price,
                size=quotes.ask_size,
                instruction="POST_ONLY",
            )
            if result:
                self.bot_state.open_ask_id = result.get("id")
                self.bot_state.open_ask_price = quotes.ask_price
                self.risk_manager.record_order()
                log.info(
                    f"[ASK] {quotes.ask_size:.4f} @ ${quotes.ask_price:.0f} "
                    f"(fair=${quotes.fair_price:.1f} obi={quotes.obi:+.2f})"
                )

        self.bot_state.last_quote_time = time.time()

    async def emergency_exit(self):
        """Emergency position close: cancel all, query REST real position, then close.

        Uses REST position (not internal tracking) to avoid reduce_only rejection
        when internal state diverges from exchange. Includes anti-reentrancy and
        price protection via IOC limit order.

        Handles exchange maintenance modes:
        - CANCEL_ONLY: only cancellations allowed, wait and retry
        - POST_ONLY: try aggressive POST_ONLY limit order instead of IOC
        """
        # Anti-reentrancy: 10s cooldown between attempts
        now = time.time()
        if self.bot_state.emergency_exit_in_progress:
            log.debug("[EMERGENCY EXIT] Already in progress, skipping")
            return
        if now - self.bot_state.emergency_exit_last_attempt < 10:
            log.debug("[EMERGENCY EXIT] Cooldown active, skipping")
            return

        self.bot_state.emergency_exit_in_progress = True
        self.bot_state.emergency_exit_last_attempt = now

        try:
            # Check system state before attempting exit
            sys_state = (await self.client.get_system_state()).lower()
            if sys_state == "cancel_only":
                log.warning(
                    "[EMERGENCY EXIT] Exchange in CANCEL_ONLY mode. "
                    "Cancelling orders, will retry exit when system recovers."
                )
                await self.client.cancel_all(self.market_name)
                self.bot_state.clear_orders()
                # Extend cooldown to 30s — don't spam during maintenance
                self.bot_state.emergency_exit_last_attempt = time.time() + 20
                return

            log.warning("[EMERGENCY EXIT] Cancelling all orders...")
            await self.client.cancel_all(self.market_name)
            self.bot_state.clear_orders()

            # Query REST for real position — do NOT trust internal tracking
            positions = await self.client.get_positions()
            btc_pos = [p for p in positions if p.get("market") == self.market_name]

            if not btc_pos or abs(float(btc_pos[0].get("size", 0))) < 0.00005:
                log.info("[EMERGENCY EXIT] No position on exchange, syncing local state")
                self._reset_position_state()
                return

            real_size = abs(float(btc_pos[0]["size"]))
            real_side = btc_pos[0].get("side", "")
            close_side = "SELL" if real_side == "LONG" else "BUY"

            # Sync internal state to REST value
            self.bot_state.net_position = real_size if real_side == "LONG" else -real_size

            log.warning(
                f"[EMERGENCY EXIT] Exchange position: {real_side} {real_size:.4f} BTC. "
                f"Sending {close_side} {real_size:.4f}"
            )

            # Determine order instruction based on system state
            bbo = await self.client.get_bbo()
            result = None

            if sys_state == "post_only":
                # POST_ONLY mode: use aggressive POST_ONLY limit at BBO
                log.warning("[EMERGENCY EXIT] Exchange in POST_ONLY mode, using POST_ONLY limit")
                if bbo:
                    if close_side == "SELL":
                        limit_price = bbo["ask"]  # Place at best ask (aggressive sell)
                    else:
                        limit_price = bbo["bid"]  # Place at best bid (aggressive buy)
                    result = await self.client.place_order(
                        side=close_side,
                        price=limit_price,
                        size=real_size,
                        instruction="POST_ONLY",
                        order_type="Limit",
                        reduce_only=True,
                    )
            else:
                # Normal mode: IOC limit with slippage protection
                # Fix 4: $5 was too tight — emergency exit fires during volatile
                # conditions where BBO can move $10-20 between fetch and submit.
                # Hendershott & Riordan (2013): institutional urgency exits use
                # 5-10bps tolerance. $30 ≈ 3bps on $97k BTC — covers most
                # microstructure moves while preventing absurd fill prices.
                if bbo:
                    slippage = 30.0
                    if close_side == "SELL":
                        limit_price = bbo["bid"] - slippage
                    else:
                        limit_price = bbo["ask"] + slippage

                    result = await self.client.place_order(
                        side=close_side,
                        price=limit_price,
                        size=real_size,
                        instruction="IOC",
                        order_type="Limit",
                        reduce_only=True,
                    )
                else:
                    # Fallback: market IOC if BBO unavailable
                    result = await self.client.place_order(
                        side=close_side,
                        price=0,
                        size=real_size,
                        instruction="IOC",
                        order_type="Market",
                        reduce_only=True,
                    )

            if result:
                self.risk_manager.record_order()
                # Verify close after 1s
                await asyncio.sleep(1)
                positions2 = await self.client.get_positions()
                btc_pos2 = [p for p in positions2 if p.get("market") == self.market_name]
                if not btc_pos2 or abs(float(btc_pos2[0].get("size", 0))) < 0.00005:
                    log.info("[EMERGENCY EXIT] Position confirmed closed")
                    self._reset_position_state()
                else:
                    remaining = abs(float(btc_pos2[0].get("size", 0)))
                    log.error(f"[EMERGENCY EXIT] Position still open: {remaining:.4f} BTC")
            else:
                log.error("[EMERGENCY EXIT] Failed to send exit order!")
                if sys_state not in ("ok", "unknown"):
                    # System degraded — extend cooldown to avoid hammering
                    self.bot_state.emergency_exit_last_attempt = time.time() + 20
        finally:
            self.bot_state.emergency_exit_in_progress = False

    def _reset_position_state(self):
        """Reset all position-related state after confirmed close."""
        self.bot_state.net_position = 0.0
        self.bot_state.position_entry_time = 0
        self.bot_state.tighten_mode = False
        self.bot_state.avg_entry_price = 0.0
        self.bot_state.unrealized_pnl = 0.0

    async def _cancel_all_on_circuit_break(self):
        """Cancel all outstanding orders AND close position when circuit breaker fires.

        5 consecutive losses = market moving against us. Must:
        1. Cancel all outstanding orders (prevent further fills)
        2. Close any existing position (stop bleeding during 60s cooldown)
        Sitting with an open position during cooldown = more losses.
        """
        if self.bot_state.open_bid_id or self.bot_state.open_ask_id:
            log.warning("[CIRCUIT BREAKER] Cancelling all open orders to prevent reversal")
            try:
                await self.client.cancel_all(self.market_name)
            except Exception as e:
                log.error(f"Cancel on circuit break failed: {e}")
            self.bot_state.open_bid_id = None
            self.bot_state.open_bid_price = 0.0
            self.bot_state.open_ask_id = None
            self.bot_state.open_ask_price = 0.0

        # Close any existing position immediately
        if self.bot_state.has_position:
            log.warning("[CIRCUIT BREAKER] Position open during breaker — force closing")
            await self.emergency_exit()

    # =========================================================================
    # Fill / Order Handling
    # =========================================================================

    async def on_fill(self, fill: dict):
        """Process a fill event."""
        side = fill.get("side", "")
        price = float(fill.get("price", 0) or fill.get("fill_price", 0))
        size = float(fill.get("size", 0) or fill.get("fill_size", 0))
        order_id = fill.get("order_id", fill.get("id", ""))

        if price == 0 or size == 0:
            return

        # Determine if maker or taker
        is_maker = fill.get("liquidity", "").upper() == "MAKER"
        fill_type = "maker" if is_maker else "taker"

        # Track fees (Pro: 0.003% maker, 0.02% taker)
        notional = price * size
        fee = self.risk_manager.record_fee(notional, is_maker=is_maker)

        # Update PnL tracker (for PnL math only, NOT position authority)
        summary = self.pnl_tracker.on_fill(side, price, size)

        # Save pre-fill position for closing-fill detection (Fix A below)
        pre_fill_pos = self.bot_state.net_position

        # Apply immediate position delta from this fill.
        # Before this fix: on_fill() did NOT update position → relied entirely on
        # WS position channel which lags fills → 51% mismatch rate (392/768 fills
        # in Log2). Wrong position → wrong inventory skewing → wrong quote direction.
        # The WS position channel uses SET (absolute), not increment, so when it
        # arrives later it simply overrides this delta with the correct value.
        # REST reconciliation (every 2s + after each fill) provides final correction.
        if side == "BUY":
            self.bot_state.net_position += size
        elif side == "SELL":
            self.bot_state.net_position -= size

        self.bot_state.total_trades += 1
        if is_maker:
            self.bot_state.maker_fills += 1
        else:
            self.bot_state.taker_fills += 1

        # Update realized PnL
        self.bot_state.realized_pnl_session = summary["cumulative_pnl"]
        if summary["realized_pnl"] != 0:
            self.risk_manager.update_pnl(summary["realized_pnl"])

        # Circuit breaker: cancel all outstanding orders immediately
        # Prevents stale orders from being filled and reversing position
        if self.risk_manager.is_circuit_breaker_active():
            await self._cancel_all_on_circuit_break()

        # Clear matched order ID
        if order_id == self.bot_state.open_bid_id:
            self.bot_state.open_bid_id = None
            self.bot_state.open_bid_price = 0.0
        elif order_id == self.bot_state.open_ask_id:
            self.bot_state.open_ask_id = None
            self.bot_state.open_ask_price = 0.0

        # Cancel ALL remaining orders after any fill.
        # After a fill, the remaining order reflects the pre-fill fair price
        # and is mispriced due to inventory change (gamma*inv_ratio*sigma²).
        # Data: avg_loss=2.4x avg_win (229 fills, 19min), stale opposite-side
        # orders are the primary source of adverse selection.
        # Ref: Guilbaud & Pham (2013) — optimal MM adjusts all quotes after fill.
        # Ref: Cartea & Jaimungal (2015) — inventory-dependent optimal quotes
        # must update immediately when inventory changes.
        # Pro cost: 1 cancel_all per fill ≈ 12/min << 800/s limit.
        remaining_id = self.bot_state.open_bid_id or self.bot_state.open_ask_id
        if remaining_id:
            log.debug("[FILL-CANCEL] Cancelling stale orders after fill")
            try:
                await self.client.cancel_all(self.market_name)
            except Exception as e:
                log.debug(f"Post-fill cancel failed: {e}")
            self.bot_state.open_bid_id = None
            self.bot_state.open_bid_price = 0.0
            self.bot_state.open_ask_id = None
            self.bot_state.open_ask_price = 0.0

        # Fix 3 (revised): Only trigger fill_event for CLOSING fills.
        # Original: fill_event fired for ALL fills → immediate requote.
        # Problem: during rapid price drops, entry fills trigger cascading
        # adverse accumulation. Live data 2025-02-07: 6 BUY fills in 6s
        # during $86 drop, accumulating 0.00568 BTC long → sold at -$1.02.
        # Ref: Hendershott, Jones & Menkveld (2011) — MMs pull quotes during
        # fast moves to avoid adverse selection ("quote pulling" defense).
        # Fix: only closing fills (position-reducing) trigger immediate requote.
        # Entry fills keep the natural 3s refresh_interval delay as protection.
        is_closing_fill = (
            (side == "BUY" and pre_fill_pos < -self.base_size * 0.5) or
            (side == "SELL" and pre_fill_pos > self.base_size * 0.5)
        )
        if is_closing_fill:
            self._fill_event.set()
            log.debug("[FILL-REQUOTE] Closing fill → immediate requote")

        # Log — show exchange position (bot_state), not PnLTracker position
        rpnl = summary["realized_pnl"]
        log.info(
            f"[FILL] {fill_type.upper()} {side} {size:.4f} @ ${price:.1f} | "
            f"pos={self.bot_state.net_position:+.4f} | "
            f"rPnL=${rpnl:+.4f} cumPnL=${summary['cumulative_pnl']:+.4f}"
        )

        # Write to CSV
        self.csv_writer.write_fill(
            side=side,
            price=price,
            size=size,
            maker_taker=fill_type,
            net_position=self.bot_state.net_position,
            unrealized_pnl=self.bot_state.unrealized_pnl,
            realized_pnl=rpnl,
            obi=self.bot_state.last_obi,
            sigma=self.bot_state.last_sigma,
            fair_price=self.bot_state.last_fair_price,
            bid_quote=self.bot_state.open_bid_price,
            ask_quote=self.bot_state.open_ask_price,
        )

        # Post-fill forced reconciliation: verify position with REST
        # Data: 8 mismatches in 19min (7 sign flips) because WS position
        # updates lag behind fills. Without this, bot quotes wrong direction
        # for up to 5s (now 2s with reduced interval, but immediate is better).
        # Cost: 1 GET /positions per fill ≈ 12/min << 120 GET/s limit.
        await self._reconcile_position()

    async def _handle_order_update(self, order: dict):
        """Handle order status changes (POST_ONLY rejection, etc.)."""
        order_id = order.get("id", "")
        status = order.get("status", "")
        cancel_reason = order.get("cancel_reason", "")

        if status == "CLOSED" and cancel_reason:
            # POST_ONLY rejection: order would have crossed
            if "POST_ONLY" in cancel_reason.upper() or "WOULD_EXECUTE" in cancel_reason.upper():
                log.info(f"[ORDER] POST_ONLY rejected (would cross): {order_id}")
            else:
                log.debug(f"[ORDER] Cancelled: {order_id} reason={cancel_reason}")

            # Clear from tracked orders
            if order_id == self.bot_state.open_bid_id:
                self.bot_state.open_bid_id = None
                self.bot_state.open_bid_price = 0.0
            elif order_id == self.bot_state.open_ask_id:
                self.bot_state.open_ask_id = None
                self.bot_state.open_ask_price = 0.0

    # =========================================================================
    # Data Polling Fallbacks
    # =========================================================================

    async def _poll_bbo(self):
        """Fallback: poll BBO via REST when WS is stale."""
        bbo = await self.client.get_bbo()
        if bbo:
            self.market_state.update_bbo(
                bbo["bid"], bbo["ask"],
                bbo.get("bid_size", 0), bbo.get("ask_size", 0)
            )
            self.quote_engine.update(bbo)
            self._bbo_event.set()

    async def _reconcile_position(self):
        """Reconcile local position state with REST API."""
        try:
            positions = await self.client.get_positions()
            btc_pos = [p for p in positions if p.get("market") == self.market_name]

            if btc_pos:
                pos = btc_pos[0]
                rest_size = abs(float(pos.get("size", 0)))
                rest_side = pos.get("side", "NONE")
                rest_net = rest_size if rest_side == "LONG" else (
                    -rest_size if rest_side == "SHORT" else 0.0
                )
            else:
                rest_net = 0.0

            local_net = self.bot_state.net_position

            if abs(rest_net - local_net) > 0.00005:
                log.warning(
                    f"[RECONCILE] Position mismatch! "
                    f"local={local_net:+.4f} vs REST={rest_net:+.4f}. "
                    f"Correcting to REST value."
                )
                self.bot_state.net_position = rest_net
                if btc_pos:
                    self.bot_state.avg_entry_price = float(
                        btc_pos[0].get("average_entry_price", 0)
                    )
                if abs(rest_net) > 0.00005 and self.bot_state.position_entry_time == 0:
                    self.bot_state.position_entry_time = time.time()
                elif abs(rest_net) < 0.00005:
                    self.bot_state.position_entry_time = 0
            else:
                log.debug(f"[RECONCILE] OK: pos={rest_net:+.4f}")
        except Exception as e:
            log.debug(f"[RECONCILE] Failed: {e}")

    async def _fetch_orderbook(self):
        """Fetch orderbook depth for OBI calculation."""
        ob = await self.client.get_orderbook(
            depth=self.config.get("obi", {}).get("depth", 5)
        )
        if ob:
            self.market_state.update_orderbook(ob["bids"], ob["asks"])
            self.quote_engine.update(
                {"bid": self.market_state.best_bid, "ask": self.market_state.best_ask},
                ob
            )

    # =========================================================================
    # Display & Status
    # =========================================================================

    def _print_banner(self):
        """Print startup banner."""
        paradex_cfg = self.config.get("paradex", {})
        strategy = self.config.get("strategy", {})
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        profile = paradex_cfg.get("profile", "retail").upper()

        log.info("=" * 60)
        log.info(f"  Paradex Market Making Bot [{mode}]")
        log.info("=" * 60)
        log.info(f"  Market:     {self.market_name}")
        log.info(f"  Profile:    {profile} ({'0% fees' if profile == 'RETAIL' else '0.003% maker / 0.02% taker'})")
        log.info(f"  Env:        {paradex_cfg.get('env', 'mainnet')}")
        log.info(f"  Base Size:  {self.base_size} BTC")
        log.info(f"  Max Pos:    {self.max_position} BTC")
        log.info(f"  Refresh:    {self.refresh_interval}s")
        log.info(f"  Gamma:      {strategy.get('gamma', 0.3)}")
        log.info(f"  Kappa:      {strategy.get('kappa', 1.5)}")
        obi_cfg = self.config.get("obi", {})
        log.info(f"  OBI:        {'ON' if obi_cfg.get('enabled', True) else 'OFF'} "
                 f"(delta={obi_cfg.get('delta', 0.3)}, threshold={obi_cfg.get('threshold', 0.3)})")
        log.info("=" * 60)

    async def _display_account_info(self):
        """Fetch and display account info."""
        balance = await self.client.get_balance()
        if balance is not None:
            log.info(f"  Balance: ${balance:.2f} USDC")

        positions = await self.client.get_positions()
        btc_pos = [p for p in positions if p.get("market") == self.market_name]
        if btc_pos:
            pos = btc_pos[0]
            size = abs(float(pos.get("size", 0)))
            side = pos.get("side", "NONE")
            log.info(f"  Position: {side} {size:.4f} BTC")
            if side == "LONG":
                self.bot_state.net_position = size
            elif side == "SHORT":
                self.bot_state.net_position = -size
            self.bot_state.avg_entry_price = float(pos.get("average_entry_price", 0))
            if self.bot_state.has_position:
                self.bot_state.position_entry_time = time.time()
        else:
            log.info("  Position: None")

    def _print_status(self):
        """Print current status to console."""
        self.status_printer.print_status(
            position=self.bot_state.net_position,
            unrealized_pnl=self.bot_state.unrealized_pnl,
            realized_pnl=self.bot_state.realized_pnl_session,
            bid=self.bot_state.open_bid_price or None,
            ask=self.bot_state.open_ask_price or None,
            spread=self.market_state.spread,
            obi=self.bot_state.last_obi,
            sigma=self.bot_state.last_sigma,
            orders_used=self.risk_manager.get_usage(),
            total_trades=self.bot_state.total_trades,
            maker_fills=self.bot_state.maker_fills,
        )

    # =========================================================================
    # Shutdown
    # =========================================================================

    async def graceful_shutdown(self):
        """Graceful shutdown: cancel orders, try maker exit, print stats."""
        self.running = False
        log.info("")
        log.info("=" * 60)
        log.info("  SHUTTING DOWN")
        log.info("=" * 60)

        # Flush pending vol-pause tracking tasks
        await self.vol_pause_logger.flush_pending()

        if self.client:
            # Cancel all orders
            log.info("Cancelling all open orders...")
            await self.client.cancel_all(self.market_name)
            self.bot_state.clear_orders()

            # Close position if any — query REST real position, don't trust internal state
            positions = await self.client.get_positions()
            btc_pos = [p for p in positions if p.get("market") == self.market_name]

            if btc_pos and abs(float(btc_pos[0].get("size", 0))) > 0.00005:
                real_size = abs(float(btc_pos[0]["size"]))
                real_side = btc_pos[0].get("side", "")
                close_side = "SELL" if real_side == "LONG" else "BUY"
                log.info(f"Closing position: {close_side} {real_size:.4f} BTC (from REST)")

                # Try limit order first (maker)
                bbo = await self.client.get_bbo()
                if bbo:
                    price = bbo["ask"] if close_side == "SELL" else bbo["bid"]
                    result = await self.client.place_order(
                        side=close_side,
                        price=price,
                        size=real_size,
                        instruction="POST_ONLY",
                        reduce_only=True,
                    )
                    if result:
                        await asyncio.sleep(6)
                        # Re-query REST to check if closed
                        positions2 = await self.client.get_positions()
                        btc_pos2 = [
                            p for p in positions2
                            if p.get("market") == self.market_name
                            and abs(float(p.get("size", 0))) > 0.00005
                        ]
                        if btc_pos2:
                            remaining = abs(float(btc_pos2[0]["size"]))
                            remaining_side = btc_pos2[0].get("side", "")
                            remaining_close = "SELL" if remaining_side == "LONG" else "BUY"
                            log.info(f"Maker exit incomplete, IOC closing {remaining:.4f} BTC")
                            await self.client.cancel_all(self.market_name)
                            await self.client.place_order(
                                side=remaining_close,
                                price=0,
                                size=remaining,
                                instruction="IOC",
                                order_type="Market",
                                reduce_only=True,
                            )
            else:
                log.info("No position on exchange to close")

            # Close client
            await self.client.close()

        # Print final statistics
        stats = self.pnl_tracker.get_stats()
        log.info("")
        log.info("=" * 60)
        log.info("  SESSION SUMMARY")
        log.info("=" * 60)
        log.info(f"  Total Fills:    {stats['total_trades']}")
        log.info(f"  Closing Trades: {stats['closing_trades']}")
        log.info(f"  Total PnL:      ${stats['total_pnl']:+.4f}")
        log.info(f"  Win Rate:       {stats['win_rate']:.1%}")
        log.info(f"  Profit Factor:  {stats['profit_factor']:.2f}")
        log.info(f"  Avg PnL/Trade:  ${stats['avg_pnl']:+.4f}")
        log.info(f"  Max Drawdown:   ${stats['max_drawdown']:.4f}")
        log.info(f"  Maker Fills:    {self.bot_state.maker_fills}/{self.bot_state.total_trades}")
        rate_usage = self.risk_manager.get_usage()
        log.info(f"  Orders Used:    {rate_usage['day']}/day")
        log.info(f"  Total Fees:     ${rate_usage['fees']:.4f}")
        log.info(f"  Net PnL:        ${stats['total_pnl'] - rate_usage['fees']:+.4f} (after fees)")
        log.info("=" * 60)

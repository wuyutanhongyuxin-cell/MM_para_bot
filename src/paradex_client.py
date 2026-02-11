"""Paradex API Client: REST + WebSocket wrapper over paradex-py SDK.

Provides a unified async interface for the market making bot.
Uses paradex-py SDK for auth/signing/orders, with aiohttp fallback.

Authentication:
    - Retail Profile: POST /auth?token_usage=interactive -> 0% fees
    - Pro Profile: Standard SDK auth -> batch orders, higher rate limits
"""

import asyncio
import json
import logging
import time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

import aiohttp

log = logging.getLogger("MM-BOT")

# SDK imports (graceful fallback)
_SDK_AVAILABLE = False
try:
    from paradex_py import ParadexSubkey
    from paradex_py.common.order import Order, OrderSide, OrderType
    from paradex_py.api.ws_client import ParadexWebsocketChannel

    _SDK_AVAILABLE = True
    log.info("paradex-py SDK loaded successfully")
except ImportError:
    log.warning("paradex-py SDK not available, using direct REST/WS")
    ParadexSubkey = None
    Order = None
    OrderSide = None
    OrderType = None
    ParadexWebsocketChannel = None


def _get_env(env_name: str):
    """Get paradex-py environment object."""
    if not _SDK_AVAILABLE:
        return None
    try:
        if env_name == "mainnet":
            from paradex_py.environment import PROD
            return PROD
        else:
            from paradex_py.environment import TESTNET
            return TESTNET
    except ImportError:
        try:
            from paradex_py.environment import Environment
            return Environment.PROD if env_name == "mainnet" else Environment.TESTNET
        except ImportError:
            return None


class ParadexClient:
    """Unified Paradex API client supporting both SDK and direct REST."""

    def __init__(
        self,
        l2_private_key: str,
        l2_address: str = "",
        env: str = "mainnet",
        market: str = "BTC-USD-PERP",
        profile: str = "retail",
        dry_run: bool = False,
    ):
        self.l2_private_key = l2_private_key
        self.l2_address = l2_address
        self.env_name = env
        self.market = market
        self.profile = profile
        self.dry_run = dry_run

        # API base URLs
        if env == "mainnet":
            self.base_url = "https://api.prod.paradex.trade/v1"
            self.ws_url = "wss://ws.api.prod.paradex.trade/v1"
        else:
            self.base_url = "https://api.testnet.paradex.trade/v1"
            self.ws_url = "wss://ws.api.testnet.paradex.trade/v1"

        # SDK instance
        self._paradex = None
        self._jwt_token: Optional[str] = None
        self._jwt_expires_at: int = 0

        # WS state
        self._ws_connected = False
        self._ws_callbacks: Dict[str, Callable] = {}
        self._ws_task: Optional[asyncio.Task] = None

        # Persistent HTTP session (aiohttp docs: "Don't create a session per request")
        self._session: Optional[aiohttp.ClientSession] = None

        # Reconnection
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    # =========================================================================
    # Initialization & Authentication
    # =========================================================================

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Return persistent HTTP session, creating if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def connect(self):
        """Initialize SDK, authenticate, and connect WebSocket."""
        # Step 0: Create persistent HTTP session
        await self._ensure_session()

        # Step 1: Initialize SDK
        if _SDK_AVAILABLE:
            try:
                env_obj = _get_env(self.env_name)
                if self.l2_address:
                    self._paradex = ParadexSubkey(
                        env=env_obj,
                        l2_private_key=self.l2_private_key,
                        l2_address=self.l2_address,
                    )
                else:
                    self._paradex = ParadexSubkey(
                        env=env_obj,
                        l2_private_key=self.l2_private_key,
                    )
                log.info(f"SDK initialized ({self.env_name})")
            except Exception as e:
                log.warning(f"SDK init failed: {e}, using REST fallback")
                self._paradex = None

        # Step 2: Authenticate
        if self.profile == "retail":
            await self._auth_interactive()
        else:
            await self._auth_standard()

        # Step 3: Connect WebSocket
        await self._ws_connect()

    async def _auth_interactive(self):
        """Authenticate with interactive token for Retail Profile (0% fees).

        Uses SDK for signing, then manually calls /auth?token_usage=interactive.
        Also monkey-patches SDK's auth() so its 4-minute auto-refresh
        continues using interactive auth instead of reverting to standard.
        """
        if self._paradex:
            try:
                # Get auth headers from SDK (includes signature)
                auth_headers = self._paradex.account.auth_headers()
                session = await self._ensure_session()
                url = f"{self.base_url}/auth?token_usage=interactive"
                headers = {"Content-Type": "application/json", **auth_headers}
                async with session.post(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._jwt_token = data.get("jwt_token")
                        self._decode_jwt_expiry()
                        # Inject interactive JWT into SDK (correct method names)
                        self._inject_sdk_token(self._jwt_token)
                        # Patch SDK auth refresh to use interactive forever
                        self._patch_sdk_interactive_auth()
                        log.info("Retail auth OK (interactive token, 0% fees)")
                        return
                    else:
                        text = await resp.text()
                        log.error(f"Interactive auth failed: {resp.status} {text}")
            except Exception as e:
                log.error(f"Interactive auth error: {e}")

        # Fallback: standard auth
        log.warning("Falling back to standard auth (may have fees)")
        await self._auth_standard()

    def _inject_sdk_token(self, jwt_token: str):
        """Inject JWT token into SDK's account and HTTP client."""
        if not self._paradex:
            return
        # Set on account object (SDK method: set_jwt_token, not set_token)
        self._paradex.account.set_jwt_token(jwt_token)
        # Set on HTTP client headers so all SDK requests use it
        self._paradex.api_client.client.headers.update(
            {"Authorization": f"Bearer {jwt_token}"}
        )
        # Reset auth timestamp so SDK doesn't immediately trigger refresh
        self._paradex.api_client.auth_timestamp = time.time()

    def _patch_sdk_interactive_auth(self):
        """Monkey-patch SDK's auth() to always use interactive token.

        The SDK's _validate_auth() calls self.auth() every 4 minutes.
        By default, auth() calls POST /auth (standard = Pro profile).
        This patch makes it call POST /auth?token_usage=interactive instead.
        """
        if not self._paradex:
            return

        from paradex_py.api.models import AuthSchema as _AuthSchema
        api_client = self._paradex.api_client
        parent = self

        def interactive_auth():
            headers = api_client.account.auth_headers()
            res = api_client.post(
                api_url=api_client.api_url,
                path="auth",
                headers=headers,
                params={"token_usage": "interactive"},
            )
            data = _AuthSchema().load(res, unknown="exclude", partial=True)
            api_client.auth_timestamp = time.time()
            api_client.account.set_jwt_token(data.jwt_token)
            api_client.client.headers.update(
                {"Authorization": f"Bearer {data.jwt_token}"}
            )
            parent._jwt_token = data.jwt_token
            parent._decode_jwt_expiry()
            log.info("SDK auth refreshed (interactive token, 0% fees)")

        api_client.auth = interactive_auth

    async def _auth_standard(self):
        """Standard SDK authentication (Pro Profile)."""
        if self._paradex:
            try:
                # SDK auto-authenticates on init, just verify
                if hasattr(self._paradex, 'api_client'):
                    result = self._paradex.api_client.fetch_account_summary()
                    if result:
                        log.info("Standard auth OK (Pro Profile)")
                        return
            except Exception as e:
                log.warning(f"SDK auth check failed: {e}")

        # Manual REST auth fallback
        await self._auth_rest_fallback()

    async def _auth_rest_fallback(self):
        """Direct REST auth when SDK is unavailable."""
        log.info("Attempting direct REST authentication...")
        # This requires starknet signing which is complex without SDK
        # For now, log a warning
        log.error(
            "Direct REST auth requires paradex-py SDK for Starknet signing. "
            "Please install: pip install paradex-py"
        )

    def _decode_jwt_expiry(self):
        """Decode JWT to extract expiry timestamp."""
        if not self._jwt_token:
            return
        try:
            import base64
            parts = self._jwt_token.split('.')
            if len(parts) >= 2:
                payload = parts[1]
                # Pad base64
                payload += '=' * (4 - len(payload) % 4)
                decoded = json.loads(base64.b64decode(payload))
                self._jwt_expires_at = decoded.get("exp", 0)
        except Exception:
            self._jwt_expires_at = int(time.time()) + 3600  # Default 1hr

    async def _ensure_auth(self) -> bool:
        """Re-authenticate if JWT is about to expire."""
        if self._jwt_token and self._jwt_expires_at > int(time.time()) + 120:
            return True
        log.info("JWT expiring, re-authenticating...")
        if self.profile == "retail":
            await self._auth_interactive()
        else:
            await self._auth_standard()
        return self._jwt_token is not None

    def _headers(self) -> Dict[str, str]:
        """HTTP headers with auth."""
        h = {"Content-Type": "application/json"}
        if self._jwt_token:
            h["Authorization"] = f"Bearer {self._jwt_token}"
        return h

    # =========================================================================
    # REST API Methods
    # =========================================================================

    async def _rest_get(self, path: str, params: dict = None) -> Optional[Dict]:
        """Generic REST GET (uses persistent session)."""
        await self._ensure_auth()
        try:
            session = await self._ensure_session()
            url = f"{self.base_url}{path}"
            async with session.get(url, headers=self._headers(), params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                text = await resp.text()
                log.warning(f"GET {path} failed: {resp.status} {text}")
                return None
        except Exception as e:
            log.error(f"GET {path} error: {e}")
            return None

    async def _rest_post(self, path: str, data: dict = None) -> Optional[Dict]:
        """Generic REST POST (uses persistent session)."""
        await self._ensure_auth()
        try:
            session = await self._ensure_session()
            url = f"{self.base_url}{path}"
            async with session.post(
                url, headers=self._headers(), json=data or {}
            ) as resp:
                if resp.status in (200, 201):
                    return await resp.json()
                text = await resp.text()
                log.warning(f"POST {path} failed: {resp.status} {text}")
                return None
        except Exception as e:
            log.error(f"POST {path} error: {e}")
            return None

    async def _rest_delete(self, path: str) -> bool:
        """Generic REST DELETE (uses persistent session)."""
        await self._ensure_auth()
        try:
            session = await self._ensure_session()
            url = f"{self.base_url}{path}"
            async with session.delete(url, headers=self._headers()) as resp:
                return resp.status in (200, 204)
        except Exception as e:
            log.error(f"DELETE {path} error: {e}")
            return False

    async def get_bbo(self) -> Optional[Dict]:
        """Get Best Bid/Offer.

        Returns: {"bid": float, "ask": float, "bid_size": float, "ask_size": float}
        """
        # Try SDK first
        if self._paradex:
            try:
                data = self._paradex.api_client.fetch_bbo(self.market)
                if data:
                    return {
                        "bid": float(data.get("bid", 0)),
                        "ask": float(data.get("ask", 0)),
                        "bid_size": float(data.get("bid_size", 0)),
                        "ask_size": float(data.get("ask_size", 0)),
                    }
            except Exception as e:
                log.debug(f"SDK fetch_bbo failed: {e}")

        # REST fallback
        data = await self._rest_get(f"/bbo/{self.market}")
        if data:
            return {
                "bid": float(data.get("bid", 0)),
                "ask": float(data.get("ask", 0)),
                "bid_size": float(data.get("bid_size", 0)),
                "ask_size": float(data.get("ask_size", 0)),
            }
        return None

    async def get_orderbook(self, depth: int = 10) -> Optional[Dict]:
        """Get orderbook depth.

        Returns: {"bids": [(price, size), ...], "asks": [(price, size), ...]}
        """
        if self._paradex:
            try:
                data = self._paradex.api_client.fetch_orderbook(
                    self.market, params={"depth": depth}
                )
                if data:
                    bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
                    asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]
                    return {"bids": bids, "asks": asks}
            except Exception as e:
                log.debug(f"SDK fetch_orderbook failed: {e}")

        data = await self._rest_get(f"/orderbook/{self.market}", {"depth": depth})
        if data:
            bids = [(float(b[0]), float(b[1])) for b in data.get("bids", [])]
            asks = [(float(a[0]), float(a[1])) for a in data.get("asks", [])]
            return {"bids": bids, "asks": asks}
        return None

    async def get_account(self) -> Optional[Dict]:
        """Get account summary (balance, equity, etc.)."""
        if self._paradex:
            try:
                return self._paradex.api_client.fetch_account_summary()
            except Exception as e:
                log.debug(f"SDK fetch_account_summary failed: {e}")

        return await self._rest_get("/account_summary")

    async def get_positions(self) -> List[Dict]:
        """Get current positions."""
        if self._paradex:
            try:
                data = self._paradex.api_client.fetch_positions()
                return data.get("results", []) if isinstance(data, dict) else data or []
            except Exception as e:
                log.debug(f"SDK fetch_positions failed: {e}")

        data = await self._rest_get("/positions")
        if data:
            return data.get("results", []) if isinstance(data, dict) else []
        return []

    async def get_open_orders(self) -> List[Dict]:
        """Get currently open orders."""
        if self._paradex:
            try:
                data = self._paradex.api_client.fetch_orders(
                    params={"market": self.market}
                )
                results = data.get("results", []) if isinstance(data, dict) else data or []
                return [o for o in results if o.get("status") == "OPEN"]
            except Exception as e:
                log.debug(f"SDK fetch_orders failed: {e}")

        data = await self._rest_get("/orders", {"market": self.market, "status": "OPEN"})
        if data:
            return data.get("results", []) if isinstance(data, dict) else []
        return []

    async def get_balance(self) -> Optional[float]:
        """Get USDC balance."""
        if self._paradex:
            try:
                data = self._paradex.api_client.fetch_balances()
                results = data.get("results", []) if isinstance(data, dict) else data or []
                for item in results:
                    if item.get("token") == "USDC":
                        return float(item.get("size", 0))
            except Exception as e:
                log.debug(f"SDK fetch_balances failed: {e}")

        data = await self._rest_get("/balance")
        if data:
            for item in data.get("results", []):
                if item.get("token") == "USDC":
                    return float(item.get("size", 0))
        return None

    async def place_order(
        self,
        side: str,
        price: float,
        size: float,
        instruction: str = "POST_ONLY",
        order_type: str = "Limit",
        reduce_only: bool = False,
    ) -> Optional[Dict]:
        """Place an order.

        Args:
            side: "BUY" or "SELL"
            price: Limit price (0 for market orders)
            size: Order size in BTC
            instruction: "POST_ONLY", "GTC", or "IOC"
            order_type: "Limit" or "Market"
            reduce_only: If True, only reduces position

        Returns:
            Order response dict or None
        """
        if self.dry_run:
            log.info(
                f"[DRY-RUN] Would place {order_type} {side} {size:.4f} "
                f"@ ${price:.1f} ({instruction})"
            )
            return {
                "id": f"dry_{int(time.time()*1000)}",
                "status": "OPEN",
                "side": side,
                "price": str(price),
                "size": str(size),
                "dry_run": True,
            }

        if self._paradex and _SDK_AVAILABLE and Order:
            try:
                # Build order using SDK
                order_side = OrderSide.Buy if side.upper() == "BUY" else OrderSide.Sell
                otype = OrderType.Market if order_type == "Market" else OrderType.Limit

                order_params = {
                    "market": self.market,
                    "order_type": otype,
                    "order_side": order_side,
                    "size": Decimal(str(size)),
                    "instruction": instruction,
                    "reduce_only": reduce_only,
                }
                if order_type != "Market":
                    order_params["limit_price"] = Decimal(str(int(price)))

                order = Order(**order_params)
                result = self._paradex.api_client.submit_order(order=order)
                if result:
                    oid = result.get("id", "?")
                    log.info(
                        f"Order placed: {side} {size:.4f} @ ${price:.0f} "
                        f"({instruction}) id={oid}"
                    )
                    return result
            except Exception as e:
                log.error(f"SDK submit_order failed: {e}")
                # Don't fallback to REST for orders (signing required)
                return None

        # REST fallback (requires SDK for signing)
        log.error("Cannot place order without SDK (signature required)")
        return None

    async def place_batch_orders(self, orders: list) -> list:
        """Place multiple orders rapidly (Pro mode: 800 req/s).

        Places orders individually via SDK submit_order — at Pro rate limits
        this is effectively instant and avoids needing to separate sign/submit.

        Args:
            orders: List of dicts with {side, price, size, instruction, reduce_only}

        Returns:
            List of order response dicts
        """
        results = []
        for o in orders:
            r = await self.place_order(
                side=o["side"],
                price=o["price"],
                size=o["size"],
                instruction=o.get("instruction", "POST_ONLY"),
                reduce_only=o.get("reduce_only", False),
            )
            if r:
                results.append(r)
        return results

    async def cancel_batch_orders(self, order_ids: list) -> bool:
        """Cancel multiple orders in a single batch request (Pro mode).

        Args:
            order_ids: List of order ID strings to cancel

        Returns:
            True if request succeeded
        """
        if not order_ids:
            return True

        if self.dry_run:
            log.info(f"[DRY-RUN] Would batch cancel {len(order_ids)} orders")
            return True

        # Try REST batch cancel (uses persistent session)
        try:
            session = await self._ensure_session()
            url = f"{self.base_url}/orders/batch"
            headers = self._headers()
            payload = {"order_ids": order_ids}
            async with session.delete(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    log.debug(f"Batch cancelled {len(order_ids)} orders")
                    return True
                text = await resp.text()
                log.warning(f"Batch cancel failed: {resp.status} {text}")
        except Exception as e:
            log.error(f"Batch cancel error: {e}")

        # Fallback: cancel individually
        for oid in order_ids:
            await self.cancel_order(oid)
        return True

    async def cancel_order(self, order_id: str, market: str = None) -> bool:
        """Cancel a single order by ID."""
        if self.dry_run:
            log.info(f"[DRY-RUN] Would cancel order {order_id}")
            return True

        if self._paradex:
            try:
                self._paradex.api_client.cancel_order(order_id=order_id)
                return True
            except Exception as e:
                log.debug(f"SDK cancel_order failed: {e}")

        return await self._rest_delete(f"/orders/{order_id}")

    async def cancel_all(self, market: str = None) -> bool:
        """Cancel all open orders."""
        mkt = market or self.market
        if self.dry_run:
            log.info(f"[DRY-RUN] Would cancel all orders for {mkt}")
            return True

        if self._paradex:
            try:
                self._paradex.api_client.cancel_all_orders({"market": mkt})
                log.info(f"All orders cancelled for {mkt}")
                return True
            except Exception as e:
                log.debug(f"SDK cancel_all failed: {e}")

        data = await self._rest_post(f"/orders/cancel_all", {"market": mkt})
        return data is not None

    async def get_system_state(self) -> str:
        """Get system status: 'ok', 'maintenance', 'cancel_only'."""
        if self._paradex:
            try:
                data = self._paradex.api_client.fetch_system_state()
                return data.get("status", "ok") if data else "ok"
            except Exception:
                pass

        data = await self._rest_get("/system/state")
        if data:
            return data.get("status", "ok")
        return "unknown"

    # =========================================================================
    # WebSocket
    # =========================================================================

    async def _ws_connect(self):
        """Connect WebSocket via SDK."""
        if not self._paradex:
            log.warning("No SDK available for WebSocket")
            return

        try:
            connected = await self._paradex.ws_client.connect()
            if connected:
                self._ws_connected = True
                self._reconnect_delay = 1.0
                log.info("WebSocket connected")
            else:
                log.warning("WebSocket connection failed")
        except Exception as e:
            log.error(f"WS connect error: {e}")

    async def _ws_reconnect(self):
        """Reconnect WebSocket with exponential backoff."""
        self._ws_connected = False
        log.info(f"WS reconnecting in {self._reconnect_delay:.0f}s...")
        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(
            self._reconnect_delay * 2, self._max_reconnect_delay
        )
        await self._ws_connect()
        # Re-subscribe all channels
        for channel_name, cb in self._ws_callbacks.items():
            await self._subscribe_channel(channel_name, cb)

    async def _subscribe_channel(self, channel_name: str, callback: Callable):
        """Internal: subscribe to a WS channel by name."""
        if not self._paradex or not _SDK_AVAILABLE or not ParadexWebsocketChannel:
            return

        channel_map = {
            "bbo": ParadexWebsocketChannel.BBO,
            "fills": ParadexWebsocketChannel.FILLS,
            "orders": ParadexWebsocketChannel.ORDERS,
            "positions": ParadexWebsocketChannel.POSITIONS,
            "order_book": ParadexWebsocketChannel.ORDER_BOOK,
        }

        channel = channel_map.get(channel_name)
        if not channel:
            log.warning(f"Unknown WS channel: {channel_name}")
            return

        params = {}
        if channel_name in ("bbo", "fills", "order_book"):
            params["market"] = self.market
        if channel_name == "orders":
            params["market"] = "ALL"
        if channel_name == "order_book":
            params["depth"] = 10

        try:
            await self._paradex.ws_client.subscribe(
                channel, callback=callback, params=params
            )
            self._ws_callbacks[channel_name] = callback
            log.info(f"Subscribed to WS channel: {channel_name}")
        except Exception as e:
            log.error(f"WS subscribe {channel_name} failed: {e}")

    async def subscribe_bbo(self, callback: Callable):
        """Subscribe to BBO updates."""
        await self._subscribe_channel("bbo", callback)

    async def subscribe_fills(self, callback: Callable):
        """Subscribe to fill notifications."""
        await self._subscribe_channel("fills", callback)

    async def subscribe_orders(self, callback: Callable):
        """Subscribe to order status updates."""
        await self._subscribe_channel("orders", callback)

    async def subscribe_positions(self, callback: Callable):
        """Subscribe to position updates."""
        await self._subscribe_channel("positions", callback)

    async def subscribe_orderbook(self, callback: Callable):
        """Subscribe to orderbook depth updates."""
        await self._subscribe_channel("order_book", callback)

    async def pump_ws(self) -> bool:
        """Pump one WS message. Returns True if message processed."""
        if not self._paradex or not self._ws_connected:
            return False
        try:
            return await self._paradex.ws_client.pump_once()
        except Exception as e:
            log.warning(f"WS pump error: {e}")
            await self._ws_reconnect()
            return False

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def close(self):
        """Clean up all connections."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        if self._paradex:
            try:
                await self._paradex.close()
            except Exception:
                pass
        log.info("Client closed")

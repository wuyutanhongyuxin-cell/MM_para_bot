"""Entry point for Paradex Market Making Bot.

Usage:
    python -m src.main                          # Live trading
    python -m src.main --dry-run                # Simulation mode
    python -m src.main --testnet --dry-run      # Testnet simulation
    python -m src.main --config my_config.yaml  # Custom config
"""

import argparse
import asyncio
import logging
import signal
import sys

from .bot import SpreadCaptureBot

log = logging.getLogger("MM-BOT")


def main():
    parser = argparse.ArgumentParser(
        description="Paradex BTC-USD-PERP Market Making Bot"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulation mode: no real orders placed"
    )
    parser.add_argument(
        "--testnet", action="store_true",
        help="Use Paradex testnet instead of mainnet"
    )
    args = parser.parse_args()

    # Create bot
    bot = SpreadCaptureBot(
        config_path=args.config,
        dry_run=args.dry_run,
        testnet=args.testnet,
    )

    # Run with proper signal handling
    if sys.platform == "win32":
        # Windows: asyncio.run + KeyboardInterrupt
        try:
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            log.info("Interrupted by user")
    else:
        # Unix: proper signal handling
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def _shutdown():
            log.info("Shutdown signal received")
            asyncio.ensure_future(bot.graceful_shutdown())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown)

        try:
            loop.run_until_complete(bot.run())
        except KeyboardInterrupt:
            loop.run_until_complete(bot.graceful_shutdown())
        finally:
            loop.close()


if __name__ == "__main__":
    main()

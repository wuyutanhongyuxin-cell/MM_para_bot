#!/usr/bin/env python3
"""Check Paradex account balance and positions.

Usage:
    python scripts/check_balance.py
    python scripts/check_balance.py --testnet
"""

import argparse
import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.paradex_client import ParadexClient


async def check(env: str):
    load_dotenv()
    l2_key = os.getenv("PARADEX_L2_PRIVATE_KEY", "")
    l2_addr = os.getenv("PARADEX_L2_ADDRESS", "")

    if not l2_key:
        print("Error: PARADEX_L2_PRIVATE_KEY not set in .env")
        return

    client = ParadexClient(
        l2_private_key=l2_key,
        l2_address=l2_addr,
        env=env,
        profile="retail",
    )

    try:
        await client.connect()

        print("=" * 40)
        print(f"  Paradex Account ({env})")
        print("=" * 40)

        # Balance
        balance = await client.get_balance()
        if balance is not None:
            print(f"  USDC Balance: ${balance:.2f}")

        # Account info
        account = await client.get_account()
        if account:
            equity = account.get("equity", account.get("account_value", "?"))
            margin = account.get("initial_margin_requirement", "?")
            print(f"  Equity:       ${equity}")
            print(f"  Margin Used:  ${margin}")

        # Positions
        positions = await client.get_positions()
        if positions:
            print()
            print("  Open Positions:")
            for pos in positions:
                market = pos.get("market", "?")
                side = pos.get("side", "?")
                size = pos.get("size", "0")
                entry = pos.get("average_entry_price", "?")
                upnl = pos.get("unrealized_pnl", "0")
                print(f"    {market}: {side} {size} @ ${entry} (uPnL: ${upnl})")
        else:
            print("  No open positions")

        # Open orders
        orders = await client.get_open_orders()
        if orders:
            print()
            print("  Open Orders:")
            for order in orders:
                side = order.get("side", "?")
                price = order.get("price", "?")
                size = order.get("remaining_size", order.get("size", "?"))
                otype = order.get("type", "?")
                inst = order.get("instruction", "?")
                print(f"    {side} {size} @ ${price} ({otype} {inst})")
        else:
            print("  No open orders")

        # System state
        state = await client.get_system_state()
        print(f"\n  System: {state}")

        print("=" * 40)

    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description="Check Paradex balance & positions")
    parser.add_argument("--testnet", action="store_true", help="Use testnet")
    args = parser.parse_args()

    env = "testnet" if args.testnet else "mainnet"
    asyncio.run(check(env))


if __name__ == "__main__":
    main()

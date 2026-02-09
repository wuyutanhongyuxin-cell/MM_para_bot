#!/usr/bin/env python3
"""Analyze trades.csv and output performance statistics.

Usage:
    python scripts/analyze_trades.py
    python scripts/analyze_trades.py --file path/to/trades.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime


def load_trades(filepath: str) -> list:
    """Load trades from CSV file."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        sys.exit(1)

    trades = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)
    return trades


def analyze(trades: list):
    """Compute and print trade statistics."""
    if not trades:
        print("No trades found.")
        return

    total = len(trades)
    realized_pnls = []
    maker_count = 0
    taker_count = 0
    buy_count = 0
    sell_count = 0
    hourly_pnl = defaultdict(float)
    consecutive_losses = 0
    max_consecutive_losses = 0
    hold_times = []

    cumulative_pnl = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for t in trades:
        rpnl = float(t.get("realized_pnl", 0))
        realized_pnls.append(rpnl)
        cumulative_pnl += rpnl

        if cumulative_pnl > peak:
            peak = cumulative_pnl
        dd = peak - cumulative_pnl
        if dd > max_drawdown:
            max_drawdown = dd

        if t.get("maker_taker") == "maker":
            maker_count += 1
        else:
            taker_count += 1

        if t.get("side") == "BUY":
            buy_count += 1
        else:
            sell_count += 1

        # Hourly distribution
        try:
            ts = datetime.fromisoformat(t.get("timestamp", ""))
            hour = ts.hour
            hourly_pnl[hour] += rpnl
        except (ValueError, TypeError):
            pass

        # Consecutive losses
        if rpnl < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        elif rpnl > 0:
            consecutive_losses = 0

    # Filter trades with realized PnL (closing trades only)
    closing_trades = [p for p in realized_pnls if p != 0]
    wins = [p for p in closing_trades if p > 0]
    losses = [p for p in closing_trades if p < 0]

    total_profit = sum(wins)
    total_loss = abs(sum(losses))
    net_pnl = sum(realized_pnls)

    # Print report
    print("=" * 60)
    print("  TRADE ANALYSIS REPORT")
    print("=" * 60)
    print()

    print(f"  Total Fills:        {total}")
    print(f"  Closing Trades:     {len(closing_trades)}")
    print(f"  Buys / Sells:       {buy_count} / {sell_count}")
    print()

    print(f"  Net PnL:            ${net_pnl:+.4f}")
    print(f"  Total Profit:       ${total_profit:+.4f}")
    print(f"  Total Loss:         ${total_loss:.4f}")
    print(f"  Profit Factor:      {total_profit / total_loss:.2f}" if total_loss > 0 else "  Profit Factor:      inf")
    print()

    if closing_trades:
        win_rate = len(wins) / len(closing_trades)
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        expectancy = net_pnl / len(closing_trades)

        print(f"  Win Rate:           {win_rate:.1%}")
        print(f"  Avg Win:            ${avg_win:+.4f}")
        print(f"  Avg Loss:           ${avg_loss:.4f}")
        print(f"  Expectancy:         ${expectancy:+.4f}/trade")
    print()

    print(f"  Max Drawdown:       ${max_drawdown:.4f}")
    print(f"  Max Consec Losses:  {max_consecutive_losses}")
    print()

    maker_pct = maker_count / total * 100 if total > 0 else 0
    print(f"  Maker Fills:        {maker_count} ({maker_pct:.0f}%)")
    print(f"  Taker Fills:        {taker_count} ({100-maker_pct:.0f}%)")
    print()

    # Hourly PnL distribution
    if hourly_pnl:
        print("  PnL by Hour (UTC):")
        print("  " + "-" * 40)
        for hour in sorted(hourly_pnl.keys()):
            pnl = hourly_pnl[hour]
            bar = "+" * int(abs(pnl) * 100) if pnl > 0 else "-" * int(abs(pnl) * 100)
            print(f"    {hour:02d}:00  ${pnl:+.4f}  {bar}")

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze trading performance")
    parser.add_argument("--file", default="trades.csv", help="Path to trades CSV")
    args = parser.parse_args()

    trades = load_trades(args.file)
    analyze(trades)


if __name__ == "__main__":
    main()

"""Comprehensive sanity check for MM bot fills data."""
import csv
import sys
import io
from datetime import datetime
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

csv_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\Downloads\fills_2026-02-11_10-18-18_to_2026-02-11_13-53-25.csv"

with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fills = list(reader)

# Reverse to chronological order (CSV is newest first)
fills.reverse()

print(f"{'='*70}")
print(f"SANITY CHECK — {csv_path.split(chr(92))[-1]}")
print(f"{'='*70}")

# ============ 1. BASIC STATS ============
print(f"\n{'='*70}")
print("1. BASIC STATS")
print(f"{'='*70}")

total = len(fills)
t_first = fills[0]["created_at"]
t_last = fills[-1]["created_at"]
dt_first = datetime.fromisoformat(t_first.replace("Z", "+00:00"))
dt_last = datetime.fromisoformat(t_last.replace("Z", "+00:00"))
runtime_sec = (dt_last - dt_first).total_seconds()
runtime_min = runtime_sec / 60

print(f"Total fills: {total}")
print(f"Time range: {t_first} → {t_last}")
print(f"Runtime: {runtime_min:.1f} min ({runtime_sec:.0f}s)")
print(f"Fill rate: {total/runtime_min:.1f} fills/min")

total_btc = sum(float(f["size"]) for f in fills)
total_usd = sum(float(f["price"]) * float(f["size"]) for f in fills)
print(f"Total volume: {total_btc:.4f} BTC (${total_usd:,.0f} notional)")

buys = [f for f in fills if f["side"] == "BUY"]
sells = [f for f in fills if f["side"] == "SELL"]
print(f"Buys: {len(buys)} ({100*len(buys)/total:.1f}%), Sells: {len(sells)} ({100*len(sells)/total:.1f}%)")

makers = [f for f in fills if f["liquidity"] == "MAKER"]
takers = [f for f in fills if f["liquidity"] == "TAKER"]
print(f"Maker: {len(makers)} ({100*len(makers)/total:.1f}%), Taker: {len(takers)} ({100*len(takers)/total:.1f}%)")

# Check flags for interactive vs pro
interactive = [f for f in fills if "interactive" in str(f.get("flags", ""))]
print(f"Interactive (retail) fills: {len(interactive)}, Pro fills: {total - len(interactive)}")

# ============ 2. FEE ANALYSIS ============
print(f"\n{'='*70}")
print("2. FEE ANALYSIS")
print(f"{'='*70}")

total_fees = sum(float(f["fee"]) for f in fills)
maker_fees = sum(float(f["fee"]) for f in fills if f["liquidity"] == "MAKER")
taker_fees = sum(float(f["fee"]) for f in fills if f["liquidity"] == "TAKER")
print(f"Total fees: ${total_fees:.6f}")
print(f"  Maker fees: ${maker_fees:.6f}")
print(f"  Taker fees: ${taker_fees:.6f}")
print(f"Avg fee per fill: ${total_fees/total:.6f}")

# Effective fee rates
maker_notional = sum(float(f["price"]) * float(f["size"]) for f in makers)
taker_notional = sum(float(f["price"]) * float(f["size"]) for f in takers)
if maker_notional > 0:
    print(f"Effective maker rate: {100*maker_fees/maker_notional:.4f}% (expected 0.003%)")
if taker_notional > 0:
    print(f"Effective taker rate: {100*taker_fees/taker_notional:.4f}% (expected 0.02%)")

# ============ 3. PnL ANALYSIS ============
print(f"\n{'='*70}")
print("3. PnL ANALYSIS")
print(f"{'='*70}")

rpnls = [(float(f["realized_pnl"]), f) for f in fills]
nonzero_rpnls = [(r, f) for r, f in rpnls if abs(r) > 1e-10]
print(f"Fills with realized PnL: {len(nonzero_rpnls)} / {total}")

gross_pnl = sum(r for r, _ in rpnls)
net_pnl = gross_pnl - total_fees
print(f"Gross PnL (before fees): ${gross_pnl:.6f}")
print(f"Net PnL (after fees):    ${net_pnl:.6f}")
print(f"PnL per hour: ${net_pnl / (runtime_min/60):.6f}")
print(f"PnL per fill: ${net_pnl / total:.6f}")

# ============ 4. WIN/LOSS ANALYSIS ============
print(f"\n{'='*70}")
print("4. WIN/LOSS ANALYSIS")
print(f"{'='*70}")

wins = [r for r, _ in nonzero_rpnls if r > 0]
losses = [r for r, _ in nonzero_rpnls if r < 0]
print(f"Winning fills: {len(wins)}")
print(f"Losing fills: {len(losses)}")
if wins and losses:
    print(f"Win rate: {100*len(wins)/(len(wins)+len(losses)):.1f}%")
    print(f"Avg win:  ${sum(wins)/len(wins):.6f}")
    print(f"Avg loss: ${sum(losses)/len(losses):.6f}")
    print(f"Win/Loss ratio: {abs(sum(wins)/len(wins) / (sum(losses)/len(losses))):.2f}")
    print(f"Profit factor: {sum(wins) / abs(sum(losses)):.3f}")
    print(f"Largest win:  ${max(wins):.6f}")
    print(f"Largest loss: ${min(losses):.6f}")

# Consecutive wins/losses
max_consec_win = 0
max_consec_loss = 0
cur_win = 0
cur_loss = 0
for r, _ in nonzero_rpnls:
    if r > 0:
        cur_win += 1
        cur_loss = 0
        max_consec_win = max(max_consec_win, cur_win)
    elif r < 0:
        cur_loss += 1
        cur_win = 0
        max_consec_loss = max(max_consec_loss, cur_loss)
print(f"Max consecutive wins: {max_consec_win}")
print(f"Max consecutive losses: {max_consec_loss}")

# ============ 5. TAKER ANALYSIS ============
print(f"\n{'='*70}")
print("5. TAKER FILL ANALYSIS (emergency exits)")
print(f"{'='*70}")

taker_rpnls = [(float(f["realized_pnl"]), f) for f in takers]
taker_nonzero = [(r, f) for r, f in taker_rpnls if abs(r) > 1e-10]
print(f"Taker fills: {len(takers)}")
if taker_nonzero:
    taker_wins = sum(1 for r, _ in taker_nonzero if r > 0)
    taker_losses = sum(1 for r, _ in taker_nonzero if r < 0)
    print(f"  Taker wins: {taker_wins}, losses: {taker_losses}")
    taker_total_pnl = sum(r for r, _ in taker_rpnls)
    print(f"  Taker total PnL: ${taker_total_pnl:.6f}")
    for r, f in taker_rpnls:
        print(f"    {f['created_at'][:19]} {f['side']} {f['size']} @ ${f['price']} rPnL=${r:.6f}")

# ============ 6. POSITION TRACKING ============
print(f"\n{'='*70}")
print("6. POSITION TRACKING")
print(f"{'='*70}")

pos = 0.0
max_long = 0.0
max_short = 0.0
positions = []
for f in fills:
    size = float(f["size"])
    if f["side"] == "BUY":
        pos += size
    else:
        pos -= size
    positions.append(pos)
    max_long = max(max_long, pos)
    max_short = min(max_short, pos)

print(f"Max long position: {max_long:.5f} BTC")
print(f"Max short position: {max_short:.5f} BTC")
print(f"Final position: {pos:.5f} BTC")
over_50pct = sum(1 for p in positions if abs(p) > 0.005)
print(f"Times position > 50% max (0.005): {over_50pct} ({100*over_50pct/len(positions):.1f}%)")

# Sign flips without going through zero
sign_flips = 0
for i in range(1, len(positions)):
    if positions[i-1] > 0.0003 and positions[i] < -0.0003:
        sign_flips += 1
    elif positions[i-1] < -0.0003 and positions[i] > 0.0003:
        sign_flips += 1
print(f"Position sign flips (skip zero): {sign_flips}")

# ============ 7. TIME ANALYSIS ============
print(f"\n{'='*70}")
print("7. TIME ANALYSIS")
print(f"{'='*70}")

# Fills per 5-minute bucket
buckets = defaultdict(int)
pnl_buckets = defaultdict(float)
for f in fills:
    dt = datetime.fromisoformat(f["created_at"].replace("Z", "+00:00"))
    bucket = dt.strftime("%H:%M")[:4] + "0"  # 10-min buckets
    buckets[bucket] += 1
    pnl_buckets[bucket] += float(f["realized_pnl"])

print("10-min buckets (fills | cumPnL contribution):")
cum = 0
for bucket in sorted(buckets.keys()):
    cum += pnl_buckets[bucket]
    bar = "█" * (buckets[bucket] // 2)
    print(f"  {bucket}: {buckets[bucket]:3d} fills | pnl=${pnl_buckets[bucket]:+.4f} | cum=${cum:+.4f} {bar}")

# Gaps (>2 min between fills = possible cooldown)
print(f"\nGaps > 2 minutes (possible cooldowns):")
gap_count = 0
for i in range(1, len(fills)):
    dt1 = datetime.fromisoformat(fills[i-1]["created_at"].replace("Z", "+00:00"))
    dt2 = datetime.fromisoformat(fills[i]["created_at"].replace("Z", "+00:00"))
    gap = (dt2 - dt1).total_seconds()
    if gap > 120:
        gap_count += 1
        print(f"  {fills[i-1]['created_at'][:19]} → {fills[i]['created_at'][:19]}: {gap:.0f}s ({gap/60:.1f}min)")
if gap_count == 0:
    print("  None")

# ============ 8. PnL CURVE ============
print(f"\n{'='*70}")
print("8. PnL CURVE (cumulative)")
print(f"{'='*70}")

cum_pnl = 0
cum_pnl_net = 0
checkpoints = [0, len(fills)//4, len(fills)//2, 3*len(fills)//4, len(fills)-1]
for i, (r, f) in enumerate(rpnls):
    cum_pnl += r
    cum_pnl_net = cum_pnl - sum(float(fills[j]["fee"]) for j in range(i+1))
    if i in checkpoints:
        print(f"  Fill #{i+1:3d}: gross=${cum_pnl:+.6f} net=${cum_pnl_net:+.6f} ({f['created_at'][:19]})")

# Final
print(f"\n  FINAL: gross=${gross_pnl:+.6f}  fees=${total_fees:.6f}  net=${net_pnl:+.6f}")

# ============ 9. VERDICT ============
print(f"\n{'='*70}")
print("9. SANITY CHECK VERDICT")
print(f"{'='*70}")

issues = []
if len(takers) / total > 0.10:
    issues.append(f"HIGH taker ratio: {100*len(takers)/total:.1f}% (target <10%)")
if len(wins) > 0 and len(losses) > 0:
    pf = sum(wins) / abs(sum(losses))
    if pf < 0.8:
        issues.append(f"LOW profit factor: {pf:.3f} (negative expectation)")
    elif pf < 1.0:
        issues.append(f"MARGINAL profit factor: {pf:.3f} (near breakeven)")
    wr = len(wins) / (len(wins) + len(losses))
    if wr < 0.40:
        issues.append(f"LOW win rate: {100*wr:.1f}%")
    avg_wl = abs(sum(wins)/len(wins) / (sum(losses)/len(losses)))
    if avg_wl < 0.5:
        issues.append(f"BAD win/loss ratio: {avg_wl:.2f} (losses much larger than wins)")
if net_pnl < 0:
    issues.append(f"NEGATIVE net PnL: ${net_pnl:.6f}")
if sign_flips > 0:
    issues.append(f"Position sign flips: {sign_flips}")
if max_consec_loss >= 8:
    issues.append(f"HIGH consecutive losses: {max_consec_loss}")

if not issues:
    print("✓ No major red flags detected")
else:
    for iss in issues:
        print(f"✗ {iss}")

print(f"\nOverall: {'NEGATIVE EXPECTATION' if net_pnl < -0.01 else 'MARGINAL' if abs(net_pnl) < 0.05 else 'POSITIVE'}")

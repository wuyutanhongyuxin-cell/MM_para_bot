"""Analyze MM bot log file for key patterns."""
import re
import sys
import statistics

log_path = sys.argv[1] if len(sys.argv) > 1 else r"E:\01——grid_log\para_logs\run_2026-02-11_155736.log"

with open(log_path, encoding="utf-8") as f:
    lines = f.readlines()

obi_values = []
one_sided_bid = 0  # bid blocked (ask-only)
one_sided_ask = 0  # ask blocked (bid-only)
both_sided = 0
total_quotes = 0

bid_pat = re.compile(r'bid=\$(\d+)x(\d+\.\d+)')
ask_pat = re.compile(r'ask=\$(\d+)x(\d+\.\d+)')
obi_pat = re.compile(r'obi=([+-]?\d+\.\d+)')

for line in lines:
    if '[QUOTE]' in line:
        total_quotes += 1
        m = obi_pat.search(line)
        if m:
            obi_values.append(float(m.group(1)))

        bm = bid_pat.search(line)
        am = ask_pat.search(line)
        if bm and am:
            bid_size = float(bm.group(2))
            ask_size = float(am.group(2))
            if bid_size == 0 and ask_size > 0:
                one_sided_bid += 1
            elif ask_size == 0 and bid_size > 0:
                one_sided_ask += 1
            elif bid_size > 0 and ask_size > 0:
                both_sided += 1

print(f"Total quotes: {total_quotes}")
print(f"Both-sided: {both_sided} ({100*both_sided/total_quotes:.1f}%)")
print(f"Bid blocked (ask-only): {one_sided_bid} ({100*one_sided_bid/total_quotes:.1f}%)")
print(f"Ask blocked (bid-only): {one_sided_ask} ({100*one_sided_ask/total_quotes:.1f}%)")
total_one = one_sided_bid + one_sided_ask
print(f"Total one-sided: {total_one} ({100*total_one/total_quotes:.1f}%)")
print()

# OBI stats
abs_obi = [abs(v) for v in obi_values]
above_04 = sum(1 for v in abs_obi if v > 0.40)
above_02 = sum(1 for v in abs_obi if v > 0.20)
print(f"OBI mean: {statistics.mean(obi_values):+.3f}")
print(f"OBI abs mean: {statistics.mean(abs_obi):.3f}")
print(f"OBI abs median: {statistics.median(abs_obi):.3f}")
print(f"OBI |>0.40|: {above_04} ({100*above_04/len(obi_values):.1f}%)")
print(f"OBI |>0.20|: {above_02} ({100*above_02/len(obi_values):.1f}%)")
print(f"OBI range: [{min(obi_values):+.3f}, {max(obi_values):+.3f}]")
print()

# Fills analysis
fills = []
for line in lines:
    if '[FILL]' in line:
        m_rpnl = re.search(r'rPnL=\$([+-]?\d+\.\d+)', line)
        m_cum = re.search(r'cumPnL=\$([+-]?\d+\.\d+)', line)
        m_side = re.search(r'(MAKER|TAKER) (BUY|SELL)', line)
        if m_rpnl and m_cum and m_side:
            fills.append({
                'rpnl': float(m_rpnl.group(1)),
                'cum': float(m_cum.group(1)),
                'type': m_side.group(1),
                'side': m_side.group(2),
            })

print(f"Total fills: {len(fills)}")
if fills:
    makers = sum(1 for f in fills if f['type'] == 'MAKER')
    takers = sum(1 for f in fills if f['type'] == 'TAKER')
    print(f"Maker: {makers}, Taker: {takers}")
    print(f"Final cumPnL: ${fills[-1]['cum']:.4f}")

    # Consecutive losses
    consec = 0
    max_consec = 0
    breaker_count = 0
    for f in fills:
        if f['rpnl'] < 0:
            consec += 1
            if consec > max_consec:
                max_consec = consec
            if consec >= 5:
                breaker_count += 1
        elif f['rpnl'] > 0:
            consec = 0
    print(f"Max consecutive losses: {max_consec}")

    # Win/loss analysis
    wins = [f['rpnl'] for f in fills if f['rpnl'] > 0]
    losses = [f['rpnl'] for f in fills if f['rpnl'] < 0]
    print(f"Winning fills: {len(wins)}, avg=${statistics.mean(wins):.4f}" if wins else "No wins")
    print(f"Losing fills: {len(losses)}, avg=${statistics.mean(losses):.4f}" if losses else "No losses")
    print(f"Win rate: {100*len(wins)/(len(wins)+len(losses)):.1f}%" if wins and losses else "N/A")

print()
# Emergency exits
emergencies = [l.strip() for l in lines if 'EMERGENCY EXIT' in l and 'Exchange position' in l]
print(f"Emergency exits: {len(emergencies)}")
for e in emergencies:
    print(f"  {e}")

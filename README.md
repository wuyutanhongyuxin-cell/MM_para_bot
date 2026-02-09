# Paradex BTC-USD-PERP Market Making Bot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Platform-Paradex-green.svg" alt="Paradex">
  <img src="https://img.shields.io/badge/Strategy-Market_Making-orange.svg" alt="Strategy">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center"><b>基于简化 Avellaneda-Stoikov 算法的 Paradex 永续合约做市机器人</b></p>
<p align="center"><b>A simplified Avellaneda-Stoikov market making bot for Paradex perpetual contracts</b></p>

---

## 策略简介 | Strategy Overview

本机器人采用 **做市策略（Market Making）** 而非方向性交易。核心盈利逻辑：

> 在 BTC-USD-PERP 的买卖两侧同时挂 POST_ONLY 限价单，捕获 bid-ask spread。配合 Paradex Retail Profile 的 **零手续费**（Maker 0% + Taker 0%），做市利润不被费用侵蚀。通过库存管理控制仓位风险，用反向 OBI（Order Book Imbalance）信号微调报价位置。

This bot uses a **market making strategy** instead of directional trading. It places POST_ONLY limit orders on both sides of the BTC-USD-PERP orderbook to capture the bid-ask spread. Combined with Paradex Retail Profile's **zero fees** (Maker 0% + Taker 0%), spread profits are not eroded by fees.

---

## 架构 | Architecture

```
                    ┌──────────────────────┐
                    │    Paradex Exchange   │
                    │  BTC-USD-PERP Market  │
                    └──────┬──────┬────────┘
                      REST │      │ WebSocket
                           │      │ (BBO, Fills, Orders)
                    ┌──────┴──────┴────────┐
                    │   ParadexClient       │
                    │  (paradex-py SDK)     │
                    └──────┬──────┬────────┘
                           │      │
              ┌────────────┘      └────────────┐
              │                                │
    ┌─────────┴─────────┐          ┌───────────┴──────────┐
    │   QuoteEngine     │          │    RiskManager       │
    │ Avellaneda-Stoikov│          │ Rate Limits + PnL    │
    │ + OBI Overlay     │          │ + Inventory Timeout  │
    └─────────┬─────────┘          └───────────┬──────────┘
              │                                │
              └────────────┬───────────────────┘
                           │
                    ┌──────┴──────┐
                    │ SpreadCapture│
                    │     Bot      │
                    │ (State Machine)│
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         ┌────┴───┐  ┌────┴───┐  ┌────┴────┐
         │ Logger │  │ State  │  │  CSV    │
         │Console │  │Tracker │  │ Writer  │
         └────────┘  └────────┘  └─────────┘
```

---

## 快速开始 | Quick Start

### 环境要求 | Requirements

- Python 3.10+
- Paradex 账户（已 onboard，有 USDC 余额）
- L2 Private Key（从 Paradex 导出）

### 1. 安装 | Install

```bash
git clone https://github.com/wuyutanhongyuxin-cell/MM_para_bot.git
cd MM_para_bot

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 | Configure

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 Paradex L2 私钥：

```ini
PARADEX_L2_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE
```

<details>
<summary>如何获取 L2 私钥 | How to get L2 Private Key</summary>

1. 登录 [Paradex](https://app.paradex.trade/)
2. 点击右上角钱包图标
3. 选择 "Export Private Key"
4. 输入密码验证
5. 复制显示的私钥
</details>

### 3. 模拟运行 | Dry Run (Recommended First)

```bash
python -m src.main --dry-run
```

Dry-run 模式会连接真实市场数据，但不会实际下单。你可以安全地验证配置和策略逻辑。

### 4. 正式运行 | Live Trading

```bash
python -m src.main
```

使用 Testnet 测试：

```bash
python -m src.main --testnet
```

自定义配置文件：

```bash
python -m src.main --config my_config.yaml
```

---

## 配置说明 | Configuration

所有策略参数在 `config.yaml` 中配置：

### 核心做市参数 | Core Market Making

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `gamma` | 0.3 | 风险厌恶系数，越大越保守 | 0.1 - 0.5 |
| `kappa` | 1.5 | 波动率敏感度，影响 spread 宽度 | 1.0 - 3.0 |
| `min_half_spread` | 1.0 | 最小半 spread（$），报价下限 | 1.0 - 3.0 |
| `refresh_interval` | 8.0 | 报价刷新间隔（秒） | 5.0 - 15.0 |
| `base_size` | 0.0003 | 基础订单量（BTC） | 0.0003 - 0.001 |
| `max_position` | 0.001 | 最大净仓位（BTC） | 0.001 - 0.01 |

### OBI Overlay

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | true | 是否启用 OBI 信号 |
| `alpha` | 0.3 | EMA 平滑系数（0=慢, 1=快） |
| `threshold` | 0.3 | 激活阈值：\|OBI\| > 此值才生效 |
| `delta` | 0.3 | 信号强度（fair price 偏移幅度） |
| `depth` | 5 | 使用前 N 档 orderbook |

### 风控 | Risk Management

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_loss_per_hour` | $0.50 | 小时最大亏损 |
| `max_loss_per_day` | $1.00 | 日最大亏损 |
| `inventory_timeout` | 120s | 库存软超时 → 缩紧 spread |
| `emergency_timeout` | 300s | 库存硬超时 → IOC 强平 |
| `max_unrealized_loss` | $0.30 | 最大未实现亏损 → 触发强平 |

### 频率限制 | Rate Limits (Retail Profile)

| 参数 | 设定值 | Paradex 限制 |
|------|--------|-------------|
| 每秒 | 2 | 3 |
| 每分钟 | 25 | 30 |
| 每小时 | 280 | 300 |
| 每天 | 950 | 1000 |

> 留有安全余量，避免触发自动切换到 Pro Profile。

---

## 策略算法详解 | Algorithm Details

### Simplified Avellaneda-Stoikov

<details>
<summary>展开查看完整算法 | Click to expand</summary>

#### Step 1: Fair Price（公允价格）

$$\text{fair\_price} = \text{mid\_price} - \gamma \cdot Q \cdot \sigma^2$$

其中：
- $\gamma$：风险厌恶系数
- $Q$：当前净仓位
- $\sigma$：价格波动率（mid price 差值的标准差）

当持多仓（Q > 0）时，fair price 降低，鼓励卖出；反之亦然。

#### Step 2: OBI Contrarian Overlay

$$\text{if } |OBI_{smooth}| > threshold:$$
$$\quad \text{fair\_price} \mathrel{+}= -\delta \cdot OBI_{smooth} \cdot \text{spread}$$

OBI 为正（买压）时，反向下移 fair price（逆向思维：大众在买，我们提供卖盘）。

#### Step 3: Half Spread

$$\text{half\_spread} = \max(\text{min\_spread},\; \text{spread} \times 0.4 + \kappa \cdot \sigma)$$

$$\text{if } \text{inv\_ratio} > 0.5: \quad \text{half\_spread} \mathrel{\times}= (1 + \text{inv\_ratio})$$

库存越重，spread 越宽（降低被继续填单的概率）。

#### Step 4: Quote Prices

$$\text{bid} = \lfloor \text{fair} - \text{half\_spread} \rfloor_{\text{tick}}$$
$$\text{ask} = \lceil \text{fair} + \text{half\_spread} \rceil_{\text{tick}}$$

强制不穿越 BBO：$\text{bid} \leq \text{best\_bid}$，$\text{ask} \geq \text{best\_ask}$

#### Step 5: Size Skewing

| 情况 | bid_size | ask_size |
|------|----------|----------|
| 无仓位 | base_size | base_size |
| 多头 (Q>0) | base × max(0, 1-Q/max) | base_size |
| 空头 (Q<0) | base_size | base × max(0, 1+Q/max) |

</details>

### 状态机 | State Machine

```
IDLE (无仓位)
  └─ BBO更新 + 风控OK + refresh到期 → 双边挂单
QUOTING (双边挂单)
  ├─ bid成交 → INVENTORY_LONG
  ├─ ask成交 → INVENTORY_SHORT
  └─ 双边成交 → IDLE (完美 spread 捕获!)
INVENTORY_LONG / SHORT
  ├─ 出场单成交 → IDLE
  ├─ 120s超时 → 缩紧spread
  ├─ 300s超时 → IOC紧急平仓
  └─ 浮亏>$0.30 → IOC紧急平仓
```

---

## 风控体系 | Risk Management

### 多层保护 | Multi-Layer Protection

1. **频率限制**：4个时间窗口的订单计数器，预留安全余量
2. **PnL 限制**：小时/日亏损上限，超限自动停止交易
3. **库存超时**：持仓超 120s 缩紧报价，超 300s IOC 强制平仓
4. **浮亏保护**：未实现亏损超阈值立即强平
5. **时段过滤**：只在高流动性时段（UTC 7-16）交易
6. **POST_ONLY**：保证所有订单都是 Maker，永远不穿越对手盘

### 紧急平仓流程 | Emergency Exit

```
触发条件:
  - 库存持有 > 300秒
  - 未实现亏损 > $0.30
  - Ctrl+C 退出信号

执行流程:
  1. 撤销所有挂单
  2. 尝试 Maker 限价平仓（3秒等待）
  3. 若未成交 → IOC 市价强平
  4. 打印最终统计
```

---

## 交易日志分析 | Trade Analysis

每笔成交自动记录到 `trades.csv`。使用分析脚本：

```bash
python scripts/analyze_trades.py
python scripts/analyze_trades.py --file path/to/trades.csv
```

输出包括：
- 总交易数、胜率、利润因子、期望值
- Maker vs Taker 填充比例
- 按小时分布的 PnL
- 最大连续亏损
- 最大回撤

### 检查余额 | Check Balance

```bash
python scripts/check_balance.py
python scripts/check_balance.py --testnet
```

---

## Paradex 平台特点 | Platform Notes

### Retail vs Pro Profile

| 特性 | Retail | Pro |
|------|--------|-----|
| Maker 费率 | **0%** | 0.003% |
| Taker 费率 | **0%** | 0.02% |
| Speed Bump | 500ms 下单 / 300ms 撤单 | 无 |
| 频率限制 | 3/s, 30/min, 300/hr, 1000/day | 800/s |
| Batch Orders | 不支持 | 支持 |
| 获取方式 | `?token_usage=interactive` | 默认 |

> 本 bot 默认使用 **Retail Profile**（零费用），这是做市策略的核心优势。

### XP 收益 | XP Rewards

做市行为可获得 Paradex Quote Quality XP 奖励，进一步增加策略收益。

### BTC-USD-PERP 参数

| 参数 | 值 |
|------|-----|
| 最小订单量 | 0.0003 BTC |
| Tick Size | $1 |
| 最大杠杆 | 20x |
| Funding 周期 | 8小时 |

---

## 项目结构 | Project Structure

```
MM_para_bot/
├── README.md               # 本文档
├── config.yaml             # 策略配置
├── .env.example            # 环境变量模板
├── requirements.txt        # Python 依赖
├── LICENSE                 # MIT
├── src/
│   ├── main.py             # 入口
│   ├── bot.py              # SpreadCaptureBot 主类
│   ├── quote_engine.py     # 报价引擎
│   ├── risk_manager.py     # 风控管理
│   ├── state.py            # 状态管理
│   ├── paradex_client.py   # API 封装
│   ├── logger.py           # 日志系统
│   └── utils.py            # 工具函数
├── tests/
│   ├── test_quote_engine.py
│   ├── test_risk_manager.py
│   └── test_dry_run.py
└── scripts/
    ├── analyze_trades.py   # 交易分析
    └── check_balance.py    # 余额检查
```

---

## FAQ / Troubleshooting

<details>
<summary>Q: 为什么用做市而不是方向性交易？</summary>

A: 经过 273 笔实盘交易的诊断，方向性交易（基于 OBI 信号）胜率仅 45.1%，利润因子 0.78，期望值为负。核心发现是 OBI 没有方向预测能力，但 Maker 出场能赚取 spread。因此转向做市策略，不预测方向，只捕获 spread。
</details>

<details>
<summary>Q: POST_ONLY 被拒绝怎么办？</summary>

A: 如果你的限价单会立即成交（穿越对手盘），Paradex 会自动拒绝。Bot 会在下次 BBO 更新时重新报价，这是正常行为。
</details>

<details>
<summary>Q: 启动资金需要多少？</summary>

A: 最低 $50-100 USDC 即可运行（base_size=0.0003 BTC，约 $30 名义值，20x 杠杆下保证金约 $1.5）。推荐 $100 以上以留有充足的风控余量。
</details>

<details>
<summary>Q: 会不会超出频率限制被切换到 Pro Profile？</summary>

A: 不会。Bot 内置 4 层频率限制计数器，所有限制都预留了安全余量（例如小时限制设为 280，Paradex 实际限制 300）。
</details>

<details>
<summary>Q: 如何切换到 Pro Profile？</summary>

A: 在 `config.yaml` 中设置 `paradex.profile: "pro"`。Pro Profile 支持更高频率和 batch orders，但有 0.003% Maker 费用。
</details>

---

## 免责声明 | Disclaimer

- 本软件仅供学习和研究使用 / For educational and research purposes only
- 加密货币交易存在重大风险，可能导致本金全部损失 / Cryptocurrency trading carries significant risk
- 作者不对使用本软件造成的任何损失负责 / Authors are not responsible for any losses
- 请先使用 `--dry-run` 模式充分测试 / Always test with `--dry-run` first
- 建议使用小资金测试，确认稳定后再增加 / Start with small amounts

---

<p align="center">
  <b>如果这个项目对你有帮助，请给个 Star!</b><br>
  <b>If this project helps you, please give it a Star!</b>
</p>

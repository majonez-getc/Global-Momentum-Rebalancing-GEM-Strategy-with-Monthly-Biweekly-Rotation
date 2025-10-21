# Global Equity Momentum (GEM) — Automated Rebalancing Backtests

This repository contains two Python scripts that implement and compare Global Equity Momentum (GEM)-style investment strategies with different rebalancing frequencies:

- **`gem_1month.py`** → Monthly rebalancing (classic GEM style)  
- **`gem_2weeks.py`** → Biweekly rebalancing (higher-frequency variant)  

The goal is to explore how rebalancing frequency affects performance, drawdowns, and overall strategy robustness when trading major ETFs across global equity and bond markets.

---

## 🚀 Strategy Overview

The GEM (Global Equity Momentum) strategy allocates capital based on **relative momentum** between global equity ETFs and an **absolute momentum (trend) filter** to decide between risk-on and risk-off regimes.

**Key features include:**

- Momentum ranking using multiple lookback periods (3, 6, 12 months)  
- Optional trend filter using a risk benchmark (default: `SPY`)  
- Automatic rebalancing at chosen frequency (monthly / biweekly)  
- Optional transaction fees and currency conversion costs  
- Optional annual capital gains tax (Polish “Belka” tax simulation)  
- Detailed result `DataFrame` with all transactions, returns, and cumulative performance  
- Benchmark plotting against selected ETFs  

---

## 🧩 Inputs

You can customize:

- Tickers and start date  
- Momentum lookback periods  
- Transaction fees and tax rate  
- Trend filter settings  
- Rebalancing frequency  

All configuration options are stored in a **`StrategyParams`** dataclass for convenience.

---

## 📊 Outputs

Each run produces:

- **Results DataFrame** with:
  - Selected ETFs  
  - Monthly/biweekly returns  
  - Transaction history  
  - Cumulative capital over time  
- **Benchmark plot** comparing GEM vs. underlying ETFs  

**Example plot output:**


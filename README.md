Global Equity Momentum (GEM) — Automated Rebalancing Backtests

This repository contains two Python scripts that implement and compare Global Equity Momentum (GEM)-style investment strategies with different rebalancing frequencies:

gem_1month.py → Monthly rebalancing (classic GEM style)

gem_2weeks.py → Biweekly rebalancing (higher-frequency variant)

The goal is to explore how rebalancing frequency affects performance, drawdowns, and overall strategy robustness when trading major ETFs across global equity and bond markets.

🚀 Strategy Overview

The GEM (Global Equity Momentum) strategy allocates capital based on relative momentum between global equity ETFs and absolute momentum (trend filter) to decide between risk-on and risk-off regimes.

The key features include:

Momentum ranking using multiple lookback periods (3, 6, 12 months)

Optional trend filter using a risk benchmark (default: SPY)

Automatic rebalancing at chosen frequency (monthly / biweekly)

Optional transaction fees and currency conversion costs

Optional annual capital gains tax (Polish “Belka” tax simulation)

Detailed result DataFrame with all transactions, returns, and cumulative performance

Benchmark plotting against selected ETFs

🧩 Inputs

You can customize:

Tickers and start date

Momentum lookback periods

Transaction fees and tax rate

Trend filter settings

Rebalancing frequency

All configuration options are stored in a StrategyParams dataclass for convenience.

📊 Outputs

Each run produces:

Results DataFrame with:

Selected ETFs

Monthly/biweekly returns

Transaction history

Cumulative capital over time

Benchmark plot comparing GEM vs. underlying ETFs

Example plot output:

GEM (modified) vs Benchmark ETFs (normalized)
----------------------------------------------
| Value normalized to 1.0 at start date |

⚙️ How to Use

Install dependencies:

pip install yfinance pandas numpy matplotlib


Run the strategy:

python gem_1month.py


Adjust parameters in the StrategyParams section to experiment with different setups.

📈 Example Use Case

Compare long-term performance between monthly and biweekly GEM rebalancing frequencies to evaluate if more frequent rotation improves return-to-risk metrics.

🧮 Example Metrics (printed to console)

CAGR (Compound Annual Growth Rate)

Volatility

Sharpe Ratio

Max Drawdown

Total Taxes Paid (if enabled)

📂 Repository Structure
/global-momentum-rebalancing/
│
├── gem_1month.py     # GEM strategy with monthly rebalancing
├── gem_2weeks.py     # GEM strategy with biweekly rebalancing
├── README.md          # Project documentation
└── results/           # Optional folder for storing results and plots

🧠 Inspiration

Based on the Global Equity Momentum framework popularized by Gary Antonacci, extended here with:

Tax and transaction modeling

Multi-frequency rebalancing comparison

Practical implementation with modern ETF proxies

🏁 License

MIT License — feel free to use, modify, and experiment.

⭐️ If You Find It Useful

Give this repo a ⭐️ and share your own GEM tweaks or extensions!

"""
GEM Momentum Strategy (translated to English)

This script implements a simple Global Equity Momentum (GEM) style strategy.
It downloads historical adjusted prices for selected ETFs using yfinance, resamples
prices to monthly frequency, computes momentum as the average return over several
lookback periods, applies an optional trend (risk-on) filter based on a risk
ticker (default SPY), optionally applies transaction fees when switching between
assets (and when currency changes), and optionally applies an annual capital
gains tax (Poland's 'Belka' tax concept) at the end of each calendar year.

Outputs:
- A DataFrame with the strategy results (selected ETFs, transactions, monthly returns,
  cumulative capital over time) and
- Monthly price series used for benchmark comparison.

Usage:
- Adjust StrategyParams to configure tickers, dates, fees and tax rules.
- Run the module as script to see a sample run and a comparison plot.

"""
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


@dataclass
class StrategyParams:
    # ---- universal data settings ----
    start_date: str = "2010-10-31"        # date to start downloading historical prices (YYYY-MM-DD)
    end_date: Optional[str] = None         # end date (None = up to today)
    resample_rule: str = "ME"             # resampling rule for periods ("ME" = month end)

    # ---- universal strategy parameters ----
    top_n: int = 2                         # how many ETFs to pick each month (Top-N)
    momentum_periods: Tuple[int, ...] = (3, 6, 12)  # momentum lookback periods in months (composite = mean)

    # ---- fees and transaction costs ----
    with_fee: bool = True                  # whether to apply fees / currency conversion costs
    fee_rate: float = 0.01                 # one-off fee rate used when currencies / holdings change

    # ---- annual capital gains tax (Belka) ----
    apply_annual_tax: bool = True          # whether to apply annual capital gains tax
    tax_rate: float = 0.19                 # tax rate

    # ---- trend filter (optional) ----
    use_trend_filter: bool = True          # if True -> use a simple trend filter on SPY
    trend_filter_window: int = 10          # SMA window (in months) used as trend filter
    risk_filter_ticker: str = "SPY"        # ticker used as risk-on filter
    cash_proxy_ticker: str = "BIL"         # ticker used as cash proxy when comparing

    # ---- tickers and currencies ----
    tickers: Dict[str, str] = field(default_factory=lambda: {
        'Nasdaq 100': 'QQQ',
        'S&P 500': 'SPY',
        'Developed Markets': 'VEA',
        'Emerging Markets': 'VWO',
        'Long-Term Bonds': 'TLT',
        'Short-Term Bonds': 'BIL'
    })
    ticker_currency_overrides: Dict[str, str] = field(default_factory=lambda: {
        # mapping ticker -> currency (default USD; list exceptions here)
        'VEA': 'EUR'
    })

    # ---- analysis / output ----
    risk_free_rate: float = 0.03           # risk-free rate used for Sharpe ratio
    calc_start: Optional[str] = None       # date to start the simulation (if None -> start_date + 12 months)
    verbose: bool = True                   # if True -> print results and metrics


# ----- Helper functions -----

def risk_metrics(cap_series: pd.Series, risk_free_rate: float = 0.03) -> Dict[str, Any]:
    cap_series = cap_series.sort_index()
    r = cap_series.pct_change().dropna()
    years = (cap_series.index[-1] - cap_series.index[0]).days / 365.25
    start_val = float(cap_series.iloc[0])
    end_val = float(cap_series.iloc[-1])
    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 and start_val > 0 else np.nan
    vol = r.std() * np.sqrt(12)
    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else np.nan
    max_dd = (cap_series / cap_series.cummax() - 1).min()

    metrics = {
        'start_value': start_val,
        'end_value': end_val,
        'years': years,
        'cagr': cagr,
        'volatility': vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd
    }
    return metrics


def plot_vs_benchmarks(results_df: pd.DataFrame, monthly_prices: pd.DataFrame, name_map: Dict[str, str], start_date: pd.Timestamp):
    if results_df.empty:
        print("No data to plot.")
        return

    results_df['Date'] = pd.to_datetime(results_df['Date'])
    strat = results_df.set_index('Date')['Cumulative Capital'].astype(float)

    bench = monthly_prices.loc[monthly_prices.index >= pd.to_datetime(start_date)].bfill()
    bench = bench.divide(bench.iloc[0])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strat.index, strat.values, label='GEM (modified)', linewidth=2)
    for col in bench.columns:
        ax.plot(bench.index, bench[col].values, linestyle='--', linewidth=1, label=name_map.get(col, col))

    ax.set(title='GEM vs ETF (normalized)', ylabel='Value (normalized)', xlabel='Date')
    ax.legend(fontsize='small', ncol=2)
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


# ----- Main strategy function -----

def run_gem_strategy(params: StrategyParams) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], pd.Timestamp]:
    tickers = dict(params.tickers)
    if params.ticker_currency_overrides:
        ticker_currency = {t: 'USD' for t in tickers.values()}
        ticker_currency.update(params.ticker_currency_overrides)
    else:
        ticker_currency = {t: 'USD' for t in tickers.values()}

    equity = ["QQQ", "SPY", "VEA", "VWO"]
    bonds = ["TLT", "BIL"]

    start_date = params.start_date
    end_date = params.end_date or pd.to_datetime('today').strftime('%Y-%m-%d')

    data = yf.download(list(set(tickers.values())), start=start_date, end=end_date,
                       progress=False, auto_adjust=True)
    if data.empty:
        if params.verbose:
            print("No data from yfinance for the requested tickers / dates.")
        return pd.DataFrame(), pd.DataFrame(), {}, pd.Timestamp(start_date)

    price_df = data['Close'] if isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns.levels[0] else data

    monthly = price_df.resample(params.resample_rule).last()
    if monthly.index[-1] > pd.Timestamp.today().normalize():
        monthly = monthly.iloc[:-1]
    monthly_ret = monthly.pct_change()

    mom_list = []
    for p in params.momentum_periods:
        mom_list.append(monthly / monthly.shift(p) - 1)
    momentum = pd.concat(mom_list, axis=1).groupby(level=0, axis=1).mean()

    if params.calc_start:
        calc_start = pd.to_datetime(params.calc_start)
    else:
        calc_start = pd.to_datetime(start_date) + pd.DateOffset(months=12)

    results: List[Dict[str, Any]] = [{
        'Date': calc_start.strftime('%Y-%m-%d'),
        'Previous ETF': '',
        'Selected ETF': '',
        'Transaction': 'No',
        'Monthly Return': 0.0,
        'Cumulative Capital': 1.0
    }]

    capital = 1.0
    prev_set: Optional[set] = None

    # --- variables for annual tax accounting ---
    tax_rate = params.tax_rate if params.apply_annual_tax else 0.0
    capital_at_year_start = capital
    current_tax_year = calc_start.year
    total_taxes_paid = 0.0

    for date in momentum.index[momentum.index >= calc_start]:
        # if year changed -> settle tax for the previous year (if enabled)
        if params.apply_annual_tax and date.year != current_tax_year:
            gain = capital - capital_at_year_start
            if gain > 0:
                tax = gain * tax_rate
                capital -= tax
                total_taxes_paid += tax
                # add a results row for the tax settlement
                results.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Previous ETF': ','.join(prev_set) if prev_set else '',
                    'Selected ETF': '',
                    'Transaction': 'Tax',
                    'Monthly Return': '',
                    'Cumulative Capital': capital
                })
            # reset base for the new year
            capital_at_year_start = capital
            current_tax_year = date.year

        current = momentum.loc[date]
        if current.isnull().all():
            continue
        if params.risk_filter_ticker not in current.index or params.cash_proxy_ticker not in current.index:
            continue

        if params.use_trend_filter and params.risk_filter_ticker in monthly.columns:
            spy_ma = monthly[params.risk_filter_ticker].rolling(params.trend_filter_window).mean()
            is_risk_on = (monthly.loc[date, params.risk_filter_ticker] > spy_ma.loc[date]) if (date in spy_ma.index and pd.notna(spy_ma.loc[date])) else False
        else:
            is_risk_on = pd.notna(current[params.risk_filter_ticker]) and pd.notna(current[params.cash_proxy_ticker]) and current[params.risk_filter_ticker] > current[params.cash_proxy_ticker]

        pool = equity if is_risk_on else bonds
        candidates = [c for c in pool if c in current.index]
        if len(candidates) == 0:
            continue

        n = min(params.top_n, len(candidates))
        top_sel = current[candidates].nlargest(n).index.tolist()

        rets = monthly_ret.loc[date, top_sel]
        if rets.isnull().any():
            continue
        mret = float(rets.mean())

        traded = (prev_set is None) or (set(top_sel) != prev_set)

        if params.with_fee and traded:
            prev_currencies = {ticker_currency.get(t, 'USD') for t in (prev_set or [])}
            new_currencies = {ticker_currency.get(t, 'USD') for t in top_sel}
            if prev_set is not None and prev_currencies != new_currencies:
                capital *= (1 - params.fee_rate)

        capital *= (1 + mret)

        results.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Previous ETF': ','.join(prev_set) if prev_set else '',
            'Selected ETF': ','.join(top_sel),
            'Transaction': 'Yes' if traded else 'No',
            'Monthly Return': mret,
            'Cumulative Capital': capital
        })

        prev_set = set(top_sel)

    # after the loop: settle the final year (if needed)
    if params.apply_annual_tax:
        gain = capital - capital_at_year_start
        if gain > 0:
            tax = gain * tax_rate
            capital -= tax
            total_taxes_paid += tax
            # use the last momentum date as the settlement date
            last_date = momentum.index[momentum.index >= calc_start][-1]
            results.append({
                'Date': last_date.strftime('%Y-%m-%d'),
                'Previous ETF': ','.join(prev_set) if prev_set else '',
                'Selected ETF': '',
                'Transaction': 'Tax',
                'Monthly Return': '',
                'Cumulative Capital': capital
            })

    results_df = pd.DataFrame(results).sort_values('Date').reset_index(drop=True)

    years = (monthly.index[-1] - calc_start).days / 365.25 if len(monthly.index) > 1 else 0
    cagr = capital ** (1 / years) - 1 if years > 0 else np.nan

    if params.verbose:
        print(f"\nFinal result: {capital:.4f}x initial capital ({(capital - 1):.2%})")
        print(f"Compound Annual Growth Rate (CAGR): {cagr:.2%}")
        if params.apply_annual_tax:
            print(f"Total taxes paid: {total_taxes_paid:.6f}")

    if not results_df.empty:
        metrics = risk_metrics(results_df.set_index(pd.to_datetime(results_df['Date']))['Cumulative Capital'].astype(float), params.risk_free_rate)
        if params.verbose:
            print("--- Risk Metrics ---")
            print(f"Start value: {metrics['start_value']:.4f}, End value: {metrics['end_value']:.4f}")
            print(f"CAGR: {metrics['cagr']:.2%}, Volatility: {metrics['volatility']:.2%}, Sharpe: {metrics['sharpe']:.2f}, MaxDD: {metrics['max_drawdown']:.2%}")

    name_map = {v: k for k, v in tickers.items()}
    results_df['Selected ETF'] = results_df['Selected ETF'].map(lambda s: ','.join([name_map.get(x, x) for x in s.split(',')]) if s else '')
    results_df['Previous ETF'] = results_df['Previous ETF'].map(lambda s: ','.join([name_map.get(x, x) for x in s.split(',')]) if s else '')
    # format returns column (for regular rows)
    def fmt_ret(x):
        try:
            return f"{x:.2%}"
        except:
            return ""
    results_df['Monthly Return'] = results_df['Monthly Return'].map(fmt_ret)

    return results_df, monthly, name_map, calc_start


# --- example execution ---
if __name__ == "__main__":
    params = StrategyParams(
        start_date="2019-10-31",      # date to start data
        top_n=1,
        momentum_periods=(3, 6, 12),
        use_trend_filter=True,
        trend_filter_window=10,
        with_fee=True,
        fee_rate=0.01,
        apply_annual_tax=False,
        tax_rate=0.19,
        verbose=True
    )

    results, monthly_prices, names, start = run_gem_strategy(params)
    if not results.empty:
        print("--- GEM Momentum Strategy Results (with optional annual tax) ---")
        print(results.to_string(index=False))
        plot_vs_benchmarks(results, monthly_prices, names, start)

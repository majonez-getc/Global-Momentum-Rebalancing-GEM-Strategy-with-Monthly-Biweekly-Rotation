from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any
import re
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


@dataclass
class StrategyParams:
    # ---- uniwersalne ustawienia danych ----
    start_date: str = "2010-10-31"        # data, od której pobieramy historyczne ceny (YYYY-MM-DD)
    end_date: Optional[str] = None        # data końcowa (None = do "dziś")
    resample_rule: str = "2W"             # reguła resamplowania cen do okresów ("2W" = co 2 tygodnie).

    # ---- uniwersalne parametry strategii ----
    top_n: int = 2                         # ile ETF-ów wybieramy na każde rebalansowanie (Top-N)
    momentum_periods: Tuple[int, ...] = (3, 6, 12)  # okresy momentum w miesiącach (interpretowane jako miesiące)

    # ---- prowizje i koszty ----
    with_fee: bool = True                  # czy stosujemy opłaty/transakcje walutowe
    fee_rate: float = 0.01                 # jednorazowa stawka prowizji używana przy zmianie waluty/składu

    # ---- podatek Belki (roczne rozliczenie) ----
    apply_annual_tax: bool = True          # czy stosować podatek Belki rocznie
    tax_rate: float = 0.19                 # stawka podatku (Belka)

    # ---- filtr trendu (opcjonalny) ----
    use_trend_filter: bool = True          # jeżeli True -> używamy prostego filtra trendu na SPY
    trend_filter_window: int = 10          # okno SMA (w miesiącach) używane jako filtr trendu (przeliczane do okresów)
    risk_filter_ticker: str = "SPY"        # ticker użyty jako risk-on filter
    cash_proxy_ticker: str = "BIL"         # ticker użyty jako „cash proxy" w porównaniach

    # ---- tickery i waluty ----
    tickers: Dict[str, str] = field(default_factory=lambda: {
        'Nasdaq 100': 'QQQ',
        'S&P 500': 'SPY',
        'Rynki rozwinięte': 'VEA',
        'Rynki wschodzące': 'VWO',
        'Obligacje długoterminowe': 'TLT',
        'Obligacje krótkoterminowe': 'BIL'
    })
    ticker_currency_overrides: Dict[str, str] = field(default_factory=lambda: {
        # mapowanie ticker -> waluta (domyślnie USD, tu podajemy wyjątki)
        'VEA': 'EUR'
    })

    # ---- analiza/wyjście ----
    risk_free_rate: float = 0.03           # stopa wolna od ryzyka używana w wskaźniku Sharpe
    calc_start: Optional[str] = None       # data od której liczymy sim (jeśli None -> start_date + 12 mies.)
    verbose: bool = True                   # jeśli True -> drukuj rezultaty i miary


# ----- Funkcje pomocnicze -----

def risk_metrics(cap_series: pd.Series, risk_free_rate: float = 0.03) -> Dict[str, Any]:
    """
    Oblicza podstawowe miary ryzyka/zwrotu dla skumulowanego kapitału.
    Zwraca słownik z wartościami (CAGR, vol, sharpe, max_dd).
    """
    cap_series = cap_series.sort_index()
    r = cap_series.pct_change().dropna()
    years = (cap_series.index[-1] - cap_series.index[0]).days / 365.25
    start_val = float(cap_series.iloc[0])
    end_val = float(cap_series.iloc[-1])
    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 and start_val > 0 else np.nan
    vol = r.std() * np.sqrt(12)  # annualizowane z miesięcznej stopy — zgodnie z oryginałem
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


def plot_vs_benchmarks(results_df: pd.DataFrame, periodic_prices: pd.DataFrame, name_map: Dict[str, str], start_date: pd.Timestamp):
    """Wykres porównawczy skumulowanego kapitału strategii vs znormalizowane benchmarki."""
    if results_df.empty:
        print("Brak danych do wykresu.")
        return

    results_df['Data'] = pd.to_datetime(results_df['Data'])
    strat = results_df.set_index('Data')['Kapitał (skumulowany)'].astype(float)

    bench = periodic_prices.loc[periodic_prices.index >= pd.to_datetime(start_date)].bfill()
    bench = bench.divide(bench.iloc[0])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strat.index, strat.values, label='GEM (modified, 2W rebalance)', linewidth=2)
    for col in bench.columns:
        ax.plot(bench.index, bench[col].values, linestyle='--', linewidth=1, label=name_map.get(col, col))

    ax.set(title='GEM vs ETF (normalized)', ylabel='Value (normalized)', xlabel='Date')
    ax.legend(fontsize='small', ncol=2)
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


# ----- Główna funkcja strategii -----

def run_gem_strategy(params: StrategyParams) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], pd.Timestamp]:
    """
    Uruchamia strategię GEM z parametryzacją podaną w StrategyParams.

    Zwraca: (results_df, periodic_prices, name_map, calc_start_timestamp)
    """
    # --- przygotowanie tickers i walut ---
    tickers = dict(params.tickers)  # nazwa -> ticker
    if params.ticker_currency_overrides:
        ticker_currency = {t: 'USD' for t in tickers.values()}
        ticker_currency.update(params.ticker_currency_overrides)
    else:
        ticker_currency = {t: 'USD' for t in tickers.values()}

    # proste grupowanie equity vs bonds (można rozszerzyć heurystykę)
    equity = ["QQQ", "SPY", "VEA", "VWO"]
    bonds = ["TLT", "BIL"]

    # --- data pobrania ---
    start_date = params.start_date
    end_date = params.end_date or pd.to_datetime('today').strftime('%Y-%m-%d')

    # pobierz dane (close, auto_adjust=True zamapuje split/dividend)
    data = yf.download(list(set(tickers.values())), start=start_date, end=end_date,
                       progress=False, auto_adjust=True)
    if data.empty:
        if params.verbose:
            print("Brak danych z yfinance dla zadanych tickerów / dat.")
        return pd.DataFrame(), pd.DataFrame(), {}, pd.Timestamp(start_date)

    price_df = data['Close'] if isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns.levels[0] else data

    # re samplowanie do okresów (np. '2W' -> co 2 tygodnie)
    periodic = price_df.resample(params.resample_rule).last()
    # jeżeli ostatni punkt jest w przyszłości -> obetnij
    if periodic.index[-1] > pd.Timestamp.today().normalize():
        periodic = periodic.iloc[:-1]
    periodic_ret = periodic.pct_change()

    # --- Przygotuj konwersję "miesiące -> liczba okresów resamplowania" ---
    # domyślnie użytkownik podaje momentum_periods w miesiącach.
    # Przybliżenie: 1 miesiąc ≈ 4.34524 tygodnia.
    def months_to_periods(months: int, resample_rule: str) -> int:
        if 'W' in resample_rule:
            m = re.match(r'(\d*)W', resample_rule)
            weeks_per_period = int(m.group(1)) if (m and m.group(1)) else 1
            periods = int(round(months * 4.34524 / weeks_per_period))
            return max(1, periods)
        elif resample_rule.upper().endswith('M') or resample_rule.upper() in ('ME', 'M'):
            return max(1, months)
        else:
            return max(1, months)

    # --- momentum: obliczamy dla każdej z zadanych okresów i bierzemy średnią (kompozyt) ---
    mom_list = []
    for p in params.momentum_periods:
        shift_periods = months_to_periods(p, params.resample_rule)
        mom_list.append(periodic / periodic.shift(shift_periods) - 1)
    momentum = pd.concat(mom_list, axis=1).groupby(level=0, axis=1).mean()

    # --- trend filter window: przelicz okno SMA (podane w miesiącach) na liczbę okresów ---
    if 'W' in params.resample_rule:
        m = re.match(r'(\d*)W', params.resample_rule)
        weeks_per_period = int(m.group(1)) if (m and m.group(1)) else 1
        trend_window_periods = max(1, int(round(params.trend_filter_window * 4.34524 / weeks_per_period)))
    else:
        trend_window_periods = max(1, params.trend_filter_window)

    # --- ustawienia kalkulacji: kiedy zaczynamy symulację ---
    if params.calc_start:
        calc_start = pd.to_datetime(params.calc_start)
    else:
        calc_start = pd.to_datetime(start_date) + pd.DateOffset(months=12)

    # --- przygotuj wyniki i pętlę symulacji ---
    results: List[Dict[str, Any]] = [{
        'Data': calc_start.strftime('%Y-%m-%d'),
        'Poprzedni ETF': '',
        'Wybrany ETF': '',
        'Transakcja': 'Nie',
        'Miesięczny zwrot': 0.0,
        'Kapitał (skumulowany)': 1.0
    }]

    capital = 1.0
    prev_set: Optional[set] = None

    # --- zmienne do rozliczenia podatku Belki ---
    tax_rate = params.tax_rate if params.apply_annual_tax else 0.0
    capital_at_year_start = capital
    # ustaw na rok, w którym zaczynamy (calc_start może być w środku roku)
    current_tax_year = calc_start.year
    total_taxes_paid = 0.0

    for date in momentum.index[momentum.index >= calc_start]:
        # jeśli zmiana roku -> rozliczamy podatek za poprzedni rok (jeżeli włączone)
        if params.apply_annual_tax and date.year != current_tax_year:
            gain = capital - capital_at_year_start
            if gain > 0:
                tax = gain * tax_rate
                capital -= tax
                total_taxes_paid += tax
                # wpis wynikowy rozliczenia podatku
                results.append({
                    'Data': date.strftime('%Y-%m-%d'),
                    'Poprzedni ETF': ','.join(prev_set) if prev_set else '',
                    'Wybrany ETF': '',
                    'Transakcja': 'Podatek',
                    'Miesięczny zwrot': None,
                    'Kapitał (skumulowany)': capital
                })
            # reset bazy dla nowego roku
            capital_at_year_start = capital
            current_tax_year = date.year

        current = momentum.loc[date]
        if current.isnull().all():
            continue
        if params.risk_filter_ticker not in current.index or params.cash_proxy_ticker not in current.index:
            continue

        # filtr trendu (SMA na SPY)
        if params.use_trend_filter and params.risk_filter_ticker in periodic.columns:
            spy_ma = periodic[params.risk_filter_ticker].rolling(trend_window_periods).mean()
            is_risk_on = (periodic.loc[date, params.risk_filter_ticker] > spy_ma.loc[date]) if (date in spy_ma.index and pd.notna(spy_ma.loc[date])) else False
        else:
            is_risk_on = pd.notna(current[params.risk_filter_ticker]) and pd.notna(current[params.cash_proxy_ticker]) and current[params.risk_filter_ticker] > current[params.cash_proxy_ticker]

        pool = equity if is_risk_on else bonds
        candidates = [c for c in pool if c in current.index]
        if len(candidates) == 0:
            continue

        n = min(params.top_n, len(candidates))
        top_sel = current[candidates].nlargest(n).index.tolist()

        # zwroty z okresu (dla wybranego okresu resamplowania)
        rets = periodic_ret.loc[date, top_sel]
        if rets.isnull().any():
            continue
        period_ret = float(rets.mean())

        traded = (prev_set is None) or (set(top_sel) != prev_set)

        # prosty model kosztów: jeżeli następuje zmiana waluty w portfelu -> naliczamy fee
        if params.with_fee and traded:
            prev_currencies = {ticker_currency.get(t, 'USD') for t in (prev_set or [])}
            new_currencies = {ticker_currency.get(t, 'USD') for t in top_sel}
            if prev_set is not None and prev_currencies != new_currencies:
                capital *= (1 - params.fee_rate)

        capital *= (1 + period_ret)

        results.append({
            'Data': date.strftime('%Y-%m-%d'),
            'Poprzedni ETF': ','.join(prev_set) if prev_set else '',
            'Wybrany ETF': ','.join(top_sel),
            'Transakcja': 'Tak' if traded else 'Nie',
            'Miesięczny zwrot': period_ret,
            'Kapitał (skumulowany)': capital
        })

        prev_set = set(top_sel)

    # po pętli: rozlicz ostatni rok (jeśli trzeba)
    if params.apply_annual_tax:
        gain = capital - capital_at_year_start
        if gain > 0:
            tax = gain * tax_rate
            capital -= tax
            total_taxes_paid += tax
            # używamy ostatniej daty momentum jako daty rozliczenia
            last_date = momentum.index[momentum.index >= calc_start][-1]
            results.append({
                'Data': last_date.strftime('%Y-%m-%d'),
                'Poprzedni ETF': ','.join(prev_set) if prev_set else '',
                'Wybrany ETF': '',
                'Transakcja': 'Podatek',
                'Miesięczny zwrot': None,
                'Kapitał (skumulowany)': capital
            })

    results_df = pd.DataFrame(results).sort_values('Data').reset_index(drop=True)

    # statystyki końcowe
    years = (periodic.index[-1] - calc_start).days / 365.25 if len(periodic.index) > 1 else 0
    cagr = capital ** (1 / years) - 1 if years > 0 else np.nan

    if params.verbose:
        print(f"\nWynik końcowy: {capital:.4f}x kapitału początkowego ({(capital - 1):.2%})")
        print(f"Średnioroczna stopa zwrotu (CAGR): {cagr:.2%}")
        if params.apply_annual_tax:
            print(f"Suma zapłaconych podatków (Belka): {total_taxes_paid:.6f}")

    if not results_df.empty:
        metrics = risk_metrics(results_df.set_index(pd.to_datetime(results_df['Data']))['Kapitał (skumulowany)'].astype(float), params.risk_free_rate)
        if params.verbose:
            print("--- Miary Ryzyka ---")
            print(f"Start value: {metrics['start_value']:.4f}, End value: {metrics['end_value']:.4f}")
            print(f"CAGR: {metrics['cagr']:.2%}, Volatility: {metrics['volatility']:.2%}, Sharpe: {metrics['sharpe']:.2f}, MaxDD: {metrics['max_drawdown']:.2%}")

    name_map = {v: k for k, v in tickers.items()}
    # czytelne nazwy
    results_df['Wybrany ETF'] = results_df['Wybrany ETF'].map(lambda s: ','.join([name_map.get(x, x) for x in s.split(',')]) if s else '')
    results_df['Poprzedni ETF'] = results_df['Poprzedni ETF'].map(lambda s: ','.join([name_map.get(x, x) for x in s.split(',')]) if s else '')

    # formatowanie kolumny zwrotów (dla zwykłych wierszy), podatkowe wiersze mają None -> zostaną puste
    def fmt_ret(x):
        try:
            return f"{x:.2%}"
        except:
            return ""
    results_df['Miesięczny zwrot'] = results_df['Miesięczny zwrot'].map(fmt_ret)

    return results_df, periodic, name_map, calc_start


# --- przykładowe wywołanie ---
if __name__ == "__main__":
    params = StrategyParams(
        start_date="2019-10-31",      # data, od której pobieramy dane
        top_n=1,                      # wybieramy Top-1
        momentum_periods=(3, 6, 12),  # kompozyt momentum 3/6/12 miesięcy-kod przeliczy na okresy 2W
        use_trend_filter=True,        # użyjemy SMA(SPY) jako filtra
        trend_filter_window=10,       # SMA(10 miesięcy) -> przeliczone na okresy 2 W
        with_fee=True,
        fee_rate=0.01,
        apply_annual_tax=False,
        tax_rate=0.19,
        verbose=True
    )

    wyniki, periodic_prices, names, start = run_gem_strategy(params)
    if not wyniki.empty:
        print("--- Wyniki Strategii Momentum (GEM) - rebalans co 2 tygodnie, z podatkiem Belki ---")
        print(wyniki.to_string(index=False))
        plot_vs_benchmarks(wyniki, periodic_prices, names, start)

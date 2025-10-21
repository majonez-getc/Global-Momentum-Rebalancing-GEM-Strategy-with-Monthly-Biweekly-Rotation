"""
Portfolio / FIRE Simulator (parameterized)

This script simulates monthly contributions to a portfolio, applies compound
returns and inflation, and estimates when a FIRE target (25x annual expenses,
inflation-adjusted) is reached.

Configuration is done via the `PortfolioParams` dataclass. Use `contribution_years`
to set for how many years you will contribute (None = contribute for the whole
simulation horizon).
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


@dataclass
class PortfolioParams:
    # core scenario
    annual_return: float = 0.20           # expected annual return (e.g. 0.20 for 20%)
    monthly_contribution: float = 4000.0  # monthly contribution amount
    years: int = 30                       # simulation horizon in years
    annual_expenses: float = 72000.0      # annual spending used for FIRE target
    inflation: float = 0.03               # annual inflation (e.g. 0.03 = 3%)
    contribution_years: Optional[float] = None  # how many years to contribute (None = always)
    show_plots: bool = True               # whether to display matplotlib plots
    verbose: bool = True                  # whether to print textual results


def calculate_portfolio(params: PortfolioParams) -> Tuple[
        List[float], List[float], List[float], List[float], int, Optional[int]]:
    """
    Simulate monthly contributions and compound returns using values from params.

    Returns:
      (portfolio_values,
       cumulative_contributions,
       annual_gains,
       annual_gains_pct,
       months_total,
       fire_month_index)
    """
    months_total = params.years * 12
    # convert contribution_years to months (None means contribute for whole horizon)
    if params.contribution_years is None:
        months_with_contributions = None
    else:
        months_with_contributions = int(round(params.contribution_years * 12))

    monthly_return = (1 + params.annual_return) ** (1 / 12) - 1
    monthly_inflation = (1 + params.inflation) ** (1 / 12) - 1

    portfolio_values: List[float] = [0.0]
    cumulative_contributions: List[float] = [0.0]
    balance = 0.0
    annual_gains: List[float] = []
    annual_gains_pct: List[float] = []
    fire_month_index: Optional[int] = None

    fire_target = 25 * params.annual_expenses
    fire_target_real = fire_target

    for month in range(1, months_total + 1):
        # add contribution if still in contribution period (None = always contribute)
        if months_with_contributions is None or month <= months_with_contributions:
            balance += params.monthly_contribution

        # apply monthly return
        balance *= (1 + monthly_return)
        portfolio_values.append(balance)

        # update cumulative contributions
        if months_with_contributions is None or month <= months_with_contributions:
            cumulative_contributions.append(cumulative_contributions[-1] + params.monthly_contribution)
        else:
            cumulative_contributions.append(cumulative_contributions[-1])

        # update inflation-adjusted FIRE target
        fire_target_real *= (1 + monthly_inflation)

        # check FIRE achievement
        if balance >= fire_target_real and fire_month_index is None:
            fire_month_index = month

        # compute yearly results every 12 months
        if month % 12 == 0:
            total_contrib_at_year = cumulative_contributions[month]
            yearly_gain = balance - total_contrib_at_year
            if total_contrib_at_year != 0:
                yearly_gain_pct = (yearly_gain / total_contrib_at_year) * 100
            else:
                yearly_gain_pct = 0.0

            annual_gains.append(yearly_gain)
            annual_gains_pct.append(yearly_gain_pct)

    return (portfolio_values,
            cumulative_contributions,
            annual_gains,
            annual_gains_pct,
            months_total,
            fire_month_index)


def format_number(number: float) -> str:
    """Format number with thousands separator and two decimals."""
    return f"{number:,.2f}"


def display_results(params: PortfolioParams,
                    annual_gains: List[float],
                    annual_gains_pct: List[float],
                    cumulative_contributions: List[float],
                    portfolio_values: List[float],
                    fire_month_index: Optional[int]) -> None:
    """Print a yearly table and summary of the simulation."""
    final_portfolio = portfolio_values[-1]
    final_contrib = cumulative_contributions[-1]
    total_gain = final_portfolio - final_contrib

    # Header
    print(f"{'Year':<6}{'Cumulative Contributions':<28}{'Annual Gain':<18}{'Gain (%)':<12}")
    for i in range(1, len(annual_gains) + 1):
        contrib_at_year = cumulative_contributions[i * 12]
        gain = annual_gains[i - 1]
        gain_pct = annual_gains_pct[i - 1]
        print(f"{i:<6}{format_number(contrib_at_year):<28}{format_number(gain):<18}{gain_pct:<12.2f}")

    print("\nSummary:")
    print(f"Final portfolio value: {format_number(final_portfolio)}")
    print(f"Total contributions: {format_number(final_contrib)}")
    print(f"Total investment gain: {format_number(total_gain)}")

    if fire_month_index:
        fire_year = fire_month_index // 12
        fire_month = fire_month_index % 12
        if fire_month == 0:
            fire_month = 12
            fire_year -= 1
        print(f"You reach FIRE in month {fire_month} of year {fire_year} from now.")
    else:
        print("FIRE target was not reached in the simulated period.")


def create_plots(portfolio_values: List[float],
                 cumulative_contributions: List[float],
                 annual_gains: List[float],
                 params: PortfolioParams) -> None:
    """Create two plots: portfolio vs contributions, and annual gains."""
    if not params.show_plots:
        return

    plt.figure(figsize=(12, 4))

    # Plot 1: portfolio value and cumulative contributions
    plt.subplot(1, 2, 1)
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.plot(cumulative_contributions, label='Cumulative Contributions', linestyle='--')
    plt.xlabel('Months')
    plt.ylabel('Value')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)

    # Plot 2: annual gains
    plt.subplot(1, 2, 2)
    plt.bar(range(1, len(annual_gains) + 1), annual_gains, alpha=0.7)
    plt.xlabel('Year')
    plt.ylabel('Annual Gain')
    plt.title('Annual Gains')

    plt.tight_layout()
    plt.show()


def main(params: Optional[PortfolioParams] = None) -> None:
    # Use default params if none provided
    if params is None:
        params = PortfolioParams(
            annual_return=0.20,
            monthly_contribution=4000,
            years=30,
            annual_expenses=72000,
            inflation=0.03,
            contribution_years=None,  # contribute for 10 years (set to None to always contribute)
            show_plots=True,
            verbose=True
        )

    (portfolio_values,
     cumulative_contributions,
     annual_gains,
     annual_gains_pct,
     months_total,
     fire_month_index) = calculate_portfolio(params)

    if params.verbose:
        display_results(params, annual_gains, annual_gains_pct, cumulative_contributions, portfolio_values, fire_month_index)

    create_plots(portfolio_values, cumulative_contributions, annual_gains, params)


if __name__ == "__main__":
    main()

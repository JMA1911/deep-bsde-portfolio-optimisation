import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
from tqdm import trange
import os
from scipy.stats import norm

# Function to obtain market data   
def download_arithmetic_returns(tickers, start_date, end_date):
    """Downloads asset price data and computes arithmetic returns."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    adj_close = data.xs('Close', level='Price', axis=1)
    arithmetic_returns = adj_close.pct_change().dropna()
    return arithmetic_returns, adj_close

tickers = ['AAPL', 'JPM', 'XOM']
start_date = '2015-01-01'
end_date = '2025-01-01'

# Download and process data
arithmetic_returns, adj_close = download_arithmetic_returns(tickers, start_date, end_date)

def compute_performance_table(
    dates_bt,
    wealth_bt,
    portfolio_path,
    costs_bt,
    weights_path,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    dt_backtest=1/12,
    output_folder=None,
    tickers=None,
    latex_filename=None
):
    """
    Compute performance metrics and output a LaTeX table (vertical format).
    Now using arithmetic returns for performance statistics.
    """
    if latex_filename is None:
        if tickers is None:
            file_stem = "eqw_perf_table"
        else:
            file_stem = "_".join(tickers) + "_eqw_perf_table"
        latex_filename = file_stem + ".tex"

    if output_folder is not None:
        latex_filename = os.path.join(output_folder, latex_filename)

    T = len(wealth_bt) - 1
    if T <= 1:
        raise ValueError("Not enough time steps to compute statistics.")

    freq_per_year = 1 / dt_backtest

    # Use arithmetic returns
    returns = wealth_bt[1:] / wealth_bt[:-1] - 1
    cagr = (wealth_bt[-1] / initial_wealth)**(freq_per_year / T) - 1
    ann_vol = np.std(returns) * np.sqrt(freq_per_year)
    ann_excess_return = cagr - annual_rf_rate
    sharpe = ann_excess_return / ann_vol if ann_vol > 0 else np.nan

    # Probabilistic Sharpe Ratio
    benchmark_sharpe = 0.0
    if T > 1 and np.isfinite(sharpe):
        se_sharpe = np.sqrt((1 + sharpe**2 / 2) / (T - 1))
        psr = norm.cdf((sharpe - benchmark_sharpe) / se_sharpe)
    else:
        psr = np.nan

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(freq_per_year) if len(downside_returns) > 0 else np.nan
    sortino = ann_excess_return / downside_std if downside_std > 0 else np.nan

    # Max drawdown
    running_max = np.maximum.accumulate(wealth_bt)
    drawdown = (running_max - wealth_bt) / running_max
    max_dd = np.max(drawdown)

    # Average drawdown
    dd_periods = []
    current_dd = []
    for dd in drawdown:
        if dd > 0:
            current_dd.append(dd)
        elif current_dd:
            dd_periods.append(np.mean(current_dd))
            current_dd = []
    if current_dd:
        dd_periods.append(np.mean(current_dd))
    avg_drawdown = np.mean(dd_periods) if dd_periods else 0.0

    # Average recovery time (in calendar days)
    recovery_times = []
    peak_idx = 0
    for t in range(1, len(wealth_bt)):
        if wealth_bt[t] >= wealth_bt[peak_idx]:
            if t > peak_idx:
                recovery_times.append(t - peak_idx)
            peak_idx = t
    avg_recovery_days = np.mean(recovery_times) * (252 * dt_backtest) if recovery_times else np.nan

    # Calmar Ratio
    calmar = cagr / max_dd if max_dd > 0 else np.nan

    # Turnover metrics (based on dollar allocations)
    delta_u = np.diff(portfolio_path, axis=0)
    turnover_per_step = np.sum(np.abs(delta_u), axis=1) / wealth_bt[:-1]
    avg_turnover_step = np.mean(turnover_per_step)
    annual_turnover = avg_turnover_step * freq_per_year

    # Extract risky weights (drop cash weight)
    weights_risky = weights_path[:, :-1]
    std_weights = np.std(weights_risky, axis=0)
    std_weights_mean = np.mean(std_weights) * 100
    delta_w = np.diff(weights_risky, axis=0)
    abs_changes = np.mean(np.sum(np.abs(delta_w), axis=1)) * 100

    # No-trade fraction
    tolerance = 1e-6
    delta_theta = np.diff(portfolio_path, axis=0)
    unchanged_days = np.sum(np.all(np.abs(delta_theta) < tolerance, axis=1))
    no_trade_fraction = unchanged_days / (portfolio_path.shape[0] - 1)

    # Total transaction costs
    total_costs = np.sum(costs_bt)

    # VaR and Expected Shortfall (using arithmetic returns)
    confidence_level = 0.95
    if len(returns) > 0:
        var_95 = -np.percentile(returns, 100 * (1 - confidence_level))
        es_95 = -np.mean(returns[returns <= -var_95]) if np.any(returns <= -var_95) else np.nan
    else:
        var_95 = es_95 = np.nan

    var_95_pct = var_95 * 100 if np.isfinite(var_95) else np.nan
    es_95_pct = es_95 * 100 if np.isfinite(es_95) else np.nan

    # Average leverage
    risky_dollar_exposure = np.sum(np.abs(portfolio_path), axis=1)
    avg_leverage = np.mean(risky_dollar_exposure / wealth_bt)

    # Max dollar position size
    max_dollar_pos = np.max(np.abs(portfolio_path))

    # Dollar PnL per turnover
    total_pnl = wealth_bt[-1] - initial_wealth
    total_dollars_traded = np.sum(np.abs(delta_u))
    pnl_per_turnover = total_pnl / total_dollars_traded if total_dollars_traded > 0 else np.nan

    # Create vertical metrics table
    metrics_names = [
        "Terminal Wealth (\\pounds)",
        "CAGR (\\%)",
        "Annual Volatility (\\%)",
        "Sharpe Ratio",
        "Probabilistic Sharpe Ratio (\\%)",
        "Sortino Ratio",
        "Calmar Ratio",
        "Maximum Drawdown (\\%)",
        "Average Drawdown (\\%)",
        "Average Recovery Time (days)",
        "VaR 95\\% (\\%)",
        "Expected Shortfall 95\\% (\\%)",
        "Average Leverage",
        "Maximum Position Size (\\pounds)",
        "Average Turnover per Step (\\%)",
        "Annual Turnover (\\%)",
        "Standard Deviation of Portfolio Weights (\\%)",
        "Average Absolute Change in Portfolio Weights (\\%)",
        "No-Trade Fraction (\\%)",
        "Profit per Turnover (\\pounds)",
        "Total Transaction Costs (\\pounds)"
    ]

    metrics_values = [
        f"\\pounds{wealth_bt[-1]:,.2f}",
        f"{cagr * 100:.2f}\\%",
        f"{ann_vol * 100:.2f}\\%",
        f"{sharpe:.2f}",
        f"{psr * 100:.2f}\\%" if np.isfinite(psr) else "N/A",
        f"{sortino:.2f}" if np.isfinite(sortino) else "N/A",
        f"{calmar:.2f}" if np.isfinite(calmar) else "N/A",
        f"{max_dd * 100:.2f}\\%",
        f"{avg_drawdown * 100:.2f}\\%",
        f"{avg_recovery_days:.2f}" if np.isfinite(avg_recovery_days) else "N/A",
        f"{var_95_pct:.2f}\\%" if np.isfinite(var_95_pct) else "N/A",
        f"{es_95_pct:.2f}\\%" if np.isfinite(es_95_pct) else "N/A",
        f"{avg_leverage:.2f}" if np.isfinite(avg_leverage) else "N/A",
        f"\\pounds{max_dollar_pos:,.2f}" if np.isfinite(max_dollar_pos) else "N/A",
        f"{avg_turnover_step * 100:.2f}\\%",
        f"{annual_turnover * 100:.2f}\\%",
        f"{std_weights_mean:.2f}\\%" if np.isfinite(std_weights_mean) else "N/A",
        f"{abs_changes:.2f}\\%" if np.isfinite(abs_changes) else "N/A",
        f"{no_trade_fraction * 100:.2f}\\%" if np.isfinite(no_trade_fraction) else "N/A",
        f"\\pounds{pnl_per_turnover:,.2f}" if np.isfinite(pnl_per_turnover) else "N/A",
        f"\\pounds{total_costs:,.2f}"
    ]

    df_metrics = pd.DataFrame({
        "Metric": metrics_names,
        "Value": metrics_values
    })

    tabular_code = df_metrics.to_latex(
        index=False,
        escape=False,
        column_format="ll"
    )

    with open(latex_filename, "w") as f:
        f.write(tabular_code)

    print(f"LaTeX table saved to: {latex_filename}")

    # Metrics for bootstrap comparison
    hist_metrics_bootstrap = {
        "Terminal Wealth (\\pounds)": wealth_bt[-1],
        "CAGR (\\%)": cagr * 100,
        "Annual Volatility (\\%)": ann_vol * 100,
        "Sharpe Ratio": sharpe,
        "Probabilistic Sharpe Ratio (\\%)": psr * 100 if np.isfinite(psr) else np.nan,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Maximum Drawdown (\\%)": max_dd * 100,
        "Average Drawdown (\\%)": avg_drawdown * 100,
        "VaR 95\\% (\\%)": var_95_pct,
        "Expected Shortfall 95\\% (\\%)": es_95_pct,
        "Annual Turnover (\\%)": annual_turnover * 100,
        "Average Absolute Change in Portfolio Weights (\\%)": abs_changes,
        "Total Transaction Costs (\\pounds)": total_costs,
    }

    return df_metrics, hist_metrics_bootstrap

def plot_backtest_results(
    dates_bt,
    wealth_bt,
    portfolio_path,
    costs_bt,
    weights_path,
    output_dir="/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up",
    tickers=None
):
    """
    Generate and save all standard plots for the backtest results.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------
    # Ticker string for file prefixes
    # ---------------------------------------
    if tickers is None:
        ticker_prefix = "model"
    else:
        ticker_prefix = "_".join(tickers)

    def file_path(name):
        """
        Helper to generate filename with ticker prefix.
        """
        return os.path.join(output_dir, f"{ticker_prefix}_{name}.pdf")

    # -------------------------------
    # Global style tweaks
    # -------------------------------
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.6,
    })

    # Define colour palette
    single_color = "#1f77b4"  # dark blue for single-line plots

    # Define distinct colours for weights plot
    default_colors = [
        "#E66100",   # dark orange
        "#5E3C99",   # strong blue
        "#1B9E77",   # teal green
        "#E7298A",   # pink
        "#A6A600",   # olive
        "#1F78B4",   # sky blue
        "#333333",   # dark grey for risk-free
    ]

    # Compute discrete returns
    returns = wealth_bt[1:] / wealth_bt[:-1] - 1

    # Compute daily turnover series
    delta_u = np.diff(portfolio_path, axis=0)
    turnover_series = np.sum(np.abs(delta_u), axis=1) / wealth_bt[:-1]

    # ----------------------------------------
    # Plot 1: Wealth Over Time
    # ----------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(dates_bt, wealth_bt, color=single_color, linewidth=1.8)
    plt.xlabel("Date")
    plt.ylabel("Wealth (£)")
    plt.title("Wealth Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("eqw_wealth_over_time"))
    plt.close()

    # ----------------------------------------
    # Plot 2: Drawdown Over Time
    # ----------------------------------------
    running_max = np.maximum.accumulate(wealth_bt)
    drawdown = (running_max - wealth_bt) / running_max

    plt.figure(figsize=(8, 4))
    plt.plot(dates_bt, drawdown * 100, color=single_color, linewidth=1.8)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.title("Drawdown Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("eqw_drawdown_over_time"))
    plt.close()

    # ----------------------------------------
    # Plot 3: Histogram of Daily Returns
    # ----------------------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(returns * 100, bins=50, color=single_color, edgecolor='black')
    plt.xlabel("Daily Return (%)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Daily Returns")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("eqw_histogram_returns"))
    plt.close()

    # ----------------------------------------
    # Plot 4: Rolling Sharpe Ratio (Discrete)
    # ----------------------------------------
    window = 63

    ret_series = pd.Series(returns)

    # rolling daily mean and std
    roll_mean_daily = ret_series.rolling(window).mean()
    roll_std_daily = ret_series.rolling(window).std()

    # annualise
    roll_mean_ann = roll_mean_daily * 252
    roll_std_ann = roll_std_daily * np.sqrt(252)

    # subtract annual rf rate
    annual_rf_rate = 0.02
    roll_sharpe = (roll_mean_ann - annual_rf_rate) / roll_std_ann

    # match dates to rolling window
    dates_for_rolling = dates_bt[1:][window - 1 :]

    print("Rolling mean daily (typical):", roll_mean_daily.describe())
    print("Rolling std daily (typical):", roll_std_daily.describe())
    print("Rolling std annualized (typical):", roll_std_ann.describe())
    print("Rolling Sharpe annualized (typical):", roll_sharpe.describe())

    plt.figure(figsize=(8, 4))
    plt.plot(dates_for_rolling, roll_sharpe.iloc[window - 1 :], color=single_color, linewidth=1.5)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe Ratio")
    plt.title(f"Rolling Sharpe Ratio ({window}-day)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("eqw_rolling_sharpe"))
    plt.close()

    # ----------------------------------------
    # Plot 5: Line Plot of Portfolio Weights
    # ----------------------------------------
    weights_pct = weights_path * 100
    n_assets = weights_path.shape[1]

    if tickers is not None:
        labels = list(tickers) + ["Risk-free"]
    else:
        labels = [f"Asset {i+1}" for i in range(n_assets - 1)] + ["Risk-free"]

    plt.figure(figsize=(10, 5))
    for i in range(n_assets):
        color = default_colors[i % len(default_colors)]
        linestyle = "-" if i < n_assets - 1 else "--"
        plt.plot(
            dates_bt,
            weights_pct[:, i],
            label=labels[i],
            color=color,
            linestyle=linestyle,
            linewidth=1.5
        )
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.ylabel("Weight (%)")
    plt.xlabel("Date")
    plt.ylim(-300, 300)
    plt.legend(loc="upper left", frameon=True)
    plt.title("Portfolio Weights Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("eqw_weights_line_plot"))
    plt.close()

    # ----------------------------------------
    # Plot 6: Turnover Over Time
    # ----------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(dates_bt[1:], turnover_series * 100, color=single_color, linewidth=1.5)
    plt.xlabel("Date")
    plt.ylabel("Daily Turnover (%)")
    plt.title("Turnover Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("eqw_turnover_over_time"))
    plt.close()

    # ----------------------------------------
    # Plot 7: Cumulative Transaction Costs
    # ----------------------------------------
    cumulative_costs = np.cumsum(costs_bt)
    cumulative_costs_padded = np.insert(cumulative_costs, 0, 0.0)

    # Create daily cumulative costs array
    cumulative_costs_daily = np.zeros(len(dates_bt))

    # Compute rebalance indices
    n_cost_points = len(cumulative_costs_padded)
    rebalance_indices = np.linspace(
        0, len(dates_bt)-1, num=n_cost_points, dtype=int
    )

    # Fill daily array
    for i in range(len(rebalance_indices)):
        start = rebalance_indices[i]
        end = rebalance_indices[i+1] if i+1 < len(rebalance_indices) else len(dates_bt)
        cumulative_costs_daily[start:end] = cumulative_costs_padded[i]

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(dates_bt, cumulative_costs_daily, color=single_color, linewidth=1.8)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Costs (£)")
    plt.title("Cumulative Transaction Costs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("eqw_cumulative_transaction_costs"))
    plt.close()

    print(f"All plots saved in: {output_dir}")

def run_buy_and_hold_benchmark(
    adj_close,
    initial_wealth=100.0,
    risk_free_rate=0.025,
    equal_weight_riskfree=0.25,
):
    """
    Simulate a buy-and-hold strategy:
    - Invest equally in risky assets and risk-free initially.
    - No rebalancing thereafter.
    """

    dates = adj_close.index
    n_assets = adj_close.shape[1]

    equal_weight_risk = np.ones(n_assets) / n_assets
    risky_weights = (1 - equal_weight_riskfree) * equal_weight_risk

    benchmark_weights = np.append(risky_weights, equal_weight_riskfree)

    wealth_bt = [initial_wealth]
    risky_values_bt = []
    weights_bt = []
    cash_bt = []

    # Initial risky and cash allocations
    S_t = adj_close.iloc[0].values
    risky_value_t = initial_wealth * risky_weights
    cash_value_t = initial_wealth * equal_weight_riskfree

    # Save initial weights
    risky_weights_t = risky_value_t / initial_wealth
    cash_weight_t = cash_value_t / initial_wealth
    weights_full_t = np.append(risky_weights_t, cash_weight_t)
    weights_bt.append(weights_full_t)

    # Save initial risky and cash values
    risky_values_bt.append(risky_value_t)
    cash_bt.append(cash_value_t)

    for i in range(1, len(dates)):
        S_next = adj_close.iloc[i].values

        # Risky asset values evolve with price changes
        risky_value_tplus1 = risky_value_t * (S_next / S_t)

        # Cash grows at risk-free rate
        dt = 1 / 252
        cash_value_tplus1 = cash_value_t * (1 + risk_free_rate * dt)

        # Total wealth
        wealth_tplus1 = np.sum(risky_value_tplus1) + cash_value_tplus1

        # Compute current portfolio weights
        risky_weights_t = risky_value_tplus1 / wealth_tplus1
        cash_weight_t = cash_value_tplus1 / wealth_tplus1
        weights_full_t = np.append(risky_weights_t, cash_weight_t)

        # Save state
        wealth_bt.append(wealth_tplus1)
        risky_values_bt.append(risky_value_tplus1)
        cash_bt.append(cash_value_tplus1)
        weights_bt.append(weights_full_t)

        # Update for next step
        risky_value_t = risky_value_tplus1
        cash_value_t = cash_value_tplus1
        S_t = S_next

    wealth_bt = np.array(wealth_bt)
    risky_values_bt = np.array(risky_values_bt)
    cash_bt = np.array(cash_bt)
    weights_bt = np.array(weights_bt)

    # Reconstruct dollar path
    portfolio_path = np.hstack([
        risky_values_bt,
        cash_bt.reshape(-1, 1)
    ])

    # No transaction costs because we never rebalance
    costs_bt = np.zeros(len(dates))

    return dates, wealth_bt, portfolio_path, costs_bt, weights_bt

dates_bench, wealth_bench, portfolio_bench, costs_bench, weights_bench = run_buy_and_hold_benchmark(
    adj_close=adj_close,
    initial_wealth=100.0,
    risk_free_rate=0.025,
    equal_weight_riskfree=0.25,
)

output_folder = "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up"

metrics_bench_df, hist_metrics_bootstrap = compute_performance_table(
    dates_bench,
    wealth_bench,
    portfolio_bench,
    costs_bench,
    weights_bench,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    dt_backtest=1/252,
    output_folder=output_folder,
    tickers=tickers
)

plot_backtest_results(
    dates_bench,
    wealth_bench,
    portfolio_bench,
    costs_bench,
    weights_bench,
    output_dir="/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up",
    tickers=tickers
)

print(metrics_bench_df)

np.savez(
    "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up/eqw_results.npz",
    dates=np.array(dates_bench, dtype="datetime64[D]"),
    wealth=wealth_bench,
    portfolio=portfolio_bench,
    costs=costs_bench,
    weights=weights_bench,
)

def run_benchmark_on_synthetic_paths(
    synthetic_folder,
    tickers,
    n_trials,
    benchmark_func,
    benchmark_kwargs=None,
    save_results=True,
    benchmark_prefix="benchmark_path",
    start_date_bt="2015-01-01",
):
    """
    Run benchmark strategy on all synthetic paths and return
    consolidated arrays for later analysis.
    """
    
    wealth_list = []
    portfolio_list = []
    costs_list = []
    weights_list = []
    dates_common = None

    tickers_str = "_".join(tickers)

    for i in range(n_trials):
        path_file = os.path.join(
            synthetic_folder,
            f"{tickers_str}_rolling_bootstrap_path_{i}.npz"
        )

        # Load saved synthetic path
        data = np.load(path_file, allow_pickle=True)
        dates = pd.to_datetime(data["dates"])
        asset_names = list(data["tickers"])

        adj_close_synth_full = pd.DataFrame(
            data["prices"],
            index=dates,
            columns=asset_names,
        )
        cs_synth_full = pd.Series(data["cs"], index=dates)

        # Slice synthetic data to match backtest period
        adj_close_synth = adj_close_synth_full.loc[start_date_bt:]
        cs_synth = cs_synth_full.loc[start_date_bt:]

        if benchmark_kwargs is None:
            benchmark_kwargs = {}

        # Run benchmark on clipped synthetic data
        result = benchmark_func(
            adj_close=adj_close_synth,
            **benchmark_kwargs
        )

        dates_bench, wealth_bench, portfolio_bench, costs_bench, weights_bench = result

        if dates_common is None:
            dates_common = dates_bench
        else:
            assert all(dates_common == dates_bench), f"Dates mismatch in trial {i}"

        wealth_list.append(wealth_bench)
        portfolio_list.append(portfolio_bench)
        costs_list.append(costs_bench)
        weights_list.append(weights_bench)

        if save_results:
            np.savez(
                os.path.join(
                    synthetic_folder,
                    f"{tickers_str}_{benchmark_prefix}_{i}.npz"
                ),
                dates=dates_bench.values.astype("datetime64[D]"),
                wealth=wealth_bench,
                portfolio=portfolio_bench,
                costs=costs_bench,
                weights=weights_bench,
            )

    return {
        "dates": dates_common,
        "wealth_paths": np.stack(wealth_list, axis=0),
        "portfolio_paths": np.stack(portfolio_list, axis=0),
        "costs_paths": np.stack(costs_list, axis=0),
        "weights_paths": np.stack(weights_list, axis=0),
    }

summary = run_benchmark_on_synthetic_paths(
    synthetic_folder="/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up",
    tickers=tickers,
    n_trials=100,
    benchmark_func=run_buy_and_hold_benchmark,
    benchmark_kwargs=dict(
        initial_wealth=100.0,
        risk_free_rate=0.025,
        equal_weight_riskfree=0.25
    ),
    save_results=True,
    start_date_bt="2015-01-01",
)

dates = summary["dates"]
wealth_paths = summary["wealth_paths"]

def compute_bootstrap_comparison_table_from_results(
    hist_metrics: dict,
    bootstrap_results: list,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    dt_backtest=1/252,
    latex_filename=None,
    output_folder=None,
    tickers=None,
):
    """
    Compute a comparison table of historical vs bootstrap metrics,
    calculating metrics from bootstrap results directly using arithmetic returns.
    Now includes Max Drawdown, VaR, ES, and Avg Abs Change in Weights.
    """
    freq_per_year = 1 / dt_backtest

    metrics_arrays = {
        "Terminal Wealth (\\pounds)": [],
        "CAGR (\\%)": [],
        "Annual Volatility (\\%)": [],
        "Sharpe Ratio": [],
        "Sortino Ratio": [],
        "Calmar Ratio": [],
        "Maximum Drawdown (\\%)": [],
        "Average Drawdown (\\%)": [],
        "VaR 95\\% (\\%)": [],
        "Expected Shortfall 95\\% (\\%)": [],
        "Annual Turnover (\\%)": [],
        "Average Absolute Change in Portfolio Weights (\\%)": [],
        "Total Transaction Costs (\\pounds)": [],
    }

    for result in bootstrap_results:
        wealth_bt = result["wealth"]
        portfolio_path = result["theta"]
        costs_bt = result["costs"]
        weights_path = result["weights"]

        T = len(wealth_bt) - 1
        if T < 1:
            continue

        # Arithmetic returns
        returns = np.diff(wealth_bt) / wealth_bt[:-1]
        cagr = (wealth_bt[-1] / initial_wealth) ** (freq_per_year / T) - 1
        ann_vol = np.std(returns) * np.sqrt(freq_per_year)
        ann_excess_return = cagr - annual_rf_rate
        sharpe = ann_excess_return / ann_vol if ann_vol > 0 else np.nan

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(freq_per_year) if len(downside_returns) > 0 else np.nan
        sortino = ann_excess_return / downside_std if downside_std > 0 else np.nan

        # Drawdowns
        running_max = np.maximum.accumulate(wealth_bt)
        drawdown = (running_max - wealth_bt) / running_max
        max_dd = np.max(drawdown)

        dd_periods = []
        current_dd = []
        for dd in drawdown:
            if dd > 0:
                current_dd.append(dd)
            elif current_dd:
                dd_periods.append(np.mean(current_dd))
                current_dd = []
        if current_dd:
            dd_periods.append(np.mean(current_dd))
        avg_drawdown = np.mean(dd_periods) if dd_periods else 0.0

        # Calmar
        calmar = cagr / max_dd if max_dd > 0 else np.nan

        # Turnover
        delta_u = np.diff(portfolio_path, axis=0)
        turnover_per_step = np.sum(np.abs(delta_u), axis=1) / wealth_bt[:-1]
        avg_turnover_step = np.mean(turnover_per_step)
        annual_turnover = avg_turnover_step * freq_per_year

        # Average abs change in risky weights
        weights_risky = weights_path[:, :-1]
        delta_w = np.diff(weights_risky, axis=0)
        abs_changes = np.mean(np.sum(np.abs(delta_w), axis=1)) * 100

        # VaR & ES
        var_95 = -np.percentile(returns, 5)
        es_95 = -np.mean(returns[returns <= -var_95]) if np.any(returns <= -var_95) else np.nan

        # Costs
        total_costs = np.sum(costs_bt)

        # Save metrics
        metrics_arrays["Terminal Wealth (\\pounds)"].append(wealth_bt[-1])
        metrics_arrays["CAGR (\\%)"].append(cagr * 100)
        metrics_arrays["Annual Volatility (\\%)"].append(ann_vol * 100)
        metrics_arrays["Sharpe Ratio"].append(sharpe)
        metrics_arrays["Sortino Ratio"].append(sortino)
        metrics_arrays["Calmar Ratio"].append(calmar)
        metrics_arrays["Maximum Drawdown (\\%)"].append(max_dd * 100)
        metrics_arrays["Average Drawdown (\\%)"].append(avg_drawdown * 100)
        metrics_arrays["VaR 95\\% (\\%)"].append(var_95 * 100)
        metrics_arrays["Expected Shortfall 95\\% (\\%)"].append(es_95 * 100)
        metrics_arrays["Annual Turnover (\\%)"].append(annual_turnover * 100)
        metrics_arrays["Average Absolute Change in Portfolio Weights (\\%)"].append(abs_changes)
        metrics_arrays["Total Transaction Costs (\\pounds)"].append(total_costs)

    # Format output table
    rows = []
    for metric, arr in metrics_arrays.items():
        hist_val = hist_metrics.get(metric, np.nan)
        arr = np.array(arr)
        median_boot = np.median(arr)
        pct5 = np.percentile(arr, 5)
        pct95 = np.percentile(arr, 95)
        std_boot = np.std(arr, ddof=1)

        if std_boot > 0:
            z_score = (hist_val - median_boot) / std_boot
            p_value = 2 * (1 - norm.cdf(np.abs(z_score)))
        else:
            p_value = np.nan

        prob_outperf = np.mean(arr >= hist_val)

        # Formatting
        if "pounds" in metric.lower():
            fmt_hist = f"\\pounds{hist_val:,.2f}"
            fmt_median = f"\\pounds{median_boot:,.2f}"
            fmt_p5 = f"\\pounds{pct5:,.2f}"
            fmt_p95 = f"\\pounds{pct95:,.2f}"
        elif "ratio" in metric.lower():
            fmt_hist = f"{hist_val:.2f}"
            fmt_median = f"{median_boot:.2f}"
            fmt_p5 = f"{pct5:.2f}"
            fmt_p95 = f"{pct95:.2f}"
        else:
            fmt_hist = f"{hist_val:.2f}\\%"
            fmt_median = f"{median_boot:.2f}\\%"
            fmt_p5 = f"{pct5:.2f}\\%"
            fmt_p95 = f"{pct95:.2f}\\%"

        row = {
            "Metric": metric,
            "Historical": fmt_hist,
            "Bootstrap Median": fmt_median,
            "5th \\%ile": fmt_p5,
            "95th \\%ile": fmt_p95,
            "p-value": f"{p_value:.3f}" if np.isfinite(p_value) else "N/A",
            "Prob. of Outperformance (\\%)": f"{prob_outperf * 100:.2f}\\%",
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if latex_filename is None:
        file_stem = "_".join(tickers) + "_bootstrap_comparison_bench_table" if tickers else "bootstrap_comparison_bench_table"
        latex_filename = file_stem + ".tex"

    if output_folder is not None:
        latex_filename = os.path.join(output_folder, latex_filename)

    latex_code = df.to_latex(
        index=False,
        escape=False,
        column_format="lcccccc"
    )

    with open(latex_filename, "w") as f:
        f.write(latex_code)

    print(f"LaTeX table saved to: {latex_filename}")

    return df

# Convert summary into list of bootstrap trial dicts
bootstrap_results = []
for i in range(summary["wealth_paths"].shape[0]):
    bootstrap_results.append({
        "wealth": summary["wealth_paths"][i],
        "theta": summary["portfolio_paths"][i],
        "costs": summary["costs_paths"][i],
        "weights": summary["weights_paths"][i],
    })

# This is the one to feed into bootstrap comparison:
df_bootstrap = compute_bootstrap_comparison_table_from_results(
    hist_metrics=hist_metrics_bootstrap,
    bootstrap_results=bootstrap_results,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    dt_backtest=1/252,
    output_folder=output_folder,
    tickers=tickers
)

def plot_bootstrap_results(
    dates,
    bootstrap_results,
    output_dir,
    tickers=None,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    dt_backtest=1/252,
):
    """
    Plot summary results from bootstrap backtests using arithmetic returns.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Ticker prefix for filenames
    prefix = "_".join(tickers) + "_bootstrap_bench" if tickers else "bootstrap_bench"

    def file_path(name):
        return os.path.join(output_dir, f"{prefix}_{name}.pdf")

    freq_per_year = 1 / dt_backtest

    # -----------------------------------
    # Gather wealth paths
    # -----------------------------------
    wealth_paths = np.stack([r["wealth"] for r in bootstrap_results])

    # Compute statistics
    median_wealth = np.median(wealth_paths, axis=0)
    pct5 = np.percentile(wealth_paths, 5, axis=0)
    pct95 = np.percentile(wealth_paths, 95, axis=0)

    # Plot Median Wealth ± CI
    plt.figure(figsize=(8, 4))
    plt.plot(dates, median_wealth, color="#1f77b4", label="Median Wealth")
    plt.fill_between(dates, pct5, pct95, color="#1f77b4", alpha=0.3, label="90% CI")
    plt.xlabel("Date")
    plt.ylabel("Wealth (£)")
    plt.title("Median Wealth and Confidence Interval (Bootstrap)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("wealth_ci"))
    plt.close()

    # -----------------------------------
    # Histogram of Terminal Wealth
    # -----------------------------------
    terminal_wealths = wealth_paths[:, -1]

    plt.figure(figsize=(6, 4))
    plt.hist(terminal_wealths, bins=30, color="#1f77b4", edgecolor="black")
    plt.xlabel("Terminal Wealth (£)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Terminal Wealth (Bootstrap)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("terminal_wealth_hist"))
    plt.close()

    # -----------------------------------
    # Histogram of Sharpe Ratios (using arithmetic returns)
    # -----------------------------------
    sharpe_ratios = []
    for result in bootstrap_results:
        wealth_bt = result["wealth"]
        returns = np.diff(wealth_bt) / wealth_bt[:-1]
        T = len(wealth_bt) - 1
        if T < 1:
            sharpe_ratios.append(np.nan)
            continue
        cagr = (wealth_bt[-1] / initial_wealth) ** (freq_per_year / T) - 1
        ann_vol = np.std(returns) * np.sqrt(freq_per_year)
        ann_excess_return = cagr - annual_rf_rate
        sharpe = ann_excess_return / ann_vol if ann_vol > 0 else np.nan
        sharpe_ratios.append(sharpe)

    sharpe_ratios = np.array(sharpe_ratios)

    plt.figure(figsize=(6, 4))
    plt.hist(sharpe_ratios[~np.isnan(sharpe_ratios)], bins=30, color="#1f77b4", edgecolor="black")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Frequency")
    plt.title("Histogram of Sharpe Ratios (Bootstrap)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("sharpe_hist"))
    plt.close()

    # -----------------------------------
    # Histogram of Annual Turnover
    # -----------------------------------
    turnovers = []
    for result in bootstrap_results:
        wealth_bt = result["wealth"]
        theta = result["theta"]
        if len(wealth_bt) < 2:
            turnovers.append(np.nan)
            continue
        delta_u = np.diff(theta, axis=0)
        turnover_per_step = np.sum(np.abs(delta_u), axis=1) / wealth_bt[:-1]
        avg_turnover_step = np.mean(turnover_per_step)
        annual_turnover = avg_turnover_step * freq_per_year
        turnovers.append(annual_turnover * 100)

    turnovers = np.array(turnovers)

    plt.figure(figsize=(6, 4))
    plt.hist(turnovers[~np.isnan(turnovers)], bins=30, color="#1f77b4", edgecolor="black")
    plt.xlabel("Annual Turnover (%)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Annual Turnover (Bootstrap)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("turnover_hist"))
    plt.close()

    print(f"Bootstrap plots saved to {output_dir}")

plot_bootstrap_results(
    dates=summary["dates"],
    bootstrap_results=bootstrap_results,
    output_dir=output_folder,
    tickers=tickers,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    dt_backtest=1/252,
)

# Number of bootstrap trials
n_trials = summary["wealth_paths"].shape[0]

# Tickers string for file naming
tickers_str = "_".join(tickers)

# Load in simulated bootstrap paths
for i in range(n_trials):
    wealth_bt = summary["wealth_paths"][i]
    portfolio_bt = summary["portfolio_paths"][i]
    costs_bt = summary["costs_paths"][i]
    weights_bt = summary["weights_paths"][i]
    
    # Save each trial as a separate file
    np.savez(
        os.path.join(
            output_folder,
            f"{tickers_str}_parametric_bootstrap_bench_{i}.npz"
        ),
        dates=summary["dates"].values.astype("datetime64[D]"),
        wealth=wealth_bt,
        portfolio=portfolio_bt,
        costs=costs_bt,
        weights=weights_bt,
    )
    
print(f"Saved {n_trials} parametric bootstrap result files in {output_folder}")

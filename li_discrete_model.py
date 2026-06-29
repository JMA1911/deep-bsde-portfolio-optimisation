import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import pickle

def download_prices(tickers, start_date, end_date):
    """
    Download adjusted close prices for tickers from Yahoo Finance.
    Returns a DataFrame with prices.
    """
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    adj_close = data.xs('Close', level='Price', axis=1)
    return adj_close

def compute_gross_returns(prices):
    """
    Compute gross returns from adjusted close prices.
    R_t = P_t / P_{t-1}
    """
    gross_returns = prices / prices.shift(1)
    gross_returns = gross_returns.dropna()
    return gross_returns

def compute_excess_returns(gross_returns, annual_rf_rate=0.02):
    """
    Compute excess returns over gross risk-free rate.
    """
    rf_daily = annual_rf_rate / 252
    s_daily = np.exp(rf_daily)
    excess_returns = gross_returns - s_daily
    return excess_returns

def estimate_rolling_parameters(excess_returns, window):
    """
    Estimate rolling mean vector and covariance matrix of excess returns.
    """
    rolling_mean = excess_returns.rolling(window).mean()
    rolling_cov = excess_returns.rolling(window).cov()
    return rolling_mean, rolling_cov

def compute_Q(mu_t, cov_t):
    """
    Compute Q_t matrix.
    """
    mu_outer = np.outer(mu_t, mu_t)
    Q_t = cov_t + mu_outer
    return Q_t

def compute_K(Q_t, mu_t, reg=1e-1):
    Q_t += reg * np.eye(Q_t.shape[0])
    K_t = np.linalg.solve(Q_t, mu_t)
    return K_t

def compute_B(mu_t, K_t):
    """
    Compute scalar B_t.
    """
    B_t = mu_t.T @ K_t
    return B_t

def compute_alphas(s_list, B_list):
    """
    Compute the entire sequence of alpha_t backward from alpha_T = 1.
    """
    T = len(B_list)
    alpha = [None] * (T + 1)
    alpha[T] = 1.0  # terminal condition
    
    for t in reversed(range(T)):
        alpha[t] = alpha[t+1] * s_list[t]**2 * (1 - B_list[t])
    
    return alpha

def compute_gamma_star(s_list, B_list, x0, w):
    """
    Compute gamma*.
    """
    T = len(B_list)
    prod_s = np.prod(s_list)
    prod_1_minus_B = np.prod([1 - B for B in B_list])
    
    gamma_star = (
        2 * prod_s * x0
        + 1 / (w * prod_1_minus_B)
    )
    return gamma_star

def compute_betas(s_list, B_list, gamma_star, w):
    """
    Compute beta_t backwards from terminal condition.
    """
    T = len(B_list)
    beta = [None] * (T + 1)
    beta[T] = w * gamma_star
    
    for t in reversed(range(T)):
        beta[t] = beta[t+1] * s_list[t] * (1 - 2 * B_list[t])
    
    return beta

def compute_v_t(beta_tp1, alpha_tp1, w, K_t):
    """
    Compute v_t vector.
    """
    v_t = (beta_tp1 / (2 * w * alpha_tp1)) * K_t
    return v_t

def compute_optimal_policy(s_t, K_t, x_t, v_t):
    """
    Compute u_t*.
    """
    u_t_star = - s_t * K_t * x_t + v_t
    return u_t_star

def backtest_multi_period_strategy(
    prices,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    window=252,
    w=5,
    lambda_tc=0.0003,  # e.g. 3 bps per (Δu)^2
    use_transaction_costs=True
):
    """
    Backtest the multi-period mean-variance strategy on real price paths.
    """

    # ----------------------------
    # Constants
    # ----------------------------
    max_alloc = 3 * initial_wealth
    max_delta_u = 3 * initial_wealth
    wealth_floor = 1.0
    max_leverage = 1.5

    # ----------------------------
    # Pre-computations
    # ----------------------------
    gross_returns = compute_gross_returns(prices)
    excess_returns = compute_excess_returns(gross_returns, annual_rf_rate)
    rolling_mean, rolling_cov = estimate_rolling_parameters(excess_returns, window)

    rf_daily = annual_rf_rate / 252
    s_t = np.exp(rf_daily)

    assets = excess_returns.columns.tolist()
    dates = excess_returns.index[window:]
    T = len(dates)

    B_list, K_list = [], []

    for date in dates:
        mu_t = rolling_mean.loc[date].values
        cov_t = rolling_cov.loc[date].loc[assets, assets].values

        Q_t = compute_Q(mu_t, cov_t)
        K_t = compute_K(Q_t, mu_t, reg=1e-3)
        B_t = compute_B(mu_t, K_t)

        B_list.append(B_t)
        K_list.append(K_t)

    s_list = [s_t] * T
    alpha_list = compute_alphas(s_list, B_list)
    gamma_star = compute_gamma_star(s_list, B_list, initial_wealth, w)
    beta_list = compute_betas(s_list, B_list, gamma_star, w)

    # ------------------------------------------------------------
    # FINAL RUN → use fixed λ
    # ------------------------------------------------------------

    x_t = initial_wealth
    cash_t = initial_wealth
    prev_u_t = np.zeros(len(assets))

    wealth_bt = [x_t]
    portfolio_path = []
    dates_bt = []
    costs_bt = []
    weights_path = []

    for t, date in enumerate(dates[:-1]):
        next_date = dates[t+1]

        mu_t = rolling_mean.loc[date].values
        cov_t = rolling_cov.loc[date].loc[assets, assets].values
        K_t = K_list[t]

        v_t = compute_v_t(beta_list[t+1], alpha_list[t+1], w, K_t)
        u_t_star = compute_optimal_policy(s_t, K_t, x_t, v_t)
        u_t_star = np.clip(u_t_star, -max_alloc, max_alloc)

        total_exposure = np.sum(np.abs(u_t_star))
        max_exposure = max_leverage * x_t
        if total_exposure > max_exposure:
            scaling_factor = max_exposure / total_exposure
            u_t_star *= scaling_factor

        S_t = prices.loc[date].values
        S_t = np.where(S_t == 0, 1e-8, S_t)
        holdings_t = u_t_star / S_t

        risky_value_t = np.sum(u_t_star)
        cash_t = x_t - risky_value_t

        S_next = prices.loc[next_date].values
        S_next = np.where(S_next == 0, 1e-8, S_next)
        risky_value_t1 = np.sum(holdings_t * S_next)
        cash_t1 = cash_t * s_t

        delta_u = u_t_star - prev_u_t
        delta_u = np.clip(delta_u, -max_delta_u, max_delta_u)

        if not np.all(np.isfinite(delta_u)):
            print(f"NaN detected in delta_u at time {t}. Skipping step.")
            continue

        cost = lambda_tc * np.sum(delta_u ** 2) if use_transaction_costs else 0.0

        x_t1 = risky_value_t1 + cash_t1 - cost
        if not np.isfinite(x_t1):
            print(f"Non-finite wealth at time {t}. Setting wealth to floor.")
            x_t1 = wealth_floor

        x_t1 = max(x_t1, wealth_floor)

        weights_t = np.divide(
            u_t_star,
            x_t,
            out=np.zeros_like(u_t_star),
            where=(x_t != 0)
        )
        cash_weight_t = cash_t / x_t if x_t != 0 else 0.0
        weights_full_t = np.append(weights_t, cash_weight_t)

        wealth_bt.append(x_t1)
        portfolio_path.append(u_t_star)
        dates_bt.append(next_date)
        costs_bt.append(cost)
        weights_path.append(weights_full_t)

        prev_u_t = u_t_star
        x_t = x_t1

    # Add initial day
    dates_bt.insert(0, dates[0])
    initial_u_t = np.zeros(len(assets))
    initial_cash_t = initial_wealth
    initial_weights = np.append(initial_u_t / initial_wealth, initial_cash_t / initial_wealth)

    portfolio_path.insert(0, initial_u_t)
    weights_path.insert(0, initial_weights)

    return (
        dates_bt,
        np.array(wealth_bt),
        np.array(portfolio_path),
        np.array(costs_bt),
        np.array(weights_path)
    )


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
    latex_filename=None,
    arithmetic_returns_eqw=None,
    sharpe_eqw=None,
    dates_eqw=None,
    wealth_eqw=None,
):
    """
    Computes portfolio performance metrics and exports them as a LaTeX table.
    """

    if latex_filename is None:
        file_stem = "_".join(tickers) + "_multi_perf_table" if tickers else "multi_perf_table"
        latex_filename = file_stem + ".tex"

    if output_folder is not None:
        latex_filename = os.path.join(output_folder, latex_filename)

    if arithmetic_returns_eqw is not None and dates_eqw is not None:
        start_date = dates_bt[0]
        mask_eqw = dates_eqw >= start_date
        if not np.any(mask_eqw):
            raise ValueError(f"No EQW benchmark dates overlap with model backtest dates starting from {start_date}.")

        dates_eqw_matched = dates_eqw[mask_eqw]

        if wealth_eqw is not None:
            wealth_eqw_matched = wealth_eqw[mask_eqw]
            arith_returns_eqw_matched = np.diff(wealth_eqw_matched) / wealth_eqw_matched[:-1]
        else:
            arith_returns_eqw_matched = arithmetic_returns_eqw[mask_eqw[:-1]]  # Already arithmetic

        min_len = min(len(arith_returns_eqw_matched), len(wealth_bt)-1)
        arith_returns_eqw_matched = arith_returns_eqw_matched[:min_len]
        arith_returns_model = np.diff(wealth_bt) / wealth_bt[:-1]
        arith_returns_model = arith_returns_model[:min_len]
    else:
        arith_returns_eqw_matched = None
        arith_returns_model = np.diff(wealth_bt) / wealth_bt[:-1]

    T = len(arith_returns_model)
    if T <= 1:
        raise ValueError("Not enough time steps to compute statistics.")

    freq_per_year = 1 / dt_backtest

    cagr = (wealth_bt[-1] / initial_wealth)**(freq_per_year / T) - 1
    ann_vol = np.std(arith_returns_model) * np.sqrt(freq_per_year)
    ann_excess_return = cagr - annual_rf_rate
    sharpe = ann_excess_return / ann_vol if ann_vol > 0 else np.nan

    benchmark_sharpe = 0.0
    if T > 1 and np.isfinite(sharpe):
        se_sharpe = np.sqrt((1 + sharpe**2 / 2) / (T - 1))
        psr_rf = norm.cdf((sharpe - benchmark_sharpe) / se_sharpe)
    else:
        psr_rf = np.nan

    if sharpe_eqw is not None and np.isfinite(sharpe_eqw) and np.isfinite(sharpe):
        se_diff = np.sqrt((1 + sharpe**2 / 2 + 1 + sharpe_eqw**2 / 2) / (T - 1))
        psr_eqw = norm.cdf((sharpe - sharpe_eqw) / se_diff)
    else:
        psr_eqw = np.nan

    if sharpe_eqw is not None and np.isfinite(sharpe_eqw) and np.isfinite(sharpe):
        diff_sharpe = sharpe - sharpe_eqw
        t_stat_lw = diff_sharpe / se_diff
        p_value_lw = 1 - norm.cdf(t_stat_lw)
        p_value_lw = np.clip(p_value_lw, 0, 1)
    else:
        t_stat_lw = np.nan
        p_value_lw = np.nan

    if arith_returns_eqw_matched is not None and len(arith_returns_eqw_matched) == len(arith_returns_model):
        diff_returns = arith_returns_model - arith_returns_eqw_matched
        diff_mean = np.mean(diff_returns)
        diff_std = np.std(diff_returns, ddof=1)
        if diff_std > 0:
            t_stat_ret = diff_mean / (diff_std / np.sqrt(T))
            p_value_ret = 1 - norm.cdf(t_stat_ret)
            p_value_ret = np.clip(p_value_ret, 0, 1)
        else:
            t_stat_ret = np.nan
            p_value_ret = np.nan
    else:
        t_stat_ret = np.nan
        p_value_ret = np.nan

    downside_returns = arith_returns_model[arith_returns_model < 0]
    downside_std = np.std(downside_returns) * np.sqrt(freq_per_year) if len(downside_returns) > 0 else np.nan
    sortino = ann_excess_return / downside_std if downside_std > 0 else np.nan

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

    recovery_times = []
    peak_idx = 0
    for t in range(1, len(wealth_bt)):
        if wealth_bt[t] >= wealth_bt[peak_idx]:
            if t > peak_idx:
                recovery_times.append(t - peak_idx)
            peak_idx = t
    avg_recovery_days = np.mean(recovery_times) * (252 * dt_backtest) if recovery_times else np.nan

    calmar = cagr / max_dd if max_dd > 0 else np.nan

    delta_u = np.diff(portfolio_path, axis=0)
    turnover_per_step = np.sum(np.abs(delta_u), axis=1) / wealth_bt[:-1]
    avg_turnover_step = np.mean(turnover_per_step)
    annual_turnover = avg_turnover_step * freq_per_year

    weights_risky = weights_path[:, :-1]
    std_weights = np.std(weights_risky, axis=0)
    std_weights_mean = np.mean(std_weights) * 100
    delta_w = np.diff(weights_risky, axis=0)
    abs_changes = np.mean(np.sum(np.abs(delta_w), axis=1)) * 100
    tolerance = 1e-6
    delta_theta = np.diff(portfolio_path, axis=0)
    unchanged_days = np.sum(np.all(np.abs(delta_theta) < tolerance, axis=1))
    no_trade_fraction = unchanged_days / (portfolio_path.shape[0] - 1)

    total_costs = np.sum(costs_bt)

    confidence_level = 0.95
    if len(arith_returns_model) > 0:
        var_95 = -np.percentile(arith_returns_model, 100 * (1 - confidence_level))
        es_95 = -np.mean(arith_returns_model[arith_returns_model <= -var_95]) if np.any(arith_returns_model <= -var_95) else np.nan
    else:
        var_95 = es_95 = np.nan

    var_95_pct = var_95 * 100 if np.isfinite(var_95) else np.nan
    es_95_pct = es_95 * 100 if np.isfinite(es_95) else np.nan

    risky_dollar_exposure = np.sum(np.abs(portfolio_path), axis=1)
    avg_leverage = np.mean(risky_dollar_exposure / wealth_bt)
    max_dollar_pos = np.max(np.abs(portfolio_path))

    total_pnl = wealth_bt[-1] - initial_wealth
    total_dollars_traded = np.sum(np.abs(delta_u))
    pnl_per_turnover = total_pnl / total_dollars_traded if total_dollars_traded > 0 else np.nan

    metrics_names = [
        "Terminal Wealth (\\pounds)", "CAGR (\\%)", "Annual Volatility (\\%)", "Sharpe Ratio", "Sortino Ratio",
        "Calmar Ratio", "Maximum Drawdown (\\%)", "Average Drawdown (\\%)", "Average Recovery Time (days)",
        "VaR 95\\% (\\%)", "Expected Shortfall 95\\% (\\%)", "PSR vs Risk-Free (\\%)", "PSR vs B\\&H(\\%)",
        "Sharpe Difference t-stat", "Sharpe Difference p-value", "Returns Difference t-stat",
        "Returns Difference p-value", "Average Leverage", "Maximum Position Size (\\pounds)",
        "Average Turnover per Step (\\%)", "Annual Turnover (\\%)", "Profit per Turnover (\\pounds)",
        "Standard Deviation of Portfolio Weights (\\%)", "Average Absolute Change in Portfolio Weights (\\%)",
        "No-Trade Fraction (\\%)", "Total Transaction Costs (\\pounds)"
    ]

    metrics_values_raw = [
        wealth_bt[-1],
        cagr * 100,
        ann_vol * 100,
        sharpe,
        sortino if np.isfinite(sortino) else np.nan,
        calmar if np.isfinite(calmar) else np.nan,
        max_dd * 100,
        avg_drawdown * 100,
        avg_recovery_days if np.isfinite(avg_recovery_days) else np.nan,
        var_95_pct if np.isfinite(var_95_pct) else np.nan,
        es_95_pct if np.isfinite(es_95_pct) else np.nan,
        psr_rf * 100 if np.isfinite(psr_rf) else np.nan,
        psr_eqw * 100 if np.isfinite(psr_eqw) else np.nan,
        t_stat_lw if np.isfinite(t_stat_lw) else np.nan,
        p_value_lw if np.isfinite(p_value_lw) else np.nan,
        t_stat_ret if np.isfinite(t_stat_ret) else np.nan,
        p_value_ret if np.isfinite(p_value_ret) else np.nan,
        avg_leverage if np.isfinite(avg_leverage) else np.nan,
        max_dollar_pos if np.isfinite(max_dollar_pos) else np.nan,
        avg_turnover_step * 100,
        annual_turnover * 100,
        pnl_per_turnover if np.isfinite(pnl_per_turnover) else np.nan,
        std_weights_mean if np.isfinite(std_weights_mean) else np.nan,
        abs_changes if np.isfinite(abs_changes) else np.nan,
        no_trade_fraction * 100 if np.isfinite(no_trade_fraction) else np.nan,
        total_costs,
    ]

    metrics_values = [
        f"\\pounds{wealth_bt[-1]:,.2f}", f"{cagr * 100:.2f}\\%", f"{ann_vol * 100:.2f}\\%", f"{sharpe:.2f}",
        f"{sortino:.2f}" if np.isfinite(sortino) else "N/A", f"{calmar:.2f}" if np.isfinite(calmar) else "N/A",
        f"{max_dd * 100:.2f}\\%", f"{avg_drawdown * 100:.2f}\\%",
        f"{avg_recovery_days:.2f}" if np.isfinite(avg_recovery_days) else "N/A",
        f"{var_95_pct:.2f}\\%" if np.isfinite(var_95_pct) else "N/A",
        f"{es_95_pct:.2f}\\%" if np.isfinite(es_95_pct) else "N/A",
        f"{psr_rf * 100:.2f}\\%" if np.isfinite(psr_rf) else "N/A",
        f"{psr_eqw * 100:.2f}\\%" if np.isfinite(psr_eqw) else "N/A",
        f"{t_stat_lw:.2f}" if np.isfinite(t_stat_lw) else "N/A",
        f"{p_value_lw:.4f}" if np.isfinite(p_value_lw) else "N/A",
        f"{t_stat_ret:.2f}" if np.isfinite(t_stat_ret) else "N/A",
        f"{p_value_ret:.4f}" if np.isfinite(p_value_ret) else "N/A",
        f"{avg_leverage:.2f}" if np.isfinite(avg_leverage) else "N/A",
        f"\\pounds{max_dollar_pos:,.2f}" if np.isfinite(max_dollar_pos) else "N/A",
        f"{avg_turnover_step * 100:.2f}\\%", f"{annual_turnover * 100:.2f}\\%",
        f"\\pounds{pnl_per_turnover:,.2f}" if np.isfinite(pnl_per_turnover) else "N/A",
        f"{std_weights_mean:.2f}\\%" if np.isfinite(std_weights_mean) else "N/A",
        f"{abs_changes:.2f}\\%" if np.isfinite(abs_changes) else "N/A",
        f"{no_trade_fraction * 100:.2f}\\%" if np.isfinite(no_trade_fraction) else "N/A",
        f"\\pounds{total_costs:,.2f}"
    ]

    hist_metrics = dict(zip(metrics_names, metrics_values_raw))

    df_metrics = pd.DataFrame({"Metric": metrics_names, "Value": metrics_values})

    tabular_code = df_metrics.to_latex(index=False, escape=False, column_format="ll")

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    with open(latex_filename, "w") as f:
        f.write(tabular_code)

    print(f"LaTeX table saved to: {latex_filename}")

    return df_metrics, hist_metrics

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
    plt.savefig(file_path("wealth_over_time"))
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
    plt.savefig(file_path("drawdown_over_time"))
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
    plt.savefig(file_path("histogram_returns"))
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

    plt.figure(figsize=(8, 4))
    plt.plot(dates_for_rolling, roll_sharpe.iloc[window - 1 :], color=single_color, linewidth=1.5)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe Ratio")
    plt.title(f"Rolling Sharpe Ratio ({window}-day)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("rolling_sharpe"))
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
    plt.savefig(file_path("weights_line_plot"))
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
    plt.savefig(file_path("turnover_over_time"))
    plt.close()

    # ----------------------------------------
    # Plot 7: Cumulative Transaction Costs
    # ----------------------------------------
    cumulative_costs = np.cumsum(costs_bt)
    cumulative_costs_padded = np.insert(cumulative_costs, 0, 0.0)

    plt.figure(figsize=(8, 4))
    plt.plot(dates_bt, cumulative_costs_padded, color=single_color, linewidth=1.8)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Costs (£)")
    plt.title("Cumulative Transaction Costs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("cumulative_transaction_costs"))
    plt.close()

    print(f"All plots saved in: {output_dir}")

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
    dates_lists = []

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

        if benchmark_kwargs is None:
            benchmark_kwargs = {}

        # Run benchmark on entire synthetic history for correct rolling window
        result = benchmark_func(
            prices=adj_close_synth_full,
            **benchmark_kwargs
        )

        dates_bench, wealth_bench, portfolio_bench, costs_bench, weights_bench = result

        # Convert to numpy array
        dates_bench = np.array(dates_bench, dtype="datetime64[D]")

        # Discover first valid benchmark date in this trial
        first_bench_date = dates_bench[0]
        print(f"Trial {i} first benchmark date: {first_bench_date}")

        # Slice to desired trading horizon if possible
        desired_start = np.datetime64(start_date_bt)

        # The actual starting date is whichever is later:
        # (a) the desired start date, or
        # (b) the earliest benchmark date allowed by rolling window
        effective_start_date = max(desired_start, first_bench_date)

        mask = dates_bench >= effective_start_date

        dates_bench = dates_bench[mask]
        wealth_bench = wealth_bench[mask]
        portfolio_bench = portfolio_bench[mask, :]
        costs_bench = costs_bench[mask[:-1]]
        weights_bench = weights_bench[mask, :]

        dates_lists.append(dates_bench)
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
                dates=dates_bench,
                wealth=wealth_bench,
                portfolio=portfolio_bench,
                costs=costs_bench,
                weights=weights_bench,
            )

    # ---------------------------------------------------------------
    # Find the intersection of dates across all trials
    # ---------------------------------------------------------------
    common_dates = set(dates_lists[0])
    for d in dates_lists[1:]:
        common_dates &= set(d)

    if not common_dates:
        raise ValueError("No overlapping dates across trials!")

    common_dates = np.array(sorted(common_dates))
    print(f"Final common dates length: {len(common_dates)}")

    # ---------------------------------------------------------------
    # Slice all benchmark results to common dates
    # ---------------------------------------------------------------
    wealth_list_final = []
    portfolio_list_final = []
    costs_list_final = []
    weights_list_final = []

    for i in range(n_trials):
        dates_bench = dates_lists[i]

        mask = np.isin(dates_bench, common_dates)

        wealth_bench = wealth_list[i][mask]
        portfolio_bench = portfolio_list[i][mask, :]
        costs_bench = costs_list[i][mask[:-1]]
        weights_bench = weights_list[i][mask, :]

        wealth_list_final.append(wealth_bench)
        portfolio_list_final.append(portfolio_bench)
        costs_list_final.append(costs_bench)
        weights_list_final.append(weights_bench)

    return {
        "dates": common_dates,
        "wealth_paths": np.stack(wealth_list_final, axis=0),
        "portfolio_paths": np.stack(portfolio_list_final, axis=0),
        "costs_paths": np.stack(costs_list_final, axis=0),
        "weights_paths": np.stack(weights_list_final, axis=0),
    }

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
    Compute a comparison table of historical vs bootstrap metrics
    using arithmetic returns and additional risk and stability metrics.
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
        arith_returns = np.diff(wealth_bt) / wealth_bt[:-1]
        cagr = (wealth_bt[-1] / initial_wealth)**(freq_per_year / T) - 1
        ann_vol = np.std(arith_returns) * np.sqrt(freq_per_year)
        ann_excess_return = cagr - annual_rf_rate
        sharpe = ann_excess_return / ann_vol if ann_vol > 0 else np.nan

        # Sortino Ratio
        downside_returns = arith_returns[arith_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(freq_per_year) if len(downside_returns) > 0 else np.nan
        sortino = ann_excess_return / downside_std if downside_std > 0 else np.nan

        # Drawdown metrics
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

        calmar = cagr / max_dd if max_dd > 0 else np.nan

        # VaR and Expected Shortfall
        confidence_level = 0.95
        if len(arith_returns) > 0:
            var_95 = -np.percentile(arith_returns, 100 * (1 - confidence_level))
            es_95 = -np.mean(arith_returns[arith_returns <= -var_95]) if np.any(arith_returns <= -var_95) else np.nan
        else:
            var_95 = es_95 = np.nan

        # Turnover
        delta_u = np.diff(portfolio_path, axis=0)
        turnover_per_step = np.sum(np.abs(delta_u), axis=1) / wealth_bt[:-1]
        avg_turnover_step = np.mean(turnover_per_step)
        annual_turnover = avg_turnover_step * freq_per_year

        # Average absolute change in portfolio weights
        weights_risky = weights_path[:, :-1]  # exclude cash
        delta_w = np.diff(weights_risky, axis=0)
        abs_changes = np.mean(np.sum(np.abs(delta_w), axis=1)) * 100  # as %

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
        metrics_arrays["VaR 95\\% (\\%)"].append(var_95 * 100 if np.isfinite(var_95) else np.nan)
        metrics_arrays["Expected Shortfall 95\\% (\\%)"].append(es_95 * 100 if np.isfinite(es_95) else np.nan)
        metrics_arrays["Annual Turnover (\\%)"].append(annual_turnover * 100)
        metrics_arrays["Average Absolute Change in Portfolio Weights (\\%)"].append(abs_changes)
        metrics_arrays["Total Transaction Costs (\\pounds)"].append(total_costs)

    # --------------------------
    # Format output table
    # --------------------------
    rows = []

    for metric in metrics_arrays.keys():
        hist_val = hist_metrics[metric]
        arr = np.array(metrics_arrays[metric])
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

        ratios = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]

        if "wealth" in metric.lower() or "cost" in metric.lower():
            fmt_hist = f"\\pounds{hist_val:,.2f}"
            fmt_median = f"\\pounds{median_boot:,.2f}"
            fmt_p5 = f"\\pounds{pct5:,.2f}"
            fmt_p95 = f"\\pounds{pct95:,.2f}"

        elif metric in ratios:
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

    # --------------------------
    # Save LaTeX table
    # --------------------------
    if latex_filename is None:
        file_stem = "_".join(tickers) + "_bootstrap_comparison_multi_table" if tickers else "bootstrap_comparison_multi_table"
        latex_filename = file_stem + ".tex"

    if output_folder is not None:
        latex_filename = os.path.join(output_folder, latex_filename)

    latex_code = df.to_latex(index=False, escape=False, column_format="lcccccc")

    with open(latex_filename, "w") as f:
        f.write(latex_code)

    print(f"LaTeX table saved to: {latex_filename}")

    return df

def compute_bootstrap_stat_comparison_table(
    bootstrap_results_deepbsde,
    bootstrap_results_benchmark,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    dt_backtest=1/252,
    latex_filename=None,
    output_folder=None,
    tickers=None,
):
    """
    Compare DeepBSDE vs benchmark using bootstrap trials.
    Uses arithmetic returns instead of arithmetic returns.
    """
    n_trials = min(len(bootstrap_results_deepbsde), len(bootstrap_results_benchmark))
    freq_per_year = 1 / dt_backtest

    diff_sharpe = []
    diff_cagr = []

    for i in range(n_trials):
        result_deep = bootstrap_results_deepbsde[i]
        result_bench = bootstrap_results_benchmark[i]

        wealth_deep = result_deep["wealth"]
        wealth_bench = result_bench["wealth"]

        T = len(wealth_deep) - 1
        if T < 1:
            continue

        # Arithmetic returns
        arith_returns_deep = np.diff(wealth_deep) / wealth_deep[:-1]
        arith_returns_bench = np.diff(wealth_bench) / wealth_bench[:-1]

        # CAGR
        cagr_deep = (wealth_deep[-1] / initial_wealth)**(freq_per_year / T) - 1
        cagr_bench = (wealth_bench[-1] / initial_wealth)**(freq_per_year / T) - 1

        # Annual Volatility
        ann_vol_deep = np.std(arith_returns_deep) * np.sqrt(freq_per_year)
        ann_vol_bench = np.std(arith_returns_bench) * np.sqrt(freq_per_year)

        # Excess Return
        excess_return_deep = cagr_deep - annual_rf_rate
        excess_return_bench = cagr_bench - annual_rf_rate

        # Sharpe Ratio
        sharpe_deep = excess_return_deep / ann_vol_deep if ann_vol_deep > 0 else np.nan
        sharpe_bench = excess_return_bench / ann_vol_bench if ann_vol_bench > 0 else np.nan

        # Differences
        diff_sharpe.append(sharpe_deep - sharpe_bench)
        diff_cagr.append(cagr_deep - cagr_bench)

    # Compute stats
    def compute_stats(diff_array):
        diff_array = np.array(diff_array)
        mean_diff = np.nanmean(diff_array)
        std_diff = np.nanstd(diff_array, ddof=1)
        n = np.sum(~np.isnan(diff_array))

        if std_diff > 0 and n > 1:
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            p_value = 1 - norm.cdf(t_stat)
        else:
            t_stat = np.nan
            p_value = np.nan

        prob_outperf = np.mean(diff_array > 0)
        return mean_diff, t_stat, p_value, prob_outperf

    mean_diff_sharpe, t_sharpe, pval_sharpe, prob_sharpe = compute_stats(diff_sharpe)
    mean_diff_cagr, t_cagr, pval_cagr, prob_cagr = compute_stats(diff_cagr)

    # Table rows
    rows = [
        {
            "Metric": "Sharpe Ratio Difference",
            "Mean Difference": f"{mean_diff_sharpe:.4f}",
            "t-stat": f"{t_sharpe:.2f}" if np.isfinite(t_sharpe) else "N/A",
            "p-value": f"{pval_sharpe:.4f}" if np.isfinite(pval_sharpe) else "N/A",
            "Probability $>$ Benchmark (\\%)": f"{prob_sharpe * 100:.2f}\\%",
        },
        {
            "Metric": "Returns Difference (CAGR)",
            "Mean Difference": f"{mean_diff_cagr * 100:.2f}\\%",
            "t-stat": f"{t_cagr:.2f}" if np.isfinite(t_cagr) else "N/A",
            "p-value": f"{pval_cagr:.4f}" if np.isfinite(pval_cagr) else "N/A",
            "Probability $>$ Benchmark (\\%)": f"{prob_cagr * 100:.2f}\\%",
        }
    ]

    df = pd.DataFrame(rows)

    # Save LaTeX table
    if latex_filename is None:
        file_stem = "_".join(tickers) + "_bootstrap_bench_vs_multi_table" if tickers else "bootstrap_bench_vs_multi_table"
        latex_filename = file_stem + ".tex"

    if output_folder is not None:
        latex_filename = os.path.join(output_folder, latex_filename)

    latex_code = df.to_latex(index=False, escape=False, column_format="lcccc")

    with open(latex_filename, "w") as f:
        f.write(latex_code)

    print(f"LaTeX table saved to: {latex_filename}")

    return df

def plot_bootstrap_results(
    dates,
    bootstrap_results,
    output_dir,
    tickers=None,
    initial_wealth=100.0,
    annual_rf_rate=0.02,
    dt_backtest=1/252,
):
    """
    Plot summary results from bootstrap backtests using arithmetic returns.
    """

    os.makedirs(output_dir, exist_ok=True)

    if tickers is None:
        prefix = "bootstrap_multi"
    else:
        prefix = "_".join(tickers) + "_bootstrap_multi"

    def file_path(name):
        return os.path.join(output_dir, f"{prefix}_{name}.pdf")

    freq_per_year = 1 / dt_backtest

    # -----------------------------------
    # Gather wealth paths
    # -----------------------------------
    wealth_paths = np.stack([r["wealth"] for r in bootstrap_results])

    median_wealth = np.median(wealth_paths, axis=0)
    pct5 = np.percentile(wealth_paths, 5, axis=0)
    pct95 = np.percentile(wealth_paths, 95, axis=0)

    # Median wealth ± CI
    plt.figure(figsize=(8, 4))
    plt.plot(dates, median_wealth, color="#1f77b4", label="Median Wealth")
    plt.fill_between(dates, pct5, pct95, color="#1f77b4", alpha=0.3, label="90% CI")
    plt.xlabel("Date")
    plt.ylabel("Wealth (£)")
    plt.title("Bootstrap Median Wealth and Confidence Interval")
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
    # Histogram of Sharpe Ratios (Arithmetic)
    # -----------------------------------
    sharpe_ratios = []
    for result in bootstrap_results:
        wealth_bt = result["wealth"]
        T = len(wealth_bt) - 1
        if T < 1:
            sharpe_ratios.append(np.nan)
            continue

        arith_returns = np.diff(wealth_bt) / wealth_bt[:-1]
        cagr = (wealth_bt[-1] / initial_wealth) ** (freq_per_year / T) - 1
        ann_vol = np.std(arith_returns) * np.sqrt(freq_per_year)
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

if __name__ == "__main__":    
    tickers = ['AAPL', 'JPM', 'XOM']
    start_date = '2014-01-01'
    end_date = '2025-01-01'
    
    # 1. Download data
    prices = download_prices(tickers, start_date, end_date)
    
    # 2. Compute gross returns
    gross_returns = compute_gross_returns(prices)
    
    # 3. Compute excess returns
    excess_returns = compute_excess_returns(gross_returns, annual_rf_rate=0.025)
    
    # 4. Estimate rolling parameters
    window = 252
    rolling_mean, rolling_cov = estimate_rolling_parameters(excess_returns, window)
    
    # Prepare time grid
    dates = excess_returns.index[window:]
    T = len(dates)
    assets = excess_returns.columns.tolist()
    
    # Risk-free gross daily return
    rf_daily = 0.025 / 252
    s_t = np.exp(rf_daily)
    s_list = [s_t] * T

    # Pre-compute B_t and K_t lists
    B_list = []
    K_list = []
    
    for date in dates:
        mu_t = rolling_mean.loc[date].values
        cov_t = rolling_cov.loc[date].loc[assets, assets].values
        
        Q_t = compute_Q(mu_t, cov_t)
        K_t = compute_K(Q_t, mu_t, reg=1e-3)
        B_t = compute_B(mu_t, K_t)
        
        B_list.append(B_t)
        K_list.append(K_t)
    
    print("First 10 B_t values:")
    for i, b in enumerate(B_list[:10]):
        print(f"B_{i} = {b:.8f}")
    
    # Compute α, γ*, β sequences
    x0 = 100
    w = 5
    
    alpha_list = compute_alphas(s_list, B_list)
    gamma_star = compute_gamma_star(s_list, B_list, x0, w)
    beta_list = compute_betas(s_list, B_list, gamma_star, w)
    
    print("\ngamma_star:", gamma_star)
    print("alpha_t first 5:", alpha_list[:5])
    print("beta_t first 5:", beta_list[:5])
    
    # ----- Backtest using real returns -----
    # To match w and gamma from continuous must run both strategies and compute standard deivation of arithmetic-returns of wealth process, adjust to bring volatilites in alignment
    dates_bt, wealth_bt, portfolio_path, costs_bt, weights_path = backtest_multi_period_strategy(
    prices,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    window=252,
    w=5,
    lambda_tc=0.0003,   # e.g. 5 bps
    use_transaction_costs=True
)
    
    output_folder = "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up"
    tickers_str = "_".join(tickers)

    np.savez(
        os.path.join(output_folder, f"{tickers_str}_multi_backtest.npz"),
        dates=np.array(dates_bt).astype("datetime64[D]"),
        wealth=wealth_bt,
        portfolio=portfolio_path,
        costs=costs_bt,
        weights=weights_path,
    )
    
    # Load in B&H historical backtest results
    data_eqw = np.load(
        "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up/eqw_results.npz",
        allow_pickle=True
    )

    dates_eqw = data_eqw["dates"]
    wealth_eqw = data_eqw["wealth"]
    portfolio_eqw = data_eqw["portfolio"]
    costs_eqw = data_eqw["costs"]
    weights_eqw = data_eqw["weights"]
    arithmetic_returns_eqw = wealth_eqw[1:] / wealth_eqw[:-1] - 1
    T_eqw = len(wealth_eqw) - 1
    freq_per_year = 252
    cagr_eqw = (wealth_eqw[-1] / wealth_eqw[0])**(freq_per_year / T_eqw) - 1
    ann_vol_eqw = np.std(arithmetic_returns_eqw) * np.sqrt(freq_per_year)
    ann_excess_eqw = cagr_eqw - 0.025
    sharpe_eqw = ann_excess_eqw / ann_vol_eqw if ann_vol_eqw > 0 else np.nan
    output_folder = "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up"

    metrics_df, hist_metrics = compute_performance_table(
        dates_bt,
        wealth_bt,
        portfolio_path,
        costs_bt,
        weights_path,
        initial_wealth=100.0,
        annual_rf_rate=0.025,
        dt_backtest=1/252,
        output_folder=output_folder,
        tickers=tickers,
        arithmetic_returns_eqw=arithmetic_returns_eqw,
        sharpe_eqw=sharpe_eqw,
        dates_eqw=dates_eqw,
        wealth_eqw=wealth_eqw
    )

    print(metrics_df)

    plot_backtest_results(
        dates_bt,
        wealth_bt,
        portfolio_path,
        costs_bt,
        weights_path,
        output_dir="/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up",
        tickers=tickers
    )

    # Define benchmark kwargs
    benchmark_kwargs = dict(
        initial_wealth=100.0,
        annual_rf_rate=0.025,
        window=252,
        w=5,
        lambda_tc=0.0003,
        use_transaction_costs=True
    )

    # Run benchmark on synthetic paths
    bootstrap_results = run_benchmark_on_synthetic_paths(
        synthetic_folder="/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up",
        tickers=tickers,
        n_trials=100,
        benchmark_func=backtest_multi_period_strategy,
        benchmark_kwargs=benchmark_kwargs,
        save_results=True,
        benchmark_prefix="multi_period_path",
        start_date_bt="2015-01-01",
    )

    n_trials = 100
    tickers_str = "_".join(tickers)

    # Load in B&H bootstrap results
    bootstrap_benchmark_results = []
    for i in range(n_trials):
        filename = os.path.join(
            output_folder,
            f"{tickers_str}_parametric_bootstrap_bench_{i}.npz"
        )
        data = np.load(filename, allow_pickle=True)
        
        dates = pd.to_datetime(data["dates"])
        wealth = data["wealth"]
        portfolio = data["portfolio"]
        costs = data["costs"]
        weights = data["weights"]
        
        result = {
            "wealth": wealth,
            "theta": portfolio,
            "costs": costs,
            "weights": weights,
        }
        bootstrap_benchmark_results.append(result)

    # Prepare list of bootstrap result dicts
    bootstrap_results_list = []
    for i in range(bootstrap_results["wealth_paths"].shape[0]):
        result = {
            "wealth": bootstrap_results["wealth_paths"][i],
            "theta": bootstrap_results["portfolio_paths"][i],
            "costs": bootstrap_results["costs_paths"][i],
            "weights": bootstrap_results["weights_paths"][i],
        }
        bootstrap_results_list.append(result)

    output_path = os.path.join(output_folder, f"{tickers_str}_multi_bootstrap_results.pkl")

    bootstrap_save_data = {
        "results": bootstrap_results_list,
        "dates": bootstrap_results["dates"],  # common dates array from function output
    }

    with open(output_path, "wb") as f:
        pickle.dump(bootstrap_save_data, f)

    # Now pass the list
    df_bootstrap = compute_bootstrap_comparison_table_from_results(
        hist_metrics=hist_metrics,
        bootstrap_results=bootstrap_results_list,
        initial_wealth=100.0,
        annual_rf_rate=0.025,
        dt_backtest=1/252,
        output_folder=output_folder,
        tickers=tickers
    )

    df_stats = compute_bootstrap_stat_comparison_table(
        bootstrap_results_deepbsde=bootstrap_results_list,
        bootstrap_results_benchmark=bootstrap_benchmark_results,
        initial_wealth=100.0,
        annual_rf_rate=0.025,
        dt_backtest=1/252,
        output_folder=output_folder,
        tickers=tickers
    )

    plot_bootstrap_results(
        dates=bootstrap_results["dates"],
        bootstrap_results=bootstrap_results_list,
        output_dir=output_folder,
        tickers=tickers,
        initial_wealth=100.0,
        annual_rf_rate=0.025,
        dt_backtest=1/252,
    )

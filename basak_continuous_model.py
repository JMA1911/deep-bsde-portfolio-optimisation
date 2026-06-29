# Data loading and utility functions
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
from tqdm import trange
from tqdm import tqdm
from scipy.stats import norm
import os
import time
from joblib import Parallel, delayed, parallel_backend
import pickle

torch.manual_seed(0)
np.random.seed(0)

output_folder = "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up"

# Function to obtain market data   
def download_arithmetic_returns(tickers, start_date, end_date):
    """Downloads asset price data and computes arithmetic returns."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    adj_close = data.xs('Close', level='Price', axis=1)
    arithmetic_returns = adj_close.pct_change().dropna()
    return arithmetic_returns, adj_close

# Function to estimate model parameters based on historical data
def estimate_parameters(arithmetic_returns):
    """Estimates parameters for asset price simulation."""
    mu = arithmetic_returns.mean().values * 252
    sigma = arithmetic_returns.std().values * np.sqrt(252)
    corr_matrix = arithmetic_returns.corr().values
    return mu, sigma, corr_matrix

def download_state_variable(start_date, end_date):
    """
    Downloads adjusted close prices for HYG and LQD from Yahoo Finance.
    Computes and returns the credit spread (HYG - LQD) as a pd.Series.
    """
    symbols = ["HYG", "LQD"]
    data = yf.download(symbols, start=start_date, end=end_date, progress=False)["Close"]
    
    # Drop rows with any missing values
    data = data.dropna()
    
    # Compute spread
    credit_spread = data["HYG"] - data["LQD"]
    credit_spread.name = "CreditSpread"
    
    return credit_spread

def estimate_state_variable_params(credit_spread: pd.Series,
                                   arithmetic_returns: pd.DataFrame):
    """
    Estimates ABM parameters (drift and volatility) for the credit spread,
    and correlation with each asset's arithmetic return.
    """
    # First difference of credit spread (ABM increments)
    dX = credit_spread.diff().dropna()

    # Align with asset returns
    aligned = pd.concat([dX, arithmetic_returns], axis=1).dropna()
    dX_aligned = aligned.iloc[:, 0]
    R_aligned = aligned.iloc[:, 1:]  # Each column is a different asset

    # Estimate parameters
    m = dX_aligned.mean()
    nu = dX_aligned.std()
    rho = R_aligned.corrwith(dX_aligned).values  # Vector of correlations

    return float(m), float(nu), rho

# Set up your assets and dates
tickers = ['AAPL', 'JPM', 'XOM']
start_date = '2014-01-01'
end_date = '2025-01-01'

# Download full log returns and adjusted close
arithmetic_returns, adj_close = download_arithmetic_returns(tickers, start_date, end_date)

# Download full state variable (e.g. credit_spread)
credit_spread_series = download_state_variable(start_date, end_date)

# Align data
common_index = arithmetic_returns.index.intersection(credit_spread_series.index)
arithmetic_returns = arithmetic_returns.loc[common_index]
credit_spread_series = credit_spread_series.loc[common_index]

# Estimate parameters on the full dataset
mu, sigma, corr = estimate_parameters(arithmetic_returns)
m, nu, rho = estimate_state_variable_params(credit_spread_series, arithmetic_returns)

# Set initial values (use first date)
X0 = credit_spread_series.iloc[0]
S0 = adj_close.loc[common_index[0]].values

def simulate_basak_paths_real_data(S0, X0, arithmetic_returns, credit_spread_series, r, T, N, M, window_days=252):
    """
    Simulate (S_t, X_t) paths using real data for mu, sigma, m, nu, and rho 
    (all estimated via rolling window), and store the Brownian increments used.
    """
    dt = T / N
    D = len(S0)
    K = 1  # Single state variable

    S_paths = np.zeros((M, N + 1, D))
    X_paths = np.zeros((M, N + 1, K))
    mu_paths = np.zeros((M, N, D))
    sigma_paths = np.zeros((M, N, D))
    dW_S_all = np.zeros((M, N, D))
    dW_X_all = np.zeros((M, N, K))

    S_paths[:, 0, :] = S0
    X_paths[:, 0, :] = np.array([X0])

    mu_list, sigma_list, corr_list, m_list, nu_list, rho_list = [], [], [], [], [], []

    for i in range(N):
        window_returns = arithmetic_returns.iloc[i:i + window_days]
        window_credit_spread = credit_spread_series.iloc[i:i + window_days]

        mu, sigma, corr = estimate_parameters(window_returns)
        m, nu, rho = estimate_state_variable_params(window_credit_spread, window_returns)

        mu_list.append(mu)
        sigma_list.append(sigma)
        corr_list.append(corr)
        m_list.append(m)
        nu_list.append(nu)
        rho_list.append(rho)

    mu_array = np.array(mu_list)
    sigma_array = np.array(sigma_list)
    corr_array = np.array(corr_list)
    m_array = np.array(m_list)
    nu_array = np.array(nu_list)
    nu_array = np.repeat(nu_array[np.newaxis, :], M, axis=0)
    rho_array = np.array(rho_list)

    for path in range(M):
        S = np.zeros((N + 1, D))
        X = np.zeros((N + 1, K))
        mu_t = np.zeros((N, D))
        sigma_t = np.zeros((N, D))

        S[0] = S0
        X[0] = X0

        for i in range(N):
            mu_t[i] = mu_array[i]
            sigma_t[i] = sigma_array[i]
            corr = corr_array[i]
            m = m_array[i]
            nu = nu_array[path, i]
            rho = rho_array[i]  # shape (D,)

            # Cholesky of correlation matrix (guaranteed PD)
            L = np.linalg.cholesky(corr)

            # Asset Brownian increments
            Z = np.random.normal(size=D)
            dW_S = np.sqrt(dt) * (L @ Z)  # (D,)

            # State Brownian increment (correlated with dW_S)
            xi = np.random.normal()
            dW_X = np.dot(rho, dW_S) + np.sqrt(dt) * xi  # scalar

            # Simulate asset path
            S[i + 1] = S[i] + r * S[i] * dt + sigma_t[i] * S[i] * dW_S

            # Simulate state path
            X[i + 1] = X[i] + m * dt + nu * dW_X

            dW_S_all[path, i, :] = dW_S
            dW_X_all[path, i, 0] = dW_X

        S_paths[path] = S
        X_paths[path] = X
        mu_paths[path] = mu_t
        sigma_paths[path] = sigma_t

    return (
        S_paths, X_paths, mu_paths, sigma_paths,
        rho_array, nu_array, m_array,
        dW_S_all, dW_X_all
    )

def estimate_f_timevarying(mu_paths, sigma_paths, r, gamma, dt):
    """
    Compute f_t for each path and time step using time-varying mu and sigma estimated from real data.
    """
    M, N, D = mu_paths.shape
    f_vals = np.zeros((M, N))

    for m in range(M):
        for i in range(N):
            gain = 0.0
            for j in range(i, N):
                mu_j = mu_paths[m, j]      # shape (D,)
                sigma_diag = sigma_paths[m, j]  # shape (D,)
                sigma_mat = np.diag(sigma_diag)
                cov = sigma_mat @ sigma_mat.T
                cov_inv = np.linalg.inv(cov)
                excess_mu = mu_j - r       # shape (D,)
                integrand = (excess_mu.T @ cov_inv @ excess_mu) / gamma
                gain += integrand * dt
            f_vals[m, i] = gain

    return f_vals

# https://onlinelibrary.wiley.com/doi/pdf/10.1111/1540-6261.00529 Paper for derivative computation using Malliavian calculus
def compute_gradients_malliavin(
    S_paths, X_paths, f_vals, dW_S_all, dW_X_all, sigma_paths, nu_paths, dt,
    clip_val=10.0,
    avg_over_paths=30
):
    """
    Estimate ∇_S f and ∇_X f using Malliavin weights (likelihood ratio method),
    with clipping and averaging over multiple paths.
    """
    M, N_plus_1, D = S_paths.shape
    N = N_plus_1 - 1

    grad_S_all = np.zeros((M, N, D))
    grad_X_all = np.zeros((M, N, 1))

    for m in range(M):
        for t in range(N):
            f_t = f_vals[m, t]

            for d in range(D):
                S_t = S_paths[m, t, d]
                sigma_t = sigma_paths[m, t, d]
                dW_S = dW_S_all[m, t, d]

                weight = dW_S / (sigma_t * S_t * dt)
                raw_grad = f_t * weight
                grad_S_all[m, t, d] = np.clip(raw_grad, -clip_val, clip_val)

            # State var (1D)
            nu_t = nu_paths[m, t]
            dW_X = dW_X_all[m, t, 0]
            weight_X = dW_X / (nu_t * dt)
            raw_grad_X = f_t * weight_X
            grad_X_all[m, t, 0] = np.clip(raw_grad_X, -clip_val, clip_val)

    # Average over first K paths
    K = min(avg_over_paths, M)
    grad_S = grad_S_all[:K].mean(axis=0)
    grad_X = grad_X_all[:K].mean(axis=0)

    return grad_S, grad_X

def compute_theta_path(mu_path, sigma_path, grad_S, grad_X, r, gamma, rho_path, nu_path, S_path, dt, T):
    """
    Computes θₜ* for each time step in a given path using Basak (2010) formula,
    with time-varying rho and nu.
    """
    N, D = mu_path.shape
    theta_star = np.zeros((N, D))

    for t in range(N):
        mu_t = mu_path[t]
        sigma_diag = sigma_path[t]
        sigma_mat = np.diag(sigma_diag)
        cov = sigma_mat @ sigma_mat.T
        cov_inv = np.linalg.inv(cov)

        excess_mu = mu_t - r
        exp_decay = np.exp(-r * (T - t * dt))

        # Term 1: myopic component
        term1 = (1 / gamma) * (cov_inv @ excess_mu) * exp_decay

        # Hedging term
        S_t = S_path[t]  # shape (D,) # Think this need to be altered to use actually market data rather than a simulated path?w
        I_S_t = np.diag(S_t)

        grad_S_t = np.squeeze(grad_S[t])                       # shape (D,)
        grad_X_t = float(np.squeeze(grad_X[t]))                # force to scalar

        rho_t = rho_path[t]                        # shape (D,)
        nu_t = nu_path[t]                          # scalar
        nu_rho_sigma_inv = (nu_t * rho_t) / sigma_diag  # shape (D,)

        hedge_term = (I_S_t @ grad_S_t + nu_rho_sigma_inv * grad_X_t) * exp_decay

        # Final θ*
        theta_star[t] = term1 - hedge_term         # shape (D,)

    return theta_star

# Select path index
path_idx = 0

def plot_theta(theta_star, asset_names=None):
    """
    Plots the time series of each component of θₜ*.
    """
    N, D = theta_star.shape
    time = np.arange(N)

    plt.figure(figsize=(10, 6))
    for d in range(D):
        label = asset_names[d] if asset_names else f"Asset {d+1}"
        plt.plot(time, theta_star[:, d], label=label)

    plt.title("Optimal Portfolio Weights θₜ* over Time")
    plt.xlabel("Time Step")
    plt.ylabel("θₜ* (Dollar Investment)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def rolling_basak_backtest(
    adj_close,
    credit_spread_series,
    arithmetic_returns,
    initial_wealth=100.0,
    r=0.025,
    gamma=3.0,
    lambda_tc=0.1,
    window_days=252,
    T=1.0,
    N=20,
    M=1000,
    dt_backtest=1/252,
    use_transaction_costs=True
):
    """
    Rolling backtest of the Basak (2010) strategy with real data.
    Uses only Malliavin gradients (no interpolation).

    Tracks daily wealth even between rebalances.
    """
    dt = T / N
    D = adj_close.shape[1]

    rebalance_indices = []
    start_idx = window_days
    step_size = int(dt_backtest * 252)

    while start_idx < len(adj_close) - N:
        rebalance_indices.append(start_idx)
        start_idx += step_size

    wealth_bt = []
    portfolio_path = []
    weights_path = []
    dates_bt = []
    costs_bt = []

    current_wealth = initial_wealth
    prev_theta = np.zeros(D)

    # Save initial point
    initial_idx = rebalance_indices[0]
    dates_bt.append(adj_close.index[initial_idx])
    wealth_bt.append(current_wealth)
    portfolio_path.append(prev_theta)
    weights_path.append(np.append(np.zeros(D), 1.0))

    for idx in rebalance_indices:
        try:
            window_returns = arithmetic_returns.iloc[idx - window_days:idx]
            window_credit_spread = credit_spread_series.iloc[idx - window_days:idx]

            if len(window_returns) < window_days or len(window_credit_spread) < window_days:
                print(f"[Warning] Skipping index {idx} because rolling window is too short.")
                theta_0 = prev_theta
                continue

            S0 = adj_close.iloc[idx].values
            X0 = credit_spread_series.iloc[idx]

            # Simulate paths
            S_paths, X_paths, mu_paths, sigma_paths, rho_paths, nu_paths, _, dW_S, dW_X = simulate_basak_paths_real_data(
                S0=S0,
                X0=X0,
                arithmetic_returns=window_returns,
                credit_spread_series=window_credit_spread,
                r=r,
                T=T,
                N=N,
                M=M,
                window_days=window_days
            )

            f_vals = estimate_f_timevarying(mu_paths, sigma_paths, r=r, gamma=gamma, dt=dt)

            # --- Use Malliavin gradients only
            grad_S, grad_X = compute_gradients_malliavin(
                S_paths, X_paths, f_vals,
                dW_S, dW_X,
                sigma_paths, nu_paths,
                dt
            )

            real_S_path = adj_close.iloc[idx : idx + N].values
            theta_star = compute_theta_path(
                mu_path=mu_paths[0],
                sigma_path=sigma_paths[0],
                grad_S=grad_S,
                grad_X=grad_X,
                r=r, gamma=gamma,
                rho_path=rho_paths,
                nu_path=nu_paths[0],
                S_path=real_S_path,
                dt=dt, T=T
            )

            theta_0 = theta_star[0]

        except Exception as e:
            print(f"[Warning] Skipping index {idx} due to error: {repr(e)}")
            theta_0 = prev_theta
            continue

        # Compute risky exposures
        S_t = adj_close.iloc[idx].values
        S_t_safe = np.where(S_t == 0, 1e-8, S_t)
        risky_values = theta_0
        shares_t = risky_values / S_t_safe

        # Cash at rebalance
        risky_value_sum = np.sum(risky_values)
        cash_t = current_wealth - risky_value_sum

        # Transaction costs
        delta_theta = theta_0 - prev_theta
        cost = (
            lambda_tc * np.sum(delta_theta ** 2)
            if use_transaction_costs else 0.0
        )
        current_wealth -= cost
        costs_bt.append(cost)

        prev_theta = theta_0

        # Determine next rebalance
        next_idx = idx + int(dt_backtest * 252)
        if next_idx >= len(adj_close):
            next_idx = len(adj_close) - 1

        days_in_period = next_idx - idx
        price_window = adj_close.iloc[idx : idx + days_in_period + 1].values
        dates_window = adj_close.index[idx : idx + days_in_period + 1]

        S_t_safe = price_window[0, :]

        for d in range(days_in_period):
            S_t_next = price_window[d + 1, :]
            S_t_next_safe = np.where(S_t_next == 0, 1e-8, S_t_next)

            risky_value_tplus1 = np.sum(shares_t * S_t_next_safe)
            cash_tplus1 = cash_t * (1 + r * (1/252))

            current_wealth = risky_value_tplus1 + cash_tplus1
            current_wealth = max(current_wealth, 1.0)

            # Compute weights
            risky_weights_t = (
                (shares_t * S_t_next_safe) / current_wealth
                if current_wealth > 0 else np.zeros_like(theta_0)
            )
            cash_weight_t = cash_tplus1 / current_wealth if current_wealth > 0 else 1.0
            weights_full = np.append(risky_weights_t, cash_weight_t)

            dates_bt.append(dates_window[d + 1])
            wealth_bt.append(current_wealth)
            portfolio_path.append(theta_0)
            weights_path.append(weights_full)

            # Step forward
            cash_t = cash_tplus1
            S_t_safe = S_t_next_safe

    dates_bt = pd.DatetimeIndex(dates_bt)

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
    dt_backtest=1/252,
    output_folder=None,
    tickers=None,
    latex_filename=None,
    arithmetic_returns_eqw_matched=None,
    sharpe_eqw=None,
    dates_eqw=None,
    wealth_eqw=None,
):
    """
    Computes portfolio performance metrics and exports them as a LaTeX table.
    """
    if latex_filename is None:
        if tickers is None:
            file_stem = "cont_perf_table"
        else:
            file_stem = "_".join(tickers) + "_cont_perf_table"
        latex_filename = file_stem + ".tex"
    if output_folder is not None:
        latex_filename = os.path.join(output_folder, latex_filename)

    dates_bt = pd.to_datetime(dates_bt).tz_localize(None)
    dates_eqw = pd.DatetimeIndex(dates_eqw).tz_localize(None)

    if wealth_eqw is not None and dates_eqw is not None:
        common_dates = np.intersect1d(dates_bt.values, dates_eqw)

        if len(common_dates) < 10:
            print(f"[Warning] Only {len(common_dates)} overlapping dates between model and EQW benchmark.")
            arithmetic_returns_eqw_matched = None
            arithmetic_returns_model = wealth_bt[1:] / wealth_bt[:-1] - 1
        else:
            mask_model = np.isin(dates_bt, common_dates)
            mask_eqw = np.isin(dates_eqw, common_dates)

            wealth_bt_common = wealth_bt[mask_model]
            wealth_eqw_common = wealth_eqw[mask_eqw]

            arithmetic_returns_model = wealth_bt_common[1:] / wealth_bt_common[:-1] - 1
            arithmetic_returns_eqw_matched = wealth_eqw_common[1:] / wealth_eqw_common[:-1] - 1

            min_len = min(len(arithmetic_returns_model), len(arithmetic_returns_eqw_matched))
            arithmetic_returns_model = arithmetic_returns_model[:min_len]
            arithmetic_returns_eqw_matched = arithmetic_returns_eqw_matched[:min_len]
    else:
        arithmetic_returns_eqw_matched = None
        arithmetic_returns_model = wealth_bt[1:] / wealth_bt[:-1] - 1

    T = len(arithmetic_returns_model)
    if T <= 1:
        raise ValueError("Not enough time steps to compute statistics.")

    freq_per_year = 1 / dt_backtest

    cagr = (wealth_bt[-1] / initial_wealth)**(freq_per_year / T) - 1
    ann_vol = np.std(arithmetic_returns_model) * np.sqrt(freq_per_year)
    ann_excess_return = cagr - annual_rf_rate
    sharpe = ann_excess_return / ann_vol if ann_vol > 0 else np.nan

    benchmark_sharpe = 0.0
    if T > 1 and np.isfinite(sharpe):
        se_sharpe = np.sqrt((1 + sharpe**2 / 2) / (T - 1))
        psr_rf = norm.cdf((sharpe - benchmark_sharpe) / se_sharpe)
    else:
        psr_rf = np.nan

    if sharpe_eqw is not None and np.isfinite(sharpe_eqw) and np.isfinite(sharpe):
        se_diff = np.sqrt(
            (1 + sharpe**2 / 2) / (T - 1) +
            (1 + sharpe_eqw**2 / 2) / (T - 1)
        )
        psr_eqw = norm.cdf((sharpe - sharpe_eqw) / se_diff)
    else:
        psr_eqw = np.nan

    if sharpe_eqw is not None and np.isfinite(sharpe_eqw) and np.isfinite(sharpe):
        diff_sharpe = sharpe - sharpe_eqw
        se_diff = np.sqrt(
            (1 + sharpe**2 / 2) / (T - 1) +
            (1 + sharpe_eqw**2 / 2) / (T - 1)
        )
        t_stat_lw = diff_sharpe / se_diff
        p_value_lw = 1 - norm.cdf(t_stat_lw)
        p_value_lw = np.clip(p_value_lw, 0, 1)
    else:
        t_stat_lw = np.nan
        p_value_lw = np.nan

    if arithmetic_returns_eqw_matched is not None and len(arithmetic_returns_eqw_matched) == len(arithmetic_returns_model):
        diff_returns = arithmetic_returns_model - arithmetic_returns_eqw_matched
        diff_mean = np.mean(diff_returns)
        diff_std = np.std(diff_returns, ddof=1)
        t_stat_ret = np.nan
        p_value_ret = np.nan
        if diff_std > 0:
            t_stat_ret = diff_mean / (diff_std / np.sqrt(T))
            p_value_ret = 1 - norm.cdf(t_stat_ret)
            p_value_ret = np.clip(p_value_ret, 0, 1)
    else:
        t_stat_ret = np.nan
        p_value_ret = np.nan

    downside_returns = arithmetic_returns_model[arithmetic_returns_model < 0]
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
    if len(arithmetic_returns_model) > 0:
        var_95 = -np.percentile(arithmetic_returns_model, 100 * (1 - confidence_level))
        es_95 = -np.mean(arithmetic_returns_model[arithmetic_returns_model <= -var_95]) if np.any(arithmetic_returns_model <= -var_95) else np.nan
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
        "Terminal Wealth (\\pounds)",
        "CAGR (\\%)",
        "Annual Volatility (\\%)",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Maximum Drawdown (\\%)",
        "Average Drawdown (\\%)",
        "Average Recovery Time (days)",
        "VaR 95\\% (\\%)",
        "Expected Shortfall 95\\% (\\%)",
        "PSR vs Risk-Free (\\%)",
        "PSR vs B\\&H (\\%)",
        "Sharpe Difference t-stat",
        "Sharpe Difference p-value",
        "Returns Difference t-stat",
        "Returns Difference p-value",
        "Average Leverage",
        "Maximum Position Size (\\pounds)",
        "Average Turnover per Step (\\%)",
        "Annual Turnover (\\%)",
        "Profit per Turnover (\\pounds)",
        "Standard Deviation of Portfolio Weights (\\%)",
        "Average Absolute Change in Portfolio Weights (\\%)",
        "No-Trade Fraction (\\%)",
        "Total Transaction Costs (\\pounds)",
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
        f"\\pounds{wealth_bt[-1]:,.2f}",
        f"{cagr * 100:.2f}\\%",
        f"{ann_vol * 100:.2f}\\%",
        f"{sharpe:.2f}",
        f"{sortino:.2f}" if np.isfinite(sortino) else "N/A",
        f"{calmar:.2f}" if np.isfinite(calmar) else "N/A",
        f"{max_dd * 100:.2f}\\%",
        f"{avg_drawdown * 100:.2f}\\%",
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
        f"{avg_turnover_step * 100:.2f}\\%",
        f"{annual_turnover * 100:.2f}\\%",
        f"\\pounds{pnl_per_turnover:,.2f}" if np.isfinite(pnl_per_turnover) else "N/A",
        f"{std_weights_mean:.2f}\\%" if np.isfinite(std_weights_mean) else "N/A",
        f"{abs_changes:.2f}\\%" if np.isfinite(abs_changes) else "N/A",
        f"{no_trade_fraction * 100:.2f}\\%" if np.isfinite(no_trade_fraction) else "N/A",
        f"\\pounds{total_costs:,.2f}",
    ]

    hist_metrics = dict(zip(metrics_names, metrics_values_raw))

    df_metrics = pd.DataFrame({
        "Metric": metrics_names,
        "Value": metrics_values
    })

    tabular_code = df_metrics.to_latex(
        index=False,
        escape=False,
        column_format="ll"
    )

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
    Generates and saves standard backtest plots (wealth, drawdown, returns, Sharpe, weights, turnover, costs).
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

    single_color = "#1f77b4"

    default_colors = [
        "#E66100",
        "#5E3C99",
        "#1B9E77",
        "#E7298A",
        "#A6A600",
        "#1F78B4",
        "#333333",
    ]

    returns = wealth_bt[1:] / wealth_bt[:-1] - 1

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
    plt.savefig(file_path("con_wealth_over_time"))
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
    plt.savefig(file_path("con_drawdown_over_time"))
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
    plt.savefig(file_path("con_histogram_returns"))
    plt.close()

    # ----------------------------------------
    # Plot 4: Rolling Sharpe Ratio
    # ----------------------------------------
    window = 63

    ret_series = pd.Series(returns)
    roll_mean_daily = ret_series.rolling(window).mean()
    roll_std_daily = ret_series.rolling(window).std()

    roll_mean_ann = roll_mean_daily * 252
    roll_std_ann = roll_std_daily * np.sqrt(252)

    annual_rf_rate = 0.02
    roll_sharpe = (roll_mean_ann - annual_rf_rate) / roll_std_ann

    dates_for_rolling = dates_bt[1:][window - 1 :]

    plt.figure(figsize=(8, 4))
    plt.plot(dates_for_rolling, roll_sharpe.iloc[window - 1 :], color=single_color, linewidth=1.5)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe Ratio")
    plt.title(f"Rolling Sharpe Ratio ({window}-day)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("con_rolling_sharpe"))
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
    plt.savefig(file_path("con_weights_line_plot"))
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
    plt.savefig(file_path("con_turnover_over_time"))
    plt.close()

    # ----------------------------------------
    # Plot 7: Cumulative Transaction Costs
    # ----------------------------------------
    cumulative_costs = np.cumsum(costs_bt)
    cumulative_costs_padded = np.insert(cumulative_costs, 0, 0.0)

    cumulative_costs_daily = np.zeros(len(dates_bt))

    n_cost_points = len(cumulative_costs_padded)
    rebalance_indices = np.linspace(
        0, len(dates_bt)-1, num=n_cost_points, dtype=int
    )

    for i in range(len(rebalance_indices)):
        start = rebalance_indices[i]
        end = rebalance_indices[i+1] if i+1 < len(rebalance_indices) else len(dates_bt)
        cumulative_costs_daily[start:end] = cumulative_costs_padded[i]

    plt.figure(figsize=(8, 4))
    plt.plot(dates_bt, cumulative_costs_daily, color=single_color, linewidth=1.8)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Costs (£)")
    plt.title("Cumulative Transaction Costs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path("con_cumulative_transaction_costs"))
    plt.close()

    print(f"All plots saved in: {output_dir}")

# Loading in B&H historical backtest results
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
ann_excess_eqw = cagr_eqw - 0.02
sharpe_eqw = ann_excess_eqw / ann_vol_eqw if ann_vol_eqw > 0 else np.nan

run_historical = False

if run_historical:
    start_time = time.time()
    dates_bt, wealth_bt, theta_path, costs_bt, weights_path = rolling_basak_backtest(
        adj_close=adj_close,
        credit_spread_series=credit_spread_series,
        arithmetic_returns=arithmetic_returns,
        initial_wealth=100.0,
        r=0.025,
        gamma=3.0,
        lambda_tc=0.0003,
        window_days=252,
        T=1.0,
        N=30,
        M=30,
        dt_backtest=1/252,
        use_transaction_costs=True,
    )

    np.savez(
        "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up/basak_backtest_daily.npz",
        dates=dates_bt,
        wealth=wealth_bt,
        theta=theta_path,
        costs=costs_bt,
        weights=weights_path,
    )

    output_folder = "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up"

    # Compute metrics table
    metrics_df, hist_metrics_cont = compute_performance_table(
        dates_bt,
        wealth_bt,
        theta_path,
        costs_bt,
        weights_path,
        initial_wealth=100.0,
        annual_rf_rate=0.025,
        dt_backtest=1/252,
        output_folder=output_folder,
        tickers=tickers,
        arithmetic_returns_eqw_matched=arithmetic_returns_eqw,
        sharpe_eqw=sharpe_eqw,
        dates_eqw=dates_eqw,
        wealth_eqw=wealth_eqw
    )

    print(metrics_df)

    plot_backtest_results(
            dates_bt,
            wealth_bt,
            theta_path,
            costs_bt,
            weights_path,
            output_dir="/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up",
            tickers=tickers
        )

    end_time = time.time()  # ⏱️ End timing
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

run_bootstrap = True

def run_single_benchmark_trial(i, synthetic_folder, tickers, benchmark_func, benchmark_kwargs, benchmark_prefix, start_date_bt, save_results):
    """Run the historical backtest backtest algorithm on a single synthetic path."""
    tickers_str = "_".join(tickers)
    path_file = os.path.join(synthetic_folder, f"{tickers_str}_rolling_bootstrap_path_{i}.npz")

    data = np.load(path_file, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    asset_names = list(data["tickers"])

    adj_close_synth_full = pd.DataFrame(data["prices"], index=dates, columns=asset_names)
    credit_spread_synth_full = pd.Series(data["cs"], index=dates)
    arithmetic_returns = adj_close_synth_full.pct_change().dropna()

    if benchmark_kwargs is None:
        benchmark_kwargs = {}

    result = benchmark_func(
        adj_close=adj_close_synth_full,
        credit_spread_series=credit_spread_synth_full,
        arithmetic_returns=arithmetic_returns,
        **benchmark_kwargs
    )

    dates_bench, wealth_bench, portfolio_bench, costs_bench, weights_bench = result
    dates_bench = np.array(dates_bench, dtype="datetime64[D]")

    desired_start = np.datetime64(start_date_bt)
    effective_start_date = max(desired_start, dates_bench[0])
    mask = dates_bench >= effective_start_date

    dates_bench = dates_bench[mask]
    wealth_bench = wealth_bench[mask]
    portfolio_bench = portfolio_bench[mask, :]
    costs_bench = costs_bench[mask[:-1]]
    weights_bench = weights_bench[mask, :]

    if save_results:
        np.savez(
            os.path.join(synthetic_folder, f"{tickers_str}_{benchmark_prefix}_{i}.npz"),
            dates=dates_bench,
            wealth=wealth_bench,
            portfolio=portfolio_bench,
            costs=costs_bench,
            weights=weights_bench
        )
    
    print(f"Running trial {i} in process {os.getpid()}")

    return dates_bench, wealth_bench, portfolio_bench, costs_bench, weights_bench

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

    tickers_str = "_".join(tickers)

    # Run all trials in parallel
    with parallel_backend('loky', n_jobs=-1):
        results = Parallel(n_jobs=-1)(
            delayed(run_single_benchmark_trial)(
                i,
                synthetic_folder,
                tickers,
                benchmark_func,
                benchmark_kwargs,
                benchmark_prefix,
                start_date_bt,
                save_results
            )
            for i in tqdm(range(n_trials), desc="Running bootstrap backtests")
        )

    # Unpack results
    dates_lists = []
    wealth_list = []
    portfolio_list = []
    costs_list = []
    weights_list = []

    for dates_bench, wealth_bench, portfolio_bench, costs_bench, weights_bench in results:
        dates_lists.append(dates_bench)
        wealth_list.append(wealth_bench)
        portfolio_list.append(portfolio_bench)
        costs_list.append(costs_bench)
        weights_list.append(weights_bench)

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

data_hist = np.load(
    "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up/basak_backtest_daily.npz",
    allow_pickle=True
)

dates_hist = pd.to_datetime(data_hist["dates"])
wealth_hist = data_hist["wealth"]
theta_hist = data_hist["theta"]
costs_hist = data_hist["costs"]
weights_hist = data_hist["weights"]

# Compute historical metrics
_, hist_metrics = compute_performance_table(
    dates_hist,
    wealth_hist,
    theta_hist,
    costs_hist,
    weights_hist,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    dt_backtest=1/252,
    output_folder=output_folder,
    tickers=tickers,
    arithmetic_returns_eqw_matched=arithmetic_returns_eqw,
    sharpe_eqw=sharpe_eqw,
    dates_eqw=dates_eqw,
    wealth_eqw=wealth_eqw,
)

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
    Compute a comparison table of historical vs bootstrap metrics.
    Uses arithmetic returns and includes new metrics:
    Max Drawdown, VaR, ES, and Avg Abs Change in Portfolio Weights.
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
        "Average Absolute Change in Portfolio Weights (\\%)": [],
        "Annual Turnover (\\%)": [],
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
        returns = wealth_bt[1:] / wealth_bt[:-1] - 1

        # Metrics
        cagr = (wealth_bt[-1] / initial_wealth)**(freq_per_year / T) - 1
        ann_vol = np.std(returns) * np.sqrt(freq_per_year)
        ann_excess_return = cagr - annual_rf_rate
        sharpe = ann_excess_return / ann_vol if ann_vol > 0 else np.nan

        # Sortino
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(freq_per_year) if len(downside_returns) > 0 else np.nan
        sortino = ann_excess_return / downside_std if downside_std > 0 else np.nan

        # Max and average drawdown
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

        # VaR and ES at 95%
        var_95 = -np.percentile(returns, 5)
        es_95 = -np.mean(returns[returns <= -var_95]) if np.any(returns <= -var_95) else np.nan

        # Average abs change in weights
        weights_risky = weights_path[:, :-1]
        delta_w = np.diff(weights_risky, axis=0)
        abs_changes = np.mean(np.sum(np.abs(delta_w), axis=1)) * 100

        # Turnover
        delta_u = np.diff(portfolio_path, axis=0)
        turnover_per_step = np.sum(np.abs(delta_u), axis=1) / wealth_bt[:-1]
        avg_turnover_step = np.mean(turnover_per_step)
        annual_turnover = avg_turnover_step * freq_per_year

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

        # Format
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

    if latex_filename is None:
        if tickers is None:
            file_stem = "bootstrap_comparison_cont_table"
        else:
            file_stem = "_".join(tickers) + "_bootstrap_comparison_cont_table"
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
    Compute a comparison table of DeepBSDE bootstrap vs Benchmark bootstrap
    using arithmetic returns.
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

        # Use arithmetic returns
        returns_deep = wealth_deep[1:] / wealth_deep[:-1] - 1
        returns_bench = wealth_bench[1:] / wealth_bench[:-1] - 1

        # Compute CAGR
        cagr_deep = (wealth_deep[-1] / initial_wealth)**(freq_per_year / T) - 1
        cagr_bench = (wealth_bench[-1] / initial_wealth)**(freq_per_year / T) - 1

        # Compute Sharpe using arithmetic returns
        ann_vol_deep = np.std(returns_deep) * np.sqrt(freq_per_year)
        ann_vol_bench = np.std(returns_bench) * np.sqrt(freq_per_year)

        excess_return_deep = cagr_deep - annual_rf_rate
        excess_return_bench = cagr_bench - annual_rf_rate

        sharpe_deep = excess_return_deep / ann_vol_deep if ann_vol_deep > 0 else np.nan
        sharpe_bench = excess_return_bench / ann_vol_bench if ann_vol_bench > 0 else np.nan

        # Store differences
        diff_sharpe.append(sharpe_deep - sharpe_bench)
        diff_cagr.append(cagr_deep - cagr_bench)

    # Compute statistics
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

    rows = []

    rows.append({
        "Metric": "Sharpe Ratio Difference",
        "Mean Difference": f"{mean_diff_sharpe:.4f}",
        "t-stat": f"{t_sharpe:.2f}" if np.isfinite(t_sharpe) else "N/A",
        "p-value": f"{pval_sharpe:.4f}" if np.isfinite(pval_sharpe) else "N/A",
        "Probability $>$ Benchmark (\\%)": f"{prob_sharpe*100:.2f}\\%"
    })

    rows.append({
        "Metric": "Returns Difference (CAGR)",
        "Mean Difference": f"{mean_diff_cagr*100:.2f}\\%",
        "t-stat": f"{t_cagr:.2f}" if np.isfinite(t_cagr) else "N/A",
        "p-value": f"{pval_cagr:.4f}" if np.isfinite(pval_cagr) else "N/A",
        "Probability $>$ Benchmark (\\%)": f"{prob_cagr*100:.2f}\\%"
    })

    df = pd.DataFrame(rows)

    if latex_filename is None:
        file_stem = "_".join(tickers) + "_bootstrap_bench_vs_cont_table" if tickers else "bootstrap_bench_vs_cont_table"
        latex_filename = file_stem + ".tex"

    if output_folder is not None:
        latex_filename = os.path.join(output_folder, latex_filename)

    latex_code = df.to_latex(
        index=False,
        escape=False,
        column_format="lcccc"
    )

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
    Plot summary results from bootstrap backtests.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Ticker prefix for filenames
    if tickers is None:
        prefix = "bootstrap_cont"
    else:
        prefix = "_".join(tickers) + "_bootstrap_cont"

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
    # Histogram of Sharpe Ratios
    # -----------------------------------
    sharpe_ratios = []
    for result in bootstrap_results:
        wealth_bt = result["wealth"]
        arithmetic_returns = wealth_bt[1:] / wealth_bt[:-1] - 1
        T = len(wealth_bt) - 1
        if T < 1:
            sharpe_ratios.append(np.nan)
            continue
        cagr = (wealth_bt[-1] / initial_wealth) ** (freq_per_year / T) - 1
        ann_vol = np.std(arithmetic_returns) * np.sqrt(freq_per_year)
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

if run_bootstrap:
    start_time = time.time()
    output_folder = "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up"

    # Define benchmark kwargs
    benchmark_kwargs = dict(
        initial_wealth=100.0,
        r=0.025,
        gamma=3.0,
        lambda_tc=0.0003,
        T=1.0,
        N=30,
        M=30,
        dt_backtest=1/252,
        use_transaction_costs=True
    )

    # Run benchmark on synthetic paths
    bootstrap_results = run_benchmark_on_synthetic_paths(
        synthetic_folder="/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up",
        tickers=tickers,
        n_trials=100,
        benchmark_func=rolling_basak_backtest,
        benchmark_kwargs=benchmark_kwargs,
        save_results=True,
        benchmark_prefix="continuous_path",
        start_date_bt="2015-01-01",
    )

    n_trials = 100
    tickers_str = "_".join(tickers)

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

    output_path = os.path.join(output_folder, f"{tickers_str}_cont_bootstrap_results.pkl")

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

    end_time = time.time()  # ⏱️ End timing
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


output_folder = "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up"

# Reload saved bootstrap benchmark results from disk (B&H)
bootstrap_benchmark_results = []
for i in range(5):
    filename = os.path.join(
        output_folder,
        f"{'_'.join(tickers)}_parametric_bootstrap_bench_{i}.npz"
    )
    data = np.load(filename, allow_pickle=True)
    
    result = {
        "wealth": data["wealth"],
        "theta": data["portfolio"],
        "costs": data["costs"],
        "weights": data["weights"],
    }
    bootstrap_benchmark_results.append(result)

# Reload saved bootstrap results from disk
with open(os.path.join(output_folder, f"{'_'.join(tickers)}_cont_bootstrap_results.pkl"), "rb") as f:
    data = pickle.load(f)

bootstrap_results_list = data["results"]
dates = data["dates"]

# Load historical benchmark
data_hist = np.load(
    os.path.join(output_folder, "basak_backtest_daily.npz"),
    allow_pickle=True
)
dates_bt = pd.to_datetime(data_hist["dates"])
wealth_bt = data_hist["wealth"]
theta_path = data_hist["theta"]
costs_bt = data_hist["costs"]
weights_path = data_hist["weights"]

# Compute metrics for historical strategy
_, hist_metrics = compute_performance_table(
    dates_bt, wealth_bt, theta_path, costs_bt, weights_path,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    dt_backtest=1/252,
    output_folder=output_folder,
    tickers=tickers,
    arithmetic_returns_eqw_matched=arithmetic_returns_eqw,
    sharpe_eqw=sharpe_eqw,
    dates_eqw=dates_eqw,
    wealth_eqw=wealth_eqw,
)

# Now generate comparison tables and plots
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
    dates=dates,
    bootstrap_results=bootstrap_results_list,
    output_dir=output_folder,
    tickers=tickers,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    dt_backtest=1/252,
)

# 1. Data loading and utility functions
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
from tqdm import trange
import os
from scipy.stats import norm
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm 
import glob
import pickle

torch.manual_seed(0)
np.random.seed(0)

output_folder = "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up"

# =======================
# Normalisation Helpers
# =======================
def normalize_series(series):
    safe_std = max(series.std(), 1e-2)
    return (series - series.mean()) / safe_std

def normalize_tensor(tensor):
    if tensor.numel() <= 1:
        return tensor  # skip normalizing scalar inputs
    safe_std = max(tensor.std(), 1e-2)
    return (tensor - tensor.mean()) / safe_std

# =======================
# Base Network Class
# =======================
class BaseNetwork(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Could increase number of hidden layers later
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t, W, S, X):
        t = normalize_tensor(t if t.dim() == 2 else t.unsqueeze(1))
        W = normalize_tensor(W if W.dim() == 2 else W.unsqueeze(1))
        X = normalize_tensor(X if X.dim() == 2 else X.unsqueeze(1))
        S = normalize_tensor(S)
        x = torch.cat([t, W, S, X], dim=1)
        return self.model(x)

# =======================
# Network Specialisations
# =======================
class PolicyNetwork(BaseNetwork):
    pass

class ZNet(BaseNetwork):
    pass

class SecondDerivNet(BaseNetwork):
    pass

class Deep2BSDE:
    def __init__(self, mu, sigma, corr_matrix, m, nu, rho_vector, X0, S0, T=1.0, N=100, n_paths=1, use_transaction_costs=True, anchor_penalty_weight=10.0):
        # Simulation settings
        self.T = T
        self.N = N
        self.dt = T / N
        self.n_paths = n_paths
        self.n_assets = 3
        self.use_transaction_costs = use_transaction_costs

        # Model parameters
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)
        self.corr_matrix = torch.tensor(corr_matrix, dtype=torch.float32)
        self.cholesky = torch.linalg.cholesky(self.corr_matrix)

        self.r = 0.025
        self.lambda_tc = 1e-5
        self.gamma = 3.0
        self.anchor_penalty_weight = anchor_penalty_weight

        self.m = float(m.iloc[0]) if isinstance(m, pd.Series) else float(m)
        self.nu = float(nu.iloc[0]) if isinstance(nu, pd.Series) else float(nu)
        self.rho = np.array(rho_vector)
        self.X0 = float(X0.iloc[0]) if isinstance(X0, pd.Series) else float(X0)

        self.S0 = torch.tensor(S0, dtype=torch.float32)
        self.W0 = 100.0

        self.time_grid = np.linspace(0, T, N + 1)

        self.policy_net = PolicyNetwork(input_dim=6, hidden_dim=64)
        self.z_net = ZNet(input_dim=6, hidden_dim=64, output_dim=1)
        self.second_deriv_net = SecondDerivNet(input_dim=6, hidden_dim=64, output_dim=1)

        self.raw_theta_anchor = None

    def simulate_paths(self, batch_size=64, jitter_mu=0.005, jitter_sigma=0.002, jitter_m=0.001, jitter_nu=0.001, jitter_rho=0.02):
        """Simulates forward paths for assets, wealth, and state variable under the policy network."""

        device = torch.device("cpu")
        self.batch_size = batch_size

        W_paths = np.zeros((batch_size, self.N + 1))
        S_paths = np.zeros((batch_size, self.N + 1, self.n_assets))
        X_paths = np.zeros((batch_size, self.N + 1))
        theta_paths = np.zeros((batch_size, self.N + 1, self.n_assets))
        dW_paths = np.zeros((batch_size, self.N, self.n_assets)) 

        for path in range(batch_size):
            S = np.zeros((self.N + 1, self.n_assets))
            X = np.zeros(self.N + 1)
            W = np.zeros(self.N + 1)

            S[0], X[0], W[0] = self.S0, self.X0, self.W0
            theta_prev = None

            # Apply jitter to parameters per path
            mu = self.mu.cpu().numpy() + np.random.normal(0, jitter_mu, size=self.n_assets)
            sigma = self.sigma.cpu().numpy() + np.random.normal(0, jitter_sigma, size=self.n_assets)
            sigma = np.clip(sigma, 1e-4, 1.0)  # Prevent negative or too small vol

            m = self.m + np.random.normal(0, jitter_m)
            nu = self.nu + np.random.normal(0, jitter_nu)
            nu = max(1e-4, nu)  # Ensure positive

            rho_perturbed = self.rho + np.random.normal(0, jitter_rho, size=self.rho.shape)
            rho_perturbed = np.clip(rho_perturbed, -1.0, 1.0)

            for i in range(self.N):
                t_tensor = torch.tensor([[self.time_grid[i]]], dtype=torch.float32).to(device)
                W_tensor = torch.tensor([[max(1e-6, min(W[i], 1e6))]], dtype=torch.float32).to(device)
                S_tensor = torch.tensor(np.clip(S[i], 1e-6, 1e6), dtype=torch.float32).reshape(1, -1).to(device)
                X_tensor = torch.tensor([[max(1e-6, min(X[i], 1e6))]], dtype=torch.float32).to(device)

                raw_theta = self.policy_net(t_tensor, W_tensor, S_tensor, X_tensor)
                raw_theta = torch.nan_to_num(raw_theta, nan=0.0, posinf=1e6, neginf=-1e6)
                theta_tensor = raw_theta * W_tensor
                theta_np = theta_tensor.detach().cpu().numpy().flatten()

                Z = np.random.normal(0, 1, self.n_assets)
                dW_vector = np.sqrt(self.dt) * (self.cholesky.cpu().numpy() @ Z)

                S[i+1] = S[i] + mu * S[i] * self.dt + sigma * S[i] * dW_vector
                S[i+1] = np.clip(S[i+1], 1e-6, 1e6)

                dZ = np.random.normal(0, np.sqrt(self.dt))
                rho_unit = rho_perturbed / np.linalg.norm(rho_perturbed)
                orth_component = dW_vector - np.dot(dW_vector, rho_unit) * rho_unit
                dW_tilde = np.dot(rho_perturbed, dW_vector) + dZ * np.linalg.norm(orth_component)
                dW_tilde = np.nan_to_num(dW_tilde, nan=0.0)

                X[i+1] = X[i] + m * self.dt + nu * dW_tilde
                X[i+1] = np.clip(X[i+1], -1e-6, 0)
                # -70, -10

                drift = self.r * W[i] + np.dot(theta_np, (mu - self.r))
                diffusion = np.dot(theta_np * sigma, dW_vector)

                if self.use_transaction_costs:
                    if theta_prev is None:
                        transaction_cost = 0.0
                    else:
                        delta_theta = theta_np - theta_prev
                        transaction_cost = self.lambda_tc * np.sum(delta_theta ** 2) / self.dt
                    theta_prev = theta_np.copy()
                else:
                    transaction_cost = 0.0

                W[i+1] = W[i] + drift * self.dt + diffusion - transaction_cost
                W[i+1] = max(1e-6, min(W[i+1], 1e6))

                S_paths[path, i+1] = S[i+1]
                theta_paths[path, i] = theta_np
                dW_paths[path, i] = dW_vector

            theta_paths[path, -1] = theta_paths[path, -2]
            W_paths[path] = W
            X_paths[path] = X

        return W_paths, S_paths, X_paths, theta_paths, dW_paths

    def train(self, epochs=1000, batch_size=64, lr=1e-3):
        """Trains policy, Z, and Gamma networks using a stepwise backward BSDE scheme 
        (Huré DBDP1-style). The loss enforces local BSDE consistency by matching 
        the stepwise update of Y across all time steps, including drift, diffusion, 
        transaction cost penalty, and optional anchor penalty."""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.autograd.set_detect_anomaly(True)

        self.policy_net.to(device)
        self.z_net.to(device)
        self.second_deriv_net.to(device)

        optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) +
            list(self.second_deriv_net.parameters()) +
            list(self.z_net.parameters()),
            lr=lr
        )

        loss_history = []
        delta_theta_history = []

        # cache device-tensors for parameters used each step
        mu_vec = self.mu.to(device)              # (n_assets,)
        sigma_vec = self.sigma.to(device)        # (n_assets,)

        for epoch in trange(epochs):
            W_paths_np, S_paths_np, X_paths_np, theta_paths_np, dW_paths_np = self.simulate_paths(
                batch_size,
                jitter_mu=0.005, jitter_sigma=0.002, jitter_m=0.001, jitter_nu=0.001, jitter_rho=0.02
            )

            W_paths = torch.tensor(W_paths_np, dtype=torch.float32, device=device)
            S_paths = torch.tensor(S_paths_np, dtype=torch.float32, device=device)
            X_paths = torch.tensor(X_paths_np, dtype=torch.float32, device=device)
            dW_paths = torch.tensor(dW_paths_np, dtype=torch.float32, device=device)

            W_paths = torch.clamp(W_paths, min=1e-6, max=1e6)
            W_T = W_paths[:, -1:]
            Y = W_T - 0.5 * self.gamma * W_T**2
            Y = torch.clamp(Y, min=-1e6, max=1e6)

            loss_epoch = 0.0
            delta_theta_norms = []

            for i in reversed(range(self.N)):
                t_tensor = torch.full((batch_size, 1), self.time_grid[i], dtype=torch.float32, device=device)
                W_i = torch.clamp(W_paths[:, i:i+1], min=1e-6, max=1e6)     # (B,1)
                S_i = torch.clamp(S_paths[:, i], min=1e-6, max=1e6)         # (B,n_assets)
                X_i = torch.clamp(X_paths[:, i:i+1], min=1e-6, max=1e6)     # (B,1)
                dW_i = dW_paths[:, i]                                       # (B,n_assets)

                # Networks
                Z = self.z_net(t_tensor, W_i, S_i, X_i)                     # expect scalar per sample
                Gamma = self.second_deriv_net(t_tensor, W_i, S_i, X_i)      # expect scalar per sample
                raw_theta = self.policy_net(t_tensor, W_i, S_i, X_i)        # (B,n_assets)

                # sanitize numerics
                Z = torch.nan_to_num(Z, nan=0.0, posinf=1e6, neginf=-1e6).reshape(-1, 1)         # (B,1)
                Gamma = torch.nan_to_num(Gamma, nan=0.0, posinf=1e6, neginf=-1e6).reshape(-1, 1) # (B,1)
                raw_theta = torch.nan_to_num(raw_theta, nan=0.0, posinf=1e6, neginf=-1e6)        # (B,n_assets)

                # Scale policy to dollars
                theta = raw_theta * W_i                                                          # (B,n_assets)

                # carry previous raw theta for FD penalty
                if i == self.N - 1:
                    raw_theta_prev = raw_theta.detach().clone()

                # Drift of wealth: r W + theta · (mu - r)
                drift = self.r * W_i + torch.sum(theta * (mu_vec - self.r), dim=1, keepdim=True)  # (B,1)

                # FD transaction penalty in generator (units of per-time drift)
                delta_raw_theta = raw_theta - raw_theta_prev                                      # (B,n_assets)
                delta_theta_dollars = delta_raw_theta * W_i                                       # (B,n_assets)
                delta_theta_norm = torch.norm(delta_theta_dollars, dim=1).mean().item()
                delta_theta_norms.append(delta_theta_norm)

                if self.use_transaction_costs:
                    penalty = -self.lambda_tc * torch.sum(delta_theta_dollars ** 2, dim=1, keepdim=True) / (self.dt ** 2)
                else:
                    penalty = torch.zeros_like(drift)

                raw_theta_prev = raw_theta.detach().clone()

                # Optional anchor penalty (applied only at t=0)
                if i == 0 and self.raw_theta_anchor is not None:
                    anchor_tensor = self.raw_theta_anchor.clone().detach().to(dtype=torch.float32, device=device)
                    if anchor_tensor.shape != raw_theta.shape:
                        anchor_tensor = anchor_tensor.expand_as(raw_theta)
                    anchor_theta = anchor_tensor * W_i
                    anchor_penalty = self.anchor_penalty_weight * torch.sum((theta - anchor_theta) ** 2, dim=1, keepdim=True)
                else:
                    anchor_penalty = torch.zeros_like(drift)

                # Wealth diffusion: theta^T sigma  (scalar per sample)
                wealth_diffusion = torch.sum(theta * sigma_vec, dim=1, keepdim=True)              # (B,1)

                # Correct generator f:
                # f = (drift + penalty) * Z + 0.5 * (wealth_diffusion^2) * Gamma + anchor_penalty
                f_i = (drift + penalty) * Z + 0.5 * (wealth_diffusion ** 2) * Gamma + anchor_penalty  # (B,1)

                # Martingale increment: Z * (theta^T sigma dW)
                Z_dot_dW = Z * torch.sum((theta * sigma_vec) * dW_i, dim=1, keepdim=True)         # (B,1)

                # Stepwise update
                Y_pred = torch.clamp(Y - f_i * self.dt + Z_dot_dW, min=-1e6, max=1e6)

                loss = torch.mean((Y_pred - Y.detach()) ** 2)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss_epoch += loss.item()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                Y = Y_pred.detach()

            loss_history.append(loss_epoch / self.N)
            delta_theta_history.append(np.mean(delta_theta_norms))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Avg Step Loss: {loss_epoch / self.N:.6f}, Avg Δθ norm: {np.mean(delta_theta_norms):.6f}")

        return loss_history, W_paths, theta_paths_np


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

# Define tickers and dates
tickers = ['AAPL', 'JPM', 'XOM']
start_date = '2014-01-01'
end_date = '2025-01-01'
tickers_str = "_".join(tickers)

# Download and process data
arithmetic_returns, adj_close = download_arithmetic_returns(tickers, start_date, end_date)

# Use only training data to estimate parameters
arithmetic_returns_train = arithmetic_returns.loc[:'2021-12-31']
adj_close_test = adj_close.loc['2022-01-01':]

# Parameter estimation and model defined
mu, sigma, corr = estimate_parameters(arithmetic_returns)
start = "2014-01-01"
end = "2025-01-01"

credit_spread = download_state_variable(start, end)
# Trim cs series to match training window
cs_train = credit_spread.loc[arithmetic_returns_train.index]

m, nu, rho = estimate_state_variable_params(credit_spread, arithmetic_returns)

S0 = adj_close.loc[arithmetic_returns_train.index[0]].values
model = Deep2BSDE(mu=mu, sigma=sigma, corr_matrix=corr, m=m, nu=nu, rho_vector=rho, X0=cs_train.iloc[0], S0=S0, N=30, use_transaction_costs=True)

# Train fixed model
RUN_FIXED_MODEL = False

if RUN_FIXED_MODEL:
    loss_history_fix, W_paths_fix, theta_paths_fix = model.train(
        epochs=10,      # More epochs for meaningful training
        batch_size=256,
        lr=1e-3
    )

# Plot fixed model
def plot_fixed_loss(loss_history, title="Loss Over Epochs (Fixed Window)", save_path="fixed_loss_plot.pdf"):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()  # ensures proper spacing
    plt.savefig(save_path, format='pdf')  # save as vector graphic
    plt.show()  

# Plot theta evolution
def plot_theta_paths(theta_paths, time_grid, n_assets=3, n_samples=5):
    for asset in range(n_assets):
        plt.figure(figsize=(8, 4))
        for i in range(min(n_samples, theta_paths.shape[0])):
            plt.plot(time_grid, theta_paths[i, :, asset], label=f"Path {i+1}")
        plt.title(f"$\\theta_t$ Path for Asset {asset + 1}")
        plt.xlabel("Time")
        plt.ylabel(f"$\\theta_t$ Asset {asset + 1}")
        plt.grid(True)
        plt.legend()
        plt.show()

# Plot fixed window results
if RUN_FIXED_MODEL:
    plot_fixed_loss(loss_history_fix, save_path="/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up/fixed_loss_plot.pdf")
    plot_theta_paths(theta_paths_fix, model.time_grid)

# Function for model that re-estimates parameters and retrains model every n days
def run_rolling_training(
    arithmetic_returns,
    credit_spread,
    adj_close,
    rolling_window_days=100,    
    rebalancing_freq=50,
    epochs_per_window=3,
    batch_size=64,
    lr=1e-3,
    N=30,
    use_transaction_costs=True,
    anchor_penalty_weight=10.0  # Make it adjustable
):
    """
    Runs rolling-window training of the Deep2BSDE model, with parameter re-estimation,
    warm starts, and anchor penalties across windows.
    """
    trading_dates = arithmetic_returns.index

    rolling_dates = trading_dates[::rebalancing_freq]
    rolling_dates = [
        d for d in rolling_dates
        if d - pd.Timedelta(days=rolling_window_days) >= trading_dates[0]
    ]

    theta_trajectories = []
    dates_used = []
    final_outputs = []
    loss_histories = []
    theta_anchor_prev_window = None
    previous_model = None  # <-- store previous model for warm start

    for date in rolling_dates:
        print(f"\nTraining window ending {date.date()}")

        np.random.seed(0)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)

        # Ensure window_end is a valid trading day
        if date not in trading_dates:
            idx = trading_dates.get_indexer([date], method="pad")
            if idx[0] == -1:
                print(f"Skipping window ending {date.date()} — no trading day found.")
                continue
            window_end = trading_dates[idx[0]]
        else:
            window_end = date

        # Compute window_start and align to trading days
        window_start_raw = window_end - pd.Timedelta(days=rolling_window_days)
        idx_start = trading_dates.get_indexer([window_start_raw], method="backfill")
        if idx_start[0] == -1:
            print(f"Skipping window ending {window_end.date()} — no trading day found for window start.")
            continue
        window_start = trading_dates[idx_start[0]]

        # Extract slices
        window_returns = arithmetic_returns.loc[window_start:window_end]
        window_cs = credit_spread.loc[window_start:window_end]

        if window_returns.empty or len(window_returns) < 60:
            print(f"Skipping window ending {window_end.date()} — insufficient data.")
            continue

        # Estimate parameters
        mu, sigma, corr = estimate_parameters(window_returns)
        m, nu, rho = estimate_state_variable_params(window_cs, window_returns)
        X0 = window_cs.iloc[0]

        # Find nearest S0 value
        adj_close_dates = adj_close.index
        idx_S0 = adj_close_dates.get_indexer([window_start], method="pad")
        if idx_S0[0] == -1:
            print(f"Skipping window ending {window_end.date()} — no adj_close data for window start.")
            continue
        S0 = adj_close.loc[adj_close_dates[idx_S0[0]]].values

        # Instantiate new model for this window
        model = Deep2BSDE(
            mu=mu,
            sigma=sigma,
            corr_matrix=corr,
            m=m,
            nu=nu,
            rho_vector=rho,
            X0=X0,
            S0=S0,
            N=N,
            use_transaction_costs=use_transaction_costs,
            anchor_penalty_weight=anchor_penalty_weight
        )

        # Warm-start: load weights from previous model
        if previous_model is not None:
            model.policy_net.load_state_dict(previous_model.policy_net.state_dict())
            model.z_net.load_state_dict(previous_model.z_net.state_dict())
            model.second_deriv_net.load_state_dict(previous_model.second_deriv_net.state_dict())

        # Anchor smoothing penalty from last window's theta
        model.raw_theta_anchor = theta_anchor_prev_window

        try:
            loss_history, W_paths, theta_paths = model.train(
                epochs=epochs_per_window,
                batch_size=batch_size,
                lr=lr
            )
        except Exception as e:
            print(f"Training failed for window ending {window_end.date()}: {e}")
            continue

        # Store final raw_theta (dollar allocation) from this window
        W_final_clamped = np.clip(W_paths[:, -1:], 1e-6, None)
        theta_anchor_prev_window = theta_paths[:, -1, :] / W_final_clamped

        # Save results
        theta_trajectories.append(theta_paths)
        dates_used.append(window_end)
        loss_histories.append(loss_history)
        W_T = W_paths[:, -1]
        theta_T = theta_paths[:, -1, :]
        final_outputs.append((W_T, theta_T, model))

        previous_model = model  # for warm-starting next window

    return theta_trajectories, dates_used, final_outputs, loss_histories

RUN_ROLLING_MODEL = True

# Train rolling model
if RUN_ROLLING_MODEL:
    theta_trajectories, dates_used, final_outputs, loss_histories = run_rolling_training(
        arithmetic_returns,
        credit_spread=credit_spread,
        adj_close = adj_close,
        rolling_window_days=252,
        rebalancing_freq=252,
        epochs_per_window=5,  # adjust for deeper training
        batch_size=128,
        lr=1e-3,
        N=30, use_transaction_costs=True # Consistent with daily rebalancing, could be computationally costly
    )

    # Extract trained Deep2BSDE models
    rolling_dates = dates_used
    trained_models = [output[2] for output in final_outputs]  # Each output

# Plotting loss evolution for rolling model
def plot_rolling_losses(loss_histories, filename="rolling_losses_stepwise.png", dpi=300, show=True):
    plt.figure(figsize=(10, 5))
    for i, losses in enumerate(loss_histories):
        plt.plot(losses, label=f"Window {i+1}")
    plt.title("Loss per Epoch for Each Rolling Window")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

# Plot rolling training losses
if RUN_ROLLING_MODEL:
    plot_rolling_losses(loss_histories, filename = "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up/rolling_losses_stepwise.png", show=True)

def run_backtest_dynamic(
    adj_close,
    credit_spread,
    rolling_dates,
    trained_models,
    initial_wealth=100.0,
    use_transaction_costs=True,
    tolerance=0.0,
    cache_path=None,  # ← NEW: path to save/load results
    force_recompute=False  # ← NEW: force rerun even if cache exists
):
    """
    Backtest Deep2BSDE model with daily evaluation of the policy.

    - Run policy network every day.
    - Rebalance only if theta changes.
    - Wealth changes every day due to returns.
    - Saves results to .npz file if cache_path is given.
    """

    # ========================
    # 1. Try loading cache
    # ========================
    if cache_path is not None and os.path.exists(cache_path) and not force_recompute:
        print(f"📂 Loading cached backtest from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            pd.to_datetime(data["dates"]),
            data["wealth"],
            data["theta"],
            data["costs"],
            data["weights"]
        )

    # ========================
    # 2. Run fresh backtest
    # ========================
    print("🚀 Running new backtest...")
    time_index = adj_close.index
    n_assets = adj_close.shape[1]
    dates_bt = []
    theta_bt = []
    W_bt = [initial_wealth]
    costs_bt = []
    weights_bt = []

    # Start with first model
    current_model_idx = 0
    current_model = trained_models[current_model_idx]
    current_model.policy_net.eval()

    # Initial state
    W_t = initial_wealth
    S_t = adj_close.iloc[0].values
    X_t = credit_spread.loc[:time_index[0]].iloc[-1].item()
    r_f = current_model.r
    lambda_tc = 0.0003
    dt = 1/252
    theta_prev = np.zeros(n_assets)

    for i in range(1, len(time_index)):
        date = time_index[i]
        S_next = adj_close.iloc[i].values
        X_t = credit_spread.loc[:date].iloc[-1].item()

        # Switch model if using rolling
        if len(trained_models) > 1 and current_model_idx + 1 < len(rolling_dates):
            if date >= rolling_dates[current_model_idx + 1]:
                current_model_idx += 1
                current_model = trained_models[current_model_idx]
                current_model.policy_net.eval()
                dt = current_model.dt

        # Run policy every day
        t_tensor = torch.tensor([[i / len(time_index)]], dtype=torch.float32)
        W_tensor = torch.tensor([[W_t]], dtype=torch.float32)
        S_tensor = torch.tensor(S_t, dtype=torch.float32).unsqueeze(0)
        X_tensor = torch.tensor([[float(X_t)]], dtype=torch.float32)

        with torch.no_grad():
            raw_theta = current_model.policy_net(
                t_tensor, W_tensor, S_tensor, X_tensor
            )
            theta_scaled = raw_theta * W_tensor
            theta_t = theta_scaled.numpy().flatten()

        # Determine whether to rebalance
        delta_theta = theta_t - theta_prev
        if np.all(np.abs(delta_theta) < tolerance):
            transaction_cost = 0.0
            theta_effective = theta_prev.copy()
        else:
            transaction_cost = lambda_tc * np.sum(delta_theta ** 2) if use_transaction_costs else 0.0
            theta_effective = theta_t.copy()
            theta_prev = theta_t.copy()

        # Calculate risky and cash values
        risky_value_t = np.sum(theta_effective)
        cash_value_t = W_t - risky_value_t
        cash_value_tplus1 = cash_value_t * (1 + r_f * dt)

        S_t_safe = np.clip(S_t, 1e-6, None)
        shares = theta_effective / S_t_safe
        risky_value_tplus1 = np.sum(shares * S_next)

        # Compute new wealth
        W_t_new = cash_value_tplus1 + risky_value_tplus1 - transaction_cost
        W_t_new = max(W_t_new, 1e-8)

        # Save outputs
        dates_bt.append(date)
        theta_bt.append(theta_effective)
        W_bt.append(W_t_new)
        costs_bt.append(transaction_cost)

        # Compute portfolio weights
        risky_weights = theta_effective / W_t if W_t > 0 else np.zeros_like(theta_effective)
        cash_weight = (W_t - np.sum(theta_effective)) / W_t if W_t > 0 else 1.0
        weights_bt.append(np.append(risky_weights, cash_weight))

        # Update
        S_t = S_next
        W_t = W_t_new

    # ========================
    # 3. Save to cache
    # ========================
    if cache_path is not None:
        np.savez(
            cache_path,
            dates=np.array(dates_bt),
            wealth=np.array(W_bt[1:]),  # skip initial
            theta=np.array(theta_bt),
            costs=np.array(costs_bt),
            weights=np.array(weights_bt)
        )
        print(f"💾 Backtest results saved to: {cache_path}")

    return (
        pd.to_datetime(dates_bt),
        np.array(W_bt[1:]),
        np.array(theta_bt),
        np.array(costs_bt),
        np.array(weights_bt)
    )

# Backtest for rolling model
if RUN_ROLLING_MODEL:
    # Define adjusted close test data starting from first rolling window
    adj_close_test = adj_close.loc[dates_used[0]:]
    # Backtest using the trained rolling models
    cache_path = os.path.join(output_folder, f"{tickers_str}_deepbsde_backtest.npz")

    dates_bt, wealth_bt, theta_bt, costs_bt, weights_bt = run_backtest_dynamic(
        adj_close=adj_close_test,
        credit_spread=credit_spread,
        rolling_dates=dates_used,
        trained_models=trained_models,
        initial_wealth=100.0,
        use_transaction_costs=True,
        tolerance=0.0,
        cache_path=cache_path,
        force_recompute=True  # Change to True if you want to force rerun
    )

    plt.figure(figsize=(10, 5))
    plt.plot(dates_bt, wealth_bt)
    plt.title("Backtested Wealth Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Wealth")
    plt.grid(True)
    plt.show()

# Backtest for fixed model
if RUN_FIXED_MODEL:
    dates_bt, wealth_bt, theta_bt, costs_bt, weights_bt = run_backtest_dynamic(
        adj_close_test,  # only future data
        credit_spread,
        rolling_dates=[adj_close_test.index[0]],
        trained_models=[model],
        initial_wealth=100.0,
        use_transaction_costs=True
    )

    plt.figure(figsize=(10, 5))
    plt.plot(dates_bt, wealth_bt)
    plt.title("Backtested Wealth Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Wealth")
    plt.grid(True)
    plt.show()

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
    # ----------------------------------------------------
    # Construct filename automatically if tickers provided
    # ----------------------------------------------------
    if latex_filename is None:
        if tickers is None:
            file_stem = "bs_perf_table"
        else:
            file_stem = "_".join(tickers) + "_bs_perf_table"

        latex_filename = file_stem + ".tex"

    if output_folder is not None:
        latex_filename = os.path.join(output_folder, latex_filename)

    # -----------------------------------------------
    # 1. Align B&H benchmark series if provided
    # -----------------------------------------------
    if arithmetic_returns_eqw is not None and dates_eqw is not None:
        start_date = dates_bt[0]
        mask_eqw = dates_eqw >= start_date

        if not np.any(mask_eqw):
            raise ValueError(
                f"No B&H benchmark dates overlap with model backtest dates starting from {start_date}."
            )

        dates_eqw_matched = dates_eqw[mask_eqw]

        if wealth_eqw is not None:
            wealth_eqw_matched = wealth_eqw[mask_eqw]
            arithmetic_returns_eqw_matched = wealth_eqw_matched[1:] / wealth_eqw_matched[:-1] - 1
        else:
            arithmetic_returns_eqw_matched = arithmetic_returns_eqw[mask_eqw[:-1]]

        min_len = min(len(arithmetic_returns_eqw_matched), len(wealth_bt)-1)
        arithmetic_returns_eqw_matched = arithmetic_returns_eqw_matched[:min_len]
        arithmetic_returns_model = wealth_bt[1:] / wealth_bt[:-1] - 1
        arithmetic_returns_model = arithmetic_returns_model[:min_len]
    else:
        arithmetic_returns_eqw_matched = None
        arithmetic_returns_model = wealth_bt[1:] / wealth_bt[:-1] - 1

    # -------------------------
    # 2. Main performance metrics
    # -------------------------
    T = len(arithmetic_returns_model)
    if T <= 1:
        raise ValueError("Not enough time steps to compute statistics.")

    freq_per_year = 1 / dt_backtest

    cagr = (wealth_bt[-1] / initial_wealth)**(freq_per_year / T) - 1
    ann_vol = np.std(arithmetic_returns_model) * np.sqrt(freq_per_year)
    ann_excess_return = cagr - annual_rf_rate
    sharpe = ann_excess_return / ann_vol if ann_vol > 0 else np.nan

    # Probabilistic Sharpe Ratio vs cash
    benchmark_sharpe = 0.0
    if T > 1 and np.isfinite(sharpe):
        se_sharpe = np.sqrt((1 + sharpe**2 / 2) / (T - 1))
        psr_rf = norm.cdf((sharpe - benchmark_sharpe) / se_sharpe)
    else:
        psr_rf = np.nan

    # Probabilistic Sharpe Ratio vs EQW
    if sharpe_eqw is not None and np.isfinite(sharpe_eqw) and np.isfinite(sharpe):
        se_diff = np.sqrt(
            (1 + sharpe**2 / 2) / (T - 1) +
            (1 + sharpe_eqw**2 / 2) / (T - 1)
        )
        psr_eqw = norm.cdf((sharpe - sharpe_eqw) / se_diff)
    else:
        psr_eqw = np.nan

    # -------------------------
    # Sharpe ratio differences
    # -------------------------
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

    # -------------------------
    # Paired t-test for mean returns difference
    # -------------------------
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

    # -------------------------
    # Other metrics
    # -------------------------
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
    tolerance = 0.0
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

    # -----------------------------
    # Store both formatted and raw
    # -----------------------------

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

    # --------------------------------------------------
    # Ticker string for file prefixes
    # --------------------------------------------------
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
    plt.savefig(file_path("bs_wealth_over_time"))
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
    plt.savefig(file_path("bs_drawdown_over_time"))
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
    plt.savefig(file_path("bs_histogram_returns"))
    plt.close()

    # ----------------------------------------
    # Plot 4: Rolling Sharpe Ratio (Discrete)
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
    plt.savefig(file_path("bs_rolling_sharpe"))
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
    plt.savefig(file_path("bs_weights_line_plot"))
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
    plt.savefig(file_path("bs_turnover_over_time"))
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
    plt.savefig(file_path("bs_cumulative_transaction_costs"))
    plt.close()

    print(f"All plots saved in: {output_dir}")

if RUN_ROLLING_MODEL:
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
    arithmetic_returns_eqw = wealth_eqw[1:] / wealth_eqw[:-1] - 1  # Arithmetic returns
    T_eqw = len(wealth_eqw) - 1
    freq_per_year = 252
    cagr_eqw = (wealth_eqw[-1] / wealth_eqw[0])**(freq_per_year / T_eqw) - 1
    ann_vol_eqw = np.std(arithmetic_returns_eqw) * np.sqrt(freq_per_year)
    ann_excess_eqw = cagr_eqw - 0.02
    sharpe_eqw = ann_excess_eqw / ann_vol_eqw if ann_vol_eqw > 0 else np.nan

    output_folder = "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up"

    # Compute metrics table
    metrics_df, hist_metrics_deepbsde = compute_performance_table(
        dates_bt,
        wealth_bt,
        theta_bt,
        costs_bt,
        weights_bt,
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
    # Save metrics to disk for reuse
    with open(os.path.join(output_folder, "hist_metrics_deepbsde.pkl"), "wb") as f:
        pickle.dump(hist_metrics_deepbsde, f)

    print(metrics_df)

    plot_backtest_results(
        dates_bt,
        wealth_bt,
        theta_bt,
        costs_bt,
        weights_bt,
        output_dir="/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up",
        tickers=tickers
    )

def _run_single_bootstrap_trial(
    path_file,
    i,
    initial_wealth,
    window_days,
    epochs_per_window,
    batch_size,
    lr,
    use_transaction_costs,
    run_rolling_training_flag,
    output_folder,
    tickers,
    trained_models=None,         # NEW
    rolling_dates=None           # NEW
):
    """
    Runs a single bootstrap backtest trial on synthetic data.
    """

    print(f"\n=== Bootstrap Trial {i+1} ===")
    print(f"Loading synthetic path: {os.path.basename(path_file)}")

    data = np.load(path_file, allow_pickle=True)
    dates = pd.to_datetime(data['dates'])
    prices = pd.DataFrame(data['prices'], index=dates, columns=data['tickers'])
    cs = pd.Series(data['cs'], index=dates)

    # Trim synthetic data if using pretrained models
    if trained_models is not None and rolling_dates is not None:
        first_model_start_date = rolling_dates[0]
        prices = prices.loc[first_model_start_date:]
        cs = cs.loc[first_model_start_date:]
        print(f"📅 Trimmed synthetic path to start from {first_model_start_date.date()} to align with pretrained models.")
    else:
        print(f"📅 Using full synthetic path: {dates[0].date()} to {dates[-1].date()}")

    print(f"✅ Backtest will run from {prices.index[0].date()} to {prices.index[-1].date()} using {len(prices)} days of data.")

    try:
        synthetic_arithmetic_returns = prices.pct_change().dropna()

        if run_rolling_training_flag or trained_models is None or rolling_dates is None:
            print("🔁 Performing rolling training on this synthetic path...")

            trained_models_results = run_rolling_training(
                arithmetic_returns=synthetic_arithmetic_returns,
                credit_spread=cs,
                adj_close=prices,
                rolling_window_days=window_days,
                rebalancing_freq=window_days,
                epochs_per_window=epochs_per_window,
                batch_size=batch_size,
                lr=lr,
                N=252,
                use_transaction_costs=use_transaction_costs
            )

            theta_trajectories, dates_used, final_outputs, loss_histories = trained_models_results

            if any(np.isnan(history).any() for history in loss_histories):
                print(f"Trial {i+1}: Training produced NaNs. Skipping.")
                return None

            trained_models_this_trial = [output[2] for output in final_outputs]
            rolling_dates_input = dates_used
        else:
            print("🧠 Using pre-trained models from historical training...")
            trained_models_this_trial = trained_models
            rolling_dates_input = rolling_dates

    except Exception as e:
        print(f"Trial {i+1}: Training failed with exception:\n{e}")
        return None

    # =======================
    # Run backtest
    # =======================
    dates_bt, wealth_bt, theta_bt, costs_bt, weights_bt = run_backtest_dynamic(
        adj_close=prices,
        credit_spread=cs,
        rolling_dates=rolling_dates_input,
        trained_models=trained_models_this_trial,
        initial_wealth=initial_wealth,
        use_transaction_costs=use_transaction_costs,
        cache_path=f"cache_bootstrap_trial_{i}.pkl",
        force_recompute=True
    )

    # Save
    tickers_str = "_".join(tickers)
    npz_path = os.path.join(output_folder, f"{tickers_str}_parametric_bootstrap_deepbsde_{i}.npz")
    print(f"Saving results for Trial {i+1} to: {npz_path}")
    np.savez(
        npz_path,
        dates=dates_bt,
        wealth=wealth_bt,
        portfolio=theta_bt,
        costs=costs_bt,
        weights=weights_bt
    )

    return {
        "dates": dates_bt,
        "wealth": wealth_bt,
        "theta": theta_bt,
        "costs": costs_bt,
        "weights": weights_bt,
        "trained_models": trained_models_this_trial
    }

def run_parametric_bootstrap_general_parallel(
    synthetic_path_files,
    initial_wealth=100.0,
    use_transaction_costs=True,
    window_days=252,
    epochs_per_window=5,
    batch_size=64,
    lr=1e-3,
    run_rolling_training_flag=True,
    output_folder=None,
    tickers=None,
    n_jobs=-1,
    trained_models=None,          # NEW
    rolling_dates=None            # NEW
):
    """
    Runs multiple bootstrap trials in parallel and collects results.
    """

    print(f"Launching parallel bootstrap on {len(synthetic_path_files)} trials...")
    os.makedirs(output_folder, exist_ok=True)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_bootstrap_trial)(
            path_file,
            i,
            initial_wealth,
            window_days,
            epochs_per_window,
            batch_size,
            lr,
            use_transaction_costs,
            run_rolling_training_flag,
            output_folder,
            tickers,
            trained_models=trained_models,
            rolling_dates=rolling_dates
        )
        for i, path_file in enumerate(tqdm(synthetic_path_files, desc="Bootstrap Trials"))
    )

    results = [r for r in results if r is not None]
    all_trained_models = [r["trained_models"] for r in results]

    print(f"\nCompleted {len(results)} successful bootstrap trials out of {len(synthetic_path_files)}.")
    return results, all_trained_models

def simulate_synthetic_market(
    mu,
    sigma,
    corr_matrix,
    m,
    nu,
    rho_vector,
    S0,
    X0,
    N,
    start_date,
    tickers,
    cs_floor=1e-3,
    cs_cap=200,
    price_floors=None,
    price_caps=None,
):
    """
    Simulates synthetic asset prices and state variable using estimated dynamics.
    """
    dt = 1/252
    n_assets = len(S0)

    time_index = pd.date_range(start=start_date, periods=N+1, freq='B')

    S = np.zeros((N+1, n_assets))
    X = np.zeros(N+1)

    S[0] = S0
    X[0] = X0

    cholesky = np.linalg.cholesky(corr_matrix)
    rho_unit = rho_vector / np.linalg.norm(rho_vector)

    # Default price caps/floors if not specified
    if price_floors is None:
        price_floors = np.full(n_assets, 1e-3)
    if price_caps is None:
        price_caps = np.full(n_assets, 1e5)

    for i in range(N):
        # Generate correlated shocks
        Z = np.random.normal(size=n_assets)
        dW_vector = np.sqrt(dt) * (cholesky @ Z)

        # Simulate asset prices
        S[i+1] = S[i] + mu * S[i] * dt + sigma * S[i] * dW_vector
        # Clamp each asset individually
        S[i+1] = np.clip(S[i+1], 1e-6, 1e6)

        # Generate orthogonal increment
        dZ = np.random.normal(0, np.sqrt(dt))
        orth_component = dW_vector - np.dot(dW_vector, rho_unit) * rho_unit
        dW_tilde = np.dot(rho_vector, dW_vector) + dZ * np.linalg.norm(orth_component)

        # Simulate LEVEL cs directly
        X[i+1] = X[i] + m * dt + nu * dW_tilde
        X[i+1] = np.clip(X[i+1], -70, -10)

    synthetic_adj_close = pd.DataFrame(S, index=time_index, columns=tickers)
    synthetic_cs = pd.Series(X, index=time_index)

    return synthetic_adj_close, synthetic_cs

def generate_rolling_synthetic_paths(
    tickers,
    synthetic_save_folder,
    estimation_start_date="2013-01-01",
    simulation_start_date="2014-01-01",
    simulation_end_date="2025-01-01",
    n_desired=100,
):
    """
    Generate rolling synthetic market paths using yearly rolling parameter estimation.
    """
    dt = 1 / 252

    # Ensure all required historical data is available
    arithmetic_returns, adj_close = download_arithmetic_returns(
        tickers, start_date=estimation_start_date, end_date=simulation_end_date
    )
    credit_spread = download_state_variable(
        start_date=estimation_start_date, end_date=simulation_end_date
    )

    # Cap and floor values
    #price_floors = np.maximum(1e-3, adj_close.quantile(0.005) * 0.8)
    #price_caps = adj_close.quantile(0.995) * 1.5
    #cs_floor = max(1e-3, credit_spread.quantile(0.005).item() * 0.8)
    #cs_cap = credit_spread.quantile(0.995).item() * 1.5

    sim_start_year = pd.to_datetime(simulation_start_date).year
    sim_end_year = pd.to_datetime(simulation_end_date).year

    for path_idx in range(n_desired):
        print(f"\n🔁 Simulating rolling path {path_idx+1}/{n_desired}...")

        sim_prices_list = []
        sim_cs_list = []

        # Initial S0 and X0 from historical data on simulation_start_date
        S0_date = adj_close.index[adj_close.index.get_indexer([simulation_start_date], method="nearest")[0]]
        X0_date = credit_spread.index[credit_spread.index.get_indexer([simulation_start_date], method="nearest")[0]]
        prev_S = adj_close.loc[S0_date].values
        prev_X = float(credit_spread.loc[X0_date])

        for year in range(sim_start_year, sim_end_year):
            # Estimation window (previous calendar year)
            est_start = f"{year - 1}-01-01"
            est_end = f"{year}-01-01"

            # Simulation window
            sim_start = f"{year}-01-01"
            sim_end = f"{year + 1}-01-01"
            sim_dates = pd.bdate_range(sim_start, sim_end)
            N = len(sim_dates) - 1

            # Estimate parameters
            returns_est = arithmetic_returns.loc[est_start:est_end]
            cs_est = credit_spread.loc[est_start:est_end]
            mu, sigma, corr_matrix = estimate_parameters(returns_est)
            m, nu, rho_vector = estimate_state_variable_params(cs_est, returns_est)

            # Simulate one year forward
            sim_prices, sim_cs = simulate_synthetic_market(
                mu=mu,
                sigma=sigma,
                corr_matrix=corr_matrix,
                m=m,
                nu=nu,
                rho_vector=rho_vector,
                S0=prev_S,
                X0=prev_X,
                N=N,
                start_date=sim_start,
                tickers=tickers,
                cs_floor=None,
                cs_cap=None,
                price_floors=None,
                price_caps=None
            )
            # Remove last row to prevent overlapping date with next year
            if year < sim_end_year - 1:
                sim_prices = sim_prices.iloc[:-1]
                sim_cs = sim_cs.iloc[:-1]

            # Append and update for next block
            sim_prices_list.append(sim_prices)
            sim_cs_list.append(sim_cs)
            prev_S = sim_prices.iloc[-1].values
            prev_X = sim_cs.iloc[-1]

        # Combine and save
        full_sim_prices = pd.concat(sim_prices_list)
        full_sim_cs = pd.concat(sim_cs_list)

        tickers_str = "_".join(tickers)
        save_path = os.path.join(
            synthetic_save_folder,
            f"{tickers_str}_rolling_bootstrap_path_{path_idx}.npz"
        )
        np.savez(
            save_path,
            dates=full_sim_prices.index.values.astype("datetime64[D]"),
            prices=full_sim_prices.values,
            cs=full_sim_cs.values,
            tickers=np.array(tickers)
        )
        print(f"✔️ Saved: {save_path}")

    print(f"\n✅ All {n_desired} rolling synthetic paths saved.")

def plot_synthetic_paths_preview(synthetic_save_folder, tickers, max_paths=5):
    """
    Plots the first `max_paths` synthetic paths for each asset and the cs.
    
    Parameters:
    - synthetic_save_folder: directory where .npz files are saved
    - tickers: list of asset tickers
    - max_paths: number of synthetic paths to plot (default: 5)
    """
    files = sorted([
        f for f in os.listdir(synthetic_save_folder)
        if f.endswith(".npz") and "rolling_bootstrap_path" in f
    ])[:max_paths]

    if not files:
        print("❌ No synthetic .npz files found in folder.")
        return

    n_assets = len(tickers)
    fig, axs = plt.subplots(n_assets + 1, 1, figsize=(12, 3 * (n_assets + 1)), sharex=True)

    for i, file in enumerate(files):
        data = np.load(os.path.join(synthetic_save_folder, file))
        prices = data['prices']
        cs = data['cs']
        dates = pd.to_datetime(data['dates'])

        for j in range(n_assets):
            axs[j].plot(dates, prices[:, j], label=f'Path {i+1}')
            axs[j].set_ylabel(tickers[j])
            axs[j].grid(True)

        axs[-1].plot(dates, cs, label=f'Path {i+1}')
        axs[-1].set_ylabel("cs")
        axs[-1].grid(True)

    for ax in axs:
        ax.legend()

    axs[-1].set_xlabel("Date")
    plt.suptitle("Preview of Synthetic Asset Prices and cs (First 5 Paths)")
    plt.tight_layout()
    plt.show()

SIMULATE_PATHS = False

output_folder = "/Users/jackadams/Documents/University of Leeds/Semester 2/Dissertation/Write Up"
synthetic_save_folder = output_folder # You can change this if needed

if SIMULATE_PATHS:
    print("\n=== Simulating Synthetic Market Paths (Rolling Window) ===")

    # Parameters for rolling simulation
    estimation_start_date = "2013-01-01"          # For first estimation window (2013 data)
    simulation_start_date = "2014-01-01"          # Start of simulated path
    simulation_end_date = "2025-01-01"            # End of simulated path
    n_desired = 100                    # Number of synthetic paths

    for f in glob.glob(os.path.join(synthetic_save_folder, "*rolling_bootstrap_path_*.npz")):
        os.remove(f)

    # Run rolling simulation generator
    generate_rolling_synthetic_paths(
        tickers=tickers,
        synthetic_save_folder=synthetic_save_folder,
        estimation_start_date=estimation_start_date,
        simulation_start_date=simulation_start_date,
        simulation_end_date=simulation_end_date,
        n_desired=n_desired
    )

    print(f"\nAll {n_desired} synthetic rolling paths saved to {synthetic_save_folder}.")

    plot_synthetic_paths_preview(synthetic_save_folder, tickers, max_paths=5)

RUN_BOOTSTRAP = True

if RUN_BOOTSTRAP:
    print("\n=== Running Rolling Training and Backtest on Synthetic Paths ===")
    n_trials = 100
    synthetic_save_folder = output_folder

    synthetic_files = [
        os.path.join(
            synthetic_save_folder,
            f"{tickers_str}_rolling_bootstrap_path_{i}.npz"
        )
        for i in range(n_trials)
    ]

    bootstrap_results_rolling, all_models = run_parametric_bootstrap_general_parallel(
        synthetic_path_files=synthetic_files,
        initial_wealth=100.0,
        use_transaction_costs=True,
        window_days=252,
        epochs_per_window=5,
        batch_size=64,
        lr=1e-3,
        run_rolling_training_flag=False,
        output_folder=output_folder,
        tickers=tickers,
        n_jobs=-1,
        trained_models=trained_models,
        rolling_dates = rolling_dates
    )

    # Check: number of successful trials
    print(f"\nSuccessfully completed {len(bootstrap_results_rolling)} bootstrap trials out of {len(synthetic_files)} total.")

    # Optional: print some wealth paths summary stats
    for i, result in enumerate(bootstrap_results_rolling[:3]):
        wealth = result["wealth"]
        print(f"\nTrial {i+1} summary:")
        print(f"  Final Wealth: £{wealth[-1]:.2f}")
        print(f"  Min Wealth:   £{wealth.min():.2f}")
        print(f"  Max Wealth:   £{wealth.max():.2f}")

    # Quick diagnostic plot of wealth paths
    plt.figure(figsize=(8, 4))
    for result in bootstrap_results_rolling[:5]:  # First 5 trials
        plt.plot(result["dates"], result["wealth"], alpha=0.6)
    plt.xlabel("Date")
    plt.ylabel("Wealth (£)")
    plt.title("Sample Wealth Paths (Bootstrap Trials)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

n_trials = 100
tickers_str = "_".join(tickers)

# Loading in B&H bootstrap results for statistical comparison
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

def load_saved_deepbsde_bootstrap_results(output_folder, tickers, n_trials):
    tickers_str = "_".join(tickers)
    results = []

    for i in range(n_trials):
        filename = os.path.join(
            output_folder,
            f"{tickers_str}_parametric_bootstrap_deepbsde_{i}.npz"
        )
        data = np.load(filename, allow_pickle=True)

        result = {
            "dates": pd.to_datetime(data["dates"]),
            "wealth": data["wealth"],
            "theta": data["portfolio"],
            "costs": data["costs"],
            "weights": data["weights"],
        }
        results.append(result)

    return results

bootstrap_results_rolling = load_saved_deepbsde_bootstrap_results(
    output_folder=output_folder,
    tickers=tickers,
    n_trials=100
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

        # Sortino
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

        calmar = cagr / max_dd if max_dd > 0 else np.nan

        # Turnover
        delta_u = np.diff(portfolio_path, axis=0)
        turnover_per_step = np.sum(np.abs(delta_u), axis=1) / wealth_bt[:-1]
        avg_turnover_step = np.mean(turnover_per_step)
        annual_turnover = avg_turnover_step * freq_per_year

        # Risky weight changes
        weights_risky = weights_path[:, :-1]
        delta_w = np.diff(weights_risky, axis=0)
        abs_changes = np.mean(np.sum(np.abs(delta_w), axis=1)) * 100

        # VaR and ES
        var_95 = -np.percentile(returns, 5)
        es_95 = -np.mean(returns[returns <= -var_95]) if np.any(returns <= -var_95) else np.nan

        # Costs
        total_costs = np.sum(costs_bt)

        # Store metrics
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

    # ----------------------------
    # Build LaTeX/return table
    # ----------------------------
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
        file_stem = "_".join(tickers) + "_bootstrap_comparison_deepbsde_table" if tickers else "bootstrap_comparison_deepbsde_table"
        latex_filename = file_stem + ".tex"

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
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

with open(os.path.join(output_folder, "hist_metrics_deepbsde.pkl"), "rb") as f:
    hist_metrics_deepbsde = pickle.load(f)

# This is the one to feed into bootstrap comparison:
df_bootstrap = compute_bootstrap_comparison_table_from_results(
    hist_metrics=hist_metrics_deepbsde,
    bootstrap_results=bootstrap_results_rolling,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    dt_backtest=1/252,
    output_folder=output_folder,
    tickers=tickers
)

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
    Compare DeepBSDE vs Benchmark bootstrap metrics using arithmetic returns.
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
        returns_deep = np.diff(wealth_deep) / wealth_deep[:-1]
        returns_bench = np.diff(wealth_bench) / wealth_bench[:-1]

        # Compute CAGR
        cagr_deep = (wealth_deep[-1] / initial_wealth) ** (freq_per_year / T) - 1
        cagr_bench = (wealth_bench[-1] / initial_wealth) ** (freq_per_year / T) - 1

        # Compute Sharpe Ratios
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
        "Metric": "Sharpe Ratio Difference (Ledoit–Wolf)",
        "Mean Diff": f"{mean_diff_sharpe:.4f}",
        "t-stat": f"{t_sharpe:.2f}" if np.isfinite(t_sharpe) else "N/A",
        "p-value": f"{pval_sharpe:.4f}" if np.isfinite(pval_sharpe) else "N/A",
        "Prob DeepBSDE $>$ Benchmark (\\%)": f"{prob_sharpe * 100:.2f}\\%"
    })

    rows.append({
        "Metric": "Returns Difference (CAGR)",
        "Mean Diff": f"{mean_diff_cagr * 100:.2f}\\%",
        "t-stat": f"{t_cagr:.2f}" if np.isfinite(t_cagr) else "N/A",
        "p-value": f"{pval_cagr:.4f}" if np.isfinite(pval_cagr) else "N/A",
        "Prob DeepBSDE $>$ Benchmark (\\%)": f"{prob_cagr * 100:.2f}\\%"
    })

    df = pd.DataFrame(rows)

    if latex_filename is None:
        file_stem = "_".join(tickers) + "_bootstrap_bench_vs_deepbsde_table" if tickers else "bootstrap_bench_vs_deepbsde_table"
        latex_filename = file_stem + ".tex"

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
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

df_stats = compute_bootstrap_stat_comparison_table(
    bootstrap_results_deepbsde=bootstrap_results_rolling,
    bootstrap_results_benchmark=bootstrap_benchmark_results,
    initial_wealth=100.0,
    annual_rf_rate=0.02,
    dt_backtest=1/252,
    output_folder=output_folder,
    tickers=tickers
)

print(df_stats)

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
        prefix = "bootstrap_deepbsde"
    else:
        prefix = "_".join(tickers) + "_bootstrap_deepbsde"

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
    # Histogram of Sharpe Ratios (ARITHMETIC RETURNS)
    # -----------------------------------
    sharpe_ratios = []
    for result in bootstrap_results:
        wealth_bt = result["wealth"]
        T = len(wealth_bt) - 1
        if T < 1:
            sharpe_ratios.append(np.nan)
            continue

        # Arithmetic returns
        arithmetic_returns = wealth_bt[1:] / wealth_bt[:-1] - 1
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

plot_bootstrap_results(
    dates=bootstrap_results_rolling[0]["dates"],
    bootstrap_results=bootstrap_results_rolling,
    output_dir=output_folder,
    tickers=tickers,
    initial_wealth=100.0,
    annual_rf_rate=0.025,
    dt_backtest=1/252,
)


# Deep BSDE Portfolio Optimisation

## MSc Financial Mathematics Dissertation (Highest Dissertation Mark in MSc Financial Mathematics Cohort)

This repository contains the Python implementation and dissertation for my MSc Financial Mathematics research project completed at the University of Leeds.

The project develops a Deep Backward Stochastic Differential Equation (Deep BSDE) framework for solving the continuous-time mean-variance portfolio optimisation problem under proportional transaction costs. The implementation combines stochastic control, deep learning and historical market data to produce a scalable framework for dynamic portfolio optimisation.

---

## Research Overview

Classical continuous-time portfolio optimisation becomes computationally intractable once realistic market frictions such as transaction costs are introduced.

This project implements a Deep BSDE framework to approximate the solution of the associated Hamilton–Jacobi–Bellman equation without requiring discretisation of the state space. Performance is evaluated against established benchmark strategies using historical US market data.

Benchmark strategies include:

- Basak & Chabakauri (2010) continuous-time strategy
- Li & Ng (2000) discrete-time strategy
- Buy-and-hold portfolio

---

## Methodology

The project combines:

- Deep BSDE implementation in PyTorch
- Continuous-time benchmark implementation
- Discrete-time benchmark implementation
- Historical rolling-window backtesting
- Bootstrap robustness analysis

Testing was performed using US equity and credit market data to evaluate performance across multiple market regimes.

---

## Repository Contents

### Dissertation

**MSc Dissertation – A Deep BSDE Approach to Time-Consistent Mean-Variance Portfolio Optimisation with Transaction Costs.pdf**

The dissertation contains:

- Literature review
- Mathematical derivations
- Methodology
- Historical backtesting
- Performance evaluation
- Discussion and future research

### Python Code

| File | Description |
|------|-------------|
| `deep_bsde_model.py` | Deep BSDE implementation |
| `basak_continuous_model.py` | Basak & Chabakauri benchmark implementation |
| `li_discrete_model.py` | Li & Ng benchmark implementation |
| `bh_benchmark.py` | Buy-and-hold benchmark |

---

## Key Results

The Deep BSDE framework outperformed benchmark strategies during out-of-sample historical backtesting.

| Metric | Deep BSDE | Benchmark |
|--------|----------:|----------:|
| Annualised Return (CAGR) | **24.9%** | 16.5% |
| Sharpe Ratio | **1.47** | 0.75 |
| Annualised Volatility | **15.3%** | 18.7% |
| Maximum Drawdown | **23.0%** | 32.3% |

Performance improvements were validated using bootstrap robustness analysis.

---

## Dissertation Abstract

The non-additivity of variance poses a major challenge for mean--variance optimisation. Basak and Chabakauri (2010) derived a time-consistent allocation strategy in continuous time, but their frictionless framework lacks practical relevance. Once frictions such as transaction costs are introduced, the problem becomes nonlinear and analytically intractable, motivating numerical methods. We extend the deep BSDE methodology of Han et al. (2018); Beck et al. (2019); Huré et al. (2020) to multi-asset mean--variance optimisation with transaction costs, learning optimal time-consistent strategies in continuous time. In historical backtests, the approach outperforms both the frictionless benchmark and a passive buy-and-hold strategy, achieving a Sharpe ratio of 1.47 with statistically significant improvements in risk-adjusted performance. Robustness is confirmed through parametric bootstrap analysis, which shows consistent Sharpe gains across simulated market paths.

---

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib

---

## Author

**Jack Adams**

MSc Financial Mathematics (Distinction)

University of Leeds

LinkedIn: https://www.linkedin.com/in/jackmadams

Pandas

SciPy

PyTorch

# Portfolio Loss Distribution Simulator

- This tool simulates potential loss distributions for a portfolio under various scenarios, incorporating asset correlations, default probabilities, and recovery rates.
- It provides insights into tail risks and capital adequacy.

---

## Files
- `portfolio_loss_simulator.py`: Main script for simulating portfolio losses using Monte Carlo methods and visualizing the distribution with Plotly.
- `output.png`: Plot

---

## Libraries Used
- `numpy`
- `pandas`
- `plotly`
- `scipy.stats`

---

## Features
- **Portfolio Setup**: Defines 50 synthetic assets with random exposures ($100K-$1M), default probabilities (1-10%), recovery rates (30-60%), and a base correlation of 0.2.
- **Correlated Defaults**: Uses Cholesky decomposition to simulate correlated defaults across assets with 10,000 Monte Carlo runs.
- **Loss Calculation**: Computes total losses per simulation as exposure * (1 - recovery rate) for defaulted assets.
- **Statistics**: Calculates mean loss and tail risk percentiles (1%, 5%, 95%, 99%).
- **Visualization**: Plots the loss distribution as a histogram with mean loss and 95% VaR lines using Plotly.

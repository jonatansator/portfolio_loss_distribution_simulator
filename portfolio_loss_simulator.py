import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm  # For norm.ppf

# Step 1: Define portfolio parameters
n_assets = 50  # Number of assets in portfolio
exposure = np.random.uniform(100000, 1000000, n_assets)  # Random exposure per asset
default_prob = np.random.uniform(0.01, 0.1, n_assets)   # Random default probabilities
recovery_rate = np.random.uniform(0.3, 0.6, n_assets)   # Random recovery rates
corr_matrix = np.full((n_assets, n_assets), 0.2)        # Correlation matrix (base 0.2)
np.fill_diagonal(corr_matrix, 1)                        # Diagonal = 1

# Step 2: Simulate correlated defaults using Cholesky decomposition
n_sim = 10000  # Number of Monte Carlo simulations
Z = np.random.normal(size=(n_assets, n_sim))           # Standard normal random vars
chol = np.linalg.cholesky(corr_matrix)                 # Cholesky decomposition
X = chol @ Z                                           # Correlated random vars
threshold = norm.ppf(default_prob)                     # Default thresholds
defaults = X < threshold[:, np.newaxis]                # Boolean array of defaults

# Step 3: Calculate losses per simulation
loss_per_asset = exposure * (1 - recovery_rate)        # Loss given default per asset
sim_losses = np.zeros(n_sim)                           # Array to store total losses
for i in range(n_sim):
    sim_losses[i] = np.sum(loss_per_asset * defaults[:, i])  # Total loss per sim

# Step 4: Compute loss distribution statistics
loss_df = pd.Series(sim_losses)
pctiles = loss_df.quantile([0.01, 0.05, 0.95, 0.99])  # Tail risk percentiles
mean_loss = loss_df.mean()                             # Expected loss

# Step 5: Plot loss distribution
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=loss_df,
    histnorm='probability density',
    name='Loss Distribution',
    marker_color='#FF6B6B',  # Reddish tone for actual data
    opacity=0.7
))
fig.add_vline(x=pctiles[0.95], line_dash='dash', line_color='#4ECDC4', name='95% VaR')  # Teal for forecast
fig.add_vline(x=mean_loss, line_dash='solid', line_color='#FF6B6B', name='Mean Loss')

# Apply dark theme and styling
fig.update_layout(
    title=dict(text='Portfolio Loss Distribution', font=dict(color='white')),
    xaxis=dict(
        title=dict(text='Loss Amount', font=dict(color='white')),
        tickfont=dict(color='white'),
        gridcolor='rgba(255, 255, 255, 0.1)',
        gridwidth=0.5
    ),
    yaxis=dict(
        title=dict(text='Density', font=dict(color='white')),
        tickfont=dict(color='white'),
        gridcolor='rgba(255, 255, 255, 0.1)',
        gridwidth=0.5
    ),
    plot_bgcolor='rgb(40, 40, 40)',
    paper_bgcolor='rgb(40, 40, 40)',
    showlegend=True,
    margin=dict(l=50, r=50, t=50, b=50)
)

# Step 6: Display results
print(f"Mean Loss: ${mean_loss:,.2f}")
print(f"95% VaR: ${pctiles[0.95]:,.2f}")
print(f"99% VaR: ${pctiles[0.99]:,.2f}")
fig.show()
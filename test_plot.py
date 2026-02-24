import os

# Configure matplotlib backend for headless environments (Colab, servers)
import matplotlib
backend = os.environ.get('MPLBACKEND', 'Agg')  # Default to Agg for compatibility
if backend == 'TkAgg':
    try:
        matplotlib.use("TkAgg")
    except ImportError:
        print('Warning: TkAgg not available. Using Agg backend (non-interactive).')
        matplotlib.use('Agg')
else:
    matplotlib.use(backend)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Hardcoded reliability matrices
matrix_disabled = np.array([
    [ np.nan, 1.00, 0.94, 0.89],
    [1.00, np.nan, 0.64, 1.00],
    [1.00, 0.41, np.nan, 1.00],
    [0.81, 0.92, 1.00, np.nan]
])

matrix_enabled = np.array([
    [ np.nan,1.00, 1.00, 0.84],
    [1.00, np.nan, 0.86, 1.00],
    [1.00, 0.82, np.nan, 1.00],
    [0.94, 1.00, 1.00, np.nan]
])

# Create a figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Determine a common color scale
vmin = 0.4
vmax = 1.0

# Plot the "Battery Level Aware Disabled" matrix
sns.heatmap(matrix_disabled, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, vmin=vmin, vmax=vmax, ax=axes[0])
axes[0].set_title("Battery Level Aware Disabled DM Packet Reliability")
axes[0].set_xlabel("Destination Node ID")
axes[0].set_ylabel("Source Node ID")

# Plot the "Battery Level Aware Enabled" matrix
sns.heatmap(matrix_enabled, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, vmin=vmin, vmax=vmax, ax=axes[1])
axes[1].set_title("Battery Level Aware Enabled DM Packet Reliability")
axes[1].set_xlabel("Destination Node ID")
axes[1].set_ylabel("Source Node ID")

plt.tight_layout()
plt.show()

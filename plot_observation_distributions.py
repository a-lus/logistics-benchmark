import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Constants
LOG_DIR = os.path.expanduser("~/ray_results")
csv_file_path = os.path.join(LOG_DIR, "obs_samples_saved1.csv")
obs_names = ['Distance to pickup', 'Distance for all tasks', 'Task due', 'Task distance', 'Min distance for all tasks*', 'Avg distance for all tasks*']

# Read data from CSV file
data = pd.read_csv(csv_file_path, header=None, names=['step', 'agent_id', 'obs_0', 'obs_1', 'obs_2', 'obs_3', 'obs_4', 'obs_5'])

# Set the style of seaborn plot
sns.set_theme(style="whitegrid")

# Number of observations per agent
num_obs = 6

# Create a figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))  # Adjust size as needed
fig.suptitle('Distribution of All Observations', fontsize=16)

# Flatten axes array for easier iteration
axes = axes.flatten()

# Plot each observation
for i in range(num_obs):
    sns.histplot(data[f'obs_{i}'], bins=30, kde=True, color='blue', ax=axes[i])
    axes[i].set_title(f'{obs_names[i]}')
    axes[i].set_xlabel(f'Obs {i} Value')
    axes[i].set_ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter as needed to fit the title

# Show the plot
plt.show()

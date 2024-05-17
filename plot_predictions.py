import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Constants
LOG_DIR = os.path.expanduser("~/ray_results")
csv_file_path = os.path.join(LOG_DIR, "predictions.csv")

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# Plotting the distribution of 'Due'
axs[0].hist(df['Task time'], bins=10, color='blue', alpha=0.5)
axs[0].set_title('Distribution of Due')
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Frequency')

# Plotting the distribution of predictions
predictions = ['Prediction_0', 'Prediction_1', 'Prediction_2', 'Prediction_3', 'Prediction_4']
for pred in predictions:
    axs[1].hist(df[pred], bins=10, alpha=0.5, label=pred)
axs[1].set_title('Distribution of Predictions')
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')
axs[1].legend()

# Calculate and plot the difference between prediction and due
for pred in predictions:
    diff = df[pred] - df['Task time']
    axs[2].hist(diff, bins=10, alpha=0.5, label=f'{pred} - Due')
axs[2].set_title('Difference between Prediction and Due')
axs[2].set_xlabel('Difference')
axs[2].set_ylabel('Frequency')
axs[2].legend()

plt.tight_layout()
plt.show()
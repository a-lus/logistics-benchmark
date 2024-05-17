import os
import pandas as pd
import matplotlib.pyplot as plt

# Constants
LOG_DIR = os.path.expanduser("~/ray_results")
csv_file_path = os.path.join(LOG_DIR, "predictions_saved1.csv")
NORM_FACTOR = 1060

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Get the list of prediction and queue distance columns
prediction_columns = [col for col in df.columns if col.startswith('Prediction_')]
queue_distance_columns = [col for col in df.columns if col.startswith('Queue_distance_')]
df[queue_distance_columns] = df[queue_distance_columns].mul(NORM_FACTOR) # De-normalize queue distances


# Calculate the minimum and maximum values for predictions and 'Task time'
min_value = min(df[prediction_columns].min().min(), df['Task time'].min(), df[queue_distance_columns].min().min())
max_value = max(df[prediction_columns].max().max(), df['Task time'].max(), df[queue_distance_columns].max().max())

# Calculate the minimum and maximum values for queue distances
# min_value2 = df[queue_distance_columns].min().min()
# max_value2 = df[queue_distance_columns].max().max()

# Create subplots with shared y-axis
fig, axes = plt.subplots(nrows=len(prediction_columns), ncols=1, figsize=(10, 6), sharey='row')

# Plot each pair of prediction and queue distance
for i, (pred_col, queue_col) in enumerate(zip(prediction_columns, queue_distance_columns)):
    ax = axes[i]
    ax.plot(df['Step number'], df['Task time'], label='Task time', color='orange')  # Plot Task time on y-axis
    ax.plot(df['Step number'], df[pred_col], label=pred_col, color='royalblue')
    ax.set_ylabel('Steps')
    ax.set_ylim(min_value, max_value)  # Set y-axis limits for Task time and predictions
    # ax_twin.set_ylabel(queue_col, color='lightblue')

    ax.plot(df['Step number'], df[queue_col], label=queue_col, color='lightblue')
    # ax_twin.set_ylim(min_value2, max_value2)  # Set y-axis limits for queue distances
    # ax_twin.tick_params(axis='y', colors='lightblue')
    ax.legend(loc='upper left')
    # ax_twin.legend(loc='upper right')

    # print(df[pred_col].head())
    # print(df[queue_col].head())
    covariance = df[pred_col].cov(df[queue_col])

    # Calculate the standard deviations of 'Prediction_0' and 'Queue_distance_0'
    std_dev_prediction = df[pred_col].std()
    std_dev_queue_distance = df[queue_col].std()

    # Calculate the correlation between 'Prediction_0' and 'Queue_distance_0'
    correlation = covariance / (std_dev_prediction * std_dev_queue_distance)

    print(f'Correlation between {pred_col} and {queue_col}: {correlation}')

plt.xlabel('Step number')
plt.suptitle('Task Time, Predictions, and Queue Distances over Time', y=1.02)
plt.tight_layout()
plt.show()
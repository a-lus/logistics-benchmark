import pandas as pd
import matplotlib.pyplot as plt
import colorsys
import seaborn as sns
import os

# Constants
LOG_DIR = os.path.expanduser("~/ray_results")
csv_file_path = os.path.join(LOG_DIR, "predictions.csv")

# Read the CSV file
df = pd.read_csv(csv_file_path)

print(df['Step number'].head())

# Prepare data for plotting
x = df['Step number']  # Line number
task_time = df['Task time']
predictions = df[['Prediction_0', 'Prediction_1', 'Prediction_2', 'Prediction_3', 'Prediction_4']]

# Plotting
plt.figure(figsize=(10, 6))

# Plotting 'Due' in green
plt.plot(x, task_time, label='Task time', color='green', linestyle='-')

# Plotting predictions with hues of red
num_predictions = len(predictions.columns)
hue_start = 0  # Start hue closer to pure red
hue_step = 0.04  # Step size for hue variation
prediction_colors = [(hue_start + i * hue_step, 0.9, 0.9) for i in range(num_predictions)]  # Varying hues closer to red
prediction_colors_rgb = [colorsys.hsv_to_rgb(*color) for color in prediction_colors]  # Convert to RGB
for i, col in enumerate(predictions.columns):
    print(predictions[col].head())
    plt.plot(x, predictions[col], label=col, color=prediction_colors_rgb[i], linestyle='-')


plt.xlabel('Step number')
plt.ylabel('Value')
plt.title('Task time vs Predictions over Time')
plt.legend()
# plt.grid(True)
plt.show()
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from gymnasium.spaces import Box
from ray.rllib.algorithms.algorithm import Algorithm

from marl.marl_environment import MARLEnv

OBSERVATION_SPACE = Box(
    low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
    high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
    shape=(6,),
    dtype=np.float32)


EP_LEN = 50
experiment_name = "SamplePolicy"
LOG_DIR = "~/ray_results"
CONFIG_FILE = os.getcwd() + "/maps/01_plant/01_plant.json"
# EXPERIMENT_PATH = "/home/andreja/ray_results/LB07_learned_one_2024-05-03_09-53-07/"
# EXPERIMENT_PATH = "/home/andreja/ray_results/LB07_learned_2024-05-05_18-07-24/" # WORKING
EXPERIMENT_PATH = "/home/andreja/ray_results/LB10D_learned_qentr_2024-05-09_01-04-12/"

POLICY = "discrete_policy"  # policy_0"
USE_PRESET_POLICY = False

# Path to the checkpoint with the highest number
checkpoint_dirs = glob.glob(os.path.join(
    EXPERIMENT_PATH, '*/checkpoint_*'))
# print(checkpoint_dirs) # DEBUG
if not checkpoint_dirs:
    print(
        f"No checkpoint directories found in: {EXPERIMENT_PATH} subdirectories.")
    CHECKPOINT_PATH = ""
else:
    CHECKPOINT_PATH = max(checkpoint_dirs, key=os.path.basename)

# CHECKPOINT_PATH = "/home/andreja/ray_results/LB09_learned_2024-05-07_08-44-54/PPO_MARLEnv_51fe4_00000_0_2024-05-07_08-44-58/checkpoint_000002"

print(f"Checkpoint path: {CHECKPOINT_PATH}")
input("Press Enter to continue...")

# algorithm = config.build()
algorithm = Algorithm.from_checkpoint(CHECKPOINT_PATH)
print(f'Algorithm: {algorithm.config}')


env_config = {
    'filename': CONFIG_FILE,
    "ep_len": EP_LEN,  # number of steps = tasks per episode
    "max_queue_length": 10,
    "verbose": True,
    "shared_reward": True,
    "log_dir": LOG_DIR,
    "experiment_name": experiment_name,
    "env_metrics": True
}

env = MARLEnv(env_config)
env.reset()


obs = np.array([0.0, 0.0, 0.5, 0.5, 0, 0], dtype=np.float32)

# Create an empty list to store data dictionaries
data = []

for minq in range(0, 6):
    for avgq in range(minq, 6):
        for myq in range(0, 10):
            for dist in range(0, 10):
                obs = np.array([dist*0.1, myq*0.1, 0.2, 0.7,
                                minq*0.1, avgq*0.1], dtype=np.float32)
                action_p0 = algorithm.compute_single_action(
                    obs, policy_id=POLICY)
                print(f'action_p0: {action_p0}')
                action_pre = {0: 0.0}
                if USE_PRESET_POLICY:
                    action_pre = algorithm.compute_single_action(
                        obs, policy_id="preset_policy")

                # Create a dictionary for each row of data
                if USE_PRESET_POLICY:
                    row_data = {
                        'dist': obs[0],
                        'myq': obs[1],
                        'minq': obs[4],
                        'avgq': obs[5],
                        'action_policy_0': action_p0[0],
                        'action_preset_policy': action_pre[0]
                    }
                elif POLICY == "discrete_policy":
                    row_data = {
                        'dist': obs[0],
                        'myq': obs[1],
                        'minq': obs[4],
                        'avgq': obs[5],
                        'action_policy_0': action_p0
                    }
                else:
                    row_data = {
                        'dist': obs[0],
                        'myq': obs[1],
                        'minq': obs[4],
                        'avgq': obs[5],
                        'action_policy_0': action_p0[0]
                    }

                # Append the dictionary to the list
                data.append(row_data)

                print(f'Observation: {obs}')
                if USE_PRESET_POLICY:
                    print(
                        f'Action: {action_p0} for policy_0 and {action_pre} for preset_policy')

# Create DataFrame from the list of dictionaries
df = pd.DataFrame(data)
# Save DataFrame to a CSV file
df.to_csv('sampled.csv', index=False)

print(df.head())
input("Press Enter to continue...")

# for minq in range(0, 6):
#     for avgq in range(minq, 6):
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')

#         # Filter the DataFrame based on conditions
#         filtered_df = df[(df['minq'] == minq*0.1) & (df['avgq'] == avgq*0.1)]
#         print(f'MinQ: {minq*0.1}, AvgQ: {avgq*0.1}')
#         print(filtered_df)

#         print(f'filtered_df: {filtered_df["myq"]}')
#         print(f'filtered_df: {filtered_df["dist"]}')
#         print(f'filtered_df: {filtered_df["action_policy_0"]}')
#         print(f'filtered_df: {filtered_df["action_preset_policy"]}')

#         # Plot the data
#         ax.scatter(filtered_df['dist'], filtered_df['myq'], filtered_df['action_policy_0'], marker='o')
#         ax.scatter(filtered_df['dist'], filtered_df['myq'], filtered_df['action_preset_policy'], marker='o')
#         ax.set_xlabel('Distance')
#         ax.set_ylabel('My Queue')
#         ax.set_zlabel('Action')
#         ax.set_title(f'minq: {minq*0.1:.1f}, avgq: {avgq*0.1:.1f}')
#         plt.show()


# # Create a single figure with a grid of subplots
# fig, axes = plt.subplots(6, 6, figsize=(18, 18), subplot_kw={'projection': '3d'})

# for minq in range(0, 6):
#     for avgq in range(minq, 6):
#         # Filter the DataFrame based on conditions
#         filtered_df = df[(df['minq'] == minq*0.1) & (df['avgq'] == avgq*0.1)]

#         # Check for NaN or infinite values and replace them with zeros
#         filtered_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

#         # # Convert actions to colors
#         # colors_policy_0 = cm.viridis(filtered_df['action_policy_0'] / filtered_df['action_policy_0'].max())
#         # colors_preset_policy = cm.viridis(filtered_df['action_preset_policy'] / filtered_df['action_preset_policy'].max())

#         # # Plot the data on the corresponding subplot
#         # ax = axes[minq, avgq-minq]
#         # ax.scatter(filtered_df['dist'], filtered_df['myq'], filtered_df['action_policy_0'], c=colors_policy_0, marker='o', label='policy_0')
#         # ax.scatter(filtered_df['dist'], filtered_df['myq'], filtered_df['action_preset_policy'], c=colors_preset_policy, marker='o', label='preset_policy')

#         ax = axes[minq, avgq-minq]
#         ax.scatter(filtered_df['dist'], filtered_df['myq'], filtered_df['action_policy_0'], c='r', marker='o', label='policy_0')
#         ax.scatter(filtered_df['dist'], filtered_df['myq'], filtered_df['action_preset_policy'], c='b', marker='o', label='preset_policy')


#         # Set labels and legend
#         ax.set_xlabel('Distance')
#         ax.set_ylabel('My Queue')
#         ax.set_zlabel('Action')
#         ax.set_title(f'minq: {minq*0.1:.1f}, avgq: {avgq*0.1:.1f}')
#         ax.legend()

# plt.tight_layout()
# plt.show()

# Create a single figure with a grid of subplots
for minq in range(0, 6):
    fig, axes = plt.subplots(1, 6-minq, figsize=((6-minq)*3, 3),
                             subplot_kw={'projection': '3d'})

    for avgq in range(minq, 6):
        # Filter the DataFrame based on conditions
        filtered_df = df[(df['minq'] == minq*0.1) & (df['avgq'] == avgq*0.1)]
        ax_nr = avgq-minq
        print(f'filtered_df: {filtered_df}')
        # Plot the data on the corresponding subplot
        ax = axes[ax_nr] if 6-minq > 1 else axes
        ax.scatter(filtered_df['dist'], filtered_df['myq'],
                   filtered_df['action_policy_0'], marker='o', label='policy_0')
        if USE_PRESET_POLICY:
            ax.scatter(filtered_df['dist'], filtered_df['myq'],
                       filtered_df['action_preset_policy'], marker='o', label='preset_policy')

        # Set labels and legend
        ax.set_xlabel('Distance')
        ax.set_ylabel('My Queue')
        ax.set_zlabel('Action')
        ax.set_title(f'minq: {minq*0.1:.1f},avgq: {avgq*0.1:.1f}')
        ax.legend()

    plt.tight_layout()
    plt.show()

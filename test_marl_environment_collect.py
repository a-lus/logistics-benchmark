import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from marl.marl_environment import MARLEnv

EP_LEN = 100
experiment_name = "LogisticsBenchmarkEnvironmentTest"
LOG_DIR = os.path.expanduser("~/ray_results")
CONFIG_FILE = os.path.join(os.getcwd(), "maps/01_plant/01_plant.json")

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
episode_count = 0

# Prepare to write to CSV
csv_file_path = os.path.join(LOG_DIR, "obs_samples.csv")
predictions_path = os.path.join(LOG_DIR, "predictions.csv") 

with open(predictions_path, 'a', newline='') as csvfile:
    fieldnames = ['Episode number', 'Step number','Task Start', 'Task End', 'Deadline', 'Task time'] \
        + [f'Prediction_{aid}' for aid in env._agent_ids] \
        + [f'Queue_distance_{aid}' for aid in env._agent_ids]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    episode_count += 1
    obs_curr = {aid: 0 for aid in env._agent_ids}
    obs_prev = {aid: 0 for aid in env._agent_ids}

    for i in range(EP_LEN):
        act_dict = {}
        current_tick = env._sim.env._elapsed_steps
        obs_prev = obs_curr
        # Sample random actions for all agents
        for agent_id in env.get_agent_ids():
            act_dict[agent_id] = np.array([env.action_space_sample([0])[0]], dtype=np.float32)
        obs, rew, done, ready, info = env.step(act_dict)


        predictions = {}
        for agent_id, agent_obs in obs.items():
            predictions[agent_id] = env.predict(agent_id, env.last_assigned_task)
            obs_curr[agent_id] = agent_obs[1]

        with open(predictions_path, 'a', newline='') as csvfile:
            writer2 = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer2.writerow({
                'Episode number': episode_count,
                'Step number': i,
                'Task Start': env.last_assigned_task.pick_up,
                'Task End': env.last_assigned_task.drop_off,
                'Deadline': env.last_assigned_task.deadline,
                'Task time': env.last_assigned_task.deadline - current_tick,
                **{f'Prediction_{aid}': pred for aid, pred in predictions.items()},
                **{f'Queue_distance_{aid}': o for aid, o in obs_prev.items()}
            })


        # Write to CSV file 
        for agent_id, ob in obs.items():
            writer.writerow([i, agent_id] + list(ob))

        if done['__all__']:
            env.reset()

print("Completed writing observations to CSV.")

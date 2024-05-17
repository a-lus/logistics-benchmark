
import datetime
import glob
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from ray import air, tune
# from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium.spaces import Box
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.spaces import space_utils


from marl.marl_environment import MARLEnv
from marl.lb_callbacks import LBCallbacks


OBSERVATION_SPACE = Box(
    low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
    high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
    shape=(6,),
    dtype=np.float32)

ACTION_SPACE = Box(
    low=0,
    high=1,
    shape=(1,),
    dtype=np.float32)


def random_policy_mapping_fn(agent_id, episode, worker):
    ''' Return random_policy. '''
    return "random_policy"


def preset_policy_mapping_fn(agent_id, episode, worker):
    ''' Return preset_policy. '''
    return "preset_policy"


def constant_action_policy_mapping_fn(agent_id, episode, worker):
    ''' Return constant_action_policy. '''
    return "constant_action_policy"


def learning_policy_mapping_fn(agent_id, episode, worker):
    ''' Return policy_0. '''
    return "policy_0"


def one_policy_mapping_fn(agent_id, episode, worker):
    ''' Return policy_0 for agent 0. '''
    return "policy_0" if agent_id == 0 else "preset_policy"


def get_policy_mapping(policy_str: str):
    ''' Return the policy mapping function. '''
    if policy_str == "random_policy":
        return random_policy_mapping_fn
    elif policy_str == "preset_policy":
        return preset_policy_mapping_fn
    elif policy_str == "constant_action_policy":
        return constant_action_policy_mapping_fn
    elif policy_str == "policy_0":
        return learning_policy_mapping_fn
    elif policy_str == "policy_0_one":
        return one_policy_mapping_fn
    else:
        raise ValueError("Unknown policy mapping function.")


os.environ.setdefault("RAY_DEDUP_LOGS", "0")

for i in range(1):

    EP_LEN = 100
    experiment_name = "LBeval_" + \
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_DIR = "~/ray_results"
    CONFIG_FILE = os.getcwd() + "/maps/01_plant/01_plant.json"
    EXPERIMENT_PATH = "/home/andreja/ray_results/LB07_learned_2024-05-05_18-07-24/"
    EXPERIMENT_PATH = os.path.expanduser(EXPERIMENT_PATH)
    POLICY = "policy_0"

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

    config = (
        # AlgorithmConfig(algo_class=PPO)
        PPOConfig()
        .environment(MARLEnv, env_config=env_config, disable_env_checking=True)
        .rollouts(
            num_rollout_workers=1,
            # num_envs_per_worker=1,
            rollout_fragment_length=128,
            # num_consecutive_worker_failures_tolerance=2
        )
        .framework("torch")
        .training(
            gamma=0.9,
            lr=0.005,
            train_batch_size=1024,
            model={
                "vf_share_layers": True,
                "fcnet_hiddens": [128, 128]
            },
            vf_loss_coeff=0.5,
        )
        .multi_agent(
            policies_to_train=["policy_0"],
            policies={
                "policy_0": (
                    None,
                    OBSERVATION_SPACE,
                    ACTION_SPACE,
                    {}),
                "random_policy": (
                    None,
                    OBSERVATION_SPACE,
                    ACTION_SPACE,
                    {}),
                "preset_policy": (
                    None,
                    OBSERVATION_SPACE,
                    ACTION_SPACE,
                    {})
            },
            policy_mapping_fn=get_policy_mapping("policy_0"),)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .callbacks(LBCallbacks)
        .debugging(log_level="DEBUG")
    )

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

    # algorithm = config.build()
    algorithm = Algorithm.from_checkpoint(CHECKPOINT_PATH)
    print(f'Algorithm: {algorithm.config}')

    # policy = algorithm.get_policy(policy_id='policy_0')
    # print(policy.get_weights())


    env = MARLEnv(env_config)
    obs, info = env.reset()

    # input("Press Enter to continue...")

    for i in range(2*EP_LEN):
        act_dict = {}
        # sample policy actions for all agents
        for i, obs_ in obs.items():
            acts = algorithm.compute_single_action(obs_, policy_id=POLICY)
            act_dict[i] = np.array(acts, dtype=np.float32)

        obs, rew, done, ready, info = env.step(act_dict)

        queue_len, delays = env.get_task_generator_metrics()

        if done['__all__'] is True:
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot queue_len
            for key, values in queue_len.items():
                ax1.plot(values, label=f'Agent {key}')

            ax1.set_xlabel('Index')
            ax1.set_ylabel('Value')
            ax1.set_title('Queue Length')
            ax1.legend()

            # Plot delays as a box plot
            delays_data = [delays[key] for key in delays.keys()]
            ax2.boxplot(delays_data, labels=[
                        f'Agent {key}' for key in delays])

            ax2.set_xlabel('Agent')
            ax2.set_ylabel('Time')
            ax2.set_title('Delays')

            # Adjust spacing between subplots
            plt.tight_layout()

            # Display the plot
            plt.show()

            # print(delays)

            env.reset()

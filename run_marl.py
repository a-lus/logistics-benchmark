
import datetime
import os
import random

import numpy as np
from ray import air, tune
# from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium.spaces import Box

from marl.marl_environment import MARLEnv
from marl.lb_callbacks import LBCallbacks


def random_policy_mapping_fn(agent_id, episode, worker):
    ''' Return random_policy. '''
    return "random_policy"

def one_policy_mapping_fn(agent_id, episode, worker):
    ''' Return policy_0. '''
    return "policy_0"

def get_policy_mapping(policy_str: str):
    ''' Return the policy mapping function. '''
    if policy_str == "random_policy":
        return random_policy_mapping_fn
    elif policy_str == "policy_0":
        return one_policy_mapping_fn
    else:
        raise ValueError("Unknown policy mapping function.")

os.environ.setdefault("RAY_DEDUP_LOGS", "0")

for i in range(1):

    experiment_name = "LogisticsBenchmarkTest_" + \
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_DIR = "~/ray_results"
    CONFIG_FILE = os.getcwd() + "/maps/00_example/00_example.json"

    env_config = {
        "filename": CONFIG_FILE,
        "ep_len": 100,  # number of steps = tasks per episode
        "max_queue_length": 10,
        "verbose": True,
        # "shared_reward": True,
        "log_dir": LOG_DIR,
        "experiment_name": experiment_name,
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
            gamma=0.6,
            lr=0.005,
            train_batch_size=256,
            model={
                "vf_share_layers": True,
                "fcnet_hiddens": [128, 128]
            },
        )
        .multi_agent(
            policies_to_train=["policy_0"],
            policies={
                "policy_0": (
                    None,
                    Box(
                        low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
                        high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
                        shape=(6,),
                        dtype=np.float32,),
                    Box(
                        low=0,
                        high=1,
                        shape=(1,),
                        dtype=np.float32),
                    {}),
                "random_policy": (
                    None,
                    Box(
                        low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
                        high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
                        shape=(6,),
                        dtype=np.float32,),
                    Box(
                        low=0,
                        high=1,
                        shape=(1,),
                        dtype=np.float32),
                    {})
            },
            policy_mapping_fn=get_policy_mapping("policy_0"),)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .callbacks(LBCallbacks)
        .debugging(log_level="DEBUG")
    )

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            name=experiment_name,
            local_dir=os.path.expanduser(LOG_DIR),
            stop={"training_iteration": 200},
            verbose=3,
            # checkpoint_config=air.CheckpointConfig(checkpoint_frequency=50)
        ),
        param_space=config.to_dict(),
    ).fit()

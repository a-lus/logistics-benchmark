
import datetime
import os
import random

import numpy as np
from ray import air, tune
# from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from gymnasium.spaces import Box, Discrete

from marl.marl_environment import MARLEnv
from marl.lb_callbacks import LBCallbacks
from marl.lb_policies import RandomPolicy, PresetPolicy, ConstantActionPolicy


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

ACTION_SPACE_DISCRETE = Discrete(2)

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

def discrete_policy_mapping_fn(agent_id, episode, worker):
    ''' Return discrete_policy. '''
    return "discrete_policy"

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
    elif policy_str == "discrete_policy":
        return discrete_policy_mapping_fn
    else:
        raise ValueError("Unknown policy mapping function.")


os.environ.setdefault("RAY_DEDUP_LOGS", "0")

for i in range(2):

    EXPERIMENT_NAME = "LB10D_preset_" + \
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_DIR = "~/ray_results"
    CONFIG_FILE = os.getcwd() + "/maps/01_plant/01_plant.json"

    env_config = {
        "filename": CONFIG_FILE,
        "ep_len": 50,  # number of steps = tasks per episode
        "max_queue_length": 10,
        "verbose": True,
        "shared_reward_coeff": 0.7,
        "log_dir": LOG_DIR,
        "experiment_name": EXPERIMENT_NAME,
        "env_metrics": False,
        "fake_rewards": True
    }

    config = (
        # AlgorithmConfig(algo_class=PPO)
        PPOConfig()
        .environment(MARLEnv, env_config=env_config, disable_env_checking=True)
        .rollouts(
            num_rollout_workers=1,
            # num_envs_per_worker=1,
            rollout_fragment_length=512,
            # num_consecutive_worker_failures_tolerance=2
        )
        .framework("torch")
        .training(
            gamma=0.96,
            lr=0.000005,
            train_batch_size=512,
            sgd_minibatch_size=128,
            model={
                "fcnet_hiddens": [56, 56],
                # "vf_share_layers": True,
            },
            vf_loss_coeff=0.2,
            # entropy_coeff=0.01,
        )
        .multi_agent(
            policies_to_train=[],
            policies={
                # "policy_0": (None, OBSERVATION_SPACE, ACTION_SPACE, {}),
                # "random_policy": (RandomPolicy, OBSERVATION_SPACE, ACTION_SPACE, {}),
                "preset_policy": (PresetPolicy, OBSERVATION_SPACE, ACTION_SPACE, {}),
                # "constant_action_policy": (ConstantActionPolicy, OBSERVATION_SPACE, ACTION_SPACE,{}),
            },
            policy_mapping_fn=get_policy_mapping("preset_policy"),)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .callbacks(LBCallbacks)
        .debugging(log_level="DEBUG")
    )

    """#
    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            name=experiment_name,
            local_dir=os.path.expanduser(LOG_DIR),
            stop={"training_iteration": 50},
            verbose=3,
            # checkpoint_config=air.CheckpointConfig(checkpoint_frequency=50)
        ),
        param_space=config.to_dict(),
    ).fit()
    """#"""

    """#
    config.multi_agent(
        policies={
            "random_policy": (RandomPolicy, OBSERVATION_SPACE, ACTION_SPACE, {}),
        },
        policy_mapping_fn=get_policy_mapping("random_policy"))
    experiment_name = "LB05_random_" + \
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    env_config["experiment_name"] = experiment_name
    config.environment(MARLEnv, env_config=env_config,
                       disable_env_checking=True)

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            name=experiment_name,
            local_dir=os.path.expanduser(LOG_DIR),
            stop={"training_iteration": 20},
            verbose=3,
            # checkpoint_config=air.CheckpointConfig(checkpoint_frequency=50)
        ),
        param_space=config.to_dict(),
    ).fit()
    # """

    config.multi_agent(
        policies_to_train=["discrete_policy"],
        policies={
            # "policy_0": (None, OBSERVATION_SPACE, ACTION_SPACE, {}),
            "discrete_policy": (None, OBSERVATION_SPACE, ACTION_SPACE_DISCRETE, {}),
        },
        policy_mapping_fn=get_policy_mapping("discrete_policy"))
    EXPERIMENT_NAME = "LB10D_learned_qentr_" + \
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    env_config["experiment_name"] = EXPERIMENT_NAME
    config.environment(MARLEnv, env_config=env_config,
                       disable_env_checking=True)

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            name=EXPERIMENT_NAME,
            local_dir=os.path.expanduser(LOG_DIR),
            stop={
                "training_iteration": 600
            },
            verbose=3,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=40, checkpoint_at_end=True)
        ),
        param_space=config.to_dict(),
    ).fit()

    # config.multi_agent(
    #     policies_to_train=["policy_0"],
    #     policies={
    #         "policy_0": (None, OBSERVATION_SPACE, ACTION_SPACE, {}),
    #         "preset_policy": (PresetPolicy, OBSERVATION_SPACE, ACTION_SPACE, {}),
    #     },
    #     policy_mapping_fn=get_policy_mapping("policy_0_one"))
    # experiment_name = "LB07_learned_one_" + \
    #     datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # env_config["experiment_name"] = experiment_name
    # config.environment(MARLEnv, env_config=env_config,
    #                    disable_env_checking=True)

    # tune.Tuner(
    #     "PPO",
    #     run_config=air.RunConfig(
    #         name=experiment_name,
    #         local_dir=os.path.expanduser(LOG_DIR),
    #         stop={
    #             "training_iteration": 500
    #         },
    #         verbose=3,
    #         checkpoint_config=air.CheckpointConfig(
    #             checkpoint_frequency=50, checkpoint_at_end=True)
    #     ),
    #     param_space=config.to_dict(),
    # ).fit()

    # # Create a new Tuner instance
    # trainable = PPO(config.to_dict())
    # tuner = tune.Tuner(
    #     trainable,
    #     run_config=air.RunConfig(
    #         name=EXPERIMENT_NAME,
    #         local_dir=os.path.expanduser(LOG_DIR),
    #         stop={
    #             "training_iteration": 600
    #         },
    #         verbose=3,
    #         checkpoint_config=air.CheckpointConfig(
    #             checkpoint_frequency=40, checkpoint_at_end=True)
    #     ),
    #     param_space=config.to_dict(),
    # )

    # EXPERIMENT_NAME = "LB09_learned_2024-05-07_13-33-57/PPO_MARLEnv_b3fe9_00000_0_2024-05-07_13-34-02"
    # CHECKPOINT_PATH = os.path.expanduser(LOG_DIR) + "/" + EXPERIMENT_NAME + "/checkpoint_000004"
    # print(f"Checkpoint path: {CHECKPOINT_PATH}")
    # input("Press Enter to continue...")
    # # Restore the checkpoint
    # tuner = tune.Tuner.restore(CHECKPOINT_PATH, trainable="PPO", resume_unfinished=False)

    # # Continue training
    # tuner.fit()

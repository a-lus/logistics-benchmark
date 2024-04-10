from typing import Optional, Tuple
import os
import random
import shutil
import sys
import numpy as np

from gymnasium.spaces import Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from benchmark.simulation import Simulation


def debugging_stop(text: str = None):
    ''' Stop for debugging. '''
    input(
        f'Press Enter to continue...\n{text}' if text else 'Press Enter to continue...')


def save_config(
        config_file_path: str,
        experiment_name: str,
        log_dir: str) -> None:
    '''
    Copies the provided file to a specific directory for the experiment.
    Creates a subfolder for the experiment and copies the config file into it.
    The name of the file is preserved.
    '''

    log_dir = os.path.expanduser(log_dir)
    # Add subfolder for experiment
    log_dir = os.path.join(log_dir, experiment_name)

    print(log_dir)

    # If path does not exist, create it
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as err:
            print("Error:", err)
            sys.exit(1)

    # Check if folder exists
    if not os.path.isdir(log_dir):
        print("Error: Path is not a directory.")
        sys.exit(1)

    # Determine the filename from the provided path
    file_name = os.path.basename(config_file_path)
    destination_path = os.path.join(log_dir, file_name)

    # Copy the file
    try:
        shutil.copy(config_file_path, destination_path)
        print(f'{file_name} copied to {log_dir}')
    except FileNotFoundError as err:
        print("Error:", err)
        sys.exit(1)
    except Exception as err:
        print(f"Error: {err}")
        sys.exit(1)


class MARLEnv(MultiAgentEnv):
    """    Environment for MARL using FMS simulation.    """

    def __init__(self, config: dict):
        super().__init__()

        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.int32),
            high=np.array([100, 100, 100, 100, 100, 100], dtype=np.int32),
            shape=(6,),
            dtype=np.int32,
        )
        self.action_space = Box(
            low=0, high=1, shape=(1,), dtype=np.float32)
        self._action_space_in_preferred_format = Box(
            low=0, high=1, shape=(1,), dtype=np.float32)

        self._verbose = config["verbose"] if 'verbose' in config else True
        self._config = config

        self._sim = Simulation(self._config['filename'])
        self._tasks = []
        self._step = 0

        self._max_iterations = 200
        self._max_steps = self._config['ep_len'] if 'ep_len' in self._config else 500

        # Inicializacija (RL) agentov
        self._agent_ids = list(range(len(self._sim.agvs)))

        # Save config file
        save_config(
            config_file_path=self._config['filename'],
            experiment_name=self._config['experiment_name'],
            log_dir=self._config['log_dir']
        )

    def reset(self, *, seed=None, options=None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """ Resets the environment. """
        self._sim.soft_reset()
        self._tasks = []
        self._step = 0
        self.prnt(
            f'\n===== Resetting environment ... \
                step: {self._step}, iteration: {self._sim.env._elapsed_steps}, \
                    max_iter = {self._max_iterations}')

        obs_dict, info_dict = {}, {}
        obs_dict = self.get_obs()
        info_dict = self.get_info(obs_dict)

        self._step += 1

        # debugging_stop()  # DEBUGGING

        return obs_dict, info_dict

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        self.prnt(
            f'\n===== Stepping environment ... \
                step: {self._step}, iteration: {self._sim.env._elapsed_steps}, \
                    max_iter = {self._max_iterations}')
        self.prnt(f'action_dict: {action_dict}')

        # Execute actions
        obs_dict = self.get_obs(action_dict)
        reward_dict = self.get_rewards()
        info_dict = self.get_info(obs_dict)

        # Check termination conditions
        done_dict = ({"__all__": False} if (
            self._sim.env._elapsed_steps < self._max_iterations and
            self._step < self._max_steps) else {"__all__": True})

        self._step += 1

        return obs_dict, reward_dict, done_dict, {"__all__": False}, info_dict

    def get_rewards(self):
        rewards_dict = {}
        for agent_id in self._agent_ids:
            delays = self._sim.agvs[agent_id].delays
            # keep positive delays
            delays = [delay if delay > 0 else 0 for delay in delays]
            # clip to 100 and scale to 1
            delays = [0.01 * delay if delay < 100 else 100 for delay in delays]
            delays = [-delay for delay in delays]  # make them negative
            rewards_dict[agent_id] = np.average(
                delays) if delays else 0  # average
            self._sim.agvs[agent_id].delays = []

        rewards_dict_copy = rewards_dict.copy()
        for agent_id, reward in rewards_dict_copy.items():
            # calculate share for current agent
            share = reward / len(rewards_dict_copy)
            for other_agent_id in rewards_dict:
                if other_agent_id != agent_id:
                    # add the share to other agents
                    rewards_dict[agent_id] += share

        return rewards_dict

    def get_info(self, obs_dict: MultiAgentDict) -> MultiAgentDict:
        ''' Put values from obs_dict to info. Usefull for logging and debugging. '''
        ret_dict = {}
        if obs_dict:
            for k, val in obs_dict.items():
                ret_dict.update({k: {"vals": val,
                                     "training_enabled": True}})

        return ret_dict

    def get_obs(self, action_dict: Optional[MultiAgentDict] = None) -> MultiAgentDict:
        """ Get observation from simulator. """
        self.prnt(">>> get_obs called.")

        def has_nan(arr_list):
            for arr in arr_list:
                if np.isnan(arr).any():
                    return True
            return False

        # DISPATCH if action_dict is not None
        if action_dict:
            self.prnt(f"a: {action_dict}")
            # Check for NaN
            if has_nan(action_dict.values()):
                print("Error: NaN in action_dict.")
                sys.exit(1)

            # check which bid is the highest
            max_value = max(action_dict.values())
            max_keys = [k for k, v in action_dict.items() if v == max_value]
            agent_id = random.choice(max_keys)

            # assign to the agent (dispatch)
            self._sim.agvs[agent_id].assign_task(self._tasks[-1])
            self._tasks.pop()
        else:
            self._sim.generate_tasks()
            self._sim.releaser.get_tasks()

        # step agents
        # while _tasks empty
        while not self._tasks:
            # move agents
            self._sim.step_agents()
            self._sim.check_task_states()
            # generate tasks (and pass them to releaser)
            self._sim.generate_tasks()
            # if releaser has tasks get them
            if self._sim.releaser.get_tick_next() == self._sim.env._elapsed_steps:
                self._tasks = self._sim.releaser.get_tasks()
        # get current task
        current_task = self._tasks[-1]

        # prepare observations for current task
        obs_dict = {}

        shortest_distances_for_tasks = []
        for agent_id in self._agent_ids:
            # initialize observations for agent
            obs_dict[agent_id] = np.zeros(self.observation_space.shape[0])
            agent_position = self._sim.env.agents[agent_id].position
            # OBSERVATION 1: distance to pick-up
            obs_dict[agent_id][0] = self._sim.shortest_path_lengths[agent_position][current_task.pick_up]

            # OBSERVATION 2: shortest distance for all tasks in AGV queue
            shortest_distance_for_tasks = 0
            task_in_work = self._sim.agvs[agent_id].task_in_work
            if task_in_work:
                # first add distance for finnishing the task in work
                shortest_distance_for_tasks += self._sim.shortest_path_lengths[agent_position][task_in_work.drop_off]
                # then add distances for all tasks in queue
                queue = list(self._sim.agvs[agent_id].tasks.queue)
                if queue:
                    last_drop_off = task_in_work.drop_off
                    for task in queue:
                        shortest_distance_for_tasks += self._sim.shortest_path_lengths[last_drop_off][task.pick_up]
                        shortest_distance_for_tasks += self._sim.shortest_path_lengths[task.pick_up][task.drop_off]
                        last_drop_off = task.drop_off
            obs_dict[agent_id][1] = shortest_distance_for_tasks
            shortest_distances_for_tasks.append(shortest_distance_for_tasks)

            # OBSERVATION 3: task due (from current step)
            obs_dict[agent_id][2] = current_task.deadline - \
                self._sim.env._elapsed_steps

            # OBSERVATION 4: task distance (pick-up to drop-off)
            obs_dict[agent_id][3] = self._sim.shortest_path_lengths[current_task.pick_up][current_task.drop_off]

        for agent_id in self._agent_ids:
            remaining_shortest_distances = [d for i, d in enumerate(
                shortest_distances_for_tasks) if i != agent_id]
            # OBSERVATION 5: minimum of distance for all tasks in AGV queue (all agents except i)
            obs_dict[agent_id][4] = min(
                remaining_shortest_distances, default=0)
            # OBSERVATION 6: average of distance for all tasks in AGV queue (all agents except i)
            average_shortest_distance = int(sum(remaining_shortest_distances) / len(
                remaining_shortest_distances)) if remaining_shortest_distances else 0
            obs_dict[agent_id][5] = average_shortest_distance

        # # Get observations
        # obs_dict = {}
        # for amr in self.sim.agvs:
        #     obs = np.array(amr.get_observations())
        #     # Mask out zeros and convert to log2
        #     log2_obs = obs.copy()  # Copy obs to a new array
        #     mask = obs != 0  # Define the mask
        #     # Apply the mask and the log2 transformation
        #     log2_obs[mask] = np.log2(obs[mask])
        #     obs_dict[amr.get_id()] = log2_obs.astype(
        #         np.int32)  # Store the result

        self.prnt(f'obs_dict: {obs_dict}')

        # convert from np.int64 to np.int32 and clip to [0, 100] # TODO: check if this is needed
        for k, v in obs_dict.items():
            obs_dict[k] = np.clip(v, 0, 100).astype(np.int32)

        return obs_dict

    def prnt(self, print_input) -> None:
        """ Print if verbose. """
        if self._verbose:
            print(print_input)

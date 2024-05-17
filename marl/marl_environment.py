from typing import Optional, Tuple
import copy
import math
import os
import random
import shutil
import sys
import numpy as np

from gymnasium.spaces import Box, Discrete
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
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )
        # self.action_space = Box(
        #     low=0, high=1, shape=(1,), dtype=np.float32)
        # self._action_space_in_preferred_format = Box(
        #     low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.action_space = Discrete(2)
        self._action_space_in_preferred_format = Discrete(2)

        self._verbose = config["verbose"] if 'verbose' in config else True
        self._config = config

        self._sim = Simulation(self._config['filename'])
        self._tasks = []
        self.last_assigned_task = None
        self._step = 0  # one step is one task

        self._max_sim_steps = self._sim.settings["number_of_steps"] \
            if 'number_of_steps' in self._sim.settings else 1000        # max tasks
        self._max_steps = self._config['ep_len'] \
            if 'ep_len' in self._config else 500
        self._shared_reward_coeff = self._config['shared_reward_coeff'] \
            if 'shared_reward_coeff' in self._config else 1
        self._fake_rewards = self._config['fake_rewards'] \
            if 'fake_rewards' in self._config else False

        # Inicializacija (RL) agentov
        self._agent_ids = list(range(len(self._sim.agvs)))
        self._max_queue_length = self._config['max_queue_length'] \
            if 'max_queue_length' in self._config else 10
        self._longest_queue = 0
        self._agent_queues = {agent_id: 0 for agent_id in self._agent_ids}
        self._simulated_delays = {agent_id: 0 for agent_id in self._agent_ids}
        self.obs_dict = {}
        self.obs_prev_dict = {}

        # Environment properties (for obs)
        self.max_distance_roadmap = 0
        # Find max distance in roadmap
        for _, loc in self._sim.shortest_path_lengths.items():
            for distance in loc.values():
                if distance > self.max_distance_roadmap:
                    self.max_distance_roadmap = distance
                    # print(f'start: {start} loc: {loc}, distance: {distance}')
        self.max_distance_stations = 0
        self.min_distance_pick_up_drop_off = 1000000
        self.max_distance_pick_up_drop_off = 0
        # Find max and min distance between pick-up and drop-off stations
        for pick_up in self._sim.layout.pick_ups + self._sim.layout.pick_up_drop_offs:
            for drop_off in self._sim.layout.pick_up_drop_offs + self._sim.layout.drop_offs:
                distance = self._sim.shortest_path_lengths[pick_up][drop_off]
                if distance < self.min_distance_pick_up_drop_off and distance != 0:
                    self.min_distance_pick_up_drop_off = distance
                if distance > self.max_distance_pick_up_drop_off:
                    self.max_distance_pick_up_drop_off = distance
        # Find max distance between all stations
        self.max_distance_stations = self.max_distance_pick_up_drop_off
        for drop_off in self._sim.layout.pick_up_drop_offs + self._sim.layout.drop_offs:
            for pick_up in self._sim.layout.pick_ups + self._sim.layout.pick_up_drop_offs:
                distance = self._sim.shortest_path_lengths[drop_off][pick_up]
                if distance > self.max_distance_stations:
                    self.max_distance_stations = distance

        # Save config file
        save_config(
            config_file_path=self._config['filename'],
            experiment_name=self._config['experiment_name'],
            log_dir=self._config['log_dir']
        )

        # Test metrics
        self._env_metrics = self._config['env_metrics'] if 'env_metrics' in self._config else False
        self.metric_delay_dict = {}
        self.metric_queue_len_dict = {}

    def reset(self, *, seed=None, options=None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """ Resets the environment. """
        self._sim.soft_reset()
        self._tasks = []
        self._step = 0
        self._longest_queue = 0
        self.prnt(
            f'\n===== Resetting environment ... \
                step: {self._step}, simulation elapsed steps: {self._sim.env._elapsed_steps}, \
                    max_iter = {self._max_sim_steps}')

        info_dict = {}
        self.obs_dict = self.get_obs()
        info_dict = self.get_info(self.obs_dict)

        self._step += 1

        # debugging_stop()  # DEBUGGING

        return self.obs_dict, info_dict

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        self.prnt(
            f'\n===== Stepping environment ... \
                step: {self._step}, simulation elapsed steps: {self._sim.env._elapsed_steps}, \
                    max_iter = {self._max_sim_steps}')
        self.prnt(f'action_dict: {action_dict}')

        # Execute actions
        self.obs_prev_dict = self.obs_dict
        self.obs_dict = self.get_obs(action_dict)

        # Check termination conditions
        done_dict = {"__all__": False}

        self._step += 1

        # Check if simulation is finished
        reward_dict = {}
        if self._step > self._max_steps:
            done_dict = {"__all__": True}
            self.finish_sim()
        elif self._sim.env._elapsed_steps > self._max_sim_steps:
            done_dict = {"__all__": True}
            # for agent_id in reward_dict:
            #     reward_dict[agent_id] -= 5
        elif self._longest_queue > self._max_queue_length:
            done_dict = {"__all__": True}
            self.finish_sim()

        # Get rewards and info
        reward_dict = self.get_rewards(action_dict) if not self._fake_rewards else self.get_fake_rewards(action_dict)
        info_dict = self.get_info(self.obs_dict)

        if done_dict["__all__"]:
            self.prnt(
                f'\n===== Simulation finished ... \
                    step: {self._step}, simulation elapsed steps: {self._sim.env._elapsed_steps}, \
                        max_iter = {self._max_sim_steps}')
            # debugging_stop()  # DEBUGGING

        return self.obs_dict, reward_dict, done_dict, {"__all__": False}, info_dict

    def get_rewards(self, actions: MultiAgentDict) -> MultiAgentDict:
        ''' Calculate rewards for agents. '''

        self.save_task_generator_metrics()

        rewards_dict = {}
        for agent_id in self._agent_ids:
            delays = self._sim.agvs[agent_id].delays
            # keep positive delays
            delays = [delay if delay > 0 else -10 for delay in delays]
            # average delays
            delay = np.average(delays) if delays else 0
            # clip to [0, 100]
            delay = np.clip(delay, -10, 100)
            rewards_dict[agent_id] = -delay
            self._sim.agvs[agent_id].delays = []

        # Rewards for queue length
        total_tasks = sum(
            queue_len for queue_len in self._agent_queues.values())
        queues_entropy_max = -math.log(1/len(self._agent_queues))
        p = {}
        epsilon = 0.0001

        rewards_dict_copy = copy.deepcopy(rewards_dict)
        for agent_id, reward in rewards_dict_copy.items():
            # calculate share for current agent
            share = reward / len(rewards_dict_copy)
            for other_agent_id in rewards_dict:
                if other_agent_id != agent_id:
                    # add the share to other agents
                    rewards_dict[other_agent_id] += share * \
                        self._shared_reward_coeff

            # Rewards for queue length
            p[agent_id] = self._agent_queues[agent_id] / \
                total_tasks if total_tasks > 0 else 0
            
            # Add reward for bidding too low if empty
            if self.obs_prev_dict[agent_id][1] == 0 and actions[agent_id] < 0.3:
                rewards_dict[agent_id] -= 20
            
            # # Add reward for bidding too high if full
            # if self.obs_prev_dict[agent_id][1] > self.obs_prev_dict[agent_id][5] and actions[agent_id] > 0.7:
            #     rewards_dict[agent_id] -= 20

        if total_tasks > len(self._agent_queues):
            queues_entropy = - \
                sum(p[agent_id] * math.log(p[agent_id] + epsilon)
                    for agent_id in p)
            normalized_queues_entropy = queues_entropy / queues_entropy_max
            alpha = 70
            for agent_id in rewards_dict:
                rewards_dict[agent_id] -= alpha * normalized_queues_entropy

        # Normalize rewards (linearly from -1 to 0)
        max_reward = 250
        for agent_id, reward in rewards_dict.items():
            reward = np.clip(reward, -max_reward, 0)
            rewards_dict[agent_id] = reward / (max_reward)

        self.prnt(f'rewards_dict: {rewards_dict}')

        return rewards_dict

    def get_info(self, obs_dict: MultiAgentDict) -> MultiAgentDict:
        ''' Put values from obs_dict to info. Usefull for logging and debugging. '''
        ret_dict = {}
        if obs_dict:
            for k, val in obs_dict.items():
                ret_dict.update({k: {"vals": val,
                                     "training_enabled": True}})

        return ret_dict

    def finish_sim(self) -> None:
        ''' Finish simulation. '''
        self.prnt(">>> finish_sim called.")
        start = self._sim.env._elapsed_steps
        tasks_to_finish = True

        while tasks_to_finish:
            self._sim.step_agents()
            self._sim.check_task_states()

            tasks_to_finish = False
            for agent_id in self._agent_ids:
                if self._sim.agvs[agent_id].task_in_work or self._sim.agvs[agent_id].tasks:
                    tasks_to_finish = True
                    break
        
        for agent_id in self._agent_ids:
            self.prnt(f'Delays for agent {agent_id}: {self._sim.agvs[agent_id].delays}')

        self.prnt(
            f'Simulation finished in {self._sim.env._elapsed_steps - start} steps.')

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
            self.last_assigned_task = self._tasks[-1]

            if self._fake_rewards:
                self._simulate_delays(self._tasks[-1])

            # print(f":=:=:=:=:=:=:=: Agent {agent_id} assigned task {self._tasks[-1]}")
            # for aid in self._agent_ids:
            #     prediction = self.predict(aid, self._tasks[-1])
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

        # # # self.predict(0, current_task)
        # # # debugging_stop()

        # prepare observations for current task
        obs_dict = {}

        self._longest_queue = 0
        shortest_distances_normalized = []
        shortest_distances_normalization_factor = 2 * \
            self._max_queue_length * self.max_distance_stations
        for agent_id in self._agent_ids:
            # initialize observations for agent
            obs_dict[agent_id] = np.zeros(self.observation_space.shape[0])
            agent_position = self._sim.env.agents[agent_id].position
            # OBSERVATION 1: distance to pick-up
            obs_dict[agent_id][0] = self._sim.shortest_path_lengths[agent_position][current_task.pick_up] / \
                self.max_distance_roadmap

            tasks_in_work = 0
            # OBSERVATION 2: shortest distance for all tasks in AGV queue
            shortest_distance_for_tasks = 0
            task_in_work = self._sim.agvs[agent_id].task_in_work
            if task_in_work:
                # first add distance for finishing the task in work
                shortest_distance_for_tasks += self._sim.shortest_path_lengths[agent_position][task_in_work.drop_off]
                # then add distances for all tasks in queue
                # queue = list(self._sim.agvs[agent_id].tasks)
                queue = self._sim.agvs[agent_id].tasks
                if queue:
                    last_drop_off = task_in_work.drop_off
                    for task in queue:
                        tasks_in_work += 1
                        shortest_distance_for_tasks += self._sim.shortest_path_lengths[last_drop_off][task.pick_up]
                        shortest_distance_for_tasks += self._sim.shortest_path_lengths[task.pick_up][task.drop_off]
                        last_drop_off = task.drop_off
            if tasks_in_work > self._longest_queue:
                self._longest_queue = tasks_in_work
            self._agent_queues[agent_id] = tasks_in_work    # log queue length
            shortest_distance_normalized = shortest_distance_for_tasks / \
                shortest_distances_normalization_factor
            obs_dict[agent_id][1] = shortest_distance_normalized
            shortest_distances_normalized.append(shortest_distance_normalized)

            # OBSERVATION 3: task due (from current step)
            min_time = self.min_distance_pick_up_drop_off + \
                self._sim.task_generators[0].time_buffer_min
            max_time = self.max_distance_pick_up_drop_off + \
                self._sim.task_generators[0].time_buffer_max
            obs_dict[agent_id][2] = (
                (current_task.deadline - self._sim.env._elapsed_steps) - min_time) / (max_time - min_time)

            # OBSERVATION 4: task distance (pick-up to drop-off)
            obs_dict[agent_id][3] = self._sim.shortest_path_lengths[current_task.pick_up][current_task.drop_off] / \
                self.max_distance_stations

        for agent_id in self._agent_ids:
            remaining_shortest_distances = [d for i, d in enumerate(
                shortest_distances_normalized) if i != agent_id]
            # OBSERVATION 5: minimum of distance for all tasks in AGV queue (all agents except i)
            obs_dict[agent_id][4] = min(
                remaining_shortest_distances, default=0)*4
            # OBSERVATION 6: average of distance for all tasks in AGV queue (all agents except i)
            average_shortest_distance = sum(remaining_shortest_distances) / len(
                remaining_shortest_distances) if remaining_shortest_distances else 0
            obs_dict[agent_id][5] = average_shortest_distance*2

        # Clip all observations to [0, 1]
        for agent_id, obs in obs_dict.items():
            obs_dict[agent_id] = np.clip(obs, 0.0, 1.0)

        # For DEBUGGING: check if any obs > 1
        for agent_id, obs in obs_dict.items():
            if np.any(obs > 1):
                debugging_stop(
                    f'Error: obs > 1 for agent {agent_id}. Obs: {obs}')

        self.prnt(f'obs_dict: {obs_dict}')

        return obs_dict

    def prnt(self, print_input) -> None:
        """ Print if verbose. """
        if self._verbose:
            print(print_input)

    def get_sim_iterations(self) -> int:
        """ Get number of simulation iterations."""
        return self._sim.env._elapsed_steps

    def get_longest_queue(self) -> int:
        """ Get longest queue."""
        return self._longest_queue

    def save_task_generator_metrics(self):
        '''
        Save metrics for task generator.
        Saving list of delays and queue length for each agent.
        '''
        if self._env_metrics:
            for agent_id in self._agent_ids:
                if agent_id in self.metric_delay_dict:
                    self.metric_delay_dict[agent_id] += self._sim.agvs[agent_id].delays
                    self.metric_queue_len_dict[agent_id].append(len(
                        self._sim.agvs[agent_id].tasks))
                else:
                    self.metric_delay_dict[agent_id] = self._sim.agvs[agent_id].delays
                    self.metric_queue_len_dict[agent_id] = [len(
                        self._sim.agvs[agent_id].tasks)]

    def get_task_generator_metrics(self):
        ''' Return metrics for task generator. '''
        if self._env_metrics:
            return self.metric_queue_len_dict, self.metric_delay_dict
        else:
            self.prnt("Error: Environment metrics are not enabled.")
            return None, None

    def get_sim_iterations_per_task(self):
        ''' Return number of simulation iterations per task. '''
        if self._env_metrics:
            return self._sim.env._elapsed_steps / self._step
        else:
            self.prnt("Error: Environment metrics are not enabled.")
            return None

    def predict(self, agent_id: int, task) -> int:
        ''' Predict the time for the task. '''
        # print(f'\n*** STARTING PREDICTION ***\n*** Copy sim time before task execution: {self._sim.env._elapsed_steps}')
        current_sim = copy.deepcopy(self._sim)
        # print(f'agents plan before assignment: {current_sim.agvs[agent_id].tasks}; task_in_work: {current_sim.agvs[agent_id].task_in_work}')
        current_sim.agvs[agent_id].assign_task(task)
        # print(f'agents plan after assignment: {current_sim.agvs[agent_id].tasks}; task_in_work: {current_sim.agvs[agent_id].task_in_work}')

        agent_path = [current_sim.env.agents[agent_id].position]

        starting = current_sim.env._elapsed_steps
        current_sim.step_agents()
        current_sim.check_task_states()
        agent_path.append(current_sim.env.agents[agent_id].position)

        # print(f'agents plan after one step: {current_sim.agvs[agent_id].tasks}; task_in_work: {current_sim.agvs[agent_id].task_in_work}')

        while current_sim.agvs[agent_id].task_in_work or current_sim.agvs[agent_id].tasks:
            current_sim.step_agents()
            current_sim.check_task_states()
            agent_path.append(current_sim.env.agents[agent_id].position)

        task_time = current_sim.env._elapsed_steps - starting
        delay = current_sim.env._elapsed_steps - task.deadline
        # if task_time < 10:
        #     debugging_stop(f'-_-_-_-_-_-_-_-_-_-_-  agent path: {agent_path}')

        # print(f'*** Copy sim time after task execution: {current_sim.env._elapsed_steps} (starting at {starting})')
        print(f'Predicted time for task for {agent_id}: {task_time}, \
delay: {delay} (current sim time: {current_sim.env._elapsed_steps}, \
deadline: {task.deadline})')
        return task_time

    def _simulate_delays(self, task):
        ''' Return simulated delays. '''

        # for agent_id in self._agent_ids:
        #     self._simulated_delays[agent_id] = self.predict(agent_id, task)

        for agent_id in self._agent_ids:
            agent_position = self._sim.env.agents[agent_id].position
            last_drop_off = agent_position
            # estimate delay based on difference between task deadline, 
            # task due (obs_prev[2]), task distance (obs_prev[3]),
            # and agent's queue length (obs_prev[1])
            estimated_distance_for_tasks = 0
            task_in_work = self._sim.agvs[agent_id].task_in_work
            if task_in_work:
                # first add distance for finishing the task in work
                estimated_distance_for_tasks += self._sim.shortest_path_lengths[agent_position][task_in_work.drop_off]
                # then add distances for all tasks in queue
                # queue = list(self._sim.agvs[agent_id].tasks)
                queue = self._sim.agvs[agent_id].tasks
                last_drop_off = task_in_work.drop_off
                if queue:
                    for tsk in queue:
                        estimated_distance_for_tasks += self._sim.shortest_path_lengths[last_drop_off][tsk.pick_up]
                        estimated_distance_for_tasks += self._sim.shortest_path_lengths[tsk.pick_up][tsk.drop_off]
                        last_drop_off = tsk.drop_off
            estimated_distance_for_tasks += self._sim.shortest_path_lengths[last_drop_off][task.pick_up]
            estimated_distance_for_tasks += self._sim.shortest_path_lengths[task.pick_up][task.drop_off]
            
            due_time = task.deadline - self._sim.env._elapsed_steps

            self._simulated_delays[agent_id] = estimated_distance_for_tasks - due_time


    def get_fake_rewards(self, actions: MultiAgentDict = None) -> MultiAgentDict:
        ''' Return fake rewards. '''

        rewards_dict = {}
        # for agent_id in self._agent_ids:
        #     delay = self._simulated_delays[agent_id]
        #     correct_bid = math.exp(-0.1*(delay + 10)**2)
        #     rewards_dict[agent_id] = -abs(correct_bid - actions[agent_id][0])
            # print(f'Agent {agent_id} delay: {delay}, reward: {rewards_dict[agent_id]};  exp = {-0.1*abs(delay + 10)}')
        
        for agent_id in self._agent_ids:
            delay = self._simulated_delays[agent_id]
            if delay > 0:
                rewards_dict[agent_id] = -1
            else:
                rewards_dict[agent_id] = 1
        
        # Rewards for queue length
        total_tasks = sum(
            queue_len for queue_len in self._agent_queues.values())
        queues_entropy_max = -math.log(1/len(self._agent_queues))
        p = {}
        epsilon = 0.0001

        for agent_id, queues in self._agent_queues.items():
            # Rewards for queue length
            p[agent_id] = queues / \
                total_tasks if total_tasks > 0 else 0

        if total_tasks > len(self._agent_queues):
            queues_entropy = - \
                sum(p[agent_id] * math.log(p[agent_id] + epsilon)
                    for agent_id in p)
            normalized_queues_entropy = queues_entropy / queues_entropy_max
            alpha = 0.5
            for agent_id in rewards_dict:
                rewards_dict[agent_id] -= alpha * normalized_queues_entropy

        self.prnt(f'[get_fake_rewards] rewards_dict: {rewards_dict}')


        return rewards_dict


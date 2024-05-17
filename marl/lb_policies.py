from typing import Dict, List, Optional, Union

import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Box
from ray.rllib.evaluation import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights, TensorStructType, TensorType


class RandomPolicy(Policy):
    """Hand-coded policy that returns random actions.
    From: https://github.com/ray-project/ray/blob/443395bf3e97ab0653b6d6000c06d829babbe740/rllib/examples/policy/random_policy.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.get("ignore_action_bounds", False) and isinstance(
            self.action_space, Box
        ):
            self.action_space_for_sampling = Box(
                -float("inf"),
                float("inf"),
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_for_sampling = self.action_space

    @override(Policy)
    def init_view_requirements(self):
        super().init_view_requirements()
        vr = self.view_requirements[SampleBatch.INFOS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False

    @override(Policy)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List["Episode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ):
        obs_batch_size = len(tree.flatten(obs_batch)[0])
        actions = [
            self.action_space_for_sampling.sample()
            for _ in range(obs_batch_size)
        ]
        actions = [2 * (action - 0.5) for action in actions]
        actions = np.array(actions).reshape(obs_batch_size, -1)

        return actions, [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions: Union[List[TensorType], TensorType],
        obs_batch: Union[List[TensorType], TensorType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Optional[Union[List[TensorType], TensorType]] = None,
        prev_reward_batch: Optional[Union[List[TensorType], TensorType]] = None,
        actions_normalized: bool = True,
        in_training: bool = True,
    ) -> TensorType:
        return np.random.random(size=len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(
                    lambda s: s[None], self.observation_space.sample()
                ),
            }
        )
    

class PresetPolicy(Policy):
    """Hand-coded policy that returns predefined actions.
    From: https://github.com/ray-project/ray/blob/443395bf3e97ab0653b6d6000c06d829babbe740/rllib/examples/policy/random_policy.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        if self.config.get("ignore_action_bounds", False) and isinstance(
            self.action_space, Box
        ):
            self.action_space_for_sampling = Box(
                -float("inf"),
                float("inf"),
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_for_sampling = self.action_space

    @override(Policy)
    def init_view_requirements(self):
        super().init_view_requirements()
        # Disable for_training and action attributes for SampleBatch.INFOS column
        # since it can not be properly batched.
        vr = self.view_requirements[SampleBatch.INFOS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False

    @override(Policy)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List["Episode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ):
        '''
        Observations:
        obs0: my_distance_to_pick_up_location
        obs1: my_distance_for_all_tasks
        obs2: tsk_until_due
        obs3: tsk_distance
        obs4: min_plan
        obs5: avg_plan
        
        '''
        actions = []

        # print(f"obs_batch = {obs_batch}") # DEBUG

        # Extract the observations
        for observations in obs_batch:
        #  = obs_batch.get('observations', [])

            obs0, obs1, _, _, obs4, obs5 = observations
            obs_min = obs4/4
            obs_avg = obs5/2

            if obs1 == 0:           # If there are no tasks
                action = 1 - 0.1 * obs0
                # print(f'PresetPolicy. condition: obs1 == 0, action = {action}')
            elif obs1 <= obs_min:      # If my distance to all tasks is less than the min plan
                action = 0.6 + 0.2 * (obs_min - obs1 - obs0)
                # print(f'PresetPolicy. condition: obs1 <= obs4, action = {action}')
            elif obs1 < obs_avg:       # If my distance to all tasks is less than the avg plan
                action = 0.2 + 0.2 * (obs_avg - obs1 - obs0)
                # print(f'PresetPolicy. condition: obs1 < obs5, action = {action}')
            else:
                action = 0.1 - 0.1 * (obs_avg - obs1)
                # print(f'PresetPolicy. condition: else, action = {action}')
            actions.append(action)

        # subtract 0.5 from actions and multiply by 2
        actions = [2 * (action - 0.5) for action in actions]
        # Convert the list of actions into a NumPy array
        actions = np.array(actions).reshape(len(obs_batch), -1)

        # print(f'PresetPolicy. actions = {actions}') # DEBUG
        return actions, [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions: Union[List[TensorType], TensorType],
        obs_batch: Union[List[TensorType], TensorType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Optional[Union[List[TensorType], TensorType]] = None,
        prev_reward_batch: Optional[Union[List[TensorType], TensorType]] = None,
        actions_normalized: bool = True,
        in_training: bool = True,
    ) -> TensorType:
        return np.zeros_like(actions, dtype=np.float32)

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(
                    lambda s: s[None], self.observation_space.sample()
                ),
            }
        )

class ConstantActionPolicy(Policy):
    """
    A simple constant action policy for demonstration purposes.
    Always returns a constant action value of 0.5.
    """

    def __init__(self, observation_space, action_space, config):
        super(ConstantActionPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            config=config)

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List["Episode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ):
        # Number of observations dictates number of actions
        num_actions = len(obs_batch)
        # Create a list of 0.5 of the appropriate shape
        constant_actions = [1.0] * num_actions
        actions = np.array(constant_actions).reshape(num_actions, -1)
        print(f'ConstantActionPolicy. actions = {actions}') # DEBUG
        # print action_space
        print(f'ConstantActionPolicy. action_space = {self.action_space}')
        return actions, [], {}  # actions, state_outs, extra_fetches

    def learn_on_batch(self, samples):
        # This policy does not learn, so just return an empty dictionary.
        return {}

    def compute_log_likelihoods(
        self,
        actions: Union[List[TensorType], TensorType],
        obs_batch: Union[List[TensorType], TensorType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Optional[Union[List[TensorType], TensorType]] = None,
        prev_reward_batch: Optional[Union[List[TensorType], TensorType]] = None,
        actions_normalized: bool = True,
        in_training: bool = True,
    ) -> TensorType:
        # Log likelihoods of the actions are zero since the policy is deterministic.
        return np.zeros_like(actions, dtype=np.float32)

    def get_weights(self):
        # This policy does not have any weights
        return {}

    def set_weights(self, weights):
        # This policy does not set any weights
        pass

    def get_initial_state(self):
        # Return an empty list as this policy does not maintain state.
        return []

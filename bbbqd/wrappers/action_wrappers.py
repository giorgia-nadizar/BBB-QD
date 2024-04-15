import numpy as np
from evogym.envs import *
from gym.core import ObsType
from typing import Any


# noinspection PyUnresolvedReferences
class ActionSpaceCorrectionWrapper(gym.Wrapper):
    """ the observation space of evogym is given wrong. this wrapper corrects it """

    def __init__(self, env: EvoGymBase):
        super().__init__(env)
        self.action_space = spaces.Box(np.ones_like(env.action_space.low) * -0.4,
                                       np.ones_like(env.action_space.high) * 0.6,
                                       dtype=np.float64)


# noinspection PyUnresolvedReferences
class ActionWrapper(gym.Wrapper):
    """ base class for action wrappers """

    def __init__(self, env: EvoGymBase, rescale_action: bool = True):
        super().__init__(env)
        self.robot_structure = env.world.objects['robot'].get_structure()
        self.active_voxels = self.robot_structure == 3
        self.active_voxels += self.robot_structure == 4
        self.robot_bounding_box = self.env.world.objects['robot'].get_structure().shape
        self.rescale = rescale_action

    def rescale_action(self, action: np.ndarray) -> np.ndarray:
        if self.rescale:
            low, high = self.env.action_space.low[0], self.env.action_space.high[0]
            normalized_action = action / 2 + .5
            rescaled_action = normalized_action * (high - low) + low
            return rescaled_action
        else:
            return action


# noinspection PyUnresolvedReferences

class LocalActionWrapper(ActionWrapper):
    """ flattens the action before feeding it to the inner step method: it can be fed in a 2d array """

    def __init__(self, env: EvoGymBase, rl: bool = False):
        super().__init__(env)
        n_active_voxels = self.active_voxels.sum()
        self.action_space = spaces.Box(low=self.env.action_space.low[0] * np.ones(n_active_voxels, ),
                                       high=self.env.action_space.high[0] * np.ones(n_active_voxels, ),
                                       shape=(n_active_voxels,), dtype=np.float64)
        self.action_size = 1
        self.rl = rl

    # takes actions only for the active voxels (can be fed in a 1D or 2D array)
    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        action = action.flatten()
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        if self.rl:
            reward = np.array([reward] * self.active_voxels.sum())
            done = np.array([done] * self.active_voxels.sum())
        # info = [info] * self.active_voxels.sum()
        return obs, reward, done, info


# noinspection PyUnresolvedReferences

class GlobalActionWrapper(ActionWrapper):
    """ only takes the values computed for the voxels that are actually active (it expects a number of actions
    matching the grid size, regardless of where voxels are) """

    def __init__(self, env: EvoGymBase, fixed_body: bool = False):
        super().__init__(env)
        self.action_space = spaces.Box(low=self.env.action_space.low[0] * np.ones(self.robot_structure.size),
                                       high=self.env.action_space.high[0] * np.ones(self.robot_structure.size),
                                       shape=(self.robot_structure.size,), dtype=np.float64)
        self.fixed_body = fixed_body
        self.action_size = self.robot_structure.size if not fixed_body else self.active_voxels.sum()

    # takes actions for all voxels and uses only the needed ones
    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        if not self.fixed_body:
            action = action[self.active_voxels.flatten()]
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        return obs, reward, done, info


class ActionSkipWrapper(gym.Wrapper):
    """ repeats an actions for skip timesteps """

    def __init__(self, env: EvoGymBase, skip: int):
        super().__init__(env)
        self.skip = skip

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        obs, reward, done, info = None, None, None, None
        total_reward = 0
        total_info = []
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            total_info.append(info)
            if done:
                break
        return obs, total_reward, done, {key: np.asarray([i[key] for i in total_info]) for key in total_info[0]}

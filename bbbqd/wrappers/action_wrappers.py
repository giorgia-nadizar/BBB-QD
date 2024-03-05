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

    def __init__(self, env: EvoGymBase):
        super().__init__(env)
        self.robot_structure = env.world.objects['robot'].get_structure()
        self.active_voxels = self.robot_structure == 3
        self.active_voxels += self.robot_structure == 4
        self.robot_bounding_box = self.env.world.objects['robot'].get_structure().shape


# noinspection PyUnresolvedReferences
class LocalActionWrapper(ActionWrapper):
    """ flattens the action before feeding it to the inner step method: it can be fed in a 2d array """

    def __init__(self, env: EvoGymBase, **kwargs):
        super().__init__(env)
        self.kwargs = kwargs
        n_active_voxels = self.active_voxels.sum()
        self.action_space = spaces.Box(low=self.env.action_space.low[0] * np.ones(n_active_voxels, ),
                                       high=self.env.action_space.high[0] * np.ones(n_active_voxels, ),
                                       shape=(n_active_voxels,), dtype=np.float64)
        self.action_size = n_active_voxels

    # takes actions only for the active voxels (can be fed in a 1D or 2D array)
    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        action = action.flatten()
        obs, reward, done, info = self.env.step(action)
        if 'rl' in self.kwargs:
            reward = np.array([reward] * self.active_voxels.sum())
            done = np.array([done] * self.active_voxels.sum())
        info = [info] * self.active_voxels.sum()
        return obs, reward, done, info


# noinspection PyUnresolvedReferences
class GlobalActionWrapper(ActionWrapper):
    """ only takes the values computed for the voxels that are actually active (it expects a number of actions
    matching the grid size, regardless of where voxels are) """

    def __init__(self, env: EvoGymBase, **kwargs):
        super().__init__(env)
        self.kwargs = kwargs
        self.action_space = spaces.Box(low=self.env.action_space.low[0] * np.ones(self.robot_structure.size),
                                       high=self.env.action_space.high[0] * np.ones(self.robot_structure.size),
                                       shape=(self.robot_structure.size,), dtype=np.float64)
        self.action_size = self.robot_structure.size

    # takes actions for all voxels and uses only the needed ones
    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        action = action[self.active_voxels.flatten()]
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


class ActionSkipWrapper(gym.Wrapper):
    """ repeats an actions for skip timesteps """

    def __init__(self, env: EvoGymBase, skip: int):
        super().__init__(env)
        self.skip = skip

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        obs, reward, done, info = None, None, None, None
        total_reward = 0
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

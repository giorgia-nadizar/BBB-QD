from typing import Tuple, List, Any, Callable, Dict

import gym
import numpy as np
from gym.core import ActType, ObsType


def get_descriptors_extractor_function(descriptors: List[str]) -> Callable[[Dict[str, Any]], np.ndarray]:
    def _descriptors_extractor(info: Dict[str, Any]) -> np.ndarray:
        descriptors_values = []
        for descriptor in descriptors:
            descriptors_values.extend(info[descriptor])
        return np.asarray(descriptors_values)

    return _descriptors_extractor


class CenterVelocityWrapper(gym.Wrapper):

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        info["velocity"] = self.env.get_vel_com_obs("robot")
        return obs, reward, done, info


class CenterPositionWrapper(gym.Wrapper):

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        info["position"] = self.env.get_pos_com_obs("robot")
        return obs, reward, done, info

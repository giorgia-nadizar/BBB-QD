from typing import Tuple

import gym
import numpy as np
from gym.core import ActType, ObsType


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


class CenterAngleWrapper(gym.Wrapper):

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        original_angle = self.env.get_ort_obs("robot")
        angle = original_angle + np.pi if original_angle <= np.pi else original_angle - np.pi
        info["angle"] = angle
        return obs, reward, done, info


class FloorContactWrapper(gym.Wrapper):
    floor_level: float = 0.105

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        masses_positions = self.env.object_pos_at_time(self.env.get_time(), 'robot')
        info["floor_contact"] = np.asarray([np.min(masses_positions[1]) <= self.floor_level])
        return obs, reward, done, info

from typing import Tuple, Union, List, Any

import gym
import numpy as np
from gym.core import ActType, ObsType

from bbbqd.behavior.behavior_utils import detect_ground_contact
from bbbqd.environments.environments_utils import extract_ground_profile_masses_ids


class CenterVelocityWrapper(gym.Wrapper):

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        info["velocity"] = self.env.get_vel_com_obs("robot")
        return obs, reward, done, info


class ObjectVelocityWrapper(gym.Wrapper):

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        info["object_velocity"] = self.env.get_vel_com_obs("package")
        return obs, reward, done, info


class CenterPositionWrapper(gym.Wrapper):

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        info["position"] = self.env.get_pos_com_obs("robot")
        return obs, reward, done, info


class ObjectPositionWrapper(gym.Wrapper):

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        info["object_position"] = self.env.get_pos_com_obs("package")
        return obs, reward, done, info


class CenterAngleWrapper(gym.Wrapper):

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        original_angle = self.env.get_ort_obs("robot")
        angle = original_angle + np.pi if original_angle <= np.pi else original_angle - np.pi
        info["angle"] = angle
        return obs, reward, done, info


class ObjectAngleWrapper(gym.Wrapper):

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        original_angle = self.env.get_ort_obs("package")
        angle = np.abs(original_angle - np.pi)
        info["object_angle"] = angle / np.pi
        return obs, reward, done, info


class FloorContactWrapper(gym.Wrapper):
    offset: float = 0.005
    side_length: float = 0.1
    ground_masses_ids: List[List[Any]] = None

    def reset(self, **kwargs) -> Union[ObsType, Tuple[ObsType, dict]]:
        reset_result = super().reset(**kwargs)
        self.ground_masses_ids = extract_ground_profile_masses_ids(self.env, self.side_length)
        return reset_result

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        robot_position = self.env.object_pos_at_time(self.env.get_time(), "robot")
        robot_structure = self.env.world.objects['robot'].get_structure()
        ground = self.env.object_pos_at_time(self.env.get_time(), "ground")
        ground_contact = detect_ground_contact(robot_position, robot_structure, ground, self.ground_masses_ids,
                                               self.side_length, self.offset)
        info["floor_contact"] = np.asarray([ground_contact])
        return obs, reward, done, info


class WallsContactWrapper(gym.Wrapper):
    left_wall: float = 0.1
    right_wall: float = 0.6
    offset: float = 0.005

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        masses_positions = self.env.object_pos_at_time(self.env.get_time(), "robot")
        left_contact_ys = masses_positions[1][masses_positions[0] <= (self.left_wall + self.offset)]
        right_contact_ys = masses_positions[1][masses_positions[0] >= (self.right_wall - self.offset)]
        if len(left_contact_ys) == 0 or len(right_contact_ys) == 0:
            info["walls_contact"] = np.asarray([None])
        else:
            avg_left_contact_y = np.mean(left_contact_ys)
            avg_right_contact_y = np.mean(right_contact_ys)
            slope = (avg_right_contact_y - avg_left_contact_y) / (self.right_wall - self.left_wall)
            angle = np.arctan(np.abs(slope))
            info["walls_contact"] = np.asarray([angle])
        return obs, reward, done, info

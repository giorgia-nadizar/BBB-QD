from typing import Tuple, Union, List, Any

import gym
import numpy as np
from gym.core import ActType, ObsType

from bbbqd.behavior.behavior_utils import detect_ground_contact


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

        # extract all ground masses
        ground_masses_lists = []
        allowed_ground_names = ["ground", "new_object_2", "terrain"]  # some envs have different names
        for i in range(1, 8):
            allowed_ground_names.append(f"platform_{i}")
        for ground_name in allowed_ground_names:
            try:
                masses = self.env.object_pos_at_time(self.env.get_time(), ground_name)
                structure = self.env.world.objects[ground_name].get_structure()
                ground_masses_lists.append((masses, structure))
            except ValueError:
                pass
        assert len(ground_masses_lists) > 0

        # extract ids of masses
        self.ground_masses_ids = []
        for ground_masses, ground_structure in ground_masses_lists:
            current_profile = []

            structure_column_idx = -1
            structure_row_idx = np.argmax(ground_structure[:, structure_column_idx + 1] > 0)

            first_column = ground_masses[:, ground_masses[0] == np.min(ground_masses[0])]
            current_mass = first_column[:, first_column[1] == np.max(first_column[1])]
            current_profile.append(current_mass)
            descending = False

            while True:
                # look for neighbors
                masses_above = ground_masses[:,
                               (np.abs(ground_masses[1] - current_mass[1]) < 2 * self.side_length) &
                               (np.abs(ground_masses[0] - current_mass[0]) < self.side_length / 2) &
                               (ground_masses[1] > current_mass[1])
                               ]
                masses_right = ground_masses[:,
                               (np.abs(ground_masses[0] - current_mass[0]) < 2 * self.side_length) &
                               (np.abs(ground_masses[1] - current_mass[1]) < self.side_length / 2) &
                               (ground_masses[0] > current_mass[0])
                               ]
                masses_below = ground_masses[:,
                               (np.abs(ground_masses[1] - current_mass[1]) < 2 * self.side_length) &
                               (np.abs(ground_masses[0] - current_mass[0]) < self.side_length / 2) &
                               (ground_masses[1] < current_mass[1])
                               ]

                if not descending and len(masses_above[0]) > 0:
                    current_profile.append(masses_above[:, masses_above[1] == np.min(masses_above[1])])
                    structure_row_idx -= 1
                elif len(masses_right[0]) > 0 and ground_structure[structure_row_idx, structure_column_idx + 1] > 0:
                    current_profile.append(masses_right[:, masses_right[0] == np.min(masses_right[0])])
                    structure_column_idx += 1
                    descending = False
                elif len(masses_below[0]) > 0:
                    current_profile.append(masses_below[:, masses_below[1] == np.max(masses_below[1])])
                    descending = True
                    structure_row_idx += 1
                else:
                    break

                current_mass = current_profile[len(current_profile) - 1]

            current_profile_ids = [np.where((ground_masses[0] == m[0]) & (ground_masses[1] == m[1]))[0][0].astype(int)
                                   for m in current_profile]
            self.ground_masses_ids.append(current_profile_ids)

        return reset_result

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        robot = self.env.object_pos_at_time(self.env.get_time(), "robot")
        ground = self.env.object_pos_at_time(self.env.get_time(), "ground")
        ground_contact = False
        for ground_ids in self.ground_masses_ids:
            current_ground_masses = ground[:, ground_ids]
            ground_contact = detect_ground_contact(robot,
                                                   current_ground_masses,
                                                   side=self.side_length,
                                                   offset=self.offset)
            if ground_contact:
                break
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

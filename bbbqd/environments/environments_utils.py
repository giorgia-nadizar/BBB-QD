import gym
import numpy as np
from typing import Union

from evogym.envs import EvoGymBase
from gym import Env


def extract_ground_profile_masses_ids(env: Union[EvoGymBase, Env, gym.Wrapper], side_length: float = 0.1):
    # extract all ground masses
    ground_masses_lists = []
    allowed_ground_names = ["ground", "new_object_2", "terrain"]  # some envs have different names
    for i in range(1, 8):
        allowed_ground_names.append(f"platform_{i}")
    for ground_name in allowed_ground_names:
        try:
            masses = env.object_pos_at_time(env.get_time(), ground_name)
            structure = env.world.objects[ground_name].get_structure()
            ground_masses_lists.append((masses, structure))
        except ValueError:
            pass
    assert len(ground_masses_lists) > 0

    # extract ids of masses
    ground_masses_ids = []
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
                           (np.abs(ground_masses[1] - current_mass[1]) < 2 * side_length) &
                           (np.abs(ground_masses[0] - current_mass[0]) < side_length / 2) &
                           (ground_masses[1] > current_mass[1])
                           ]
            masses_right = ground_masses[:,
                           (np.abs(ground_masses[0] - current_mass[0]) < 2 * side_length) &
                           (np.abs(ground_masses[1] - current_mass[1]) < side_length / 2) &
                           (ground_masses[0] > current_mass[0])
                           ]
            masses_below = ground_masses[:,
                           (np.abs(ground_masses[1] - current_mass[1]) < 2 * side_length) &
                           (np.abs(ground_masses[0] - current_mass[0]) < side_length / 2) &
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
        ground_masses_ids.append(current_profile_ids)

    return ground_masses_ids

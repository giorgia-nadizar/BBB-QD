import numpy as np
from evogym.envs import *
from gym.core import ActType
from typing import Any
from bbbqd.body.body_utils import structure_corners, polygon_area, voxel_velocity, two_d_idx_of, moore_neighbors, \
    one_hot_structure


class ObservationWrapper(gym.Wrapper):
    """ base class for observation wrappers """

    def __init__(self, env: EvoGymBase, obs_details: Dict[str, Any]):
        super().__init__(env)
        self.robot_structure = env.world.objects['robot'].get_structure()
        self.robot_bounding_box = self.env.world.objects['robot'].get_structure().shape
        self.robot_structure_one_hot = one_hot_structure(self.robot_structure, self.robot_bounding_box)
        self.observe_structure = obs_details.get('observe_structure', False)
        self.observe_voxel_volume = obs_details.get('observe_voxel_volume', False)
        self.observe_voxel_vel = obs_details.get('observe_voxel_vel', False)
        self.observe_time = obs_details.get('observe_time', False)
        self.observe_time_interval = obs_details.get('observe_time_interval', 1)

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self.observe()

    def step(self, action: ActType) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return self.observe(), reward, done, info

    def observe(self) -> np.ndarray:
        raise NotImplementedError

    def volumes_from_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns a 2d np array with volumes of each voxel (or -9999 if the voxel is not in the body)
        and a mask of the voxels that are in the body
        """
        pos = self.env.object_pos_at_time(self.env.get_time(), 'robot')
        corners = structure_corners(pos, self.robot_structure, self.robot_bounding_box)
        volumes = np.ones_like(self.robot_structure, dtype=float) * -9999
        mask = np.zeros_like(self.robot_structure)
        for idx, corners in enumerate(corners):
            if corners is None:
                continue
            x, y = two_d_idx_of(idx, self.robot_bounding_box)
            volumes[x, y] = polygon_area([x[0] for x in corners], [x[1] for x in corners])
            mask[x, y] = 1
        return volumes, mask

    def velocities_from_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns a 2d numpy array with velocity of each voxel (or -9999 if the voxel is not in the body)
        and a mask of the voxels that are in the body
        """
        vel = self.env.object_vel_at_time(self.env.get_time(), 'robot')
        corners = structure_corners(vel, self.robot_structure, self.robot_bounding_box)
        velocities = np.ones((self.robot_bounding_box[0], self.robot_bounding_box[1], 2)) * -9999
        mask = np.zeros_like(self.robot_structure)
        for idx, corners in enumerate(corners):
            if corners is None:
                continue
            x, y = two_d_idx_of(idx, self.robot_bounding_box)
            vx, vy = voxel_velocity([x[0] for x in corners], [x[1] for x in corners])
            velocities[x, y, 0] = vx
            velocities[x, y, 1] = vy
            mask[x, y] = 1
        return velocities, mask


class LocalObservationWrapper(ObservationWrapper):
    """ this wrapper returns the local observation of the body: local observation of all active voxels in a nested list
    (the outer list has size n_voxels, each inner list has size obs_len, derived from the neighborhood size multiplied
    by the n of sensors in each voxel)"""

    def __init__(self, env: EvoGymBase, obs_details: Dict[str, Any]):
        super().__init__(env, obs_details=obs_details)
        self.observation_range = obs_details.get('observation_range', 1)
        self.nr_voxel_in_neighborhood = (2 * self.observation_range + 1) ** 2
        # get the observation space
        obs_len = 0
        if self.observe_structure:
            obs_len += self.nr_voxel_in_neighborhood * 5
        if self.observe_voxel_volume:
            obs_len += self.nr_voxel_in_neighborhood
        if self.observe_voxel_vel:
            obs_len += self.nr_voxel_in_neighborhood * 2
        if self.observe_time:
            obs_len += 1
        assert obs_len > 0, 'No observation selected'

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float64)
        self.observation_size = obs_len

    # returns an array with two dimensions
    def observe(self) -> np.ndarray:
        # get the raw observations
        if self.observe_voxel_volume:
            volumes, volumes_mask = self.volumes_from_pos()
        if self.observe_voxel_vel:
            velocities, velocities_mask = self.velocities_from_pos()
        # walk through the structure and create the observation per each voxel
        observations = []
        for idx in range(self.robot_structure.size):
            x, y = two_d_idx_of(idx, self.robot_bounding_box)
            if self.robot_structure[x, y] in [0, 1, 2]:
                continue
            obs = []
            # get the neighbors
            neighbors = moore_neighbors(x, y, self.observation_range, self.robot_bounding_box)
            # get the volumes of the neighbors
            for neighbor in neighbors:
                if neighbor[0] == -1:  # neighbor is only -1 when it is out of bounds
                    # observe structure
                    if self.observe_structure:
                        obs.extend([1, 0, 0, 0, 0])
                    # observe volume
                    if self.observe_voxel_volume:
                        obs.append(0)
                    # observe velocity
                    if self.observe_voxel_vel:
                        obs.extend([0, 0])
                else:
                    # observe structure
                    if self.observe_structure:
                        obs.extend(self.robot_structure_one_hot[neighbor[1][0], neighbor[1][1], :])
                    # observe volume
                    if self.observe_voxel_volume:
                        # if the voxel is not in the body, then insert 0
                        if volumes_mask[neighbor[1][0], neighbor[1][1]] == 0:
                            obs.append(0)
                        else:
                            obs.append(volumes[neighbor[1][0], neighbor[1][1]])
                    # observe velocity
                    if self.observe_voxel_vel:
                        # if the voxel is not in the body, then insert 0
                        if velocities_mask[neighbor[1][0], neighbor[1][1]] == 0:
                            obs.extend([0, 0])
                        else:
                            obs.extend(velocities[neighbor[1][0], neighbor[1][1], :])
            if self.observe_time:
                period = self.observe_time_interval
                env_time = self.env.get_time() % period
                obs.append(env_time)
            # observation is ready
            obs = np.asarray(obs)
            observations.append(obs)
        # observations is a list of observations, we need to convert it to a numpy array
        observations = np.asarray(observations)
        return observations


class GlobalObservationWrapper(ObservationWrapper):
    """ this wrapper returns the global observation of the body
    (the observations of each voxel one after the other, read by row from above)
    """

    def __init__(self, env: EvoGymBase, obs_details: Dict[str, Any], fixed_body: bool = False):
        super().__init__(env, obs_details=obs_details)
        self.fixed_body = fixed_body
        active_voxels = self.robot_structure == 3
        active_voxels += self.robot_structure == 4
        self.n_active_voxels = active_voxels.sum()
        # get the observation space
        size_to_multiply = self.n_active_voxels if self.fixed_body else self.robot_structure.size
        obs_len = 0
        if self.observe_structure:
            obs_len += 5 * size_to_multiply
        if self.observe_voxel_volume:
            obs_len += 1 * size_to_multiply
        if self.observe_voxel_vel:
            obs_len += 2 * size_to_multiply
        if self.observe_time:
            obs_len += 1
        assert obs_len > 0, 'No observation selected'

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float64)
        self.observation_size = obs_len

    # returns a unidimensional array
    def observe(self) -> np.ndarray:
        # get the raw observations
        if self.observe_voxel_volume:
            volumes, volumes_mask = self.volumes_from_pos()
        if self.observe_voxel_vel:
            velocities, velocities_mask = self.velocities_from_pos()
        # walk through the bounding box and create the observation per each grid
        obs = []
        for idx in range(self.robot_structure.size):
            # find the position of the voxel
            x, y = two_d_idx_of(idx, self.robot_bounding_box)
            # observe structure
            if self.observe_structure:
                if (self.fixed_body and self.robot_structure_one_hot[x, y, 0] == 0) or not self.fixed_body:
                    obs.extend(self.robot_structure_one_hot[x, y, :])
            # observe volume
            if self.observe_voxel_volume:
                # if the voxel is not in the body, then insert 0 (unless the body is fixed)
                if volumes_mask[x, y] == 0:
                    if not self.fixed_body:
                        obs.append(0)
                else:
                    obs.append(volumes[x, y])
            # observe velocity
            if self.observe_voxel_vel:
                # if the voxel is not in the body, then insert 0 (unless the body is fixed)
                if velocities_mask[x, y] == 0:
                    if not self.fixed_body:
                        obs.extend([0, 0])
                else:
                    obs.extend(velocities[x, y, :])
        if self.observe_time:
            period = self.observe_time_interval
            env_time = self.env.get_time() % period
            obs.append(env_time)
        # observation is ready
        obs = np.asarray(obs)
        return obs

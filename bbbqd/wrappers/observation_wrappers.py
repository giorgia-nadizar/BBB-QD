from evogym.envs import *
from gym.core import ActType
from typing import Any


class ObservationWrapper(gym.Wrapper):
    """ base class for observation wrappers """

    def __init__(self, env: EvoGymBase):
        super().__init__(env)
        self.robot_structure = env.world.objects['robot'].get_structure()
        self.robot_bounding_box = self.env.world.objects['robot'].get_structure().shape
        self.robot_structure_one_hot = self.get_one_hot_structure()

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self.observe()

    def step(self, action: ActType) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return self.observe(), reward, done, info

    def observe(self) -> np.ndarray:
        raise NotImplementedError

    def get_volumes_from_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns a 2d np array with volumes of each voxel (or -9999 if the voxel is not in the body)
        and a mask of the voxels that are in the body
        """
        pos = self.env.object_pos_at_time(self.env.get_time(), 'robot')
        structure_corners = self.get_structure_corners(pos)
        volumes = np.ones_like(self.robot_structure, dtype=float) * -9999
        mask = np.zeros_like(self.robot_structure)
        for idx, corners in enumerate(structure_corners):
            if corners is None:
                continue
            x, y = self.two_d_idx_of(idx)
            area = self.polygon_area([x[0] for x in corners], [x[1] for x in corners])
            volumes[x, y] = area
            mask[x, y] = 1
        return volumes, mask

    def get_velocities_from_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns a 2d numpy array with velocity of each voxel (or -9999 if the voxel is not in the body)
        and a mask of the voxels that are in the body
        """
        vel = self.env.object_vel_at_time(self.env.get_time(), 'robot')
        structure_corners = self.get_structure_corners(vel)
        velocities = np.ones((self.robot_bounding_box[0], self.robot_bounding_box[1], 2)) * -9999
        mask = np.zeros_like(self.robot_structure)
        for idx, corners in enumerate(structure_corners):
            if corners is None:
                continue
            x, y = self.two_d_idx_of(idx)
            vx, vy = self.voxel_velocity([x[0] for x in corners], [x[1] for x in corners])
            velocities[x, y, 0] = vx
            velocities[x, y, 1] = vy
            mask[x, y] = 1
        return velocities, mask

    def get_structure_corners(self, observation: np.ndarray) -> List[List[List[float]]]:
        """ process the observation to each voxel's corner that exists in reading order (left to right, top to bottom)
        evogym returns basic observation for each point mass (corner of voxel) in reading order.
        but for the voxels that are neighbors -- share a corner, only one observation is returned.
        this function takes any observation in that form and returns a 2d list.
        each element of the list is a list of corners of the voxel and show the observation for 4 corners of the voxel.
        """
        f_structure = self.robot_structure.flatten()
        pointer_to_masses = 0
        structure_corners = [None] * self.robot_bounding_box[0] * self.robot_bounding_box[1]
        for idx, val in enumerate(f_structure):
            if val == 0:
                continue
            else:
                if pointer_to_masses == 0:
                    structure_corners[idx] = [[observation[0, 0], observation[1, 0]],
                                              [observation[0, 1], observation[1, 1]],
                                              [observation[0, 2], observation[1, 2]],
                                              [observation[0, 3], observation[1, 3]]]
                    pointer_to_masses += 4
                else:
                    # check the 2d location and find out whether this voxel has a neighbor in to its left or up
                    x, y = self.two_d_idx_of(idx)
                    left_idx = self.one_d_idx_of(x, y - 1)
                    up_idx = self.one_d_idx_of(x - 1, y)
                    upright_idx = self.one_d_idx_of(x - 1, y + 1)
                    upleft_idx = self.one_d_idx_of(x - 1, y - 1)
                    if (y - 1 >= 0 and x - 1 >= 0 and structure_corners[left_idx] is not None
                            and structure_corners[up_idx] is not None):
                        # both neighbors are occupied, only the bottom right point mass is new
                        structure_corners[idx] = [structure_corners[up_idx][2],
                                                  structure_corners[up_idx][3],
                                                  structure_corners[left_idx][3],
                                                  [observation[0, pointer_to_masses],
                                                   observation[1, pointer_to_masses]]]
                        pointer_to_masses += 1
                    elif (y - 1 >= 0 and structure_corners[left_idx] is not None and y + 1 < self.robot_bounding_box[
                        1] and x - 1 >= 0 and structure_corners[upright_idx] is not None
                          and self.robot_structure[x, y + 1] != 0):
                        # left and up right are occupied, bottom right point mass is new
                        # (connected to up right through right neighbor)
                        structure_corners[idx] = [structure_corners[left_idx][1],
                                                  structure_corners[upright_idx][2],
                                                  structure_corners[left_idx][3],
                                                  [observation[0, pointer_to_masses],
                                                   observation[1, pointer_to_masses]]]
                        pointer_to_masses += 1
                    elif y - 1 >= 0 and structure_corners[left_idx] is not None:
                        # only the left neighbor is occupied, top right and bottom right point masses are new
                        structure_corners[idx] = [structure_corners[left_idx][1],
                                                  [observation[0, pointer_to_masses],
                                                   observation[1, pointer_to_masses]],
                                                  structure_corners[left_idx][3],
                                                  [observation[0, pointer_to_masses + 1],
                                                   observation[1, pointer_to_masses + 1]]]
                        pointer_to_masses += 2
                    elif x - 1 >= 0 and structure_corners[up_idx] is not None:
                        # only the up neighbor is occupied, bottom left and bottom right point masses are new
                        structure_corners[idx] = [structure_corners[up_idx][2],
                                                  structure_corners[up_idx][3],
                                                  [observation[0, pointer_to_masses],
                                                   observation[1, pointer_to_masses]],
                                                  [observation[0, pointer_to_masses + 1],
                                                   observation[1, pointer_to_masses + 1]]]
                        pointer_to_masses += 2
                    elif (y + 1 < self.robot_bounding_box[1] and x - 1 >= 0
                          and structure_corners[upright_idx] is not None and self.robot_structure[x, y + 1] != 0):
                        # only the up right neighbor is occupied, top left, bottom left, and bottom right point masses
                        # are new (connected to upright through right neighbor)
                        structure_corners[idx] = [
                            [observation[0, pointer_to_masses], observation[1, pointer_to_masses]],
                            structure_corners[upright_idx][2],
                            [observation[0, pointer_to_masses + 1], observation[1, pointer_to_masses + 1]],
                            [observation[0, pointer_to_masses + 2], observation[1, pointer_to_masses + 2]]]
                        pointer_to_masses += 3
                    else:
                        # no neighbors are occupied, all four point masses are new
                        structure_corners[idx] = [
                            [observation[0, pointer_to_masses], observation[1, pointer_to_masses]],
                            [observation[0, pointer_to_masses + 1], observation[1, pointer_to_masses + 1]],
                            [observation[0, pointer_to_masses + 2], observation[1, pointer_to_masses + 2]],
                            [observation[0, pointer_to_masses + 3], observation[1, pointer_to_masses + 3]]]
                        pointer_to_masses += 4

        return structure_corners

    @staticmethod
    def polygon_area(x: List[float], y: List[float]) -> float:
        """ Calculates the area of an arbitrary polygon given its vertices in x and y (list) coordinates.
        assumes the order is wrong """
        x[0], x[1] = x[1], x[0]
        y[0], y[1] = y[1], y[0]
        correction = x[-1] * y[0] - y[-1] * x[0]
        main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
        area = 0.5 * np.abs(main_area + correction)
        return area

    @staticmethod
    def voxel_velocity(x: List[float], y: List[float]) -> Tuple[float, float]:
        """ Calculates the velocity of a voxel given its 4 corners' velocities """
        return (x[0] + x[1] + x[2] + x[3]) / 4.0, (y[0] + y[1] + y[2] + y[3]) / 4.0

    def two_d_idx_of(self, idx: int) -> Tuple[int, int]:
        """
        returns 2d index of a 1d index
        """
        return idx // self.robot_bounding_box[0], idx % self.robot_bounding_box[1]

    def one_d_idx_of(self, x: int, y: int) -> int:
        """
        returns 1d index of a 2d index
        """
        return x * self.robot_bounding_box[0] + y

    def get_moore_neighbors(self, x: int, y: int, observation_range: int) -> List[Tuple[float, List[int]]]:
        """
        returns the 8 neighbors of a voxel in the structure
        """
        neighbors = []
        min_obs_range = observation_range * -1
        max_obs_range = observation_range + 1
        for i in range(min_obs_range, max_obs_range):
            for j in range(min_obs_range, max_obs_range):
                if 0 <= x + i < self.robot_bounding_box[0] and 0 <= y + j < self.robot_bounding_box[1]:
                    neighbors.append((1.0, [x + i, y + j]))
                else:
                    neighbors.append((-1.0, None))
        return neighbors

    def get_one_hot_structure(self) -> np.ndarray:
        """
        returns a one-hot encoding of the structure
        """
        one_hot = np.zeros((self.robot_bounding_box[0], self.robot_bounding_box[1], 5))
        for i in range(self.robot_bounding_box[0]):
            for j in range(self.robot_bounding_box[1]):
                one_hot[i, j, int(self.robot_structure[i, j])] = 1
        return one_hot


class LocalObservationWrapper(ObservationWrapper):
    """ this wrapper returns the local observation of the body: local observation of all active voxels in a nested list
    (the outer list has size n_voxels, each inner list has size obs_len, derived from the neighborhood size multiplied
    by the n of sensors in each voxel)"""

    def __init__(self, env: EvoGymBase, **kwargs):
        super().__init__(env)
        self.kwargs = kwargs
        self.nr_voxel_in_neighborhood = (2 * self.kwargs.get('observation_range', 1) + 1) ** 2
        # get the observation space
        obs_len = 0
        if self.kwargs.get('observe_structure', False):
            obs_len += self.nr_voxel_in_neighborhood * 5
        if self.kwargs.get('observe_voxel_volume', False):
            obs_len += self.nr_voxel_in_neighborhood
        if self.kwargs.get('observe_voxel_vel', False):
            obs_len += self.nr_voxel_in_neighborhood * 2
        if self.kwargs.get('observe_time', False):
            obs_len += 1
        assert obs_len > 0, 'No observation selected'

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float64)
        self.observation_size = obs_len

    # returns an array with two dimensions
    def observe(self) -> np.ndarray:
        # get the raw observations
        if self.kwargs.get('observe_voxel_volume', False):
            volumes, volumes_mask = self.get_volumes_from_pos()
        if self.kwargs.get('observe_voxel_vel', False):
            velocities, velocities_mask = self.get_velocities_from_pos()
        # walk through the structure and create the observation per each voxel
        observations = []
        for idx in range(self.robot_structure.size):
            x, y = self.two_d_idx_of(idx)
            if self.robot_structure[x, y] in [0, 1, 2]:
                continue
            obs = []
            # get the neighbors
            neighbors = self.get_moore_neighbors(x, y, self.kwargs.get('observation_range', 2))
            # get the volumes of the neighbors
            for neighbor in neighbors:
                if neighbor[0] == -1:  # neighbor is only -1 when it is out of bounds
                    # observe structure
                    if self.kwargs.get('observe_structure', False):
                        obs.extend([1, 0, 0, 0, 0])
                    # observe volume
                    if self.kwargs.get('observe_voxel_volume', False):
                        obs.append(0)
                    # observe velocity
                    if self.kwargs.get('observe_voxel_vel', False):
                        obs.extend([0, 0])
                else:
                    # observe structure
                    if self.kwargs.get('observe_structure', False):
                        obs.extend(self.robot_structure_one_hot[neighbor[1][0], neighbor[1][1], :])
                    # observe volume
                    if self.kwargs.get('observe_voxel_volume', False):
                        # if the voxel is not in the body, then insert 0
                        if volumes_mask[neighbor[1][0], neighbor[1][1]] == 0:
                            obs.append(0)
                        else:
                            obs.append(volumes[neighbor[1][0], neighbor[1][1]])
                    # observe velocity
                    if self.kwargs.get('observe_voxel_vel', False):
                        # if the voxel is not in the body, then insert 0
                        if velocities_mask[neighbor[1][0], neighbor[1][1]] == 0:
                            obs.extend([0, 0])
                        else:
                            obs.extend(velocities[neighbor[1][0], neighbor[1][1], :])
            if self.kwargs.get('observe_time', False):
                period = self.kwargs.get('observe_time_interval', 1)
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

    def __init__(self, env: EvoGymBase, **kwargs):
        super().__init__(env)
        self.kwargs = kwargs
        # get the observation space
        obs_len = 0
        if self.kwargs.get('observe_structure', False):
            obs_len += 5 * self.robot_structure.size
        if self.kwargs.get('observe_voxel_volume', False):
            obs_len += 1 * self.robot_structure.size
        if self.kwargs.get('observe_voxel_vel', False):
            obs_len += 2 * self.robot_structure.size
        if self.kwargs.get('observe_time', False):
            obs_len += 1
        assert obs_len > 0, 'No observation selected'

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float64)
        self.observation_size = obs_len

    # returns a unidimensional array
    def observe(self) -> np.ndarray:
        # get the raw observations
        if self.kwargs.get('observe_voxel_volume', False):
            volumes, volumes_mask = self.get_volumes_from_pos()
        if self.kwargs.get('observe_voxel_vel', False):
            velocities, velocities_mask = self.get_velocities_from_pos()
        # walk through the bounding box and create the observation per each grid
        obs = []
        for idx in range(self.robot_structure.size):
            # find the position of the voxel
            x, y = self.two_d_idx_of(idx)
            # observe structure
            if self.kwargs.get('observe_structure', False):
                obs.extend(self.robot_structure_one_hot[x, y, :])
            # observe volume
            if self.kwargs.get('observe_voxel_volume', False):
                # if the voxel is not in the body, then insert 0
                # print(f"{volumes_mask[x,y]} -> {volumes[x,y]}")
                if volumes_mask[x, y] == 0:
                    obs.append(0)
                else:
                    obs.append(volumes[x, y])
            # observe velocity
            if self.kwargs.get('observe_voxel_vel', False):
                # if the voxel is not in the body, then insert 0
                if velocities_mask[x, y] == 0:
                    obs.extend([0, 0])
                else:
                    obs.extend(velocities[x, y, :])
        if self.kwargs.get('observe_time', False):
            period = self.kwargs.get('observe_time_interval', 1)
            env_time = self.env.get_time() % period
            obs.append(env_time)
        # observation is ready
        obs = np.asarray(obs)
        return obs

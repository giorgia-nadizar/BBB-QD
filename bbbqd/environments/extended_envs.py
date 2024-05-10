import os

import numpy as np
import pkg_resources
from evogym import EvoWorld
from evogym.envs import CarrySmallRect, PackageBase
from gym import spaces
from gym.envs.registration import register


class CustomCarrySmallRect(CarrySmallRect):

    def __init__(self, body, connections=None):
        # make world
        super().__init__(np.asarray([[3]]), connections)
        self.world = EvoWorld.from_json(
            pkg_resources.resource_filename('bbbqd.environments', os.path.join('envs/CustomCarrier-v0.json')))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        PackageBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size

        self.action_space = spaces.Box(low=0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(6 + num_robot_points,), dtype=np.float)

        # threshhold height
        self.thresh_height = 3.0 * self.VOXEL_SIZE


register(
    id='CustomCarrier-v0',
    entry_point='bbbqd.environments.extended_envs:CustomCarrySmallRect',
    max_episode_steps=500
)

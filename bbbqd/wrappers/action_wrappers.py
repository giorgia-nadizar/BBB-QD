from evogym.envs import *


class ActionSpaceCorrectionWrapper(gym.Wrapper[EvoGymBase]):
    """ the observation space of evogym is given wrong. this wrapper corrects it """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(np.ones_like(env.action_space.low) * -0.4,
                                       np.ones_like(env.action_space.high) * 0.6, dtype=np.float64)


class ActionWrapper(gym.Wrapper):
    """ base class for action wrappers """

    def __init__(self, env):
        super().__init__(env)
        self.robot_structure = env.world.objects['robot'].get_structure()
        self.active_voxels = self.robot_structure == 3
        self.active_voxels += self.robot_structure == 4
        self.robot_bounding_box = self.env.world.objects['robot'].get_structure().shape


class LocalActionWrapper(ActionWrapper):
    """ flattens the action before feeding it to the inner step method (it expects it in a nested list) """

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.kwargs = kwargs
        self.action_space = spaces.Box(low=self.env.action_space.low[0] * np.ones(1, ),
                                       high=self.env.action_space.high[0] * np.ones(1, ),
                                       shape=(1,), dtype=np.float64)  # not sure about this one

    def step(self, action):
        action = action.flatten()
        obs, reward, done, info = self.env.step(action)
        if 'rl' in self.kwargs:
            reward = np.array([reward] * self.active_voxels.sum())
            done = np.array([done] * self.active_voxels.sum())
        info = [info] * self.active_voxels.sum()
        return obs, reward, done, info


class GlobalActionWrapper(ActionWrapper):
    """ only takes the values computed for the voxels that are actually active (it expects a number of actions
    matching the grid size, regardless of where voxels are """

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.kwargs = kwargs
        self.action_space = spaces.Box(low=self.env.action_space.low[0] * np.ones(self.robot_structure.size),
                                       high=self.env.action_space.high[0] * np.ones(self.robot_structure.size),
                                       shape=(self.robot_structure.size,), dtype=np.float64)

    def step(self, action):
        action = action[self.active_voxels.flatten()]
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


class ActionSkipWrapper(gym.core.Wrapper):
    """ repeats an actions for skip timesteps """

    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        obs, reward, done, info = None, None, None, None
        total_reward = 0
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

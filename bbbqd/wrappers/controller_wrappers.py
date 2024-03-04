from bbbqd.wrappers.action_wrappers import *
from bbbqd.wrappers.observation_wrappers import *


class ControllerWrapper(gym.Wrapper):
    def __init__(self, env: EvoGymBase, skip: int = 0):
        super().__init__(env)
        self.env = ActionSpaceCorrectionWrapper(env)
        if skip > 0:
            self.env = ActionSkipWrapper(self.env, skip)


class GlobalWrapper(ControllerWrapper):

    def __init__(self, env: EvoGymBase, skip: int = 0, **kwargs):
        super().__init__(env, skip)
        self.env = GlobalObservationWrapper(GlobalActionWrapper(self.env), **kwargs)


class LocalWrapper(ControllerWrapper):
    def __init__(self, env: EvoGymBase, skip: int = 0, **kwargs):
        super().__init__(env, skip)
        self.env = LocalObservationWrapper(LocalActionWrapper(self.env), **kwargs)

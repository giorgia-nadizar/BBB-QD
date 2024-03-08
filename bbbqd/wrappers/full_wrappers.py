from __future__ import annotations

from bbbqd.wrappers.action_wrappers import *
from bbbqd.wrappers.observation_wrappers import *


class FullWrapper(gym.Wrapper):
    def __init__(self, env: EvoGymBase, skip: int = 0):
        super().__init__(env)
        self.env = ActionSpaceCorrectionWrapper(env)
        if skip > 0:
            self.env = ActionSkipWrapper(self.env, skip)


class GlobalWrapper(FullWrapper):

    def __init__(self, env: EvoGymBase, obs_details: Dict[str, Any], skip: int = 0, fixed_body: bool = False):
        super().__init__(env, skip)
        self.env = GlobalObservationWrapper(
            GlobalActionWrapper(self.env, fixed_body=fixed_body),
            fixed_body=fixed_body, obs_details=obs_details
        )


class LocalWrapper(FullWrapper):
    def __init__(self, env: EvoGymBase, obs_details: Dict[str, Any], skip: int = 0):
        super().__init__(env, skip)
        self.env = LocalObservationWrapper(LocalActionWrapper(self.env), obs_details=obs_details)

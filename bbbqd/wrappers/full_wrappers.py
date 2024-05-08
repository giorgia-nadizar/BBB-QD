from __future__ import annotations

from bbbqd.wrappers.action_wrappers import *
from bbbqd.wrappers.observation_wrappers import *
from bbbqd.wrappers.qd_wrappers import CenterVelocityWrapper, CenterPositionWrapper, CenterAngleWrapper, \
    FloorContactWrapper, ObjectVelocityWrapper, ObjectPositionWrapper, WallsContactWrapper


class FullWrapper(gym.Wrapper):
    def __init__(self, env: EvoGymBase, skip: int = 0, qd_wrappers: List[str] = None):
        super().__init__(env)
        self.env = ActionSpaceCorrectionWrapper(env)
        if qd_wrappers is not None:
            if "velocity" in qd_wrappers:
                self.env = CenterVelocityWrapper(self.env)
            if "position" in qd_wrappers:
                self.env = CenterPositionWrapper(self.env)
            if "object_velocity" in qd_wrappers:
                self.env = ObjectVelocityWrapper(self.env)
            if "object_position" in qd_wrappers:
                self.env = ObjectPositionWrapper(self.env)
            if "angle" in qd_wrappers:
                self.env = CenterAngleWrapper(self.env)
            if "floor_contact" in qd_wrappers:
                self.env = FloorContactWrapper(self.env)
            if "walls_contact" in qd_wrappers:
                self.env = WallsContactWrapper(self.env)
        if skip > 0:
            self.env = ActionSkipWrapper(self.env, skip)


class GlobalWrapper(FullWrapper):

    def __init__(self, env: EvoGymBase, obs_details: Dict[str, Any], skip: int = 0, fixed_body: bool = False,
                 qd_wrappers: List[str] = None):
        super().__init__(env, skip, qd_wrappers)
        self.env = GlobalObservationWrapper(
            GlobalActionWrapper(self.env, fixed_body=fixed_body),
            fixed_body=fixed_body, obs_details=obs_details
        )


class LocalWrapper(FullWrapper):
    def __init__(self, env: EvoGymBase, obs_details: Dict[str, Any], skip: int = 0, qd_wrappers: List[str] = None):
        super().__init__(env, skip, qd_wrappers)
        self.env = LocalObservationWrapper(LocalActionWrapper(self.env), obs_details=obs_details)

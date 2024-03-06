# taken from https://github.com/mertan-a/gecco-23
from __future__ import annotations

from typing import Union

import numpy as np

from bbbqd.wrappers.action_wrappers import *
from bbbqd.wrappers.full_wrappers import FullWrapper, GlobalWrapper, LocalWrapper
from bbbqd.wrappers.observation_wrappers import *


def make_env(config: Dict[str, Any], body: np.ndarray = None) -> Union[gym.Env, FullWrapper]:
    if body is None:
        body = np.array(config["body"])
    env = gym.make(config["env_name"], body=body)
    if config["controller"] == "global":
        env = GlobalWrapper(env, skip=config.get("skip", 0), **config["flags"])
    elif config["controller"] == "local":
        env = LocalWrapper(env, skip=config.get("skip", 0), **config["flags"])
    return env

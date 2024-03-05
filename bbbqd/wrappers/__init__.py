# taken from https://github.com/mertan-a/gecco-23
from __future__ import annotations

from typing import Union

from bbbqd.wrappers.action_wrappers import *
from bbbqd.wrappers.full_wrappers import FullWrapper, GlobalWrapper, LocalWrapper
from bbbqd.wrappers.observation_wrappers import *


def make_env(config: Dict[str, Any]) -> Union[gym.Env, FullWrapper]:
    env = gym.make(config["env_name"], body=np.array(config["body"]))
    if config["controller"] == "global":
        env = GlobalWrapper(env, **config["flags"])
    elif config["controller"] == 'local':
        env = LocalWrapper(env, **config["flags"])
    return env

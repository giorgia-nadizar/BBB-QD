# partially taken from https://github.com/mertan-a/gecco-23
from __future__ import annotations

from typing import Union

from bbbqd.wrappers.action_wrappers import *
from bbbqd.wrappers.full_wrappers import FullWrapper, GlobalWrapper, LocalWrapper
from bbbqd.wrappers.observation_wrappers import *


def make_env(config: Dict[str, Any], body: np.ndarray = None) -> Union[gym.Env, FullWrapper]:
    if body is None:
        if config.get("body") is not None:
            body = np.array(config["body"])
        else:
            body = np.ones((config["grid_size"], config["grid_size"])) * 3
    qd_wrappers = config.get("qd_wrappers", [])
    if isinstance(qd_wrappers, str):
        qd_wrappers = [qd_wrappers]
    env = gym.make(config["env_name"], body=body)
    fixed_body = config.get("fixed_body", False)
    if config["controller"] == "global":
        env = GlobalWrapper(env,
                            skip=config.get("skip", 0),
                            fixed_body=fixed_body,
                            obs_details=config["flags"],
                            qd_wrappers=qd_wrappers
                            )
    elif config["controller"] == "local":
        env = LocalWrapper(env,
                           skip=config.get("skip", 0),
                           obs_details=config["flags"],
                           qd_wrappers=qd_wrappers
                           )
    return env

from typing import Union, Dict, Any

import numpy as np

from bbbqd.body.bodies import has_actuator
from bbbqd.brain.controllers import ControllerWrapper, Controller
from bbbqd.wrappers import make_env


# TODO: will also provide descriptors
def evaluate_controller_and_body(controller: Union[Controller, ControllerWrapper],
                                 body: Union[np.ndarray, None],
                                 config: Dict[str, Any],
                                 render: bool = False) -> float:
    if body is not None and not has_actuator(body):
        print("Body with no actuator, negative infinity fitness.")
        return -np.infty

    env = make_env(config, body)
    cumulative_reward = 0
    obs = env.reset()
    for _ in range(config["episode_length"]):
        action = controller.compute_action(obs)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        if render:
            env.render()
        if done:
            break

    env.close()
    return cumulative_reward


def evaluate_controller(controller: Union[Controller, ControllerWrapper],
                        config: Dict[str, Any],
                        render: bool = False) -> float:
    return evaluate_controller_and_body(controller, None, config, render)

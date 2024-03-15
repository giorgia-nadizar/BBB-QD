from typing import Union, Dict, Any, Callable, Tuple

import numpy as np

from bbbqd.body.bodies import has_actuator
from bbbqd.brain.controllers import ControllerWrapper, Controller
from bbbqd.wrappers import make_env


def evaluate_controller_and_body(controller: Union[Controller, ControllerWrapper],
                                 body: Union[np.ndarray, None],
                                 config: Dict[str, Any],
                                 descriptors_functions: Tuple[
                                     Callable[[Dict[str, Any]], np.ndarray], Callable[[np.ndarray], np.ndarray]
                                 ] = None,
                                 render: bool = False) -> Union[Tuple[float, np.ndarray], float]:
    if body is not None and not has_actuator(body):
        print("Body with no actuator, negative infinity fitness.")
        return -np.infty

    env = make_env(config, body)
    cumulative_reward = 0
    obs = env.reset()
    descriptors = []
    for _ in range(config["episode_length"]):
        action = controller.compute_action(obs)
        obs, reward, done, info = env.step(action)
        if descriptors_functions is not None:
            descriptors.append(descriptors_functions[0](info))
        cumulative_reward += reward
        if render:
            env.render()
        if done:
            break

    env.close()
    if descriptors_functions is None:
        return cumulative_reward
    else:
        return cumulative_reward, descriptors_functions[1](np.asarray(descriptors))


def evaluate_controller(controller: Union[Controller, ControllerWrapper],
                        config: Dict[str, Any],
                        descriptors_extractor: Callable[[Dict[str, Any]], np.ndarray] = None,
                        render: bool = False) -> Union[Tuple[float, np.ndarray], float]:
    return evaluate_controller_and_body(controller, None, config, descriptors_extractor, render)

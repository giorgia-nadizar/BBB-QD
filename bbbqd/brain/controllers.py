from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from typing import Callable, List, Union, Dict, Any


def compute_controller_generation_fn(config: Dict[str, Any]) -> Callable[
    [Callable[[np.ndarray], np.ndarray]], Union[Controller, ControllerWrapper]]:
    def _controller_fn(control_fn: Callable[[np.ndarray], np.ndarray]) -> Union[Controller, ControllerWrapper]:
        base_controller = LocalController(control_fn) if config["controller"] == "local" \
            else GlobalController(control_fn)
        return JaxControllerWrapper(base_controller) if config.get("jax", False) else base_controller

    return _controller_fn


class Controller:

    def __init__(self, control_function: Callable[[np.ndarray], np.ndarray]):
        self.control_function = control_function

    def compute_action(self, observation: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        raise NotImplementedError


class GlobalController(Controller):

    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        return self.control_function(observation)


class LocalController(Controller):

    def compute_action(self, observation: List[np.ndarray]) -> List[np.ndarray]:
        return [self.control_function(x) for x in observation]


class ControllerWrapper:
    def __init__(self, controller: Controller):
        self.controller = controller

    def compute_action(self, observation: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        return self.controller.compute_action(observation)


class JaxControllerWrapper(ControllerWrapper):

    def compute_action(self, observation: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        # Convert observation to proper JAX format
        if isinstance(observation, List):
            jax_observation = [jnp.asarray(x) for x in observation]
        else:
            jax_observation = jnp.asarray(observation)
        jax_action = self.controller.compute_action(jax_observation)
        if isinstance(jax_action, List):
            return np.asarray([np.asarray(x) for x in jax_action])
        else:
            return np.asarray(jax_action)

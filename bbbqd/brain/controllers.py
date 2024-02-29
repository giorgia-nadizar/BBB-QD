import numpy as np
from typing import Callable, List


class Controller:

    def __init__(self, control_function: Callable[[np.ndarray], np.ndarray]):
        self.control_function = control_function


class GlobalController(Controller):

    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        return self.control_function(observation)


class LocalController(Controller):

    def compute_action(self, observation: List[np.ndarray]) -> List[np.ndarray]:
        return [self.control_function(x) for x in observation]

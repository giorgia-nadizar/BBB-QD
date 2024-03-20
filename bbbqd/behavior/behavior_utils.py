from typing import List, Any, Callable, Dict, Tuple, Union

import numpy as np

from bbbqd.behavior.behavior_descriptors import _compute_spectrum


def get_behavior_descriptors_functions(config: Dict[str, Any]) -> Tuple[
    Callable[[Dict[str, Any]], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    descriptors = config["behavior_descriptors"]
    if isinstance(descriptors, str):
        descriptors = [descriptors]
    descriptors_extractor_function = _get_descriptors_extractor_function(descriptors)
    behavior_descriptors_computing_function = _get_behavior_descriptors_computing_function(config)
    return descriptors_extractor_function, behavior_descriptors_computing_function


def _get_descriptors_extractor_function(descriptors: str) -> Callable[[Dict[str, Any]], np.ndarray]:
    existing_descriptors = ["velocity", "position", "velocity_polar"]
    for d in descriptors:
        assert d in existing_descriptors

    def _descriptors_extractor(info: Dict[str, Any]) -> np.ndarray:
        descriptors_data = []
        for descriptor in descriptors:
            if "polar" in descriptor:
                x_y = info[descriptor.replace("_polar", "")]
                descriptors_data.append(np.sqrt(x_y[0] ** 2 + x_y[1] ** 2))
                descriptors_data.append(np.arctan2(x_y[1], x_y[0]))
            else:
                descriptors_data.extend(info[descriptor].tolist())
        return np.asarray(descriptors_data)

    return _descriptors_extractor


def _get_behavior_descriptors_computing_function(config: Dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    # TODO change this
    return _compute_spectrum

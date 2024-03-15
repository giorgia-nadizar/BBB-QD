from typing import List, Any, Callable, Dict, Tuple

import numpy as np


def get_descriptors_functions(config: Dict[str, Any]) -> Tuple[
    Callable[[Dict[str, Any]], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    descriptors_extractor_function = _get_descriptors_extractor_function(config["descriptors"])
    behavior_descriptors_computing_function = _get_behavior_descriptors_computing_function(config)
    return descriptors_extractor_function, behavior_descriptors_computing_function


def _get_descriptors_extractor_function(descriptors: List[str]) -> Callable[[Dict[str, Any]], np.ndarray]:
    existing_descriptors = ["velocity", "position"]
    for descriptor in descriptors:
        assert descriptor in existing_descriptors

    def _descriptors_extractor(info: Dict[str, Any]) -> np.ndarray:
        return np.concatenate([info[d] for d in descriptors])

    return _descriptors_extractor


def _get_behavior_descriptors_computing_function(config: Dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    raise NotImplementedError

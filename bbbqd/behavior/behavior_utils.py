from typing import Any, Callable, Dict, Tuple

import numpy as np

from bbbqd.behavior.behavior_descriptors import _compute_spectra, _signal_peak, _signal_median


def get_behavior_descriptors_functions(config: Dict[str, Any]) -> Tuple[
    Callable[[Dict[str, Any]], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    descriptors = config["behavior_descriptors"]
    processing = config.get("behavior_descriptors_processing", "median")
    if isinstance(descriptors, str):
        descriptors = [descriptors]
    descriptors_extractor_function = _get_descriptors_extractor_function(descriptors)
    behavior_descriptors_computing_function = _get_behavior_descriptors_computing_function(processing)
    return descriptors_extractor_function, behavior_descriptors_computing_function


def _get_descriptors_extractor_function(descriptors: str) -> Callable[[Dict[str, Any]], np.ndarray]:
    existing_descriptors = ["velocity", "velocity_x", "velocity_y", "velocity_angle", "velocity_module",
                            "position", "position_x", "position_y",
                            "angle"]
    for d in descriptors:
        assert d in existing_descriptors

    def _descriptors_extractor(info: Dict[str, Any]) -> np.ndarray:
        descriptors_data = []
        for descriptor in descriptors:
            if descriptor == "velocity_angle":
                x_y = info["velocity"]
                descriptors_data.append(np.arctan2(x_y[1], x_y[0]))
            elif descriptor == "velocity_module":
                x_y = info["velocity"]
                descriptors_data.append(np.sqrt(x_y[0] ** 2 + x_y[1] ** 2))
            elif "x" in descriptor:
                descriptors_data.append(info[descriptor.replace("_x", "")][0])
            elif "y" in descriptor:
                descriptors_data.append(info[descriptor.replace("_y", "")][1])
            else:
                descriptors_data.extend(info[descriptor].tolist())
        return np.asarray(descriptors_data)

    return _descriptors_extractor


def _get_behavior_descriptors_computing_function(processing: str) -> Callable[[np.ndarray], np.ndarray]:
    allowed_values = ["median", "peak"]
    if processing not in allowed_values:
        raise ValueError(f"Processing should be in {allowed_values}, got {processing}")
    processing_fn = _signal_median if processing == "median" else _signal_peak

    return lambda signals: np.asarray([processing_fn(s) for s in _compute_spectra(signals)])

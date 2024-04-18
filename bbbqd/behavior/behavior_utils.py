from functools import partial
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
    behavior_descriptors_computing_function = _get_behavior_descriptors_computing_function(processing,
                                                                                           cut_off=config.get(
                                                                                               "frequency_cut_off",
                                                                                               0.4))
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
                x_y = info["velocity"].reshape(-1, info["velocity"].shape[-1])
                velocity_angle = np.arctan2(x_y[:, 1], x_y[:, 0])
                velocity_angle_later = velocity_angle.reshape(len(velocity_angle), 1)
                descriptors_data.append(velocity_angle_later)
            elif descriptor == "velocity_module":
                x_y = info["velocity"].reshape(-1, info["velocity"].shape[-1])
                velocity_module = np.sqrt(x_y[:, 0] ** 2 + x_y[:, 1] ** 2)
                velocity_module = velocity_module.reshape(len(velocity_module), 1)
                descriptors_data.append(velocity_module)
            elif "x" in descriptor:
                descriptor_data = info[descriptor.replace("_x", "")]
                descriptor_data = descriptor_data.reshape(-1, descriptor_data.shape[-1])
                descriptor_data_x = descriptor_data[:, 0]
                descriptors_data.append(descriptor_data_x.reshape(len(descriptor_data_x), 1))
            elif "y" in descriptor:
                descriptor_data = info[descriptor.replace("_y", "")]
                descriptor_data = descriptor_data.reshape(-1, descriptor_data.shape[-1])
                descriptor_data_y = descriptor_data[:, 1]
                descriptors_data.append(descriptor_data_y.reshape(len(descriptor_data_y), 1))
            else:
                descriptors_data.append(info[descriptor].reshape(-1, info[descriptor].shape[-1]))
        ddd = np.hstack(descriptors_data)
        return ddd

    return _descriptors_extractor


def _get_behavior_descriptors_computing_function(processing: str, cut_off: float) -> Callable[[np.ndarray], np.ndarray]:
    allowed_values = ["median", "peak"]
    if processing not in allowed_values:
        raise ValueError(f"Processing should be in {allowed_values}, got {processing}")
    processing_fn = partial(_signal_median if processing == "median" else _signal_peak,
                            cut_off=cut_off)
    return lambda signals: np.asarray([processing_fn(s) for s in _compute_spectra(signals)])

from functools import partial
from typing import Any, Callable, Dict, Tuple, List

import numpy as np

from bbbqd.behavior.behavior_descriptors import _signal_peak, _signal_median, _compute_spectrum

existing_descriptors = ["velocity", "velocity_x", "velocity_y", "velocity_angle", "velocity_module",
                        "position", "position_x", "position_y",
                        "angle",
                        "object_velocity", "object_velocity_x", "object_velocity_y",
                        "object_position", "object_position_x", "object_position_y",
                        "object_angle",
                        "floor_contact", "walls_contact"]


def get_behavior_descriptors_functions(config: Dict[str, Any]) -> Tuple[
    Callable[[Dict[str, Any]], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    config_descriptors = config["behavior_descriptors"]

    if isinstance(config_descriptors, str):
        config_descriptors = [config_descriptors]

    descriptors = []
    for d in config_descriptors:
        assert d in existing_descriptors
        if f"{d}_x" in existing_descriptors and f"{d}_y" in existing_descriptors:
            descriptors.extend([f"{d}_x", f"{d}_y"])
        else:
            descriptors.append(d)
    descriptors_processing_functions = [_get_descriptor_processing_fn(descriptor, config) for descriptor in descriptors]

    descriptors_extractor_fn = _get_descriptors_extractor_function(descriptors)

    def _behavior_descriptors_computing_fn(raw_descriptors: np.ndarray) -> np.ndarray:
        if raw_descriptors.shape[1] != len(descriptors):
            raise AttributeError("Descriptor processing for multi-dimensional descriptors not implemented.")
        processed_descriptors = []
        for desc_id, processing_fn in enumerate(descriptors_processing_functions):
            descriptor_signal = raw_descriptors[:, desc_id]
            processed_descriptors.append(processing_fn(descriptor_signal))
        return np.array(processed_descriptors)

    return descriptors_extractor_fn, _behavior_descriptors_computing_fn


def _get_descriptors_extractor_function(descriptors: List[str]) -> Callable[[Dict[str, Any]], np.ndarray]:
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
            elif "_x" in descriptor:
                descriptor_data = info[descriptor.replace("_x", "")]
                descriptor_data = descriptor_data.reshape(-1, descriptor_data.shape[-1])
                descriptor_data_x = descriptor_data[:, 0]
                descriptors_data.append(descriptor_data_x.reshape(len(descriptor_data_x), 1))
            elif "_y" in descriptor:
                descriptor_data = info[descriptor.replace("_y", "")]
                descriptor_data = descriptor_data.reshape(-1, descriptor_data.shape[-1])
                descriptor_data_y = descriptor_data[:, 1]
                descriptors_data.append(descriptor_data_y.reshape(len(descriptor_data_y), 1))
            else:
                descriptors_data.append(info[descriptor].reshape(-1, info[descriptor].shape[-1]))
        ddd = np.hstack(descriptors_data)
        return ddd

    return _descriptors_extractor


def _get_descriptor_processing_fn(descriptor: str, config: Dict) -> Callable[[np.ndarray], float]:
    if descriptor == "object_angle":
        return lambda angle_descriptor: float(angle_descriptor.sum()) / len(angle_descriptor)
    elif descriptor == "floor_contact":
        return lambda floor_contact: float(floor_contact.sum()) / len(floor_contact)
    elif descriptor == "walls_contact":
        return _get_walls_contact_fn(config)
    else:
        return _get_fft_fn(config)


def _get_fft_fn(config: Dict) -> Callable[[np.ndarray], float]:
    processing = config.get("behavior_descriptors_processing", "median")
    cut_off = config.get("frequency_cut_off", 0.4)
    allowed_values = ["median", "peak"]
    if processing not in allowed_values:
        raise ValueError(f"Processing should be in {allowed_values}, got {processing}")
    processing_fn = partial(_signal_median if processing == "median" else _signal_peak, cut_off=cut_off)
    return lambda signal: processing_fn(_compute_spectrum(signal))


def _get_walls_contact_fn(config: Dict) -> Callable[[np.ndarray], float]:
    walls_contact_max = config.get("walls_contact_max", np.pi / 3)

    def _walls_contact_fn(walls_contact: np.ndarray) -> float:
        walls_contact_filtered = walls_contact[walls_contact != np.array(None)]
        average_walls_contact = 0. if len(walls_contact_filtered) == 0 \
            else float(walls_contact_filtered.sum()) / len(walls_contact_filtered)
        return min(average_walls_contact, walls_contact_max) / walls_contact_max

    return _walls_contact_fn

import numpy as np
from typing import Tuple, Dict, Callable

from qdax.types import Descriptor


def _single_body_descriptor_extractor(config: Dict) -> Tuple[Callable[[np.ndarray], Descriptor], int]:
    body_descriptors = config["body_descriptors"]
    if not isinstance(body_descriptors, list):
        body_descriptors = [body_descriptors]

    def _descriptor_extractor(body: np.ndarray) -> Descriptor:
        descriptor = []
        if "n_voxels" in body_descriptors:
            descriptor.append(n_voxels(body))
        if "relative_size" in body_descriptors:
            descriptor.append(relative_size(body))
        if "width" or "height" in body_descriptors:
            width, height = width_height(body)
            if "width" in body_descriptors:
                descriptor.append(width)
            if "height" in body_descriptors:
                descriptor.append(height)
        if "relative_width" or "relative_height" in body_descriptors:
            relative_width, relative_height = relative_width_height(body)
            if "relative_width" in body_descriptors:
                descriptor.append(relative_width)
            if "relative_height" in body_descriptors:
                descriptor.append(relative_height)
        if "elongation" in body_descriptors:
            descriptor.append(elongation(body))
        if "compactness" in body_descriptors:
            descriptor.append(compactness(body))
        return np.array(descriptor)

    return _descriptor_extractor, len(body_descriptors)


def _bounding_box_limits(body: np.ndarray) -> Tuple[int, int, int, int]:
    rows = np.any(body, axis=1)
    cols = np.any(body, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return x_min, x_max, y_min, y_max


def n_voxels(body: np.ndarray) -> int:
    return (body > 0).sum()


def relative_size(body: np.ndarray) -> float:
    return n_voxels(body) / (body.shape[0] * body.shape[1])


def width_height(body: np.ndarray) -> Tuple[int, int]:
    x_min, x_max, y_min, y_max = _bounding_box_limits(body)
    return x_max - x_min + 1, y_max - y_min + 1


def relative_width_height(body: np.ndarray) -> Tuple[float, float]:
    width, height = width_height(body)
    return width / body.shape[0], height / body.shape[1]


def elongation(body: np.ndarray, n_directions: int = 8) -> float:
    if n_directions <= 0:
        raise ValueError("n_directions must be positive")
    diameters = []
    coordinates = np.where(body.transpose() > 0)
    x_coordinates = coordinates[0]
    y_coordinates = coordinates[1]
    for i in range(n_directions):
        # TODO are we sure? how many directions do we need?
        theta = i * 2 * np.pi / n_directions
        rotated_x_coordinates = x_coordinates * np.cos(theta) - y_coordinates * np.sin(theta)
        rotated_y_coordinates = x_coordinates * np.sin(theta) + y_coordinates * np.cos(theta)
        x_side = np.max(rotated_x_coordinates) - np.min(rotated_x_coordinates) + 1
        y_side = np.max(rotated_y_coordinates) - np.min(rotated_y_coordinates) + 1
        diameter = min(x_side, y_side) / max(x_side, y_side)
        diameters.append(diameter)

    return 1 - min(diameters)


def compactness(body: np.ndarray) -> float:
    convex_hull = body > 0
    new_found = True
    while new_found:
        new_found = False
        false_coordinates = np.argwhere(convex_hull == False)
        for coordinate in false_coordinates:
            x, y = coordinate[0], coordinate[1]
            adjacent_count = 0
            adjacent_coordinates = []
            for d in [-1, 1]:
                adjacent_coordinates.append((x, y + d))
                adjacent_coordinates.append((x + d, y))
                adjacent_coordinates.append((x + d, y + d))
                adjacent_coordinates.append((x + d, y - d))
            for adj_x, adj_y in adjacent_coordinates:
                if 0 <= adj_x < body.shape[0] and 0 <= adj_y < body.shape[1] and convex_hull[adj_x][adj_y]:
                    adjacent_count += 1
            if adjacent_count >= 5:
                convex_hull[x][y] = True
                new_found = True

    return (body > 0).sum() / convex_hull.sum()

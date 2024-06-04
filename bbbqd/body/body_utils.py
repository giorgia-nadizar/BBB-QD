from functools import partial
from typing import Dict, Callable, Tuple, List

import jax.numpy as jnp
import numpy as np

from bbbqd.body.bodies import encode_body_directly, encode_body_indirectly


def structure_corners(observation: np.ndarray, robot_structure: np.ndarray, robot_bounding_box: Tuple[int, int],
                      ) -> List[List[List[float]]]:
    """ process the observation to each voxel's corner that exists in reading order (left to right, top to bottom)
            evogym returns basic observation for each point mass (corner of voxel) in reading order.
            but for the voxels that are neighbors -- share a corner, only one observation is returned.
            this function takes any observation in that form and returns a 2d list.
            each element of the list is a list of corners of the voxel and show the observation for 4 corners of the voxel.
            """
    f_structure = robot_structure.flatten()
    masses_idx = 0
    structure_corners = [None] * robot_bounding_box[0] * robot_bounding_box[1]
    for idx, val in enumerate(f_structure):
        if val == 0:
            continue
        else:
            if masses_idx == 0:
                # the first voxel has order NW-NE-SW-SE
                structure_corners[idx] = [[observation[0, 0], observation[1, 0]],
                                          [observation[0, 1], observation[1, 1]],
                                          [observation[0, 2], observation[1, 2]],
                                          [observation[0, 3], observation[1, 3]]]
                masses_idx += 4
            else:
                # check the 2d location and find out whether this voxel has a neighbor to its left or up
                height, width = _two_d_idx_of(idx, robot_bounding_box)
                left_idx = _one_d_idx_of(height, width - 1, robot_bounding_box)
                up_idx = _one_d_idx_of(height - 1, width, robot_bounding_box)
                upright_idx = _one_d_idx_of(height - 1, width + 1, robot_bounding_box)
                # ?x
                # xo
                if (width - 1 >= 0 and height - 1 >= 0  # it can have neighbors W and N
                        and structure_corners[left_idx] is not None  # neighbor W
                        and structure_corners[up_idx] is not None):  # neighbor N
                    # N and W neighbors are occupied: the new mass is only SE
                    structure_corners[idx] = [structure_corners[up_idx][2],
                                              structure_corners[up_idx][3],
                                              structure_corners[left_idx][3],
                                              [observation[0, masses_idx], observation[1, masses_idx]]
                                              ]
                    masses_idx += 1
                # ??x
                # xox
                elif (width - 1 >= 0  # it can have neighbor W
                      and structure_corners[left_idx] is not None  # neighbor W
                      and width + 1 < robot_bounding_box[1]  # it can have neighbor E
                      and height - 1 >= 0  # it can have neighbor N
                      and structure_corners[upright_idx] is not None  # neighbor NE
                      and robot_structure[height, width + 1] != 0):  # non-empty neighbor E
                    # W and NE neighbors are occupied: the new mass is only SE
                    structure_corners[idx] = [structure_corners[left_idx][1],
                                              structure_corners[upright_idx][2],
                                              structure_corners[left_idx][3],
                                              [observation[0, masses_idx],
                                               observation[1, masses_idx]]]
                    masses_idx += 1
                # __
                # xo
                elif width - 1 >= 0 and structure_corners[left_idx] is not None:
                    # only W neighbor is occupied: NE and SE masses are new
                    structure_corners[idx] = [structure_corners[left_idx][1],
                                              [observation[0, masses_idx], observation[1, masses_idx]],
                                              structure_corners[left_idx][3],
                                              [observation[0, masses_idx + 1], observation[1, masses_idx + 1]]
                                              ]
                    masses_idx += 2
                # ?x
                # _o
                elif height - 1 >= 0 and structure_corners[up_idx] is not None:
                    # only N neighbor is occupied: SW and SE masses are new
                    structure_corners[idx] = [structure_corners[up_idx][2],
                                              structure_corners[up_idx][3],
                                              [observation[0, masses_idx], observation[1, masses_idx]],
                                              [observation[0, masses_idx + 1], observation[1, masses_idx + 1]]
                                              ]
                    masses_idx += 2
                # _x
                # ox
                elif (width + 1 < robot_bounding_box[1]  # it can have neighbor E
                      and height - 1 >= 0  # it can have neighbor N
                      and structure_corners[upright_idx] is not None  # it has neighbor NE
                      and robot_structure[height, width + 1] != 0):  # it has neighbor E
                    # NE and E neighbors are occupied: NW, SW, and SE masses are new
                    structure_corners[idx] = [
                        [observation[0, masses_idx], observation[1, masses_idx]],
                        structure_corners[upright_idx][2],
                        [observation[0, masses_idx + 1], observation[1, masses_idx + 1]],
                        [observation[0, masses_idx + 2], observation[1, masses_idx + 2]]]
                    masses_idx += 3
                else:
                    # no neighbors are occupied, all four point masses are new
                    structure_corners[idx] = [
                        [observation[0, masses_idx], observation[1, masses_idx]],
                        [observation[0, masses_idx + 1], observation[1, masses_idx + 1]],
                        [observation[0, masses_idx + 2], observation[1, masses_idx + 2]],
                        [observation[0, masses_idx + 3], observation[1, masses_idx + 3]]]
                    masses_idx += 4

    return structure_corners


def polygon_area(x: List[float], y: List[float]) -> float:
    """ Calculates the area of an arbitrary polygon given its vertices in x and y (list) coordinates.
    assumes the order is wrong """
    x[0], x[1] = x[1], x[0]
    y[0], y[1] = y[1], y[0]
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    area = 0.5 * np.abs(main_area + correction)
    return area


def voxel_velocity(x: List[float], y: List[float]) -> Tuple[float, float]:
    """ Calculates the velocity of a voxel given its 4 corners' velocities """
    return (x[0] + x[1] + x[2] + x[3]) / 4.0, (y[0] + y[1] + y[2] + y[3]) / 4.0


def two_d_idx_of(idx: int, robot_bounding_box: Tuple[int, int]) -> Tuple[int, int]:
    return idx // robot_bounding_box[1], idx % robot_bounding_box[1]


def one_d_idx_of(x: int, y: int, robot_bounding_box: Tuple[int, int]) -> int:
    return x * robot_bounding_box[1] + y


def moore_neighbors(x: int, y: int, observation_range: int,
                    robot_bounding_box: Tuple[int, int]) -> List[Tuple[float, List[int]]]:
    """
    returns the 8 neighbors of a voxel in the structure
    """
    neighbors = []
    min_obs_range = observation_range * -1
    max_obs_range = observation_range + 1
    for i in range(min_obs_range, max_obs_range):
        for j in range(min_obs_range, max_obs_range):
            if 0 <= x + i < robot_bounding_box[0] and 0 <= y + j < robot_bounding_box[1]:
                neighbors.append((1.0, [x + i, y + j]))
            else:
                neighbors.append((-1.0, None))
    return neighbors


def one_hot_structure(robot_structure: np.ndarray, robot_bounding_box: Tuple[int, int]) -> np.ndarray:
    """
    returns a one-hot encoding of the structure
    """
    one_hot = np.zeros((robot_bounding_box[0], robot_bounding_box[1], 5))
    for i in range(robot_bounding_box[0]):
        for j in range(robot_bounding_box[1]):
            one_hot[i, j, int(robot_structure[i, j])] = 1
    return one_hot


def compute_body_mask(config: Dict) -> jnp.ndarray:
    body_encoding = config.get("body_encoding", "direct")
    if body_encoding == "direct":
        return jnp.ones((config["grid_size"]) ** 2) * 5
    elif body_encoding == "indirect":
        return jnp.ones((config["grid_size"]) ** 2) * 4
    else:
        raise ValueError("Body encoding must be either direct or indirect.")


def compute_body_mutation_mask(config: Dict) -> jnp.ndarray:
    return config["p_mut_body"] * jnp.ones((config["grid_size"]) ** 2)


def compute_body_float_genome_length(config: Dict) -> int:
    return 0 if config.get("body_encoding", "direct") == "direct" else config["grid_size"] ** 2


def _trim_body(body: np.ndarray) -> np.ndarray:
    cols_to_remove = np.all(body == 0, axis=0)
    body = body[:, ~cols_to_remove]
    return body


def compute_body_encoding_function(config: Dict) -> Callable[[jnp.ndarray], np.ndarray]:
    body_encoding = config.get("body_encoding", "direct")
    body_trim = config.get("body_trim", False)
    if body_encoding == "direct":
        body_encoding_fn = partial(encode_body_directly, make_connected=True)
    elif body_encoding == "indirect":
        n_elements = config.get("n_body_elements", 20)
        body_encoding_fn = partial(encode_body_indirectly, n_elements=n_elements)
    else:
        raise ValueError("Body encoding must be either direct or indirect.")
    if not body_trim:
        return body_encoding_fn
    else:

        def body_encoding_with_trim_fn(body_genome: np.ndarray) -> np.ndarray:
            body = body_encoding_fn(body_genome)
            return _trim_body(body)

        return body_encoding_with_trim_fn

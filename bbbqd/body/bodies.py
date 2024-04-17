import jax.numpy as jnp
import numpy as np


def _remove_not_connected_components(body: np.ndarray) -> np.ndarray:
    body_size = len(body)
    visited = np.zeros_like(body)
    largest_connected_grid = np.zeros_like(body)

    def _bfs(x: int, y: int):
        if 0 <= x < body_size and 0 <= y < body_size:
            # already visited
            if visited[x][y] == 1:
                return 0
            visited[x][y] = 1
            if body[x][y] > 0:
                working_grid[x][y] = 1
                return 1 + _bfs(x - 1, y) + _bfs(x + 1, y) + _bfs(x, y - 1) + _bfs(x, y + 1)
            else:
                # inside body but empty
                return 0
        else:
            # outside body
            return 0

    for outer_x in range(body_size):
        for outer_y in range(body_size):
            if body[outer_x][outer_y] > 0 and not visited[outer_x][outer_y]:
                working_grid = np.zeros_like(body)
                area = _bfs(outer_x, outer_y)
                if area > np.sum(largest_connected_grid):
                    largest_connected_grid = working_grid

    return body * largest_connected_grid


def has_actuator(robot: np.ndarray) -> bool:
    return np.any(robot == 3) or np.any(robot == 4)


def encode_body_directly(body_string: jnp.ndarray, make_connected: bool = False) -> np.ndarray:
    grid_size = np.sqrt(len(body_string)).astype(int)
    grid_body = np.reshape(np.asarray(body_string), (-1, grid_size))
    if make_connected:
        grid_body = _remove_not_connected_components(grid_body)
    return grid_body


def _find_candidates(boolean_matrix: np.ndarray) -> np.ndarray:
    candidates = np.zeros_like(boolean_matrix)
    for x in range(len(boolean_matrix)):
        for y in range(len(boolean_matrix)):
            if candidates[x][y] == 1:
                continue
            if boolean_matrix[x][y] == 0 and (
                    (x - 1 >= 0 and boolean_matrix[x - 1][y] == 1) or
                    (y - 1 >= 0 and boolean_matrix[x][y - 1] == 1) or
                    (x + 1 < len(boolean_matrix) and boolean_matrix[x + 1][y] == 1) or
                    (y + 1 < len(boolean_matrix) and boolean_matrix[x][y + 1] == 1)
            ):
                candidates[x][y] = 1
    return candidates


def _floating_occupation_to_boolean(occupation: np.ndarray, n_elements: int) -> np.ndarray:
    positive_occupation = occupation - np.min(occupation) + .1
    boolean_occupation = np.zeros_like(positive_occupation)
    idx = np.unravel_index(np.argmax(positive_occupation, axis=None), positive_occupation.shape)
    boolean_occupation[idx] = 1
    elements = 1
    while elements < n_elements:
        candidates = _find_candidates(boolean_occupation)
        occupation_candidates = positive_occupation * candidates
        idx = np.unravel_index(np.argmax(occupation_candidates, axis=None), occupation_candidates.shape)
        boolean_occupation[idx] = 1
        elements += 1
    return boolean_occupation


def encode_body_indirectly(body_string: jnp.ndarray, n_elements: int) -> np.ndarray:
    if n_elements < 1:
        raise ValueError("n_elements must be at least 1")
    occupation_string, material_string = jnp.split(body_string, 2)
    material_string = material_string + jnp.ones_like(material_string)
    grid_size = np.sqrt(len(occupation_string)).astype(int)
    occupation_grid = np.reshape(np.asarray(occupation_string), (-1, grid_size))
    boolean_occupation_grid = _floating_occupation_to_boolean(occupation_grid, n_elements)
    material_grid = np.reshape(np.asarray(material_string), (-1, grid_size))
    encoded_body = (boolean_occupation_grid * material_grid).astype(int)
    connected_body = _remove_not_connected_components(encoded_body)
    return encoded_body

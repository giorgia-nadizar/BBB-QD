import jax.numpy as jnp
import numpy as np


def remove_not_connected_components(body: np.ndarray) -> np.ndarray:
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


def encode_body(body_string: jnp.ndarray, make_valid: bool = False) -> np.ndarray:
    grid_size = np.sqrt(len(body_string)).astype(int)
    grid_body = np.reshape(np.asarray(body_string), (-1, grid_size))
    if make_valid:
        grid_body = remove_not_connected_components(grid_body)
    return grid_body


if __name__ == '__main__':
    sample_body = np.asarray([
        [1, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    print(sample_body)
    print(remove_not_connected_components(sample_body))

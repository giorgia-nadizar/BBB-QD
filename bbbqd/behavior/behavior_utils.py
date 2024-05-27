import numpy as np


def detect_ground_contact(
        robot: np.ndarray,
        ground: np.ndarray,
        side: float = 0.1,
        offset: float = 0.005
) -> bool:
    min_robot_x, max_robot_x = np.min(robot[0]), np.max(robot[0])
    ground = ground[:, (ground[0] > min_robot_x - side) & (ground[0] < max_robot_x + side)]
    ground = ground[:, ground[1] > 0]
    unique_positive_ground_y_values = np.unique(ground[1])
    # if the terrain is flat we don't need complex collision detection
    if len(unique_positive_ground_y_values) == 1:
        return np.min(robot[1]) <= (unique_positive_ground_y_values[0] + offset)

    highest_terrain_point = np.max(ground[1])
    for mass_x, mass_y in robot.transpose():
        # if above tallest point, surely not in contact
        if mass_y > (highest_terrain_point + offset):
            continue
        # find all terrain masses directly near the current one
        left_ground_masses = ground[:, (ground[0] <= mass_x) & (ground[0] >= (mass_x - side))]
        right_ground_masses = ground[:, (ground[0] >= mass_x) & (ground[0] <= (mass_x + side))]
        # if below either, then it's a contact
        if mass_y <= (np.max(left_ground_masses[1]) + offset) or mass_y <= (np.max(right_ground_masses[1]) + offset):
            return True

    return False

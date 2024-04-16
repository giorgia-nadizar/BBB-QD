from typing import Dict

import jax.numpy as jnp


def compute_body_mask(config: Dict) -> jnp.ndarray:
    return jnp.ones((config["grid_size"]) ** 2) * 5

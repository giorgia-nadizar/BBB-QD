from typing import Dict

import jax.numpy as jnp


def compute_body_mask(config: Dict) -> jnp.ndarray:
    return jnp.ones((config["grid_size"]) ** 2) * 5


def compute_body_mutation_mask(config: Dict) -> jnp.ndarray:
    return config["p_mut_body"] * jnp.ones((config["grid_size"]) ** 2)

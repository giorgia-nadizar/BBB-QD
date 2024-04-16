from functools import partial
from typing import Dict, Callable

import jax.numpy as jnp
import numpy as np

from bbbqd.body.bodies import encode_body


def compute_body_mask(config: Dict) -> jnp.ndarray:
    return jnp.ones((config["grid_size"]) ** 2) * 5


def compute_body_mutation_mask(config: Dict) -> jnp.ndarray:
    return config["p_mut_body"] * jnp.ones((config["grid_size"]) ** 2)


def compute_body_encoding_function(config: Dict) -> Callable[[jnp.ndarray], np.ndarray]:
    return partial(encode_body, make_connected=True)

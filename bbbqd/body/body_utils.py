from functools import partial
from typing import Dict, Callable

import jax.numpy as jnp
import numpy as np

from bbbqd.body.bodies import encode_body_directly, encode_body_indirectly


def compute_body_mask(config: Dict) -> jnp.ndarray:
    return jnp.ones((config["grid_size"]) ** 2) * 5


def compute_body_mutation_mask(config: Dict) -> jnp.ndarray:
    return config["p_mut_body"] * jnp.ones((config["grid_size"]) ** 2)


def compute_body_encoding_function(config: Dict) -> Callable[[jnp.ndarray], np.ndarray]:
    body_encoding = config.get("body_encoding", "direct")
    if body_encoding == "direct":
        return partial(encode_body_directly, make_connected=True)
    elif body_encoding == "indirect":
        n_elements = config.get("n_elements", 10)
        return partial(encode_body_indirectly, n_elements=n_elements)
    else:
        raise ValueError("Body encoding must be either direct or indirect.")

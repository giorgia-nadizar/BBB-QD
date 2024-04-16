from functools import partial
from typing import Dict, Callable, Tuple

import jax.numpy as jnp
import numpy as np

from bbbqd.body.bodies import encode_body_directly, encode_body_indirectly


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


def compute_body_encoding_function(config: Dict) -> Callable[[jnp.ndarray], np.ndarray]:
    body_encoding = config.get("body_encoding", "direct")
    if body_encoding == "direct":
        return partial(encode_body_directly, make_connected=True)
    elif body_encoding == "indirect":
        n_elements = config.get("n_body_elements", 10)
        return partial(encode_body_indirectly, n_elements=n_elements)
    else:
        raise ValueError("Body encoding must be either direct or indirect.")

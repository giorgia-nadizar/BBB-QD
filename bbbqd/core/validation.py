from typing import Callable
import jax.numpy as jnp

import numpy as np

from bbbqd.body.body_descriptors import width_height


def count_invalid_bodies(
        genomes: jnp.ndarray,
        body_genome_size: int,
        body_encoding_fn: Callable[[np.ndarray], np.ndarray],
        max_body_width: int,
) -> int:
    results = [validate_body_width(g, body_genome_size, body_encoding_fn, max_body_width) for g in genomes]
    return len(genomes) - sum(results)


def validate_body_width(
        genome: jnp.ndarray,
        body_genome_size: int,
        body_encoding_fn: Callable[[np.ndarray], np.ndarray],
        max_body_width: int,
) -> bool:
    body_genome, _ = jnp.split(genome, [body_genome_size])
    body = body_encoding_fn(body_genome)
    body_width, _ = width_height(body)
    return body_width <= max_body_width

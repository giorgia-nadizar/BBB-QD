from typing import Tuple, Callable

import jax

from qdax.types import Genotype, RNGKey


def isoline_and_body_mutation(
        isoline_mutation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        body_mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]]
) -> Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]]:
    def _composed_variation_fn(genome1: Genotype, genome2: Genotype, random_key: RNGKey) -> Tuple[Genotype, RNGKey]:
        nn_genome1, body_genome1 = genome1.pop("body")
        nn_genome2, body_genome2 = genome2.pop("body")
        random_key, nn_key = jax.random.split(random_key)
        random_key, body_key = jax.random.split(random_key)
        body_genome3, _ = body_mutation_fn(body_genome1, random_key)
        nn_genome3, random_key = isoline_mutation_fn(nn_genome1, nn_genome2, random_key)
        genome3 = nn_genome3.copy({"body": body_genome3})
        return genome3, random_key

    return _composed_variation_fn

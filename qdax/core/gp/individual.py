from functools import partial
from typing import Tuple, Callable, Dict, Any

from jax import jit, vmap
import jax.numpy as jnp
from jax import random

from qdax.types import RNGKey, Genotype


@jit
def _identity(x: Any) -> Any:
    return x


def levels_back_transformation_function(
        n_in: int,
        n_nodes: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def _levels_back_transformation_function(genome: jnp.ndarray) -> jnp.ndarray:
        x_genes, y_genes, other_genes = jnp.split(genome, [n_nodes, 2 * n_nodes])
        x_genes = jnp.arange(n_in, n_in + n_nodes) - x_genes - 1
        y_genes = jnp.arange(n_in, n_in + n_nodes) - y_genes - 1
        return jnp.concatenate((x_genes, y_genes, other_genes))

    return _levels_back_transformation_function


def compute_mutation_mask(
        config: Dict,
        n_out: int
) -> jnp.ndarray:
    if config["solver"] == "cgp":
        return _compute_cgp_mutation_mask(config, n_out)
    if config["solver"] == "lgp":
        return _compute_lgp_mutation_mask(config)
    raise ValueError("Solver must be either cgp or lgp.")


def _compute_cgp_mutation_mask(
        config: Dict,
        n_out: int
) -> jnp.ndarray:
    in_mut_mask = config["p_mut_inputs"] * jnp.ones(config["n_nodes"])
    f_mut_mask = config["p_mut_functions"] * jnp.ones(config["n_nodes"])
    out_mut_mask = config["p_mut_outputs"] * jnp.ones(n_out)
    return jnp.concatenate((in_mut_mask, in_mut_mask, f_mut_mask, out_mut_mask))


def _compute_lgp_mutation_mask(
        config: Dict
) -> jnp.ndarray:
    n_rows = config["n_rows"]
    lhs_mask = config["p_mut_lhs"] * jnp.ones(n_rows)
    rhs_mask = config["p_mut_rhs"] * jnp.ones(n_rows)
    f_mask = config["p_mut_functions"] * jnp.ones(n_rows)
    return jnp.concatenate((lhs_mask, rhs_mask, rhs_mask, f_mask))


def compute_genome_mask(
        config: Dict,
        n_in: int,
        n_out: int
) -> jnp.ndarray:
    if config["solver"] == "cgp":
        return _compute_cgp_genome_mask(config, n_in, n_out)
    if config["solver"] == "lgp":
        return _compute_lgp_genome_mask(config, n_in)
    raise ValueError("Solver must be either cgp or lgp.")


def _compute_cgp_genome_mask(
        config: Dict,
        n_in: int,
        n_out: int
) -> jnp.ndarray:
    n_nodes = config["n_nodes"]
    if config.get("recursive", False):
        in_mask = (n_in + n_nodes) * jnp.ones(n_nodes)
    elif config.get("levels_back") is not None:
        in_mask = jnp.minimum(
            config["levels_back"] * jnp.ones(n_nodes),
            jnp.arange(n_in, n_in + n_nodes)
        )
    else:
        in_mask = jnp.arange(n_in, n_in + n_nodes)
    f_mask = config["n_functions"] * jnp.ones(n_nodes)
    out_mask = (n_in + n_nodes) * jnp.ones(n_out)
    return jnp.concatenate((in_mask, in_mask, f_mask, out_mask))


def _compute_lgp_genome_mask(
        config: Dict,
        n_in: int
) -> jnp.ndarray:
    n_rows = config["n_rows"]
    n_registers = config["n_registers"]
    lhs_mask = (n_registers - n_in) * jnp.ones(n_rows)
    rhs_mask = n_registers * jnp.ones(n_rows)
    f_mask = config["n_functions"] * jnp.ones(n_rows)
    return jnp.concatenate((lhs_mask, rhs_mask, rhs_mask, f_mask))


def generate_integer_genome(
        genome_mask: jnp.ndarray,
        rnd_key: RNGKey,
        genome_transformation_function: Callable[[jnp.ndarray], jnp.ndarray] = _identity,
        fixed_trailing: jnp.ndarray = jnp.asarray([])
) -> jnp.ndarray:
    float_genome = random.uniform(key=rnd_key, shape=genome_mask.shape)
    integer_genome = jnp.floor(float_genome * genome_mask).astype(int)
    transformed_genome = genome_transformation_function(integer_genome)
    genome_trail, _ = jnp.split(transformed_genome, [len(transformed_genome) - len(fixed_trailing)])
    return jnp.concatenate([genome_trail, fixed_trailing]).astype(int)


def generate_float_genome(
        genome_length: int,
        rnd_key: RNGKey,
) -> jnp.ndarray:
    return random.uniform(key=rnd_key, shape=(genome_length,))


def generate_mixed_genome(
        float_genome_length: int,
        genome_mask: jnp.ndarray,
        rnd_key: RNGKey,
        genome_transformation_function: Callable[[jnp.ndarray], jnp.ndarray] = _identity,
        fixed_trailing: jnp.ndarray = jnp.asarray([])
) -> jnp.ndarray:
    float_rnd_key, int_rnd_key = random.split(rnd_key)
    float_genome = generate_float_genome(float_genome_length, float_rnd_key)
    integer_genome = generate_integer_genome(genome_mask, int_rnd_key, genome_transformation_function, fixed_trailing)
    return jnp.concatenate([float_genome, integer_genome])


def generate_population(
        pop_size: int,
        genome_mask: jnp.ndarray,
        rnd_key: RNGKey,
        float_header_length: int = 0,
        genome_transformation_function: Callable[[jnp.ndarray], jnp.ndarray] = _identity,
        fixed_genome_trailing: jnp.ndarray = jnp.asarray([])
) -> jnp.ndarray:
    sub_keys = random.split(rnd_key, pop_size)
    partial_generate_genome = partial(generate_mixed_genome,
                                      float_genome_length=float_header_length,
                                      genome_mask=genome_mask,
                                      genome_transformation_function=genome_transformation_function,
                                      fixed_trailing=fixed_genome_trailing
                                      )
    vmap_generate_genome = vmap(partial_generate_genome)
    return vmap_generate_genome(rnd_key=sub_keys)


def compute_mutation_fn(
        genome_mask: jnp.ndarray,
        mutation_mask: jnp.ndarray,
        float_mutation_sigma: float = 0.1
) -> Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]]:
    def _mutation_fn(genomes: Genotype, rand_key: RNGKey) -> Tuple[Genotype, RNGKey]:
        rand_key, *mutate_keys = random.split(rand_key, len(genomes) + 1)
        mutated_genomes = vmap(
            partial(mutate_genome, genome_mask=genome_mask, mutation_mask=mutation_mask,
                    float_mutation_sigma=float_mutation_sigma),
            in_axes=(0, 0)
        )(genomes, jnp.array(mutate_keys))
        return mutated_genomes, rand_key

    return _mutation_fn


def compute_variation_fn() -> Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]]:
    def _variation_fn(genomes1: Genotype, genomes2: Genotype, rand_key: RNGKey) -> Tuple[Genotype, RNGKey]:
        rand_key, *var_keys = random.split(rand_key, len(genomes1) + 1)
        crossed_over_genomes, _ = vmap(
            lgp_one_point_crossover_genomes,
            in_axes=(0, 0, 0)
        )(genomes1, genomes2, jnp.array(var_keys))
        return crossed_over_genomes, rand_key

    return _variation_fn


def compute_variation_mutation_fn(
        genome_mask: jnp.ndarray,
        mutation_mask: jnp.ndarray,
        float_mutation_sigma: float = 0.1
) -> Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]]:
    def _variation_mutation_fn(genomes1: Genotype, genomes2: Genotype, rand_key: RNGKey) -> Tuple[Genotype, RNGKey]:
        rand_key, *var_keys = random.split(rand_key, len(genomes1) + 1)
        crossed_over_genomes, _ = vmap(
            lgp_one_point_crossover_genomes,
            in_axes=(0, 0, 0)
        )(genomes1, genomes2, jnp.array(var_keys))
        rand_key, *mutate_keys = random.split(rand_key, len(crossed_over_genomes) + 1)
        mutated_genomes = vmap(
            partial(mutate_genome, genome_mask=genome_mask, mutation_mask=mutation_mask,
                    float_mutation_sigma=float_mutation_sigma),
            in_axes=(0, 0)
        )(crossed_over_genomes, jnp.array(mutate_keys))
        return mutated_genomes, rand_key

    return _variation_mutation_fn


def mutate_genome(
        genome: jnp.ndarray,
        rnd_key: RNGKey,
        genome_mask: jnp.ndarray,
        mutation_mask: jnp.ndarray,
        genome_transformation_function: Callable[[jnp.ndarray], jnp.ndarray] = _identity,
        float_mutation_sigma: float = 0.1
) -> jnp.ndarray:
    # if the header is floating only, split the genome
    floating_genome, integer_genome = jnp.split(genome, [len(genome) - len(genome_mask)])
    float_rnd_key, integer_rnd_key = random.split(rnd_key, 2)
    # mutate floating part
    mutated_floating_genome = (floating_genome
                               + random.normal(key=float_rnd_key, shape=floating_genome.shape) * float_mutation_sigma)
    # mutate integer part
    new_genome = generate_integer_genome(genome_mask, integer_rnd_key, genome_transformation_function)
    mutation_probs = random.uniform(key=rnd_key, shape=mutation_mask.shape)
    old_ids = (mutation_probs >= mutation_mask)
    new_ids = (mutation_probs < mutation_mask)
    mutated_integer_genome = jnp.floor(integer_genome * old_ids + new_ids * new_genome)
    return jnp.concatenate([mutated_floating_genome, mutated_integer_genome])


def lgp_one_point_crossover_genomes(
        genome1: jnp.ndarray,
        genome2: jnp.ndarray,
        rnd_key: RNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert len(genome1) == len(genome2)
    rnd_key, xover_key = random.split(rnd_key, 2)
    chunk_size = int(len(genome1) / 4)
    crossover_point = random.randint(xover_key, [1], 0, chunk_size)
    ids = jnp.arange(len(genome1))
    mask1 = (ids < crossover_point) \
            | ((ids >= chunk_size) & (ids < chunk_size + crossover_point)) \
            | ((ids >= 2 * chunk_size) & (ids < 2 * chunk_size + crossover_point)) \
            | ((ids >= 3 * chunk_size) & (ids < 3 * chunk_size + crossover_point))
    mask2 = jnp.invert(mask1)
    return jnp.where(mask1, genome1, genome2), jnp.where(mask2, genome1, genome2)

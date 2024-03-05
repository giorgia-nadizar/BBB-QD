from functools import partial
from multiprocessing import Pool

import jax
import jax.numpy as jnp

from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import evaluate_controller
from bbbqd.wrappers import make_env
from qdax.core.gp.encoding import compute_encoding_function
from qdax.core.gp.individual import compute_genome_mask, compute_mutation_mask, generate_population
from qdax.core.gp.utils import update_config

config = {
    "n_nodes": 50,
    "p_mut_inputs": 0.1,
    "p_mut_functions": 0.1,
    "p_mut_outputs": 0.3,
    "solver": "cgp",
    "env_name": "Walker-v0",
    "body": [[0, 0, 0], [3, 3, 3], [3, 0, 3]],
    "episode_length": 200,
    "pop_size": 10,
    "parents_size": 10,
    "n_iterations": 100,
    "seed": 0,
    "fixed_outputs": True,
    "controller": "global",
    "flags": {
        "observe_voxel_vel": True,
        "observe_voxel_volume": True,
        "observe_time": True,
    },
    "jax": True,
    "program_wrapper": True
}

# Create environment with wrappers
env = make_env(config)

# Update config with env info
config = update_config(config, env)

# Init a random key
random_key = jax.random.PRNGKey(config["seed"])

# Init population of controllers
genome_mask = compute_genome_mask(config, config["n_in"], config["n_out"])
mutation_mask = compute_mutation_mask(config, config["n_out"])
random_key, pop_key = jax.random.split(random_key)
if config.get("fixed_outputs", False):
    fixed_outputs = jnp.arange(start=config["buffer_size"] - config["n_out"], stop=config["buffer_size"], step=1)
    population = generate_population(
        pop_size=config["parents_size"],
        genome_mask=genome_mask,
        rnd_key=pop_key,
        fixed_genome_trailing=fixed_outputs
    )
    config["p_mut_outputs"] = 0
else:
    population = generate_population(pop_size=config["parents_size"], genome_mask=genome_mask, rnd_key=pop_key)

# Define encoding function
encoding_fn = compute_encoding_function(config)

# Define function to return the proper controller
controller_creation_fn = compute_controller_generation_fn(config)

# Create controller evaluation function
controller_evaluation_fn = partial(evaluate_controller, config=config)


def _evaluate_genome(genome: jnp.ndarray) -> float:
    return controller_evaluation_fn(controller_creation_fn(encoding_fn(genome)))


genomes_list = [g for g in population]

pool_size = config.get("pool_size", config["pop_size"])

fitnesses = []
start_idx = 0
while start_idx < len(genomes_list):
    with Pool(pool_size) as p:
        current_genomes = genomes_list[start_idx:min(start_idx + pool_size, len(genomes_list))]
        current_fitnesses = p.map(_evaluate_genome, current_genomes)
        fitnesses = fitnesses + current_fitnesses
        start_idx += pool_size

print(fitnesses)

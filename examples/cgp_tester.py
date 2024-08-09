from functools import partial

import jax
import jax.numpy as jnp

from bbbqd.body.bodies import encode_body_directly
from bbbqd.body.body_utils import compute_body_mask, compute_body_encoding_function, compute_body_mutation_mask, \
    compute_body_float_genome_length
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import evaluate_controller, evaluate_controller_and_body
from bbbqd.wrappers import make_env
from qdax.core.gp.encoding import compute_encoding_function
from qdax.core.gp.individual import compute_genome_mask, generate_population
from qdax.core.gp.utils import update_config

if __name__ == '__main__':
    config = {
        "n_nodes": 50,
        "p_mut_inputs": 0.1,
        "p_mut_functions": 0.1,
        "p_mut_outputs": 0.3,
        "solver": "cgp",
        "env_name": "Walker-v0",
        "episode_length": 10,
        "pop_size": 50,
        "parents_size": 45,
        "n_iterations": 500,
        "fixed_outputs": True,
        "controller": "local",
        "flags": {
            "observe_voxel_vel": True,
            "observe_voxel_volume": True,
            "observe_time": False,
        },
        "jax": True,
        "program_wrapper": True,
        "skip": 5,
        "grid_size": 10,
        "max_env_size": 5,
        "n_body_elements": 20,
        "body_encoding": "indirect",
        "fixed_body": False,
        "p_mut_body": 0.05,
        "seed": 0
    }

    # Create environment with wrappers
    env = make_env(config)

    # Update config with env info
    config = update_config(config, env)

    # Init a random key
    random_key = jax.random.PRNGKey(config["seed"])

    # Init population (with single individual)
    body_mask = compute_body_mask(config)
    controller_mask = compute_genome_mask(config, config["n_in"], config["n_out"])
    genome_mask = jnp.concatenate([body_mask, controller_mask])
    body_float_length = compute_body_float_genome_length(config)

    random_key, pop_key = jax.random.split(random_key)
    if config.get("fixed_outputs", False):
        fixed_outputs = jnp.arange(start=config["buffer_size"] - config["n_out"], stop=config["buffer_size"], step=1)
        population = generate_population(
            pop_size=1,
            genome_mask=genome_mask,
            rnd_key=pop_key,
            fixed_genome_trailing=fixed_outputs,
            float_header_length=body_float_length
        )
    else:
        population = generate_population(pop_size=1, genome_mask=genome_mask, rnd_key=pop_key,
                                         float_header_length=body_float_length)

    individual = population[0]

    # Define encoding function
    program_encoding_fn = compute_encoding_function(config)

    # Define function to return the proper controller
    controller_creation_fn = compute_controller_generation_fn(config)

    # Body encoding function
    body_encoding_fn = compute_body_encoding_function(config)

    # Encode individual and evaluate it
    if config.get("fixed_body", True):
        evaluate_controller(controller=controller_creation_fn(program_encoding_fn(individual)),
                            config=config,
                            render=True)
    else:
        body_genome, controller_genome = jnp.split(individual, [len(body_mask) + body_float_length])
        evaluate_controller_and_body(controller=controller_creation_fn(program_encoding_fn(controller_genome)),
                                     body=body_encoding_fn(body_genome),
                                     config=config,
                                     render=True)

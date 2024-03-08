from functools import partial

import jax
import jax.numpy as jnp

from bbbqd.body.bodies import encode_body
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
        "episode_length": 200,
        "pop_size": 50,
        "parents_size": 45,
        "n_iterations": 500,
        "fixed_outputs": True,
        "controller": "global",
        "flags": {
            "observe_voxel_vel": True,
            "observe_voxel_volume": True,
            "observe_time": False,
        },
        "jax": True,
        "program_wrapper": True,
        "skip": 5,
        # "body": [[3, 3, 3], [3, 0, 3], [3, 0, 3]],
        "seed": 0,
        "fixed_body": False,
        "grid_size": 5
    }

    # Create environment with wrappers
    env = make_env(config)

    # Update config with env info
    config = update_config(config, env)

    # Init a random key
    random_key = jax.random.PRNGKey(config["seed"])

    # Init population (with single individual)
    if config.get("fixed_body", True):
        genome_mask = compute_genome_mask(config, config["n_in"], config["n_out"])
    else:
        body_mask = jnp.ones((config["grid_size"]) ** 2) * 5
        controller_mask = compute_genome_mask(config, config["n_in"], config["n_out"])
        genome_mask = jnp.concatenate([body_mask, controller_mask])

    random_key, pop_key = jax.random.split(random_key)
    if config.get("fixed_outputs", False):
        fixed_outputs = jnp.arange(start=config["buffer_size"] - config["n_out"], stop=config["buffer_size"], step=1)
        population = generate_population(
            pop_size=1,
            genome_mask=genome_mask,
            rnd_key=pop_key,
            fixed_genome_trailing=fixed_outputs
        )
    else:
        population = generate_population(pop_size=1, genome_mask=genome_mask, rnd_key=pop_key)

    individual = population[0]

    # Define encoding function
    program_encoding_fn = compute_encoding_function(config)

    # Define function to return the proper controller
    controller_creation_fn = compute_controller_generation_fn(config)

    # Body encoding function
    body_encoding_fn = partial(encode_body, make_valid=True)

    # Encode individual and evaluate it
    if config.get("fixed_body", True):
        evaluate_controller(controller=controller_creation_fn(program_encoding_fn(individual)),
                            config=config,
                            render=True)
    else:
        body_genome, controller_genome = jnp.split(individual, [config["grid_size"] ** 2])
        evaluate_controller_and_body(controller=controller_creation_fn(program_encoding_fn(controller_genome)),
                                     body=body_encoding_fn(body_genome),
                                     config=config,
                                     render=True)

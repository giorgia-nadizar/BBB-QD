from functools import partial

import jax
import jax.numpy as jnp
from flax.core import FrozenDict, frozen_dict
from typing import Callable, Union, Dict

from jax import jit

from bbbqd.body.bodies import encode_body_directly
from bbbqd.body.body_utils import compute_body_mask, compute_body_encoding_function, compute_body_mutation_mask, \
    compute_body_float_genome_length
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import evaluate_controller, evaluate_controller_and_body
from bbbqd.wrappers import make_env
from qdax.core.gp.encoding import compute_encoding_function
from qdax.core.gp.individual import compute_genome_mask, generate_population
from qdax.core.gp.utils import update_config
from qdax.core.neuroevolution.networks.networks import MLP

if __name__ == '__main__':
    config = {
        "solver": "ne",
        "env_name": "Walker-v0",
        "episode_length": 10,
        "pop_size": 50,
        "parents_size": 45,
        "n_iterations": 500,
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
        "seed": 0,
        # nn params
        "policy_hidden_layer_sizes": (20, 20),
        "sigma": 0.05,
    }

    # Create environment with wrappers
    env = make_env(config)

    # Init a random key
    random_key = jax.random.PRNGKey(config["seed"])

    # Init body genome part
    body_mask = compute_body_mask(config)

    # Init policy network
    policy_layer_sizes = config["policy_hidden_layer_sizes"] + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Compute mutation masks
    body_mutation_mask = compute_body_mutation_mask(config)
    body_float_length = compute_body_float_genome_length(config)
    body_genome_length = len(body_mask) + body_float_length


    # Define function to return the proper nn policy and controller
    def _nn_policy_creation_fn(policy_params: FrozenDict) -> Callable[[jnp.ndarray], jnp.ndarray]:
        def _nn_policy_fn(observations: jnp.ndarray) -> jnp.ndarray:
            actions = policy_network.apply(policy_params, observations)
            return actions

        return jit(_nn_policy_fn)


    controller_creation_fn = compute_controller_generation_fn(config)

    # Body encoding function
    body_encoding_fn = compute_body_encoding_function(config)

    # Generate population
    random_key, pop_key = jax.random.split(random_key)
    body_generation_fn = partial(generate_population,
                                 genome_mask=body_mask,
                                 rnd_key=pop_key,
                                 float_header_length=body_float_length)

    init_bodies = body_generation_fn(1)
    random_key, nns_key = jax.random.split(random_key)
    keys = jax.random.split(nns_key, num=1)
    fake_batch = jnp.zeros(shape=(1, env.observation_size))
    init_nns = jax.vmap(policy_network.init)(keys, fake_batch)
    population = init_nns.copy({"body": init_bodies})


    # util functions for iterating the frozen dicts and rearranging them into a list
    def _get_items_at_index(dictionary: Union[FrozenDict, Dict], index: int,
                            new_dict: Dict = {}) -> Union[FrozenDict, Dict]:
        current_keys = dictionary.keys()
        for k in current_keys:
            if isinstance(dictionary[k], FrozenDict):
                new_dict[k] = {}
                _get_items_at_index(dictionary[k], index, new_dict[k])
            else:
                new_dict[k] = dictionary[k][index]
        return new_dict


    individual = frozen_dict.freeze(_get_items_at_index(population, 0))

    # Encode individual and evaluate it
    if config.get("fixed_body", True):
        evaluate_controller(controller=controller_creation_fn(_nn_policy_creation_fn(individual)),
                            config=config,
                            render=True)
    else:
        nn_genome, body_genome = individual.pop("body")
        evaluate_controller_and_body(controller=controller_creation_fn(_nn_policy_creation_fn(nn_genome)),
                                     body=body_encoding_fn(body_genome),
                                     config=config,
                                     render=True)

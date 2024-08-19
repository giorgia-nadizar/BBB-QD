import copy
import functools
import os
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable, Union, List

import yaml
from flax.core import FrozenDict, frozen_dict
from jax import jit

from bbbqd.body.body_utils import compute_body_mask, compute_body_mutation_mask, compute_body_encoding_function, \
    compute_body_float_genome_length
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import evaluate_controller_and_body
from bbbqd.core.misc import isoline_and_body_mutation, nn_and_body_mutation
from bbbqd.wrappers import make_env
from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.emitters.mutation_operators import isoline_variation, gaussian_mutation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.gp.individual import compute_genome_mask, generate_population, compute_mutation_fn
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import RNGKey, Fitness, ExtraScores
from qdax.utils.metrics import default_ga_metrics, CSVLogger


# This is a placeholder class, all local functions will be added as its attributes
class _LocalFunctions:
    @classmethod
    def add_functions(cls, *args):
        for function in args:
            setattr(cls, function.__name__, function)
            function.__qualname__ = cls.__qualname__ + '.' + function.__name__


def run_body_evo_ga(config: Dict[str, Any]):
    # Create environment with wrappers
    env = make_env(config)

    # Update config with env info
    # config = update_config(config, env)

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

    init_bodies = body_generation_fn(config["parents_size"])
    random_key, nns_key = jax.random.split(random_key)
    keys = jax.random.split(nns_key, num=config["parents_size"])
    fake_batch = jnp.zeros(shape=(config["parents_size"], env.observation_size))
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

    def _rearrange_genomes(genomes: FrozenDict) -> List[FrozenDict]:
        return [frozen_dict.freeze(_get_items_at_index(genomes, g)) for g in range(len(genomes["body"]))]

    # Create evaluation function
    evaluation_fn = partial(evaluate_controller_and_body, config=config)

    # Define genome evaluation fn -> returns fitness and brain, body, behavior descriptors
    def _evaluate_genome(genome: FrozenDict) -> float:
        nn_genome, body_genome = genome.pop("body")
        controller = controller_creation_fn(_nn_policy_creation_fn(nn_genome))
        body = body_encoding_fn(body_genome)
        fitness = evaluation_fn(controller, body)
        return fitness

    # Add all functions to _LocalFunctions class, separating each with a comma
    _LocalFunctions.add_functions(_evaluate_genome, _nn_policy_creation_fn)

    # Define scoring fn
    def _scoring_fn(genomes: FrozenDict, rnd_key: RNGKey) -> Tuple[Fitness, ExtraScores, RNGKey]:
        genomes_list = _rearrange_genomes(genomes)
        pool_size = config.get("pool_size", config["pop_size"])
        fitnesses = []
        start_idx = 0
        while start_idx < len(genomes_list):
            with Pool(pool_size) as p:
                current_genomes = genomes_list[start_idx:min(start_idx + pool_size, len(genomes_list))]
                current_fitnesses = p.map(_evaluate_genome, current_genomes)
                fitnesses = fitnesses + current_fitnesses
                start_idx += pool_size
        return jnp.expand_dims(jnp.asarray(fitnesses), axis=1), None, rnd_key

    # Define emitter
    nn_mutation_fn = functools.partial(
        gaussian_mutation, sigma=config["sigma"]
    )
    body_mutation_fn = compute_mutation_fn(body_mask, body_mutation_mask, config.get("float_mutation_sigma", 0.1))
    mutation_fn = nn_and_body_mutation(nn_mutation_fn, body_mutation_fn)
    mixing_emitter = MixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=None,
        variation_percentage=0.0,
        batch_size=config["parents_size"]
    )

    genetic_algorithm = GeneticAlgorithm(
        scoring_function=_scoring_fn,
        emitter=mixing_emitter,
        metrics_function=default_ga_metrics,
    )

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = genetic_algorithm.init(population, config["pop_size"], random_key)

    headers = ["iteration", "max_fitness", "time", "current_time"]

    name = f"{config.get('run_name', 'trial')}_{config['seed']}"

    csv_logger = CSVLogger(
        f"../results/ga/{name}.csv",
        header=headers
    )

    for i in range(config["n_iterations"]):
        start_time = time.time()
        # main iterations
        repertoire, emitter_state, metrics, random_key = genetic_algorithm.update(repertoire, emitter_state, random_key)
        timelapse = time.time() - start_time
        current_time = datetime.now()

        # log metrics
        logged_metrics = {"time": timelapse, "iteration": i + 1, "current_time": current_time,
                          "max_fitness": metrics["max_fitness"][0]}

        csv_logger.log(logged_metrics)
        print(f"{i}\t{logged_metrics['max_fitness']}")

    os.makedirs(f"../results/ga/{name}/", exist_ok=True)
    repertoire.save(f"../results/ga/{name}/")
    with open(f"../results/ga/{name}/config.yaml", "w") as file:
        yaml.dump(config, file)


if __name__ == '__main__':
    seeds = range(10)
    envs = [  # "Walker-v0"
        "BridgeWalker-v0",
        # "Pusher-v0",
        # "UpStepper-v0",
        # "DownStepper-v0",
        # "ObstacleTraverser-v0",

        # "ObstacleTraverser-v1",
        # "Hurdler-v0",
        "PlatformJumper-v0",
        # "GapJumper-v0",
        "CaveCrawler-v0",
        "CustomCarrier-v0"
    ]
    base_cfg = {
        "p_mut_body": 0.05,
        "solver": "ne",
        "env_name": "Walker-v0",
        "episode_length": 200,
        "pop_size": 50,
        "parents_size": 45,
        "n_iterations": 4000,
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

        # nn params
        "policy_hidden_layer_sizes": (20, 20),
        "sigma": 0.05,
    }

    counter = 0
    for seed in seeds:
        for env in envs:
            counter += 1
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = seed
            cfg["env_name"] = env
            cfg["run_name"] = f"evo-body-nn-{cfg['grid_size']}x{cfg['grid_size']}-{env.replace('-v0', '').lower()}"
            print(
                f"{counter}/{len(seeds) * len(envs)} -> evo-body-{cfg['grid_size']}x{cfg['grid_size']}, "
                f"{seed}, {env.replace('-v0', '').lower()}")
            run_body_evo_ga(cfg)

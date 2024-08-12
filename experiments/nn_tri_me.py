import copy
import functools
import os
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from typing import Tuple, Dict, Any, Callable, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.core import FrozenDict, frozen_dict
from jax import jit

from bbbqd.behavior.behavior_descriptors import get_behavior_descriptors_functions
from bbbqd.body.body_descriptors import get_body_descriptor_extractor
from bbbqd.body.body_utils import compute_body_mask, compute_body_mutation_mask, compute_body_encoding_function, \
    compute_body_float_genome_length
from bbbqd.brain.brain_descriptors import get_nn_descriptor_extractor
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import evaluate_controller_and_body
from bbbqd.core.misc import isoline_and_body_mutation
from bbbqd.wrappers import make_env
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.gp.individual import generate_population, \
    compute_mutation_fn
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.tri_map_elites import TriMAPElites, sampling_function
from qdax.types import RNGKey, Fitness, ExtraScores, Descriptor
from qdax.utils.metrics import CSVLogger, default_triqd_metrics


# This is a placeholder class, all local functions will be added as its attributes
class _LocalFunctions:
    @classmethod
    def add_functions(cls, *args):
        for function in args:
            setattr(cls, function.__name__, function)
            function.__qualname__ = cls.__qualname__ + '.' + function.__name__


def run_body_evo_me(config: Dict[str, Any]):
    # Create environment with wrappers
    env = make_env(config)

    # TODO can this be removed?
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

    # Define function to return the proper nn policy and controller
    def _nn_policy_creation_fn(policy_params: FrozenDict) -> Callable[[jnp.ndarray], jnp.ndarray]:
        def _nn_policy_fn(observations: jnp.ndarray) -> jnp.ndarray:
            actions = policy_network.apply(policy_params, observations)
            return actions

        return jit(_nn_policy_fn)

    controller_creation_fn = compute_controller_generation_fn(config)

    # Body encoding function
    body_encoding_fn = compute_body_encoding_function(config)

    # Descriptors
    brain_descr_fn, _ = get_nn_descriptor_extractor(config)
    body_descr_fn, _ = get_body_descriptor_extractor(config)
    behavior_descr_fns = get_behavior_descriptors_functions(config)

    # Generate population
    random_key, bodies_key = jax.random.split(random_key)
    body_generation_fn = partial(generate_population,
                                 genome_mask=body_mask,
                                 rnd_key=bodies_key,
                                 float_header_length=body_float_length)
    init_bodies = body_generation_fn(config["parents_size"])
    random_key, nns_key = jax.random.split(random_key)
    keys = jax.random.split(nns_key, num=config["parents_size"])
    fake_batch = jnp.zeros(shape=(config["parents_size"], env.observation_size))
    init_nns = jax.vmap(policy_network.init)(keys, fake_batch)
    population = init_nns.copy({
        "body": init_bodies,
    })

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
    evaluation_fn = partial(evaluate_controller_and_body, config=config, descriptors_functions=behavior_descr_fns)

    # Define genome evaluation fn -> returns fitness and brain, body, behavior descriptors
    def _evaluate_genome(genome: FrozenDict) -> Tuple[float, np.ndarray]:
        # ignore the individual id
        nn_genome, body_genome = genome.pop("body")
        controller = controller_creation_fn(_nn_policy_creation_fn(nn_genome))
        body = body_encoding_fn(body_genome)
        brain_descriptors = brain_descr_fn(nn_genome)
        body_descriptors = body_descr_fn(body)
        fitness, behavior_descriptors = evaluation_fn(controller, body)
        return fitness, np.concatenate([brain_descriptors, body_descriptors, behavior_descriptors])

    # Add all functions to _LocalFunctions class, separating each with a comma
    _LocalFunctions.add_functions(_evaluate_genome, _nn_policy_creation_fn)

    # Define scoring fn
    def _qd_scoring_fn(genomes: FrozenDict, rnd_key: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        genomes_list = _rearrange_genomes(genomes)
        pool_size = config.get("pool_size", config["pop_size"])
        fitnesses = []
        descriptors = []
        start_idx = 0
        while start_idx < len(genomes_list):
            with Pool(pool_size) as p:
                current_genomes = genomes_list[start_idx:min(start_idx + pool_size, len(genomes_list))]
                current_fitnesses_descriptors = p.map(_evaluate_genome, current_genomes)
                for fitness, desc in current_fitnesses_descriptors:
                    fitnesses.append(fitness)
                    descriptors.append(jnp.asarray(desc))
                start_idx += pool_size

        return jnp.asarray(fitnesses), jnp.asarray(descriptors), None, rnd_key

    # Define emitter
    isoline_mutation_fn = functools.partial(
        isoline_variation, iso_sigma=config["iso_sigma"], line_sigma=config["line_sigma"]
    )
    body_mutation_fn = compute_mutation_fn(body_mask, body_mutation_mask, config.get("float_mutation_sigma", 0.1))
    variation_fn = isoline_and_body_mutation(isoline_mutation_fn, body_mutation_fn)
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=config["parents_size"]
    )

    # Sampling function
    sampling_id_fn = sampling_function(config["sampler"])

    tri_qd_metrics = partial(default_triqd_metrics, qd_offset=0)
    tri_map_elites = TriMAPElites(
        scoring_function=_qd_scoring_fn,
        emitter=mixing_emitter,
        metrics_function=tri_qd_metrics,
        descriptors_indexes1=jnp.asarray([0, 1]),
        descriptors_indexes2=jnp.asarray([2, 3]),
        descriptors_indexes3=jnp.asarray([4, 5]),
        sampling_id_function=sampling_id_fn,
    )

    brain_centroids = jnp.load("data/nn_centroids.npy")
    body_centroids = jnp.load("data/body_centroids.npy")
    behavior_centroids = jnp.load("data/behavior_centroids.npy")

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = tri_map_elites.init(
        init_genotypes=population,
        centroids1=brain_centroids,
        centroids2=body_centroids,
        centroids3=behavior_centroids,
        random_key=random_key,
        individual_id=True
    )

    headers = ["iteration", "max_fitness", "qd_score1", "qd_score2", "qd_score3", "coverage1", "coverage2", "coverage3",
               "time", "current_time", "invalid_individuals"]

    name = f"{config.get('run_name', 'trial')}_{config['seed']}"

    csv_logger = CSVLogger(
        f"../results/me_nn/{name}.csv",
        header=headers
    )

    fitness_evaluations = 0
    for i in range(config["n_iterations"]):
        start_time = time.time()
        # main iterations
        repertoire, emitter_state, metrics, random_key = tri_map_elites.update(repertoire, emitter_state, random_key)
        timelapse = time.time() - start_time
        current_time = datetime.now()

        # log metrics
        logged_metrics = {"time": timelapse, "iteration": i + 1, "current_time": current_time,
                          "max_fitness": metrics["max_fitness"],
                          "invalid_individuals": metrics["n_invalid_offspring"],
                          "qd_score1": metrics["qd_score1"], "coverage1": metrics["coverage1"],
                          "qd_score2": metrics["qd_score2"], "coverage2": metrics["coverage2"],
                          "qd_score3": metrics["qd_score3"], "coverage3": metrics["coverage3"]}

        csv_logger.log(logged_metrics)
        # print(logged_metrics["invalid_individuals"])
        fitness_evaluations = fitness_evaluations + config["parents_size"] - logged_metrics["invalid_individuals"]
        print(f"{i}\t{logged_metrics['max_fitness']}")

    os.makedirs(f"../results/me_nn/{name}/", exist_ok=True)
    repertoire.save(f"../results/me_nn/{name}/")
    with open(f"../results/me_nn/{name}/config.yaml", "w") as file:
        yaml.dump(config, file)

    i = config["n_iterations"]
    while fitness_evaluations < (config["n_iterations"] * config["parents_size"]):
        start_time = time.time()
        # main iterations
        repertoire, emitter_state, metrics, random_key = tri_map_elites.update(repertoire, emitter_state, random_key)
        timelapse = time.time() - start_time
        current_time = datetime.now()

        # log metrics
        logged_metrics = {"time": timelapse, "iteration": i + 1, "current_time": current_time,
                          "max_fitness": metrics["max_fitness"],
                          "invalid_individuals": metrics["n_invalid_offspring"],
                          "qd_score1": metrics["qd_score1"], "coverage1": metrics["coverage1"],
                          "qd_score2": metrics["qd_score2"], "coverage2": metrics["coverage2"],
                          "qd_score3": metrics["qd_score3"], "coverage3": metrics["coverage3"]}

        csv_logger.log(logged_metrics)
        # print(logged_metrics["invalid_individuals"])
        fitness_evaluations = fitness_evaluations + config["parents_size"] - logged_metrics["invalid_individuals"]
        print(f"{i}\t{logged_metrics['max_fitness']}")
        i += 1

    repertoire.save(f"../results/me_nn/{name}/extra_")


if __name__ == '__main__':
    # samplers = ["all", "s1", "s2", "s3"]
    seeds = range(1)
    samplers = ["all"]
    envs = ["Walker-v0"
            # "BridgeWalker-v0",
            # "Pusher-v0",
            # "UpStepper-v0",
            # "DownStepper-v0",
            # "ObstacleTraverser-v0",

            # "ObstacleTraverser-v1",
            # "Hurdler-v0",
            # "PlatformJumper-v0",
            # "GapJumper-v0",
            # "CaveCrawler-v0",
            # "Carrier-v0"
            ]
    envs_descriptors = {
        # "Walker-v0": {
        #     "behavior_descriptors": ["velocity_y", "floor_contact"],
        #     "qd_wrappers": ["velocity", "floor_contact"],
        #     "frequency_cut_off": 0.5
        # },
        # "Climber-v0": {
        #    "behavior_descriptors": ["floor_contact", "walls_contact"],
        #    "qd_wrappers": ["floor_contact", "walls_contact"],
        #    "body_trim": True
        # },
        # "CustomCarrier-v0": {
        #     "behavior_descriptors": ["object_angle", "floor_contact"],
        #     "qd_wrappers": ["object_angle", "floor_contact"],
        # }
    }

    base_cfg = {
        "p_mut_body": 0.05,
        "solver": "ne",
        "episode_length": 200,
        "pop_size": 5,
        "parents_size": 4,
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

        "body_descriptors": ["relative_activity", "elongation"],
        "behavior_descriptors": ["velocity_y", "floor_contact"],
        "qd_wrappers": ["velocity", "floor_contact"],
        "frequency_cut_off": 0.5,
        "brain_descriptors": ["nn_connectivity"],
        "weights_threshold": 0.25,

        # nn params
        "policy_hidden_layer_sizes": (20, 20),
        "iso_sigma": 0.005,
        "line_sigma": 0.05,
    }

    counter = 0
    for seed in seeds:
        for sampler in samplers:
            # for env in envs_descriptors.keys():
            for env in envs:
                counter += 1
                cfg = copy.deepcopy(base_cfg)
                cfg["seed"] = seed
                cfg["sampler"] = sampler
                cfg["env_name"] = env
                cfg[
                    "run_name"] = (f"evo-body-{cfg['grid_size']}x{cfg['grid_size']}-"
                                   f"{env.replace('-v0', '').lower()}-{sampler}")
                # cfg.update(envs_descriptors[env])
                print(
                    f"{counter}/{len(seeds) * len(samplers) * len(envs)} -> evo-body-"
                    f"{cfg['grid_size']}x{cfg['grid_size']}, {seed}, {sampler}")
                run_body_evo_me(cfg)

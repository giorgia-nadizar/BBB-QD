import copy
import os
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any

import numpy as np
import yaml

from bbbqd.behavior.behavior_descriptors import get_behavior_descriptors_functions
from bbbqd.body.body_descriptors import get_body_descriptor_extractor
from bbbqd.body.body_utils import compute_body_mask, compute_body_mutation_mask, compute_body_encoding_function, \
    compute_body_float_genome_length
from bbbqd.brain.brain_descriptors import get_graph_descriptor_extractor
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import evaluate_controller_and_body
from bbbqd.core.validation import count_invalid_bodies, validate_body_width
from bbbqd.wrappers import make_env
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.gp.encoding import compute_encoding_function
from qdax.core.gp.individual import compute_genome_mask, compute_mutation_mask, generate_population, \
    compute_mutation_fn, compute_variation_mutation_fn
from qdax.core.gp.utils import update_config
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

    # Update config with env info
    config = update_config(config, env)

    # Init a random key
    random_key = jax.random.PRNGKey(config["seed"])

    # Init population of controllers
    body_mask = compute_body_mask(config)
    controller_mask = compute_genome_mask(config, config["n_in"], config["n_out"])
    genome_mask = jnp.concatenate([body_mask, controller_mask])

    # Compute mutation masks
    body_mutation_mask = compute_body_mutation_mask(config)
    body_float_length = compute_body_float_genome_length(config)
    controller_mutation_mask = compute_mutation_mask(config, config["n_out"])
    mutation_mask = jnp.concatenate([body_mutation_mask, controller_mutation_mask])

    # Define encoding function
    program_encoding_fn = compute_encoding_function(config)

    # Define function to return the proper controller
    controller_creation_fn = compute_controller_generation_fn(config)

    # Body encoding function
    body_encoding_fn = compute_body_encoding_function(config)

    # Descriptors
    brain_descr_fn, _ = get_graph_descriptor_extractor(config)
    body_descr_fn, _ = get_body_descriptor_extractor(config)
    behavior_descr_fns = get_behavior_descriptors_functions(config)

    random_key, pop_key = jax.random.split(random_key)
    if config.get("fixed_outputs", False):
        fixed_outputs = jnp.arange(start=config["buffer_size"] - config["n_out"], stop=config["buffer_size"], step=1)
        population_generation_fn = partial(generate_population,
                                           genome_mask=genome_mask,
                                           rnd_key=pop_key,
                                           fixed_genome_trailing=fixed_outputs,
                                           float_header_length=body_float_length)
        config["p_mut_outputs"] = 0
    else:
        population_generation_fn = partial(generate_population,
                                           genome_mask=genome_mask,
                                           rnd_key=pop_key,
                                           float_header_length=body_float_length)

    population = population_generation_fn(config["parents_size"])

    # Create invalid checker function
    if "Climber" in config["env_name"]:
        invalidity_counter_fn = partial(count_invalid_bodies, body_genome_size=len(body_mask) + body_float_length,
                                        body_encoding_fn=body_encoding_fn, max_body_width=5)
        validity_checker = partial(validate_body_width, body_genome_size=len(body_mask) + body_float_length,
                                   body_encoding_fn=body_encoding_fn, max_body_width=5)
        # ensure initial population is all valid
        additional_population_samples = population_generation_fn(10 * config["parents_size"])
        valid_population = []
        for g in population:
            if validity_checker(g):
                valid_population.append(g)
        for g in additional_population_samples:
            if len(valid_population) == config["parents_size"]:
                break
            if validity_checker(g):
                valid_population.append(g)
        population = jnp.asarray(valid_population)
    else:
        invalidity_counter_fn = lambda _: 0

    # Create evaluation function
    evaluation_fn = partial(evaluate_controller_and_body, config=config, descriptors_functions=behavior_descr_fns)

    # Define genome evaluation fn -> returns fitness and brain, body, behavior descriptors
    def _evaluate_genome(genome: jnp.ndarray) -> Tuple[float, np.ndarray]:
        body_genome, controller_genome = jnp.split(genome, [len(body_mask) + body_float_length])
        controller = controller_creation_fn(program_encoding_fn(controller_genome))
        body = body_encoding_fn(body_genome)
        brain_descriptors = brain_descr_fn(controller_genome)
        body_descriptors = body_descr_fn(body)
        fitness, behavior_descriptors = evaluation_fn(controller, body)
        return fitness, np.concatenate([brain_descriptors, body_descriptors, behavior_descriptors])

    # Add all functions to _LocalFunctions class, separating each with a comma
    _LocalFunctions.add_functions(_evaluate_genome)

    # Define scoring fn
    def _qd_scoring_fn(genomes: jnp.ndarray, rnd_key: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        genomes_list = [g for g in genomes]
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
    mutation_fn = compute_mutation_fn(genome_mask, mutation_mask, config.get("float_mutation_sigma", 0.1))
    variation_fn = None
    variation_perc = 0.0
    if config["solver"] == "lgp":
        variation_fn = compute_variation_mutation_fn(genome_mask, mutation_mask,
                                                     config.get("float_mutation_sigma", 0.1))
        variation_perc = 1.0

    mixing_emitter = MixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=variation_fn,
        variation_percentage=variation_perc,
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
        invalidity_counter=invalidity_counter_fn
    )

    brain_centroids = jnp.load("data/brain_centroids.npy")
    body_centroids = jnp.load("data/body_centroids.npy")
    behavior_centroids = jnp.load("data/behavior_centroids.npy")

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = tri_map_elites.init(
        init_genotypes=population,
        centroids1=brain_centroids,
        centroids2=body_centroids,
        centroids3=behavior_centroids,
        random_key=random_key
    )

    headers = ["iteration", "max_fitness", "qd_score1", "qd_score2", "qd_score3", "coverage1", "coverage2", "coverage3",
               "time", "current_time", "invalid_individuals"]

    name = f"{config.get('run_name', 'trial')}_{config['seed']}"

    csv_logger = CSVLogger(
        f"../paper_results/me/{name}.csv",
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

    os.makedirs(f"../paper_results/me/{name}/", exist_ok=True)
    repertoire.save(f"../paper_results/me/{name}/")
    with open(f"../paper_results/me/{name}/config.yaml", "w") as file:
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

    repertoire.save(f"../paper_results/me/{name}/extra_")


if __name__ == '__main__':
    samplers = {
        # "all": "3b", "s1": "brain", "s2": "body", "s3": "behavior",
        "s1+s2": "brain+body", "s2+s3": "body+behavior",
        "s1+s3": "brain+behavior"
    }
    seeds = range(20)
    envs = ["Walker-v0"
            # "BridgeWalker-v0",
            #     "Pusher-v0",
            #     "UpStepper-v0",
            #     "DownStepper-v0",
            #     "ObstacleTraverser-v0",
            #
            #     "ObstacleTraverser-v1",
            #     "Hurdler-v0",
            #     "PlatformJumper-v0",
            #     "GapJumper-v0",
            #     "CaveCrawler-v0",
            # "Carrier-v0"
            ]

    base_cfg = {
        "n_nodes": 50,
        "p_mut_inputs": 0.1,
        "p_mut_functions": 0.1,
        "p_mut_outputs": 0.3,
        "p_mut_body": 0.05,
        "solver": "cgp",
        "episode_length": 200,
        "pop_size": 50,
        "parents_size": 45,
        "n_iterations": 4000,
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
        "graph_descriptors": "function_arities",
        "body_descriptors": ["relative_activity", "elongation"],
        "behavior_descriptors": ["velocity_y", "floor_contact"],
        "qd_wrappers": ["velocity", "floor_contact"],
        "frequency_cut_off": 0.5
    }

    counter = 0
    for seed in seeds:
        for sampler in samplers.keys():
            for env in envs:
                counter += 1
                cfg = copy.deepcopy(base_cfg)
                cfg["seed"] = seed
                cfg["sampler"] = sampler
                cfg["env_name"] = env
                cfg["run_name"] = f"evobb_graph_{samplers[sampler]}"
                # cfg.update(envs_descriptors[env])
                print(
                    f"{counter}/{len(seeds) * len(samplers) * len(envs)} -> evo-body-"
                    f"{cfg['grid_size']}x{cfg['grid_size']}, {seed}, {sampler}")
                run_body_evo_me(cfg)

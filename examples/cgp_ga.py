import copy
import os
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any

from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import evaluate_controller
from bbbqd.wrappers import make_env
from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.gp.encoding import compute_encoding_function
from qdax.core.gp.individual import compute_genome_mask, compute_mutation_mask, generate_population, \
    compute_mutation_fn, compute_variation_mutation_fn
from qdax.core.gp.utils import update_config
from qdax.types import RNGKey, Fitness, ExtraScores
from qdax.utils.metrics import default_ga_metrics, CSVLogger


# This is a placeholder class, all local functions will be added as its attributes
class _LocalFunctions:
    @classmethod
    def add_functions(cls, *args):
        for function in args:
            setattr(cls, function.__name__, function)
            function.__qualname__ = cls.__qualname__ + '.' + function.__name__


def run_ga(config: Dict[str, Any]):
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

    # Define genome evaluation fn
    def _evaluate_genome(genome: jnp.ndarray) -> float:
        return controller_evaluation_fn(controller_creation_fn(encoding_fn(genome)))

    # Add all functions to _LocalFunctions class, separating each with a comma
    _LocalFunctions.add_functions(_evaluate_genome)

    # Define scoring fn
    def _scoring_fn(genomes: jnp.ndarray, rnd_key: RNGKey) -> Tuple[Fitness, ExtraScores, RNGKey]:
        genomes_list = [g for g in genomes]
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
    mutation_fn = compute_mutation_fn(genome_mask, mutation_mask)
    variation_fn = None
    variation_perc = 0.0
    if config["solver"] == "lgp":
        variation_fn = compute_variation_mutation_fn(genome_mask, mutation_mask)
        variation_perc = 1.0

    mixing_emitter = MixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=variation_fn,
        variation_percentage=variation_perc,
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
        f"../results/{name}.csv",
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

    os.makedirs(f"../results/{name}/", exist_ok=True)
    repertoire.save(f"../results/{name}/")


if __name__ == '__main__':
    bodies = {
        "biped-3x2": [[3, 3, 3], [3, 0, 3]],
        "biped-4x3": [[3, 3, 3, 3], [3, 3, 3, 3], [3, 0, 0, 3]],
        "worm-5x2": [[3, 3, 3, 3, 3], [3, 3, 3, 3, 3]]
    }
    seeds = range(10)
    base_cfg = {
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
        "skip": 5
    }

    for seed in seeds:
        for body_name in bodies:
            cfg = copy.deepcopy(base_cfg)
            cfg["body"] = bodies[body_name]
            cfg["run_name"] = body_name
            cfg["seed"] = seed
            run_ga(cfg)

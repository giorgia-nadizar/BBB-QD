import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
import yaml

from bbbqd.behavior.behavior_descriptors import get_behavior_descriptors_functions
from bbbqd.body.body_descriptors import get_body_descriptor_extractor
from bbbqd.body.body_utils import compute_body_encoding_function, compute_body_float_genome_length, compute_body_mask
from bbbqd.brain.brain_descriptors import get_graph_descriptor_extractor
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import evaluate_controller_and_body
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.containers.mapelites_tri_repertoire import MapElitesTriRepertoire
from qdax.core.gp.encoding import compute_encoding_function
from qdax.types import RNGKey, Fitness, Descriptor, ExtraScores
from qdax.utils.metrics import default_triqd_metrics, CSVLogger, default_ga_metrics


# This is a placeholder class, all local functions will be added as its attributes
class _LocalFunctions:
    @classmethod
    def add_functions(cls, *args):
        for function in args:
            setattr(cls, function.__name__, function)
            function.__qualname__ = cls.__qualname__ + '.' + function.__name__


def run_task_transfer_ga(
        repertoire_path: str,
        environments: List[Tuple[str, int]],
) -> None:
    # load config
    config = yaml.safe_load(Path(f"{repertoire_path}/config.yaml").read_text())

    # other body config parameters
    body_mask_length = len(compute_body_mask(config))
    body_float_length = compute_body_float_genome_length(config)

    # encoding functions
    program_encoding_fn = compute_encoding_function(config)
    controller_creation_fn = compute_controller_generation_fn(config)
    body_encoding_fn = compute_body_encoding_function(config)

    # load repertoire
    initial_repertoire = GARepertoire.load(reconstruction_fn=lambda x: x, path=repertoire_path)

    # extract genotypes
    genotypes = initial_repertoire.genotypes
    fitnesses = initial_repertoire.fitnesses

    for env_name, episode_length in environments:
        print(f"\t{env_name}")
        config["env_name"] = env_name
        config["episode_length"] = episode_length

        # Create evaluation function
        evaluation_fn = partial(evaluate_controller_and_body, config=config)

        # Define genome evaluation fn
        def _evaluate_genome(genome: jnp.ndarray) -> float:
            body_genome, controller_genome = jnp.split(genome, [body_mask_length + body_float_length])
            controller = controller_creation_fn(program_encoding_fn(controller_genome))
            body = body_encoding_fn(body_genome)
            return evaluation_fn(controller, body)

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

        def init_and_store(genotypes: jnp.ndarray, target_path: str):
            fitnesses, extra_scores, _ = _scoring_fn(
                genotypes, None
            )
            # init the tri-repertoire
            ga_repertoire = GARepertoire.init(
                genotypes=genotypes,
                fitnesses=fitnesses,
                population_size=len(fitnesses)
            )
            ga_repertoire.save(target_path)
            headers = ["max_fitness"]
            csv_logger = CSVLogger(
                f"../paper_results/ga_transfer/{name}.csv",
                header=headers
            )
            metrics = default_ga_metrics(ga_repertoire)
            logged_metrics = {k: metrics[k] for k in headers}
            csv_logger.log(logged_metrics)

        name = f"{config['run_name']}_{config['seed']}_{env_name}"
        os.makedirs(f"../paper_results/ga_transfer/{name}/", exist_ok=True)
        init_and_store(genotypes, f"../paper_results/ga_transfer/{name}/")
        with open(f"../paper_results/ga_transfer/{name}/config.yaml", "w") as file:
            yaml.dump(config, file)


def run_task_transfer_me(
        repertoire_path: str,
        environments: List[Tuple[str, int]],
        best_n: int = None
) -> None:
    # load config
    config = yaml.safe_load(Path(f"{repertoire_path}/config.yaml").read_text())

    # other body config parameters
    body_mask_length = len(compute_body_mask(config))
    body_float_length = compute_body_float_genome_length(config)

    # encoding functions
    program_encoding_fn = compute_encoding_function(config)
    controller_creation_fn = compute_controller_generation_fn(config)
    body_encoding_fn = compute_body_encoding_function(config)

    # descriptors
    brain_descr_fn, _ = get_graph_descriptor_extractor(config)
    body_descr_fn, _ = get_body_descriptor_extractor(config)
    behavior_descr_fns = get_behavior_descriptors_functions(config)

    # metrics
    tri_qd_metrics = partial(default_triqd_metrics, qd_offset=0)

    # load repertoire
    initial_repertoire = MapElitesTriRepertoire.load(reconstruction_fn=lambda x: x, path=repertoire_path)

    # extract info which will be re-used for the new repertoires
    descriptors_indexes1 = initial_repertoire.descriptors_indexes1
    descriptors_indexes2 = initial_repertoire.descriptors_indexes2
    descriptors_indexes3 = initial_repertoire.descriptors_indexes3

    centroids1 = initial_repertoire.repertoire1.centroids
    centroids2 = initial_repertoire.repertoire2.centroids
    centroids3 = initial_repertoire.repertoire3.centroids

    # extract genotypes from each archive -> will encode them into robots and assess them for the task
    genotypes1 = initial_repertoire.repertoire1.genotypes
    genotypes2 = initial_repertoire.repertoire2.genotypes
    genotypes3 = initial_repertoire.repertoire3.genotypes

    # extract fitnesses to filter out empty cells
    fitnesses1 = initial_repertoire.repertoire1.fitnesses
    fitnesses2 = initial_repertoire.repertoire2.fitnesses
    fitnesses3 = initial_repertoire.repertoire3.fitnesses

    if best_n is not None:
        # sample only best n, if the value is available
        top_indices_1 = jnp.argsort(fitnesses1)[-best_n:]
        top_indices_2 = jnp.argsort(fitnesses2)[-best_n:]
        top_indices_3 = jnp.argsort(fitnesses3)[-best_n:]
        genotypes1 = genotypes1[top_indices_1]
        genotypes2 = genotypes2[top_indices_2]
        genotypes3 = genotypes3[top_indices_3]
    else:
        # take all genotypes
        genotypes1 = genotypes1[fitnesses1 > - jnp.inf]
        genotypes2 = genotypes2[fitnesses2 > - jnp.inf]
        genotypes3 = genotypes3[fitnesses3 > - jnp.inf]

    for env_name, episode_length in environments:
        print(f"\t{env_name}")
        config["env_name"] = env_name
        config["episode_length"] = episode_length

        # Create evaluation function
        evaluation_fn = partial(evaluate_controller_and_body, config=config, descriptors_functions=behavior_descr_fns)

        # Define genome evaluation fn -> returns fitness and brain, body, behavior descriptors
        def _evaluate_genome(genome: jnp.ndarray) -> Tuple[float, np.ndarray]:
            body_genome, controller_genome = jnp.split(genome, [body_mask_length + body_float_length])
            controller = controller_creation_fn(program_encoding_fn(controller_genome))
            body = body_encoding_fn(body_genome)
            brain_descriptors = brain_descr_fn(controller_genome)
            body_descriptors = body_descr_fn(body)
            fitness, behavior_descriptors = evaluation_fn(controller, body)
            return fitness, np.concatenate([brain_descriptors, body_descriptors, behavior_descriptors])

        _LocalFunctions.add_functions(_evaluate_genome)

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

        def init_and_store(genotypes: jnp.ndarray, target_path: str):
            fitnesses, descriptors, extra_scores, _ = _qd_scoring_fn(
                genotypes, None
            )
            # init the tri-repertoire
            tri_repertoire = MapElitesTriRepertoire.init(
                genotypes=genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                descriptors_indexes1=descriptors_indexes1,
                descriptors_indexes2=descriptors_indexes2,
                descriptors_indexes3=descriptors_indexes3,
                centroids1=centroids1,
                centroids2=centroids2,
                centroids3=centroids3,
                extra_scores=extra_scores
            )
            tri_repertoire.save(target_path)
            headers = ["max_fitness", "coverage1", "coverage2", "coverage3"]
            csv_logger = CSVLogger(
                f"../paper_results/me_transfer/{name}.csv",
                header=headers
            )
            metrics = tri_qd_metrics(tri_repertoire)
            logged_metrics = {k: metrics[k] for k in headers}
            csv_logger.log(logged_metrics)

        for rep_idx, genotypes in enumerate([genotypes1, genotypes2, genotypes3]):
            name = f"{config['run_name']}_{config['seed']}_g{rep_idx + 1}_{env_name}"
            os.makedirs(f"../paper_results/me_transfer/{name}/", exist_ok=True)
            init_and_store(genotypes, f"../paper_results/me_transfer/{name}/")
            with open(f"../paper_results/me_transfer/{name}/config.yaml", "w") as file:
                yaml.dump(config, file)


if __name__ == '__main__':

    algorithms = ["ga", "me"]
    best_n = 50

    environments = [
        # done already
        # ("BridgeWalker-v0", 200),
        # ("PlatformJumper-v0", 200),
        # ("CaveCrawler-v0", 200),
        # ("CustomCarrier-v0", 200),

        # walking
        ("BidirectionalWalker-v0", 200),

        # object manipulation
        ("CustomPusher-v0", 200),
        # ("Thrower-v0", 200), # gives error
        ("BeamToppler-v0", 200),
        ("Pusher-v1", 200),
        # ("Carrier-v1", 200), # gives error
        ("Catcher-v0", 200),
        # ("Slider-v0", 200), # gives error
        # ("Lifter-v0", 200), # gives error

        # all climb give error because too narrow

        # locomotion
        ("UpStepper-v0", 200),
        ("DownStepper-v0", 200),
        ("ObstacleTraverser-v0", 200),
        ("ObstacleTraverser-v1", 200),
        ("Hurdler-v0", 200),
        ("GapJumper-v0", 200),
        ("Traverser-v0", 200),

        # misc
        ("Flipper-v0", 200),
        ("Jumper-v0", 200),
        ("Balancer-v0", 200),
        # ("Balancer-v1", 200), # gives error

        # shape change
        ("AreaMaximizer-v0", 200),
        ("AreaMinimizer-v0", 200),
        ("WingspanMazimizer-v0", 200),
        ("HeightMaximizer-v0", 200),

    ]

    seeds = range(10, 20)
    base_name = "evobb_graph"
    if "ga" in algorithms:
        for seed in seeds:
            print(f"ga, {seed}")
            repertoire_path = f"../paper_results/ga/{base_name}_{seed}/"
            run_task_transfer_ga(repertoire_path, environments)

    if "me" in algorithms:
        samplers = ["body", "brain", "behavior", "3b"]
        for sampler in samplers:
            for seed in seeds:
                print(f"me-{sampler}, {seed}")
                repertoire_path = f"../paper_results/me/{base_name}_{sampler}_{seed}/"
                run_task_transfer_me(repertoire_path, environments, best_n)

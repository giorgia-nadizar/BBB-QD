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
from qdax.core.containers.mapelites_tri_repertoire import MapElitesTriRepertoire
from qdax.core.gp.encoding import compute_encoding_function
from qdax.types import RNGKey, Fitness, Descriptor, ExtraScores


# This is a placeholder class, all local functions will be added as its attributes
class _LocalFunctions:
    @classmethod
    def add_functions(cls, *args):
        for function in args:
            setattr(cls, function.__name__, function)
            function.__qualname__ = cls.__qualname__ + '.' + function.__name__


def run_task_transfer(
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

    # descriptors
    brain_descr_fn, _ = get_graph_descriptor_extractor(config)
    body_descr_fn, _ = get_body_descriptor_extractor(config)
    behavior_descr_fns = get_behavior_descriptors_functions(config)

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

    for env_name, episode_length in environments:
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

        for rep_idx, genotypes in [genotypes1, genotypes2, genotypes3]:
            # TODO build storage path properly
            # TODO add some other metrics to store
            init_and_store(genotypes, rep_idx)

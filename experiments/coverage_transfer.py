import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, Callable

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.core import FrozenDict
from jax import jit

from bbbqd.behavior.behavior_descriptors import get_behavior_descriptors_functions
from bbbqd.body.body_descriptors import get_body_descriptor_extractor
from bbbqd.body.body_utils import compute_body_encoding_function, compute_body_float_genome_length, compute_body_mask
from bbbqd.brain.brain_descriptors import get_graph_descriptor_extractor, get_nn_descriptor_extractor
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import evaluate_controller_and_body
from bbbqd.wrappers import make_env
from experiments.nn_task_transfer import compute_reconstruction_fn, filter_genomes, _rearrange_genomes
from qdax.core.containers.mapelites_tri_repertoire import MapElitesTriRepertoire
from qdax.core.gp.encoding import compute_encoding_function
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import RNGKey, Fitness, Descriptor, ExtraScores
from qdax.utils.metrics import default_triqd_metrics, CSVLogger


# This is a placeholder class, all local functions will be added as its attributes
class _LocalFunctions:
    @classmethod
    def add_functions(cls, *args):
        for function in args:
            setattr(cls, function.__name__, function)
            function.__qualname__ = cls.__qualname__ + '.' + function.__name__


def run_coverage_transfer_nn(
        repertoire_path: str
) -> None:
    # load config
    config = yaml.load(Path(f"{repertoire_path}/config.yaml").read_text(), Loader=yaml.FullLoader)
    env = make_env(config)

    # Init policy network
    policy_layer_sizes = config["policy_hidden_layer_sizes"] + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Define function to return the proper nn policy and controller
    def _nn_policy_creation_fn(policy_params: FrozenDict) -> Callable[[jnp.ndarray], jnp.ndarray]:
        def _nn_policy_fn(observations: jnp.ndarray) -> jnp.ndarray:
            actions = policy_network.apply(policy_params, observations)
            return actions

        return jit(_nn_policy_fn)

    # controller and body encoding fns
    controller_creation_fn = compute_controller_generation_fn(config)
    body_encoding_fn = compute_body_encoding_function(config)

    # Descriptors
    brain_descr_fn, _ = get_nn_descriptor_extractor(config)
    body_descr_fn, _ = get_body_descriptor_extractor(config)
    behavior_descr_fns = get_behavior_descriptors_functions(config)

    # metrics
    tri_qd_metrics = partial(default_triqd_metrics, qd_offset=0)

    # load repertoire
    reconstruction_fn = compute_reconstruction_fn(config, individual_id=True)
    initial_repertoire = MapElitesTriRepertoire.load(reconstruction_fn=reconstruction_fn, path=repertoire_path)

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

    # filter genotypes
    genotypes1 = filter_genomes(genotypes1, fitnesses1)
    genotypes2 = filter_genomes(genotypes2, fitnesses2)
    genotypes3 = filter_genomes(genotypes3, fitnesses3)

    # Create evaluation function
    evaluation_fn = partial(evaluate_controller_and_body, config=config, descriptors_functions=behavior_descr_fns)

    # Define genome evaluation fn -> returns fitness and brain, body, behavior descriptors
    def _evaluate_genome(genome: FrozenDict) -> Tuple[float, np.ndarray]:
        # ignore the individual id
        genome, _ = genome.pop("individual_id")
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
            f"../results/coverage/{name}.csv",
            header=headers
        )
        metrics = tri_qd_metrics(tri_repertoire)
        logged_metrics = {k: metrics[k] for k in headers}
        csv_logger.log(logged_metrics)

    for rep_idx, genotypes in enumerate([genotypes1, genotypes2, genotypes3]):
        name = f"nn_{config['run_name']}_{config['seed']}_g{rep_idx + 1}"
        os.makedirs(f"../results/coverage/{name}/", exist_ok=True)
        init_and_store(genotypes, f"../results/coverage/{name}/")
        with open(f"../results/coverage/{name}/config.yaml", "w") as file:
            yaml.dump(config, file)


def run_coverage_transfer_cgp(
        repertoire_path: str,
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

    # filter genotypes
    genotypes1 = genotypes1[fitnesses1 > - jnp.inf]
    genotypes2 = genotypes2[fitnesses2 > - jnp.inf]
    genotypes3 = genotypes3[fitnesses3 > - jnp.inf]

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
            f"../results/coverage/{name}.csv",
            header=headers
        )
        metrics = tri_qd_metrics(tri_repertoire)
        logged_metrics = {k: metrics[k] for k in headers}
        csv_logger.log(logged_metrics)

    for rep_idx, genotypes in enumerate([genotypes1, genotypes2, genotypes3]):
        name = f"cgp_{config['run_name']}_{config['seed']}_g{rep_idx + 1}"
        os.makedirs(f"../results/coverage/{name}/", exist_ok=True)
        init_and_store(genotypes, f"../results/coverage/{name}/")
        with open(f"../results/coverage/{name}/config.yaml", "w") as file:
            yaml.dump(config, file)


if __name__ == '__main__':

    seeds = range(10)
    samplers = ["all", "s1", "s2", "s3"]
    for seed in seeds:
        for sampler in samplers:
            print(f"CGP-{sampler}, {seed}")
            rep_path = f"../results/me/evo-body-10x10-floor-{sampler}_{seed}/"
            run_coverage_transfer_cgp(rep_path)

            print(f"NN-{sampler}, {seed}")
            rep_path = f"../results/me_nn/evo-body-10x10-walker-{sampler}_{seed}/"
            run_coverage_transfer_nn(rep_path)

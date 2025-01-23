import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from flax.core import FrozenDict
from jax import jit

from bbbqd.behavior.behavior_descriptors import get_behavior_descriptors_functions
from bbbqd.body.body_descriptors import get_body_descriptor_extractor
from bbbqd.body.body_utils import compute_body_encoding_function, compute_body_float_genome_length, compute_body_mask
from bbbqd.brain.brain_descriptors import get_graph_descriptor_extractor, get_nn_descriptor_extractor
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.brain.nn_descriptors import mean_and_std_activity, sparsity_and_activation_distribution
from bbbqd.core.evaluation import evaluate_controller_and_body
from bbbqd.wrappers import make_env
from experiments.nn_task_transfer import compute_reconstruction_fn, filter_null_genomes, _rearrange_genomes
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.containers.mapelites_tri_repertoire import MapElitesTriRepertoire
from qdax.core.gp.encoding import compute_encoding_function
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import RNGKey, Fitness, Descriptor, ExtraScores
from qdax.utils.metrics import default_triqd_metrics, CSVLogger, default_qd_metrics


# This is a placeholder class, all local functions will be added as its attributes
class _LocalFunctions:
    @classmethod
    def add_functions(cls, *args):
        for function in args:
            setattr(cls, function.__name__, function)
            function.__qualname__ = cls.__qualname__ + '.' + function.__name__


def run_nn_data_descr(
        repertoire_path: str,
        data_points: jnp.ndarray
) -> pd.DataFrame:
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

    # load repertoire
    reconstruction_fn = compute_reconstruction_fn(config, individual_id=True)
    initial_repertoire = MapElitesTriRepertoire.load(reconstruction_fn=reconstruction_fn, path=repertoire_path)

    nn_centroids = jnp.load("data/nn_centroids.npy")

    # extract genotypes from each archive -> will encode them into robots and assess them for the task
    genotypes1 = initial_repertoire.repertoire1.genotypes
    genotypes2 = initial_repertoire.repertoire2.genotypes
    genotypes3 = initial_repertoire.repertoire3.genotypes

    # extract fitnesses to filter out empty cells
    fitnesses1 = initial_repertoire.repertoire1.fitnesses
    fitnesses2 = initial_repertoire.repertoire2.fitnesses
    fitnesses3 = initial_repertoire.repertoire3.fitnesses

    # filter genotypes
    genotypes1 = filter_null_genomes(genotypes1, fitnesses1)
    genotypes2 = filter_null_genomes(genotypes2, fitnesses2)
    genotypes3 = filter_null_genomes(genotypes3, fitnesses3)

    # Create evaluation function
    def description_fn(genotypes) -> jnp.ndarray:
        geno_list = _rearrange_genomes(genotypes)
        desc_list = []
        for geno in geno_list:
            genome, _ = geno.pop("individual_id")
            nn_genome, body_genome = genome.pop("body")
            desc_list.append(sparsity_and_activation_distribution(policy_network, nn_genome, data_points))
        return jnp.asarray(desc_list)

    rep_names = ["brain", "body", "behavior"]
    coverages = []
    for idx, genotypes in enumerate([genotypes1, genotypes2, genotypes3]):
        descriptors = description_fn(genotypes)
        repertoire = MapElitesRepertoire.init(
            genotypes=genotypes,
            fitnesses=jnp.ones(len(descriptors)),
            descriptors=descriptors,
            centroids=nn_centroids,
            extra_scores=None
        )
        coverage = default_qd_metrics(repertoire, 0.)["coverage"]
        print(f"{rep_names[idx]} -> {coverage}")
        coverages.append(coverage)

    coverage_df = pd.DataFrame(
        {"repertoire": rep_names,
         "coverage": coverages,
         })

    return coverage_df


if __name__ == '__main__':

    data = jnp.asarray(np.load("data/nn_data_walker.npy"))

    dfs = []
    seeds = range(10)
    samplers = ["all", "s1", "s2", "s3"]
    for seed in seeds:
        for sampler in samplers:
            print(f"NN-{sampler}, {seed}")
            rep_path = f"../results/me_nn/evo-body-10x10-walker-{sampler}_{seed}/"
            coverage_df = run_nn_data_descr(rep_path, data)
            coverage_df["sampler"] = sampler
            coverage_df["seed"] = seed
            dfs.append(coverage_df)

    pd.concat(dfs).to_csv("../results/nn_activity_coverage.csv", index=False)

from functools import partial
from typing import Dict, Tuple, Callable, Union

import jax.numpy as jnp
import joblib
import numpy as np

from bbbqd.brain.nn_descriptors import mean_and_std_output_connectivity, activations_dimensionality_reduction
from qdax.core.gp.graph_utils import compute_cgp_descriptors, compute_lgp_descriptors
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import Genotype, Descriptor


def get_nn_descriptor_extractor(config: Dict, policy_network: Union[None, MLP] = None) -> Tuple[
    Callable[[Genotype], Descriptor], int]:
    assert config["solver"] == "ne"
    if "nn_connectivity" in config["brain_descriptors"]:
        weights_threshold = config.get("weights_threshold", 0.25)
        return partial(mean_and_std_output_connectivity, threshold=weights_threshold), 2
    elif "activations_dimensionality_reduction" in config["brain_descriptors"]:
        data_points = jnp.asarray(np.load(config.get("data_points", "data/nn_data_walker.npy")))
        scaler = joblib.load(config.get("scaler", "data/nn_obs_scaler.pkl"))
        pca = joblib.load(config.get("pca", "data/nn_obs_pca.pkl"))
        return partial(activations_dimensionality_reduction, policy_network=policy_network, scaler=scaler, pca=pca,
                       data_points=data_points), 2
    else:
        raise NotImplementedError


def get_graph_descriptor_extractor(config: Dict) -> Tuple[Callable[[Genotype], Descriptor], int]:
    indexes = {
        "complexity": [0],
        "inputs_usage": [1],
        "function_arities": [2, 3],
        "function_types": [4, 5, 6],
        "function_arities_fraction": [7]
    }
    list_of_descriptors = config["graph_descriptors"] if isinstance(config["graph_descriptors"], list) \
        else [config["graph_descriptors"]]
    descr_indexes = jnp.asarray([idx for desc_name in list_of_descriptors for idx in indexes[desc_name]])
    if config["solver"] == "cgp":
        single_genome_descriptor_function = partial(compute_cgp_descriptors, config=config,
                                                    descriptors_indexes=descr_indexes)
    elif config["solver"] == "lgp":
        single_genome_descriptor_function = partial(compute_lgp_descriptors, config=config,
                                                    descriptors_indexes=descr_indexes)
    else:
        raise ValueError("Solver must be either cgp or lgp.")
    # return vmap(single_genome_descriptor_function), len(descr_indexes)
    return single_genome_descriptor_function, len(descr_indexes)

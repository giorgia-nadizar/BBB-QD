from functools import partial
from typing import Dict, Tuple, Callable

import jax.numpy as jnp
from jax import vmap

from bbbqd.brain.nn_descriptors import mean_and_std_output_connectivity
from qdax.core.gp.graph_utils import compute_cgp_descriptors, compute_lgp_descriptors
from qdax.types import Genotype, Descriptor


def get_nn_descriptor_extractor(config: Dict) -> Tuple[Callable[[Genotype], Descriptor], int]:
    assert config["solver"] == "ne"
    assert "nn_connectivity" in config["brain_descriptors"]

    weights_threshold = config.get("weights_threshold", 0.25)
    return partial(mean_and_std_output_connectivity, threshold=weights_threshold), 2


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

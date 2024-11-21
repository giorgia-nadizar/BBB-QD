import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, Callable, Union, Dict

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.core import FrozenDict, frozen_dict
from jax import jit

from bbbqd.behavior.behavior_descriptors import get_behavior_descriptors_functions
from bbbqd.body.body_descriptors import get_body_descriptor_extractor
from bbbqd.body.body_utils import compute_body_encoding_function, compute_body_float_genome_length, compute_body_mask
from bbbqd.brain.brain_descriptors import get_graph_descriptor_extractor, get_nn_descriptor_extractor
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import evaluate_controller_and_body, track_experience_controller_and_body
from bbbqd.wrappers import make_env
from experiments.nn_task_transfer import compute_reconstruction_fn, filter_genomes, _rearrange_genomes
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.containers.mapelites_tri_repertoire import MapElitesTriRepertoire
from qdax.core.gp.encoding import compute_encoding_function
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import RNGKey, Fitness, Descriptor, ExtraScores
from qdax.utils.metrics import default_triqd_metrics, CSVLogger


def _get_items_at_index(dictionary: Union[FrozenDict, Dict], index: int,
                        new_dict: Dict = {}) -> Union[FrozenDict, Dict]:
    current_keys = dictionary.keys()
    for k in current_keys:
        if isinstance(dictionary[k], FrozenDict):
            new_dict[k] = {}
            _get_items_at_index(dictionary[k], index, new_dict[k])
        else:
            new_dict[k] = dictionary[k][index]
    return frozen_dict.freeze(new_dict)


def collect_nn_experience(
        repertoire_path: str
) -> np.ndarray:
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

    # load repertoire
    reconstruction_fn = compute_reconstruction_fn(config, individual_id=False)
    repertoire = GARepertoire.load(reconstruction_fn=reconstruction_fn, path=repertoire_path)

    # extract genotypes
    genotypes = repertoire.genotypes
    fitnesses = repertoire.fitnesses

    # take only best genotype
    best_genotype = _get_items_at_index(genotypes, jnp.argmax(fitnesses))
    nn_genome, body_genome = best_genotype.pop("body")
    controller = controller_creation_fn(_nn_policy_creation_fn(nn_genome))
    body = body_encoding_fn(body_genome)

    # Create tracking function
    tracking_fn = partial(track_experience_controller_and_body, config=config)

    reward, observations = tracking_fn(controller, body)

    return observations


if __name__ == '__main__':

    tasks = ["Walker-v0", "BridgeWalker-v0", "CustomCarrier-v0", "PlatformJumper-v0", "CaveCrawler-v0"]

    #
    seeds = range(10)
    for task in tasks:
        task_name = task.replace("-v0", "").lower()
        task_obs = []
        for seed in seeds:
            print(f"NN-{task}, {seed}")
            rep_path = f"../results/ga/evo-body-nn-10x10-{task_name}_{seed}/"
            task_obs.append(collect_nn_experience(rep_path))
        task_obs_matrix = np.vstack(task_obs)
        print(task_obs_matrix.shape)
        np.save(f"../experiments/data/nn_data_{task_name}.npy", task_obs_matrix)

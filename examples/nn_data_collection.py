from functools import partial
from pathlib import Path
from typing import Callable, Union, Dict, Iterable, List

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from flax.core import FrozenDict, frozen_dict
from jax import jit, vmap
from jax._src.flatten_util import ravel_pytree
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn_extra.cluster import KMedoids

from bbbqd.body.body_utils import compute_body_encoding_function
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import track_experience_controller_and_body
from experiments.nn_task_transfer import compute_reconstruction_fn
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.neuroevolution.networks.networks import MLP


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

    # Init policy network
    policy_layer_sizes = config["policy_hidden_layer_sizes"] + (1,)
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
    current_genotype = _get_items_at_index(genotypes, jnp.argmax(fitnesses))
    nn_genome, body_genome = current_genotype.pop("body")
    controller = controller_creation_fn(_nn_policy_creation_fn(nn_genome))
    body = body_encoding_fn(body_genome)

    # Create tracking function
    tracking_fn = partial(track_experience_controller_and_body, config=config)

    _, observations = tracking_fn(controller, body)

    return observations


def eval_repertoire_on_data(
        repertoire_path: str,
        data_points: np.ndarray,
) -> DataFrame:
    # load config
    config = yaml.load(Path(f"{repertoire_path}/config.yaml").read_text(), Loader=yaml.FullLoader)

    # Init policy network
    policy_layer_sizes = config["policy_hidden_layer_sizes"] + (1,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    n_inner_neurons = sum(config["policy_hidden_layer_sizes"])

    def _nn_track_activation_fn(point: jnp.ndarray, policy_params: FrozenDict) -> jnp.ndarray:
        _, inter_res = policy_network.apply(policy_params, point, capture_intermediates=True)
        flatten_res, _ = ravel_pytree(inter_res)
        activation_vals, _ = jnp.split(flatten_res, [n_inner_neurons])
        return activation_vals

    # load repertoire
    reconstruction_fn = compute_reconstruction_fn(config, individual_id=False)
    repertoire = GARepertoire.load(reconstruction_fn=reconstruction_fn, path=repertoire_path)

    # extract genotypes
    genotypes = repertoire.genotypes
    fitnesses = repertoire.fitnesses

    activations_dfs = []
    # take only best genotype
    for geno_idx in range(len(fitnesses)):
        curr_genotype = _get_items_at_index(genotypes, geno_idx)
        nn_genome, _ = curr_genotype.pop("body")
        nn_function = partial(_nn_track_activation_fn, policy_params=nn_genome)

        # evaluate NN
        jnp_data_points = jnp.asarray(data_points)
        activations = vmap(nn_function)(jnp_data_points)
        activations_df = pd.DataFrame.from_records(activations)
        activations_df["point_id"] = list(range(len(data_points)))
        activations_dfs.append(activations_df)

    return pd.concat(activations_dfs, ignore_index=True)


def experiences_collection(tasks: Iterable[str], seeds: Iterable[int], n_data_points: int) -> List[str]:
    files = []
    for task in tasks:
        task_name = task.replace("-v0", "").lower()
        task_obs = []
        for seed in seeds:
            print(f"NN-{task_name}, {seed}")
            rep_path = f"../results/ga/evo-body-nn-10x10-{task_name}_{seed}/"
            task_obs.append(collect_nn_experience(rep_path))
        task_obs_matrix = np.vstack(task_obs)

        kmedoids = KMedoids(n_clusters=n_data_points, random_state=0).fit(task_obs_matrix)
        representative_points = kmedoids.cluster_centers_
        print(f"Extracted {n_data_points} representative points for task {task_name}\n")
        np.save(f"../experiments/data/nn_data_{task_name}.npy", representative_points)
        files.append(f"../experiments/data/nn_data_{task_name}.npy")
    return files


def activations_collection(tasks: Iterable[str], seeds: Iterable[int]) -> List[str]:
    for task in tasks:
        task_name = task.replace("-v0", "").lower()
        task_dfs = []
        data_points = np.load(f"../experiments/data/nn_data_{task_name}.npy")
        for seed in seeds:
            print(f"NN-{task_name}, {seed}")
            rep_path = f"../results/ga/evo-body-nn-10x10-{task_name}_{seed}/"
            df = eval_repertoire_on_data(rep_path, data_points)
            df["seed"] = seed
            task_dfs.append(df)
        task_df = pd.concat(task_dfs, ignore_index=True)
        task_df["task"] = task_name
        task_df.to_csv(f"../experiments/data/nn_activations_{task_name}.tsv", index=False)


if __name__ == '__main__':
    # file_names = experiences_collection(
    #     n_data_points=100,
    #     tasks=["Walker-v0", "BridgeWalker-v0", "CustomCarrier-v0", "PlatformJumper-v0", "CaveCrawler-v0"],
    #     seeds=range(10),
    # )

    df_names = activations_collection(
        seeds=range(10),
        tasks=["Walker-v0", "BridgeWalker-v0", "CustomCarrier-v0", "PlatformJumper-v0", "CaveCrawler-v0"],
    )

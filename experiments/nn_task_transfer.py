import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple, Callable, Union, Dict

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.core import FrozenDict, frozen_dict
from jax import jit
from jax._src.flatten_util import ravel_pytree

from bbbqd.behavior.behavior_descriptors import get_behavior_descriptors_functions
from bbbqd.body.body_descriptors import get_body_descriptor_extractor
from bbbqd.body.body_utils import compute_body_encoding_function, compute_body_float_genome_length, compute_body_mask
from bbbqd.brain.brain_descriptors import get_graph_descriptor_extractor, get_nn_descriptor_extractor
from bbbqd.brain.controllers import compute_controller_generation_fn
from bbbqd.core.evaluation import evaluate_controller_and_body
from bbbqd.core.pytree_utils import pytree_stack
from bbbqd.wrappers import make_env
from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.containers.mapelites_tri_repertoire import MapElitesTriRepertoire
from qdax.core.gp.encoding import compute_encoding_function
from qdax.core.gp.individual import generate_population
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import RNGKey, Fitness, Descriptor, ExtraScores
from qdax.utils.metrics import default_triqd_metrics, CSVLogger, default_ga_metrics


# This is a placeholder class, all local functions will be added as its attributes
class _LocalFunctions:
    @classmethod
    def add_functions(cls, *args):
        for function in args:
            setattr(cls, function.__name__, function)
            function.__qualname__ = cls.__qualname__ + '.' + function.__name__


# util functions for iterating the frozen dicts and rearranging them into a list
def _get_items_at_index(dictionary: Union[FrozenDict, Dict], index: int,
                        new_dict: Dict = {}) -> Union[FrozenDict, Dict]:
    current_keys = dictionary.keys()
    for k in current_keys:
        if isinstance(dictionary[k], FrozenDict):
            new_dict[k] = {}
            _get_items_at_index(dictionary[k], index, new_dict[k])
        else:
            new_dict[k] = dictionary[k][index]
    return new_dict


def _rearrange_genomes(genomes: FrozenDict) -> List[FrozenDict]:
    return [frozen_dict.freeze(_get_items_at_index(genomes, g)) for g in range(len(genomes["body"]))]


def filter_genomes(genomes: FrozenDict, fitnesses: jnp.ndarray) -> FrozenDict:
    indexes_to_keep = jnp.where(fitnesses > -jnp.inf)
    list_of_genomes_to_keep = [frozen_dict.freeze(_get_items_at_index(genomes, g_idx)) for g_idx in indexes_to_keep[0]]
    genomes_to_keep = pytree_stack(list_of_genomes_to_keep)
    return genomes_to_keep


def compute_reconstruction_fn(
        config: Dict,
        individual_id: bool
):
    # Create dummy genotypes to get unravel fn
    env = make_env(config)
    random_key = jax.random.PRNGKey(config["seed"])
    body_mask = compute_body_mask(config)
    body_float_length = compute_body_float_genome_length(config)
    policy_layer_sizes = config["policy_hidden_layer_sizes"] + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    random_key, pop_key = jax.random.split(random_key)
    body_generation_fn = partial(generate_population,
                                 genome_mask=body_mask,
                                 rnd_key=pop_key,
                                 float_header_length=body_float_length)
    init_bodies = body_generation_fn(config["parents_size"])
    random_key, nns_key = jax.random.split(random_key)
    keys = jax.random.split(nns_key, num=config["parents_size"])
    fake_batch = jnp.zeros(shape=(config["parents_size"], env.observation_size))
    init_nns = jax.vmap(policy_network.init)(keys, fake_batch)
    population = init_nns.copy({"body": init_bodies})
    if individual_id:
        individual_ids = jnp.expand_dims(jnp.arange(len(init_bodies)), 1)
        population = population.copy({
            "individual_id": individual_ids,
        })

    first_genotype = frozen_dict.freeze(_get_items_at_index(population, 0))
    _, unravel_fn = ravel_pytree(first_genotype)
    return unravel_fn


def run_task_transfer_ga(
        repertoire_path: str,
        environments: List[Tuple[str, int]]
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

    # encoding functions
    controller_creation_fn = compute_controller_generation_fn(config)
    body_encoding_fn = compute_body_encoding_function(config)

    # load repertoire
    reconstruction_fn = compute_reconstruction_fn(config, individual_id=False)
    initial_repertoire = GARepertoire.load(reconstruction_fn=reconstruction_fn, path=repertoire_path)

    # extract genotypes
    genotypes = initial_repertoire.genotypes

    for env_name, episode_length in environments:
        print(f"\t{env_name}")
        config["env_name"] = env_name
        config["episode_length"] = episode_length

        # Create evaluation function
        evaluation_fn = partial(evaluate_controller_and_body, config=config)

        # Define genome evaluation fn
        def _evaluate_genome(genome: FrozenDict) -> float:
            nn_genome, body_genome = genome.pop("body")
            controller = controller_creation_fn(_nn_policy_creation_fn(nn_genome))
            body = body_encoding_fn(body_genome)
            fitness = evaluation_fn(controller, body)
            return fitness

        _LocalFunctions.add_functions(_evaluate_genome)

        # Define scoring fn
        def _scoring_fn(genomes: FrozenDict, rnd_key: RNGKey) -> Tuple[Fitness, ExtraScores, RNGKey]:
            genomes_list = _rearrange_genomes(genomes)
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
                f"../results/transfer_nn/{name}.csv",
                header=headers
            )
            metrics = default_ga_metrics(ga_repertoire)
            logged_metrics = {k: metrics[k] for k in headers}
            csv_logger.log(logged_metrics)

        name = f"ga_{config['run_name']}_{config['seed']}_{env_name}"
        os.makedirs(f"../results/transfer_nn/{name}/", exist_ok=True)
        init_and_store(genotypes, f"../results/transfer_nn/{name}/")
        with open(f"../results/transfer_nn/{name}/config.yaml", "w") as file:
            yaml.dump(config, file)


def run_task_transfer_me(
        repertoire_path: str,
        environments: List[Tuple[str, int]]
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

    for env_name, episode_length in environments:
        print(f"\t{env_name}")
        config["env_name"] = env_name
        config["episode_length"] = episode_length

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
                f"../results/transfer_nn/{name}.csv",
                header=headers
            )
            metrics = tri_qd_metrics(tri_repertoire)
            logged_metrics = {k: metrics[k] for k in headers}
            csv_logger.log(logged_metrics)

        for rep_idx, genotypes in enumerate([genotypes1, genotypes2, genotypes3]):
            name = f"me_{config['run_name']}_{config['seed']}_g{rep_idx + 1}_{env_name}"
            os.makedirs(f"../results/transfer_nn/{name}/", exist_ok=True)
            init_and_store(genotypes, f"../results/transfer_nn/{name}/")
            with open(f"../results/transfer_nn/{name}/config.yaml", "w") as file:
                yaml.dump(config, file)


if __name__ == '__main__':

    algorithms = ["ga", "me"]

    environments = [
        ("BridgeWalker-v0", 200),
        ("CustomPusher-v0", 200),
        ("UpStepper-v0", 200),
        ("DownStepper-v0", 200),
        ("ObstacleTraverser-v0", 200),

        ("ObstacleTraverser-v1", 200),
        ("Hurdler-v0", 200),
        ("PlatformJumper-v0", 200),
        ("GapJumper-v0", 200),
        ("CaveCrawler-v0", 200),
        ("CustomCarrier-v0", 200),
    ]

    seeds = range(10)
    base_name = "evo-body-nn-10x10-walker"

    # if "ga" in algorithms:
    #     for seed in seeds:
    #         print(f"ga, {seed}")
    #         repertoire_path = f"../results/ga/{base_name}_{seed}/"
    #         run_task_transfer_ga(repertoire_path, environments)

    if "me" in algorithms:
        samplers = ["all", "s1", "s2", "s3"]
        for sampler in samplers:
            for seed in seeds:
                print(f"me-{sampler}, {seed}")
                repertoire_path = f"../results/me_nn/{base_name.replace('-nn', '')}-{sampler}_{seed}/"
                run_task_transfer_me(repertoire_path, environments)

import jax.numpy as jnp
from typing import List, Tuple

import pandas as pd

from qdax.core.containers.mapelites_tri_repertoire import MapElitesTriRepertoire


def extract_best_genotype(repertoires: List[MapElitesTriRepertoire]):
    best_fitness = - jnp.inf
    best_genotype = None
    for repertoire in repertoires:
        fitnesses = repertoire.repertoire1.fitnesses
        current_best_fitness = jnp.max(fitnesses)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_genotype = repertoire.repertoire1.genotypes[jnp.argmax(fitnesses)]
    return best_genotype


def extract_original_descriptors(transferred_repertoire_path: str,
                                 original_repertoire_path: str) -> Tuple[List, List, List]:
    transferred_repertoires = [MapElitesTriRepertoire.load(reconstruction_fn=lambda x: x,
                                                           path=transferred_repertoire_path.replace("gx",
                                                                                                    f"g{rep_idx}")) for
                               rep_idx in range(1, 4)]
    best_transferred_genotype = extract_best_genotype(transferred_repertoires)

    original_repertoire = MapElitesTriRepertoire.load(reconstruction_fn=lambda x: x, path=original_repertoire_path)
    original_genotypes_1 = original_repertoire.repertoire1.genotypes
    original_genotypes_2 = original_repertoire.repertoire2.genotypes
    original_genotypes_3 = original_repertoire.repertoire3.genotypes

    try:
        index_of_best_1 = jnp.where(jnp.all(original_genotypes_1 == best_transferred_genotype, axis=1))[0].item()
        descriptors_1 = original_repertoire.repertoire1.descriptors[index_of_best_1].tolist()
    except TypeError:
        descriptors_1 = [-1, -1]

    try:
        index_of_best_2 = jnp.where(jnp.all(original_genotypes_2 == best_transferred_genotype, axis=1))[0].item()
        descriptors_2 = original_repertoire.repertoire2.descriptors[index_of_best_2].tolist()
    except TypeError:
        descriptors_2 = [-1, -1]

    try:
        index_of_best_3 = jnp.where(jnp.all(original_genotypes_3 == best_transferred_genotype, axis=1))[0].item()
        descriptors_3 = original_repertoire.repertoire3.descriptors[index_of_best_3].tolist()
    except TypeError:
        descriptors_3 = [-1, -1]

    return descriptors_1, descriptors_2, descriptors_3


if __name__ == '__main__':

    seeds = range(20)
    environments = [
        # done already
        ("BridgeWalker-v0", 200),
        ("PlatformJumper-v0", 200),
        ("CaveCrawler-v0", 200),
        ("CustomCarrier-v0", 200),

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
    samplers = ["3b", "brain", "body", "behavior"]

    for representation in ["nn", "graph"]:
        dicts_for_df = []
        for sampler in samplers:
            for seed in seeds:
                for environment, _ in environments:
                    print(f"{representation} - {sampler}, {seed}, {environment}")
                    run_name = f"evobb_{representation}_{sampler}_{seed}_{environment}"
                    original_rep_path = f"../paper_results/me/evobb_{representation}_{sampler}_{seed}/"
                    transferred_rep_path = f"../paper_results/me_transfer/evobb_{representation}_{sampler}_{seed}_gx_{environment}/"
                    d1, d2, d3 = extract_original_descriptors(transferred_rep_path, original_rep_path)
                    dicts_for_df.append({
                        "sampler": sampler,
                        "seed": seed,
                        "environment": environment,
                        "brain_descriptors_0": d1[0],
                        "brain_descriptors_1": d1[1],
                        "body_descriptors_0": d2[0],
                        "body_descriptors_1": d2[1],
                        "behavior_descriptors_0": d3[0],
                        "behavior_descriptors_1": d3[1],
                    })

        df_of_descriptors = pd.DataFrame(dicts_for_df)
        df_of_descriptors.to_csv(
            f"../paper_results/me_transfer/original_descriptors_{representation}_final.csv",
            index=False)

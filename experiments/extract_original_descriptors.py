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

    seeds = range(10)
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
    base_names = {
        "cgp": "evo-body-10x10-floor",
        "nn": "PCA-evo-body-10x10-walker"
    }
    samplers = ["all", "s1", "s2", "s3"]

    for representation in ["nn", "cgp"]:
        dicts_for_df = []
        for sampler in samplers:
            for seed in seeds:
                for environment, _ in environments:
                    print(f"{representation} - {sampler}, {seed}, {environment}")
                    base_name = base_names[representation]
                    run_name = f"{base_name}-{sampler}_{seed}"
                    original_rep_path = f"../results/me{'_nn' if representation == 'nn' else ''}/{run_name}/"
                    transferred_rep_path = (f"../results/transfer{'_nn_pca' if representation == 'nn' else ''}/"
                                            f"{'me_' if representation == 'nn' else ''}{run_name}_gx_best50_{environment}/")
                    d1, d2, d3 = extract_original_descriptors(transferred_rep_path, original_rep_path)
                    dicts_for_df.append({
                        "sampler": sampler,
                        "seed": seed,
                        "environment": environment,
                        "brain_descriptors_0": d1[0],
                        "brain_descriptors_1": d1[1],
                        "body_descriptors_0": d2[0],
                        "body_descriptors_1": d2[1],
                        "behavior_descriptors_0": d2[0],
                        "behavior_descriptors_1": d2[1],
                    })

        df_of_descriptors = pd.DataFrame(dicts_for_df)
        df_of_descriptors.to_csv(
            f"../results/transfer{'_nn_pca' if representation == 'nn' else ''}/original_descriptors.csv", index=False)

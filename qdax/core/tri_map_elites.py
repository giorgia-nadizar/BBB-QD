from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_tri_repertoire import MapElitesTriRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.map_elites import MAPElites
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


def sampling_function(function_name: str) -> Callable[[MapElitesTriRepertoire], int]:
    if "0" in function_name or "both" in function_name or "all" in function_name:
        return lambda x: 0
    elif "1" in function_name:
        return lambda x: 1
    elif "2" in function_name:
        return lambda x: 2
    elif "3" in function_name:
        return lambda x: 3
    else:
        raise ValueError("Solver must be either cgp or lgp.")


class TriMAPElites(MAPElites):
    def __init__(
            self,
            scoring_function: Callable[
                [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
            ],
            emitter: Emitter,
            metrics_function: Callable[[MapElitesTriRepertoire], Metrics],
            descriptors_indexes1: jnp.ndarray,
            descriptors_indexes2: jnp.ndarray = jnp.asarray([]),
            descriptors_indexes3: jnp.ndarray = jnp.asarray([]),
            sampling_id_function: Callable[[MapElitesTriRepertoire], int] = lambda x: 0

            # Note: sampling id semantics
            # 0 = sample from all
            # 1 = sample from first
            # 2 = sample from second
            # 3 = sample from third
    ) -> None:
        super(TriMAPElites, self).__init__(scoring_function, emitter, metrics_function)
        self._descriptors_indexes1 = descriptors_indexes1
        self._descriptors_indexes2 = descriptors_indexes2
        self._descriptors_indexes3 = descriptors_indexes3
        self._sampling_id_function = sampling_id_function

    # @partial(jax.jit, static_argnames=("self",))
    def init(
            self,
            init_genotypes: Genotype,
            centroids1: Centroid,
            centroids2: Centroid,
            centroids3: Centroid,
            random_key: RNGKey,
    ) -> Tuple[MapElitesTriRepertoire, Optional[EmitterState], RNGKey]:
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the tri-repertoire
        tri_repertoire = MapElitesTriRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            descriptors_indexes1=self._descriptors_indexes1,
            descriptors_indexes2=self._descriptors_indexes2,
            descriptors_indexes3=self._descriptors_indexes3,
            centroids1=centroids1,
            centroids2=centroids2,
            centroids3=centroids3,
            extra_scores=extra_scores
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=tri_repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return tri_repertoire, emitter_state, random_key

    # @partial(jax.jit, static_argnames=("self",))
    def update(
            self,
            tri_repertoire: MapElitesTriRepertoire,
            emitter_state: Optional[EmitterState],
            random_key: RNGKey,
    ) -> Tuple[MapElitesTriRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        sampling_id = self._sampling_id_function(tri_repertoire)
        tri_repertoire = tri_repertoire.update_sampling_mask(sampling_id)

        # generate offsprings with the emitter
        offspring, random_key = self._emitter.emit(
            tri_repertoire, emitter_state, random_key
        )
        # scores the offspring
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            offspring, random_key
        )

        # add genotypes in the repertoire
        tri_repertoire, _ = tri_repertoire.add(offspring, descriptors, fitnesses, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=tri_repertoire,
            genotypes=offspring,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(tri_repertoire)

        return tri_repertoire, emitter_state, metrics, random_key

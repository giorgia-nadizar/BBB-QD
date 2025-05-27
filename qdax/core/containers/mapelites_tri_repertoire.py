from __future__ import annotations

from functools import partial
from typing import Callable, Tuple, Optional

import flax
import jax
import jax.numpy as jnp

from bbbqd.core.pytree_utils import pytree_stack, pytree_flatten
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.types import RNGKey, Genotype, Descriptor, Fitness, ExtraScores, Centroid, Mask


class MapElitesTriRepertoire(flax.struct.PyTreeNode):
    repertoire1: MapElitesRepertoire
    repertoire2: MapElitesRepertoire
    repertoire3: MapElitesRepertoire
    descriptors_indexes1: jnp.ndarray
    descriptors_indexes2: jnp.ndarray
    descriptors_indexes3: jnp.ndarray
    sampling_mask: Mask

    def save(self, path: str = "/.") -> None:
        self.repertoire1.save(f"{path}r1_")
        self.repertoire2.save(f"{path}r2_")
        self.repertoire3.save(f"{path}r3_")
        jnp.save(path + "descriptors_indexes1.npy", self.descriptors_indexes1)
        jnp.save(path + "descriptors_indexes2.npy", self.descriptors_indexes2)
        jnp.save(path + "descriptors_indexes3.npy", self.descriptors_indexes3)

    @classmethod
    def load(cls, reconstruction_fn: Callable, path: str = "./") -> MapElitesTriRepertoire:
        repertoire1 = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path}r1_")
        repertoire2 = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path}r2_")
        repertoire3 = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path}r3_")
        descriptors_indexes1 = jnp.load(path + "descriptors_indexes1.npy")
        descriptors_indexes2 = jnp.load(path + "descriptors_indexes2.npy")
        descriptors_indexes3 = jnp.load(path + "descriptors_indexes3.npy")
        sampling_mask = jnp.ones(len(repertoire1.centroids) + len(repertoire2.centroids) + len(repertoire3.centroids))
        return cls(repertoire1, repertoire2, repertoire3, descriptors_indexes1, descriptors_indexes2,
                   descriptors_indexes3, sampling_mask)

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        fitnesses = jnp.concatenate(
            [self.repertoire1.fitnesses, self.repertoire2.fitnesses, self.repertoire3.fitnesses])
        genotypes = pytree_stack(
            [self.repertoire1.genotypes, self.repertoire2.genotypes, self.repertoire3.genotypes])
        flat_genotypes = jax.vmap(pytree_flatten)(genotypes)
        _, unique_genotypes_indexes = jnp.unique(flat_genotypes, axis=0, return_index=True, size=len(flat_genotypes),
                                                 fill_value=jnp.zeros_like(flat_genotypes[0]))
        filled_genotypes_mask = fitnesses != -jnp.inf
        unique_genotypes_mask = jnp.isin(jnp.arange(len(flat_genotypes)), unique_genotypes_indexes)
        candidate_genotypes_mask = filled_genotypes_mask & self.sampling_mask.astype(int) & unique_genotypes_mask
        p = candidate_genotypes_mask.astype(int)
        p = p / jnp.sum(p)
        random_key, subkey = jax.random.split(random_key)
        samples = jax.tree_util.tree_map(
            lambda x: jax.random.choice(subkey, x, shape=(num_samples,), p=p),
            genotypes,
        )
        return samples, random_key

    # Note: sampling id semantics
    # 0 = sample from all
    # 1 = sample from first
    # 2 = sample from second
    # 3 = sample from third
    @jax.jit
    def update_sampling_mask(self, sampling_id: int) -> MapElitesTriRepertoire:
        sampling_mask1 = jnp.ones(len(self.repertoire1.centroids)) * (sampling_id < 2)
        sampling_mask2 = jnp.ones(len(self.repertoire2.centroids)) * (sampling_id % 2 == 0)
        sampling_mask3 = jnp.ones(len(self.repertoire3.centroids)) * (sampling_id % 3 == 0)
        sampling_mask = jnp.concatenate([sampling_mask1, sampling_mask2, sampling_mask3])

        return MapElitesTriRepertoire(
            self.repertoire1,
            self.repertoire2,
            self.repertoire3,
            self.descriptors_indexes1,
            self.descriptors_indexes2,
            self.descriptors_indexes3,
            sampling_mask
        )

    @jax.jit
    def add(
            self,
            batch_of_genotypes: Genotype,
            batch_of_descriptors: Descriptor,
            batch_of_fitnesses: Fitness,
            batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[MapElitesTriRepertoire, bool]:
        descriptors1 = batch_of_descriptors.take(self.descriptors_indexes1, axis=1)
        descriptors2 = batch_of_descriptors.take(self.descriptors_indexes2, axis=1)
        descriptors3 = batch_of_descriptors.take(self.descriptors_indexes3, axis=1)
        new_repertoire1, addition_condition1 = self.repertoire1.add_and_track(
            batch_of_genotypes,
            descriptors1,
            batch_of_fitnesses,
            batch_of_extra_scores
        )
        new_repertoire2, addition_condition2 = self.repertoire2.add_and_track(
            batch_of_genotypes,
            descriptors2,
            batch_of_fitnesses,
            batch_of_extra_scores
        )
        new_repertoire3, addition_condition3 = self.repertoire3.add_and_track(
            batch_of_genotypes,
            descriptors3,
            batch_of_fitnesses,
            batch_of_extra_scores
        )
        new_triple_repertoire = MapElitesTriRepertoire(
            repertoire1=new_repertoire1,
            repertoire2=new_repertoire2,
            repertoire3=new_repertoire3,
            descriptors_indexes1=self.descriptors_indexes1,
            descriptors_indexes2=self.descriptors_indexes2,
            descriptors_indexes3=self.descriptors_indexes3,
            sampling_mask=self.sampling_mask
        )
        addition_condition = addition_condition1 + addition_condition2 + addition_condition3
        return new_triple_repertoire, addition_condition

    @classmethod
    def init(
            cls,
            genotypes: Genotype,
            fitnesses: Fitness,
            descriptors: Descriptor,
            descriptors_indexes1: jnp.ndarray,
            descriptors_indexes2: jnp.ndarray,
            descriptors_indexes3: jnp.ndarray,
            centroids1: Centroid,
            centroids2: Centroid,
            centroids3: Centroid,
            extra_scores: Optional[ExtraScores] = None,
    ) -> MapElitesTriRepertoire:
        descriptors1 = descriptors.take(descriptors_indexes1, axis=1)
        descriptors2 = descriptors.take(descriptors_indexes2, axis=1)
        descriptors3 = descriptors.take(descriptors_indexes3, axis=1)
        repertoire1 = MapElitesRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors1,
            centroids=centroids1,
            extra_scores=extra_scores
        )
        repertoire2 = MapElitesRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors2,
            centroids=centroids2,
            extra_scores=extra_scores
        )
        repertoire3 = MapElitesRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors3,
            centroids=centroids3,
            extra_scores=extra_scores
        )
        sampling_mask = jnp.ones(len(repertoire1.centroids) + len(repertoire2.centroids) + len(repertoire3.centroids))
        return MapElitesTriRepertoire(
            repertoire1=repertoire1,
            repertoire2=repertoire2,
            repertoire3=repertoire3,
            descriptors_indexes1=descriptors_indexes1,
            descriptors_indexes2=descriptors_indexes2,
            descriptors_indexes3=descriptors_indexes3,
            sampling_mask=sampling_mask
        )


class MapElitesTriRepertoireWithID(MapElitesTriRepertoire):
    n_generated_individuals: int = 0

    @classmethod
    def from_mapelites_tri_repertoire(cls,
                                      mapelites_tri_repertoire: MapElitesTriRepertoire,
                                      n_generated_individuals: int
                                      ) -> MapElitesTriRepertoireWithID:
        return cls(
            mapelites_tri_repertoire.repertoire1,
            mapelites_tri_repertoire.repertoire2,
            mapelites_tri_repertoire.repertoire3,
            mapelites_tri_repertoire.descriptors_indexes1,
            mapelites_tri_repertoire.descriptors_indexes2,
            mapelites_tri_repertoire.descriptors_indexes3,
            mapelites_tri_repertoire.sampling_mask,
            n_generated_individuals
        )

    # Same semantics as in MapElitesTriRepertoire
    @jax.jit
    def update_sampling_mask(self, sampling_id: int) -> MapElitesTriRepertoireWithID:
        updated_me_tri_repertoire = super().update_sampling_mask(sampling_id)
        return self.from_mapelites_tri_repertoire(
            updated_me_tri_repertoire,
            self.n_generated_individuals
        )

    # Checks for unicity on the id rather than on the actual genotype, neat speed up
    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        fitnesses = jnp.concatenate(
            [self.repertoire1.fitnesses, self.repertoire2.fitnesses, self.repertoire3.fitnesses])
        genotypes = pytree_stack(
            [self.repertoire1.genotypes, self.repertoire2.genotypes, self.repertoire3.genotypes])
        _, individual_ids = genotypes.pop("individual_id")
        _, unique_genotypes_indexes = jnp.unique(individual_ids, axis=0, return_index=True, size=len(individual_ids),
                                                 fill_value=jnp.zeros_like(individual_ids[0]))

        filled_genotypes_mask = fitnesses != -jnp.inf
        unique_genotypes_mask = jnp.isin(jnp.arange(len(unique_genotypes_indexes)), unique_genotypes_indexes)
        candidate_genotypes_mask = filled_genotypes_mask & self.sampling_mask.astype(int) & unique_genotypes_mask
        p = candidate_genotypes_mask.astype(int)
        p = p / jnp.sum(p)
        random_key, subkey = jax.random.split(random_key)
        samples = jax.tree_util.tree_map(
            lambda x: jax.random.choice(subkey, x, shape=(num_samples,), p=p),
            genotypes,
        )
        samples_without_ids, _ = samples.pop("individual_id")
        return samples_without_ids, random_key

    # Adds the id to genotypes wrt to super class
    @jax.jit
    def add(
            self,
            batch_of_genotypes: Genotype,
            batch_of_descriptors: Descriptor,
            batch_of_fitnesses: Fitness,
            batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[MapElitesTriRepertoireWithID, bool]:
        individual_ids = jnp.expand_dims(jnp.arange(len(batch_of_fitnesses)), 1) + self.n_generated_individuals
        genotypes_with_ids = batch_of_genotypes.copy({
            "individual_id": individual_ids,
        })
        n_generated_individuals = self.n_generated_individuals + len(batch_of_fitnesses)
        updated_me_tri_repertoire, addition_condition = super().add(
            genotypes_with_ids,
            batch_of_descriptors,
            batch_of_fitnesses,
            batch_of_extra_scores
        )
        new_me_tri_repertoire_with_id = self.from_mapelites_tri_repertoire(
            updated_me_tri_repertoire,
            n_generated_individuals
        )
        return new_me_tri_repertoire_with_id, addition_condition

    # Adds ids to the genotypes wrt to super class
    @classmethod
    def init(
            cls,
            genotypes: Genotype,
            fitnesses: Fitness,
            descriptors: Descriptor,
            descriptors_indexes1: jnp.ndarray,
            descriptors_indexes2: jnp.ndarray,
            descriptors_indexes3: jnp.ndarray,
            centroids1: Centroid,
            centroids2: Centroid,
            centroids3: Centroid,
            extra_scores: Optional[ExtraScores] = None,
    ) -> MapElitesTriRepertoireWithID:
        n_generated_individuals = len(fitnesses)
        individual_ids = jnp.expand_dims(jnp.arange(n_generated_individuals), 1)
        genotypes_with_ids = genotypes.copy({
            "individual_id": individual_ids,
        })
        me_tri_repertoire = super().init(
            genotypes_with_ids,
            fitnesses,
            descriptors,
            descriptors_indexes1,
            descriptors_indexes2,
            descriptors_indexes3,
            centroids1,
            centroids2,
            centroids3,
            extra_scores
        )
        return cls.from_mapelites_tri_repertoire(me_tri_repertoire, n_generated_individuals)

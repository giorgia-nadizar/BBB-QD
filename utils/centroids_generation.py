import jax
import jax.numpy as jnp
from typing import Union, Tuple, Callable

from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, compute_cvt_centroids_from_samples
from qdax.types import RNGKey


def compute_centroids(
        random_key: RNGKey,
        n_init_cvt_samples: int = 500000,
        n_centroids: int = 1024,
        n_descriptors: int = None,
        min_value: Union[float, jnp.ndarray] = 0.,
        max_value: Union[float, jnp.ndarray] = 1.,
        samples_condition: Callable[[jnp.ndarray], bool] = None
) -> Tuple[jnp.ndarray, RNGKey]:
    assert (isinstance(min_value, float) and isinstance(max_value, float) and n_descriptors is not None) or (
            isinstance(min_value, jnp.ndarray) and isinstance(max_value, jnp.ndarray)
            and len(min_value) == len(max_value) and samples_condition is not None)
    if isinstance(min_value, float):
        return compute_cvt_centroids(
            num_descriptors=n_descriptors,
            num_init_cvt_samples=n_init_cvt_samples,
            num_centroids=n_centroids,
            minval=min_value,
            maxval=max_value,
            random_key=random_key,
        )
    else:
        random_key, subkey = jax.random.split(random_key)
        xs = jax.random.uniform(key=subkey,
                                shape=(3 * n_init_cvt_samples, n_descriptors),
                                minval=min_value,
                                maxval=max_value)
        samples_list = []
        for x in xs:
            if samples_condition(x):
                samples_list.append(x)
            if len(samples_list) >= n_init_cvt_samples:
                break
        return compute_cvt_centroids_from_samples(
            descriptors_samples=jnp.asarray(samples_list),
            num_centroids=n_centroids,
            random_key=random_key
        )


if __name__ == '__main__':
    rnd_key = jax.random.PRNGKey(0)

    body_centroids, rnd_key = compute_centroids(
        random_key=rnd_key,
        n_descriptors=2,
        min_value=0.,
        max_value=1.
    )
    jnp.save("../experiments/data/body_centroids.npy", body_centroids)
    print("body centroids")

    behavior_centroids, rnd_key = compute_centroids(
        random_key=rnd_key,
        n_descriptors=2,
        min_value=0.,
        max_value=1.
    )
    jnp.save("../experiments/data/behavior_centroids.npy", behavior_centroids)
    print("behavior centroids")

    brain_centroids, rnd_key = compute_centroids(
        random_key=rnd_key,
        n_descriptors=2,
        min_value=jnp.zeros(2),
        max_value=jnp.ones(2),
        samples_condition=lambda arr: arr[0] + arr[1] <= 1
    )
    jnp.save("../experiments/data/brain_centroids.npy", brain_centroids)
    print("brain centroids")

    global_centroids, rnd_key = compute_centroids(
        random_key=rnd_key,
        n_descriptors=6,
        min_value=jnp.zeros(6),
        max_value=jnp.ones(6),
        samples_condition=lambda arr: arr[0] + arr[1] <= 1
    )
    jnp.save("../experiments/data/global_centroids.npy", global_centroids)
    print("global centroids")

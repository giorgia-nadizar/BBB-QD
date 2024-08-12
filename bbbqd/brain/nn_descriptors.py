import jax.numpy as jnp
from flax.core import FrozenDict


def count_weights_above_threshold_per_neuron(nn_params: FrozenDict, threshold: float) -> jnp.ndarray:
    counts = []
    for layer_name, layer_params in nn_params.items():
        if 'kernel' in layer_params:
            kernel = layer_params['kernel']
            count = jnp.sum(jnp.abs(kernel) > threshold, axis=1) / kernel.shape[1]
            counts.append(count)
    return jnp.concatenate(counts)


def mean_and_std_output_connectivity(nn_params: FrozenDict, threshold: float = 0.25) -> jnp.ndarray:
    weights_above_threshold_per_neuron = count_weights_above_threshold_per_neuron(nn_params["params"], threshold)
    mean_value, std_value = jnp.mean(weights_above_threshold_per_neuron), jnp.std(weights_above_threshold_per_neuron)
    return jnp.asarray([mean_value, std_value])

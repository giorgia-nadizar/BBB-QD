import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
import flax.linen as nn
from jax import vmap
from jax._src.flatten_util import ravel_pytree


def sparsity_index(activations: jnp.ndarray):
    l1_norm = jnp.sum(jnp.abs(activations))
    l2_norm = jnp.linalg.norm(activations)
    num_neurons = activations.size
    si = 1 - (l1_norm / (l2_norm * jnp.sqrt(num_neurons)))
    return si


def activation_distribution_index(activations: jnp.ndarray):
    # Normalize activations to form a probability distribution
    abs_activations = jnp.abs(activations)
    total_sum = jnp.sum(abs_activations)
    probabilities = abs_activations / total_sum

    # Compute entropy
    entropy = -jnp.sum(probabilities * jnp.log(probabilities + 1e-12))  # Add small epsilon for numerical stability
    max_entropy = jnp.log(activations.size)
    adi = entropy / max_entropy
    return adi


def sparsity_and_activation_distribution(policy_network: nn.Module, nn_params: FrozenDict,
                                         data_points: np.ndarray) -> jnp.ndarray:
    n_inner_neurons = sum(policy_network.layer_sizes[:-1])

    def _counting_fn(point: jnp.ndarray):
        _, inter_res = policy_network.apply(nn_params, point, capture_intermediates=True)
        flatten_res, _ = ravel_pytree(inter_res)
        activations, _ = jnp.split(flatten_res, [n_inner_neurons])
        sparsity = sparsity_index(activations)
        activation_dist = activation_distribution_index(activations)
        return jnp.asarray([sparsity, activation_dist])

    metrics = vmap(_counting_fn)(data_points)
    return jnp.mean(metrics, axis=0)


def mean_and_std_activity(policy_network: nn.Module, nn_params: FrozenDict, data_points: np.ndarray,
                          threshold: float = 0.25) -> jnp.ndarray:
    n_inner_neurons = sum(policy_network.layer_sizes[:-1])

    def _counting_fn(point: jnp.ndarray):
        _, inter_res = policy_network.apply(nn_params, point, capture_intermediates=True)
        flatten_res, _ = ravel_pytree(inter_res)
        activations, _ = jnp.split(flatten_res, [n_inner_neurons])
        return jnp.sum(jnp.abs(activations) > threshold) / n_inner_neurons

    data_activations = vmap(_counting_fn)(data_points)
    mean_value, std_value = jnp.mean(data_activations), jnp.std(data_activations)
    return jnp.asarray([mean_value, std_value * 2.])


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

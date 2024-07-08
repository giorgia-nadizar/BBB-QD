import jax.numpy as jnp
from chex import ArrayTree
from jaxlib.xla_extension import pytree
from jax.flatten_util import ravel_pytree


def pytree_stack(pytrees: ArrayTree):
    """Takes a list of trees and stacks every corresponding leaf.
    E.g., given two trees ((a, b), c) and ((a', b'), c'), returns ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in pytrees:
        leaves, treedef = pytree.flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.vstack(leaf) for leaf in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def pytree_flatten(tree: ArrayTree) -> jnp.ndarray:
    flatten_pytree, _ = ravel_pytree(tree)
    return flatten_pytree

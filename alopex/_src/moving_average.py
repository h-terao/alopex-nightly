from jax import tree_util
import chex


def update_by_ema(avg_tree: chex.ArrayTree, new_tree: chex.ArrayTree, *, momentum: float = 0.9998) -> chex.ArrayTree:
    """Update all leaves by EMA.

    Args:
        avg_tree: Average tree.
        new_tree: New tree.
        momentum: EMA momentum parameter.

    Returns:
        Updated tree.
    """

    def update_fn(avg_v, new_v):
        return new_v + momentum * (avg_v - new_v)

    return tree_util.tree_map(update_fn, avg_tree, new_tree)

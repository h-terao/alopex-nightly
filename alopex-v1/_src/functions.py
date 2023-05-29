from __future__ import annotations

import jax.numpy as jnp
import chex


def permutate(
    inputs: chex.Array, index: chex.Array, axis: int = 0, inv: bool = False
) -> chex.Array:
    """Permutate array according to the given index.

    Args:
        inputs: Array with shape (N, ...).
        index: Index array that has a shape of (N).
        inv: If True, inverse the permutation operation.
            It means that x == permutate(permutate(x, index), index, inv=True).

    Returns:
        Permutated array.
    """
    assert len(index) == inputs.shape[axis]
    ndim = inputs.ndim
    axes = [axis] + [x for x in range(ndim) if x != axis]
    inputs = jnp.transpose(inputs, axes)

    if inv:
        index = jnp.zeros_like(index).at[index].set(jnp.arange(len(index)))
    out = jnp.take(inputs, index, axis=0)

    out = jnp.transpose(out, list(range(1, axis + 1)) + [0] + list(range(axis + 1, ndim)))
    return out


def accuracy(inputs: chex.Array, labels: chex.Array, k: int = 1) -> chex.Array:
    """Top-k accuracy score.

    Args:
        inputs: Prediction with shape (..., num_classes).
        labels: One-hot encoded labels with shape (..., num_classes).

    Returns:
        An binary array with shape (...). You can obtain accuracy
        via `accuracy(...).mean()`
    """
    assert inputs.shape == labels.shape
    y = jnp.argsort(inputs)[..., -k:]
    t = jnp.argmax(labels, axis=-1, keepdims=True)
    return jnp.sum(y == t, axis=-1)

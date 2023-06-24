"""Functions"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import chex


def accuracy(
    inputs: chex.Array, labels: chex.Array, k: int = 1, keepdims: bool = False
) -> chex.Array:
    """Top-k accuracy score.

    Args:
        inputs: prediction with shape (..., num_classes).
        labels: one-hot encoded labels with shape (..., num_classes).

    Returns:
        An binary array with shape (...). You can obtain accuracy
        via `accuracy(...).mean()`
    """
    assert jnp.shape(inputs) == jnp.shape(labels)
    y = jnp.argsort(inputs)[..., -k:]
    t = jnp.argmax(labels, axis=-1, keepdims=True)
    return jnp.sum(y == t, axis=-1, keepdims=keepdims)


def kl_div(
    inputs: chex.Array,
    labels: chex.Array,
    log_label: bool = False,
    axis: int = -1,
    keepdims: bool = False,
) -> chex.Array:
    """Compute KL divergence.

    Args:
        inputs: pre-normalized prediction array.
        labels: one-hot encoded labels.
        axis: axis to reduce.
        log_label: whether label is the log space.
    """
    log_p = jax.nn.log_softmax(inputs, axis=axis)
    if log_label:
        log_p, labels = labels, jnp.exp(labels)
    else:
        log_q = jnp.log(labels)
    return jnp.sum(labels * (log_q - log_p), axis=axis, keepdims=keepdims)


def permutate(
    inputs: chex.Array, index: chex.Array, axis: int = 0, inv: bool = False
) -> chex.Array:
    """Permutate array according to the given index.

    Args:
        inputs: array with shape (N, ...).
        index: index array that has a shape of (N).
        inv: if True, inverse the permutation operation.
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

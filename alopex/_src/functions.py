"""Functions"""
from __future__ import annotations
import typing as tp

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as nn
import chex


def absolute_error(predictions: chex.Array, targets: tp.Optional[chex.Array] = None) -> chex.Array:
    """Calculates the absolute error for a set of predictions.

    Mean Absolute Error can be computed as absolute_error(a, b).mean().

    Args:
        predictions: a vector of arbitrary shape `[...]`.
        targets: a vector with shape broadcastable to that of `predictions`;
            if not provided then it is assumued to be a vector of zeros.

    Returns:
        elementwise absolute differences, with same shape as `predictions`.
    """
    chex.assert_type(predictions, float)
    if targets is not None:
        chex.assert_equal_shape((predictions, targets))
    errors = predictions - targets if targets is not None else predictions
    return jnp.absolute(errors)


l1_loss = absolute_error


def softmax_focal_loss(
    logits: chex.Array, labels: chex.Array, alpha: tp.Optional[chex.Array] = None, gamma: float = 0
) -> chex.Array:
    """Calculates the focal loss.

    References:
        [Lin et al., 2017](https://arxiv.org/abs/1708.02002v2)

    Args:
        logits: unnormalized log probabilities, with shape `[..., num_classes]`.
        labels: valid probability distributions, with a shape broadcastable to that of `logits`.
        alpha: class-wise weight of shape `[num_classes]`.
        gamma
    """
    if alpha is None:
        alpha = jnp.ones(jnp.size(logits, axis=-1), dtype=logits.dtype)
    ce_loss = -labels * nn.log_softmax(logits, axis=-1)
    pt = jnp.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return jnp.sum(focal_loss, axis=-1)


def softmax_focal_loss_with_integer_labels(
    logits: chex.Array, labels: chex.Array, alpha: tp.Optional[chex.Array] = None, gamma: float = 0
) -> chex.Array:
    num_classes = jnp.size(logits, axis=-1)
    labels = nn.one_hot(labels, num_classes, dtype=logits.dtype)
    return softmax_focal_loss(logits, labels, alpha, gamma)


def accuracy(predictions: chex.Array, labels: chex.Array, k: int = 1) -> chex.Array:
    """Calculates the classification top-k accuracy for a set of predictions.

    Top-k accuracy can be computed as accuracy(a, b).mean().

    Args:
        predictions: a vector of shape `[..., num_classes]`.
        labels: valid probability distributions, with a shape broadcastable to that of `predictions`.
        k: number of allowed classes as corrects.

    Returns:
        an accuracy vector with a shape of `[...]`.
    """
    chex.assert_type(predictions, float)
    chex.assert_equal_shape((predictions, labels))
    t = jnp.argsort(predictions)[..., -k:]
    y = jnp.argmax(labels, axis=-1, keepdims=True)
    return jnp.sum(jnp.equal(y, t), axis=-1)


def accuracy_with_integer_labels(predictions: chex.Array, labels: chex.Array, k: int = 1) -> chex.Array:
    """Calculates the classification top-k accuracy for a set of predictions.

    Top-k accuracy can be computed as accuracy_with_integer_labels(a, b).mean().

    Args:
        predictions: a vector of shape `[..., num_classes]`.
        labels: an integer vector of a shape broadcastable to `[...]`.
        k: number of allowed classes as corrects.

    Returns:
        an accuracy vector with a shape of `[...]`.
    """
    num_classes = jnp.size(predictions, axis=-1)
    labels = nn.one_hot(labels, num_classes, axis=-1)
    return accuracy(predictions, labels, k=k)


def permutate(x: chex.Array, indices: chex.Array, axis: int = 0, inv: bool = False) -> chex.Array:
    """Permutate array according to the given index.

    Args:
        x: a vector with shape `[..., n, ...]`, where n is the `axis`-th dimension.
        indices: index vector that has a shape of `[n]`.
        axis: axis to permute the vector.
        inv: if True, inverse the permutation operation.
            It means that x == permutate(permutate(x, index), index, inv=True).

    Returns:
        permutated vector with same shape as `x`.
    """
    chex.assert_equal(jnp.size(x, axis), len(indices))
    if inv:
        indices = jnp.zeros_like(indices).at[indices].set(jnp.arange(len(indices)))
    return jnp.take(x, indices, axis=axis)


def reverse_grad(x: chex.Array) -> chex.Array:
    """A functional implementation of the gradient reversal layer (GPL).

    GPL acts as an identity function during the forward propagation,
    changes the sign of gradients during the backward propagation.
    Reference: https://arxiv.org/abs/1505.07818

    Args:
        x: input vector.

    Returns:
        `x` itself.
    """

    @jax.custom_vjp
    def f(v):
        return v

    f.defvjp(lambda x: (x, ()), lambda _, g: (-g,))
    return f(x)


def gumbel_softmax(
    rng: jr.PRNGKey,
    logits: chex.Array,
    tau: float = 1,
    hard: bool = False,
    axis: int = -1,
) -> chex.Array:
    """Gumbel softmax function.

    Args:
        rng
        logits: unnormalized probabilities.
        tau: temperature parameter.
        hard: if True, the outputs will be discretized as one-hot vectors.
        axis: axis name to be normalized.
    """
    gumbels = jr.uniform(rng, jnp.shape(logits))
    gumbels = -jnp.log(-jnp.log(gumbels))
    gumbels = (logits + gumbels) / tau
    probs = nn.softmax(gumbels, axis=axis)
    if hard:
        num_classes = jnp.size(logits, axis=axis)
        one_hot = nn.one_hot(jnp.argmax(probs, axis=axis), num_classes, axis=axis)
        probs = jax.lax.stop_gradient(one_hot - probs) + probs
    return probs

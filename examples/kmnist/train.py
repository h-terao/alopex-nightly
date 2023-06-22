from __future__ import annotations
import typing as tp
import functools

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen
import optax
import alopex
from alopex.training import DynamicScale
import chex


class TrainState(tp.NamedTuple):
    step: int
    rng: chex.PRNGKey
    params: chex.ArrayTree
    state: chex.ArrayTree
    opt_state: optax.OptState
    dynamic_scale: tp.Optional[DynamicScale] = None


class MLP(linen.Module):
    dtype: chex.ArrayDType = jnp.float32

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False) -> chex.Array:
        x = jnp.reshape(x, (len(x), -1))  # Flatten.
        x = linen.Dense(1024, dtype=self.dtype)(x)
        x = linen.tanh(x)
        x = linen.Dropout(rate=0.5)(x, not is_training)
        x = linen.Dense(1024, dtype=self.dtype)(x)
        x = linen.tanh(x)
        x = linen.Dropout(rate=0.5)(x, not is_training)
        x = linen.Dense(1024, dtype=self.dtype)(x)
        x = linen.tanh(x)
        x = linen.Dense(10, dtype=self.dtype)(x)
        return x


def create_task(
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    dtype: chex.ArrayDType = jnp.float32,
):
    model = MLP(dtype)
    tx = optax.chain(
        optax.add_decayed_weights(
            weight_decay, mask=lambda params: jax.tree_util.tree_map(lambda x: x.ndim > 1, params)
        ),
        optax.sgd(learning_rate, momentum),
    )

    def init(rng, batch):
        pass

    def train_step(train_state, batch):
        pass

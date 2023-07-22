from __future__ import annotations
import typing as tp
import functools

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import tree_util
from flax import linen
import optax
import chex

if tp.TYPE_CHECKING:
    from alopex.training import DynamicScale


def train_fun_maker(
    loss_fun: tp.Callable,
    optimizer: optax.GradientTransformation,
    trainable_attrs: str | tp.Sequence[str] = "params",
    axis_name: tp.Optional[str] = None,
    *,
    gradient_accumulates: int = 1,
):
    """
    Args:
        loss_fun: a callable that compute loss from train_state and batch.
        attrs: attributes of `train_state` to compute gradients and updates by optimizer.
            Note that it should be equal to.
    """
    leaves, treedef = tree_util.tree_flatten(attrs)
    assert len(leaves) == len(set(leaves)), "All leaves of `attrs` should be unique."

    def collect_trainable_variables():
        pass

    def train_fun(train_state, batch):
        trainable_variables = {attr: getattr(train_state, attr) for attr in attrs}

        dynamic_scale = None
        if hasattr(train_state, "dynamic_scale"):
            dynamic_scale: DynamicScale = train_state.dynamic_scale

        def scan_fn(rng, batch):
            def compute_loss_and_updates(trainable_variables, rng, batch):
                replaced = train_state.replace(rng=rng, **trainable_variables)
                return loss_fun(replaced, batch)

            if dynamic_scale is None:
                grad_fn = jax.grad(compute_loss_and_updates, has_aux=True)
            else:
                grad_fn = dynamic_scale.grad(compute_loss_and_updates, has_aux=True)

            new_rng, rng = jr.split(rng)
            grads, aux = grad_fn(trainable_variables, rng, batch)
            return new_rng, (grads, aux)

        batch = tree_util.tree_map(lambda x: jnp.reshape(x, (gradient_accumulates, -1, *jnp.shape(x)[1:])), batch)
        _, outputs = jax.lax.scan(scan_fn, rng, batch)

        outputs = tree_util.tree_map(functools.partial(jnp.mean, axis=0), outputs)
        if axis_name is not None:
            outputs = jax.lax.pmean(outputs, axis_name)
        grads, (trainable_variables, log_dict) = outputs  # values: updates.

        updates, new_opt_state = optimizer.update(grads, train_state.opt_state, trainable_variables)
        new_trainable_variables = optax.apply_updates(trainable_variables, updates)
        del updates

        updates = new_trainable_variables

    return train_fun

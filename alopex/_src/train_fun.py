from __future__ import annotations
import typing as tp
import functools
import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import tree_util
import optax
import chex

if tp.TYPE_CHECKING:
    from alopex import pytypes


def get(train_state: chex.ArrayTree, name: str, default: tp.Optional[str] = None) -> tp.Any:
    if isinstance(train_state, tp.Mapping):
        return train_state.get(name, default)
    else:
        return getattr(train_state, name, default)


def replace(train_state: chex.ArrayTree, /, **kwargs) -> chex.ArrayTree:
    if isinstance(train_state, tp.Mapping):
        # Dict, OrderedDict, FrozenDict, ...
        cast_to = type(train_state)
        replaced = dict(train_state, **kwargs)
        return cast_to(replaced)
    elif dataclasses.is_dataclass(train_state):
        # dataclass.
        return dataclasses.replace(train_state, **kwargs)
    elif hasattr(train_state, "replace"):
        # Flax train_state or PyTreeNode
        return train_state.replace(**kwargs)
    elif hasattr(train_state, "_replace"):
        # NamedTuple
        return train_state._replace(**kwargs)
    else:
        raise RuntimeError("Type of train_state should be Mapping, dataclasses, PyTreeNode or NamedTuple.")


def assert_required_attrs(train_state, step_attr, trainable_attrs):
    attrs = ["rng", "opt_state", *tree_util.tree_flatten(trainable_attrs)]
    if step_attr is not None:
        attrs.append(step_attr)

    not_found_attrs = set(filter(lambda k: not hasattr(train_state, k) and k not in train_state, attrs))
    assert len(not_found_attrs) == 0, f"{not_found_attrs} are not found in train_state."


def train_fun(
    loss_fun: tp.Callable,
    optimizer: optax.GradientTransformation,
    transform: tp.Optional[tp.Callable[[chex.PRNGKey, chex.ArrayTree], chex.ArrayTree]] = None,
    step_attr: tp.Optional[str] = "step",
    trainable_attrs: str | tp.Sequence[str] = "params",
    fold_rng_key: bool = True,
    gradient_accumulates: int = 1,
    axis_name: tp.Optional[str] = None,
) -> pytypes.TrainFun:
    """Creates a train_fun that needs only one loss_fun and optimizer. The created train_fun is compatible with
        `alopex.training.train_loop`. Fork me if you need to customize the detailed flow.

    Args:
        loss_fun: a callable that computes loss from train_state and batch.
        optimizer: an optax optimizer.
        transform: a callable that transforms batches. Transformation is applied before spliting batches
            for the gradient accumulation.
        step_attr: name of the step or iteration attribute. If specified, increment the value when
            `train_step` is called.
        trainable_attrs: attributes of `train_state` to compute gradients and updates by optimizer.
            Note that treedef of trainable_attrs should be the same of tree used to initialize the state of optimizer.
        fold_rng_key: if True, fold PRNG key along device axis so that each process use different PRNG key.
        gradient_accumulates: number of gradient accumulation steps.
        axis_name: axis name along devices.

    Notes:
        - train_state should be dict, namedtuple-like, or Flax PyTreeNode object.
        - At least, train_state should have rng, dynamic_scale, opt_state, `step_attr`,
            and all members of `trainable_attrs`.
    """

    def f(train_state, batch):
        assert_required_attrs(train_state, step_attr, trainable_attrs)

        rng, new_rng = jr.split(get(train_state, "rng"))
        if axis_name is not None and fold_rng_key:
            rng = jr.fold_in(rng, jax.lax.axis_index(axis_name))

        dynamic_scale = None
        if hasattr(train_state, "dynamic_scale") or "dynamic_scale" in train_state:
            dynamic_scale: pytypes.DynamicScale = get(train_state, "dynamic_scale")

        trainable_variables = tree_util.tree_unflatten(
            tree_util.tree_structure(trainable_attrs),
            [get(train_state, attr) for attr in tree_util.tree_leaves(trainable_attrs)],
        )

        def scan_fn(rng, batch):
            def compute_loss_and_updates(trainable_variables, rng, batch):
                updates = dict(
                    zip(tree_util.tree_leaves(trainable_attrs), tree_util.tree_leaves(trainable_variables)),
                )
                replaced = replace(train_state, rng=rng, **updates)
                return loss_fun(replaced, batch)

            if dynamic_scale is None:
                grad_fn = jax.grad(compute_loss_and_updates, has_aux=True)
            else:
                grad_fn = dynamic_scale.grad(compute_loss_and_updates, has_aux=True)

            new_rng, rng = jr.split(rng)
            grads, aux = grad_fn(trainable_variables, rng, batch)
            return new_rng, (grads, aux)

        if transform is not None:
            rng, transform_rng = jr.split(rng)
            batch = transform(transform_rng, batch)

        batch = tree_util.tree_map(lambda x: jnp.reshape(x, (gradient_accumulates, -1, *jnp.shape(x)[1:])), batch)
        _, outputs = jax.lax.scan(scan_fn, rng, batch)

        outputs = tree_util.tree_map(functools.partial(jnp.mean, axis=0), outputs)
        if axis_name is not None:
            outputs = jax.lax.pmean(outputs, axis_name)

        grads, aux = outputs
        updates, opt_state = optimizer.update(grads, get(train_state, "opt_state"), trainable_variables)
        trainable_variables = optax.apply_updates(trainable_variables, updates)
        del updates

        updates, log_dict = aux
        updates["opt_state"] = opt_state
        for key, val in zip(tree_util.tree_leaves(trainable_attrs), tree_util.tree_flatten(trainable_variables)):
            updates[key] = val

        if dynamic_scale is not None:
            dynamic_scale, is_fin = dynamic_scale.update(grads)
            restores = {key: get(train_state, key) for key in updates}
            updates = tree_util.tree_map(functools.partial(jnp.where, is_fin), updates, restores)
            updates["dynamic_scale"] = dynamic_scale
            log_dict["scale"] = dynamic_scale.scale

        if step_attr is not None:
            updates[step_attr] = get(train_state, step_attr) + 1

        train_state = replace(train_state, rng=new_rng, **updates)
        return train_state, log_dict

    return f

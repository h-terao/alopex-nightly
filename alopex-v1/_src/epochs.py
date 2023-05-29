""""""
from __future__ import annotations
import typing as tp
import warnings

import jax
import jax.numpy as jnp
from jax import tree_util
import chex

from .pytypes import (
    Batch,
    Summary,
    Prediction,
    TrainState,
    TrainFun,
    EvalFun,
    PredFun,
)

__all__ = ["train_epoch", "eval_epoch", "pred_epoch"]


def _replicate(tree, devices=None):
    # Modify flax.jax_util
    devices = devices or jax.local_devices()
    return jax.device_put_replicated(tree, devices)


def _unreplicate(tree):
    # from flax.jax_util
    return tree_util.tree_map(lambda x: x[0], tree)


def _modify_batches(
    iterable: tp.Iterable,
    epoch_length: int = -1,
    prefetch: bool = False,
    devices: list[chex.Device] | None = None,
):
    devices = devices or jax.local_devices()

    def modify_batch(batch: Batch) -> list[tuple[Batch, float]]:
        num_devices = len(devices)

        leaves, treedef = tree_util.tree_flatten(batch)
        min_size = min(len(x) for x in leaves)
        min_main_size, min_sub_size = divmod(min_size, num_devices)

        main_leaves, sub_leaves = [], []
        for x in leaves:
            r = len(x) / min_size
            main_size, sub_size = int(r * min_main_size), int(r * min_sub_size)
            assert main_size * num_devices + sub_size == len(x)

            if main_size > 0:
                main_array = x[sub_size:]
                main_leaves.append(main_array.reshape(num_devices, main_size, *x.shape[1:]))

            if sub_size > 0:
                sub_array = x[:sub_size]
                sub_leaves.append(jnp.stack([sub_array] * num_devices, axis=0))

        modified_batches = []
        if main_leaves:
            main_batch = tree_util.tree_unflatten(treedef, main_leaves)
            modified_batches.append([main_batch, main_size * num_devices, False])

        if sub_leaves:
            sub_batch = tree_util.tree_unflatten(treedef, sub_leaves)
            modified_batches.append([sub_batch, sub_size, True])

        return modified_batches

    prev_batch, prev_size, prev_is_remainder = None, None, None
    for batch_idx, batch in enumerate(iterable):
        modified_batches = modify_batch(batch)
        for next_batch, next_size, is_reminder in modified_batches:
            if prefetch:
                next_batch = tree_util.tree_map(
                    lambda x: jax.device_put_sharded(list(x), devices),
                    tree=next_batch,
                )

            if prev_batch is not None:
                yield prev_batch, prev_size, prev_is_remainder

            prev_batch, prev_size, prev_is_remainder = next_batch, next_size, is_reminder

        if batch_idx + 1 == epoch_length:
            break

    if prev_batch is not None:
        yield prev_batch, prev_size, prev_is_remainder


@jax.jit
def _accumulate_scalars(
    accum_scalars: dict[str, tuple[float, float]],
    new_scalars: dict[str, float],
    weight: float = 1,
) -> dict[str, tuple[float, float]]:
    updates = {}
    for key, scalar in new_scalars.items():
        accum_scalar, accum_weight = accum_scalars.get(key, (0, 0))
        scalar = jnp.array(scalar, dtype=jnp.float32)  # If scalar is float16, overflow may cause.
        accum_scalar += scalar.mean() * weight
        accum_weight += weight
        updates[key] = (accum_scalar, accum_weight)
    return dict(accum_scalars, **updates)


def _summarize_scalars(prefix: str, accum_scalars: dict[str, tuple[float, float]], **kwargs) -> dict[str, float]:
    summary = {prefix + key: float(val / weight) for key, (val, weight) in accum_scalars.items()}
    for key, val in kwargs.items():
        summary[prefix + key] = val
    return summary


def train_epoch(
    train_fun: TrainFun,
    prefix: str | None = None,
    prefetch: bool = True,
    axis_name: str = "batch",
    devices: list[chex.Device] | None = None,
) -> tp.Callable[[TrainState, tp.Iterable[Batch], int | None], tuple[TrainState, Summary]]:
    """Creates a function that updates the model and summarizes scalars using `train_fun` until
        an iterable object stops.

    Args:
        train_fun: Training function that updates train_state.
        prefix: Prefix of scalars.
        prefetch: If True, prefetch batches.
            This flag may shortens the processing time when you use GPUs.
        axis_name: Axis name for `jax.pmap`.
        devices: Device list. If None, all visible devices are used.

    Returns:
        A wrapped version of `train_fun`.

    Example::

        def train_step(train_state, batch):
            # Update train_state and compute metrics
            new_train_state = ...
            scalars = {"loss": 0.1, "acc1": 0.9}
            return new_train_state, scalars

        train_fun = train_epoch(train_step, prefix="train/")

        # `train_fun` repeats `train_step` until `train_loader` stops.
        new_train_state, summary = train_fun(train_state, train_loader)
        assert summary == {"train/loss": 0.1, "train/acc1": 0.9}
    """
    prefix = prefix or ""
    p_train_fun = jax.pmap(train_fun, axis_name=axis_name, devices=devices)

    def wrapped(
        train_state: TrainState,
        iterable: tp.Iterable[Batch],
        epoch_length: int = -1,
    ) -> tuple[TrainState, Summary]:
        train_state = _replicate(train_state, devices)

        accum_scalars = {}
        iterable = _modify_batches(iterable, epoch_length, prefetch, devices)
        for batch, weight, is_remainder in iterable:
            if is_remainder:
                msg = (
                    "Batch size is not divisible by the number of devices for training. "
                    "Such configuration is not allowed because it causes unexpected behaviours."
                )
                raise RuntimeError(msg)

            train_state, scalars = p_train_fun(train_state, batch)
            accum_scalars = _accumulate_scalars(accum_scalars, scalars, weight)

        train_state = _unreplicate(train_state)
        summary = _summarize_scalars(prefix, accum_scalars)
        return train_state, summary

    return wrapped


def eval_epoch(
    eval_fun: EvalFun,
    prefix: str | None = None,
    prefetch: bool = True,
    axis_name: str = "batch",
    devices: list[chex.Device] | None = None,
) -> tp.Callable[[TrainState, tp.Iterable[Batch], int | None], Summary]:
    """Creates a function that summarizes scalars using `eval_fun` until
        an iterable object stops.

    Args:
        eval_fn: Evaluation function that returns metrics.
        prefix: Prefix of scalars.
        prefetch: If True, prefetch batches.
            This flag may shortens the processing time when you use GPUs.
        axis_name: Axis name for `jax.pmap`.
        devices: Device to use. If None, all visible devices are used.

    Returns:
        A wrapped version of `eval_fun`.

    """
    prefix = prefix or ""
    p_eval_fun = jax.pmap(eval_fun, axis_name=axis_name, devices=devices)

    def wrapped(
        train_state: TrainState,
        iterable: tp.Iterable[Batch],
        epoch_length: int = -1,
    ) -> Summary:
        train_state = _replicate(train_state, devices)

        accum_scalars = {}
        iterable = _modify_batches(iterable, epoch_length, prefetch, devices)
        for batch_idx, (batch, weight, is_remainder) in enumerate(iterable):
            # Only check the 1st batch (If splitted, batch_idx=1 is also the 1st batch)
            if batch_idx < 2 and is_remainder:
                msg = (
                    "Batch size is not a multiple of device count. "
                    "Because alopex splits such batches into two parts and "
                    "call `eval_fun` two times, evaluation may takes a longer time. "
                    "If your dataset is very small, ignore this warning."
                )
                warnings.warn(msg)
            scalars = p_eval_fun(train_state, batch)
            accum_scalars = _accumulate_scalars(accum_scalars, scalars, weight)

        summary = _summarize_scalars(prefix, accum_scalars)
        return summary

    return wrapped


def pred_epoch(
    pred_fun: PredFun,
    prefetch: bool = True,
    axis_name: str = "batch",
    devices: list[chex.Device] | None = None,
) -> tp.Callable[[TrainState, tp.Iterable[Batch], int | None], Prediction]:
    """Creates a function that stacks outputs of `predict_fun` until
        an iterable object stops.

    Args:
        pred_fun: A function that returns PyTrees to stack.
        prefetch: If True, prefetch batches.
            This flag may shortens the processing time when you use GPUs.
        axis_name: Axis name for `jax.pmap`.
        devices: Device to use. If None, all visible devices are used.

    Returns:
        A wrapped version of `pred_fun`.
    """
    p_pred_fun = jax.pmap(pred_fun, axis_name=axis_name, devices=devices)

    def wrapped(
        train_state: TrainState,
        iterable: tp.Iterable[Batch],
        epoch_length: int = -1,
    ) -> Prediction:
        train_state = _replicate(train_state, devices)

        outputs = []
        iterable = _modify_batches(iterable, epoch_length, prefetch, devices)
        for batch_idx, (batch, _, is_remainder) in enumerate(iterable):
            # Only check the 1st batch (If splitted, batch_idx=1 is also the 1st batch)
            if batch_idx < 2 and is_remainder:
                msg = (
                    "Batch size is not a multiple of device count. "
                    "Because alopex splits such batches into two parts and "
                    "call `predict_fun` two times, evaluation may takes a longer time. "
                    "If your dataset is very small, ignore this warning."
                )
                warnings.warn(msg)

            output = p_pred_fun(train_state, batch)
            if is_remainder:
                output = tree_util.tree_map(lambda x: x[0], output)
            else:
                output = tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), output)
            outputs.append(output)

        output = tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *outputs)
        return output

    return wrapped

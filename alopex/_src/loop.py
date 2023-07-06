"""Training and evaluation loops."""
from __future__ import annotations
import typing as tp
import warnings

import jax
import jax.numpy as jnp
from jax import tree_util
from flax import jax_utils
from einops import rearrange, repeat
import chex

__all__ = ["train_loop", "eval_loop"]


TrainState = chex.ArrayTree
TrainFun = tp.Callable[[TrainState, chex.ArrayTree], tuple[TrainState, dict[str, chex.Array]]]
TrainLoop = tp.Callable[
    [TrainState, tp.Iterable[chex.ArrayTree], tp.Optional[int]], tuple[TrainState, dict[str, chex.Array]]
]
EvalFun = tp.Callable[[TrainState, chex.ArrayTree], dict[str, chex.Array]]
EvalLoop = tp.Callable[[TrainState, tp.Iterable[chex.ArrayTree], tp.Optional[int]], dict[str, chex.Array]]


def _split_batch(iterable: tp.Iterable, max_length: int = -1, prefetch: bool = False, devices=None):
    devices = devices or jax.local_devices()
    num_devices = len(devices)

    def split(batch):
        leaves, treedef = tree_util.tree_flatten(batch)
        base_size = min(map(len, leaves))
        main_base_size, aux_base_size = divmod(base_size, num_devices)

        main_leaves = []
        aux_leaves = []
        for x in leaves:
            r = len(x) / base_size
            main_size, aux_size = int(r * main_base_size), int(r * aux_base_size)
            assert main_size * num_devices + aux_size == len(x)

            if main_size > 0:
                v = x[aux_size:]
                v = jnp.reshape(x[aux_size:], (num_devices, main_size, *jnp.shape(x)[1:]))
                v = rearrange(x[aux_size:], "(d b) ... -> d b ...", d=num_devices)
                main_leaves.append(v)

            if aux_size > 0:
                v = repeat(x[:aux_size], "... -> n ...", n=num_devices)
                aux_leaves.append(v)

        new_batch = []
        if main_leaves:
            v = tree_util.tree_unflatten(treedef, main_leaves)
            new_batch.append([v, main_size * num_devices, False])

        if aux_leaves:
            v = tree_util.tree_unflatten(treedef, aux_leaves)
            new_batch.append([v, aux_size, True])

        return new_batch

    prev_batch, prev_size, prev_is_remainder = None, None, None
    for batch_idx, batch in enumerate(iterable):
        for next_batch, actual_size, is_remainder in split(batch):
            if prefetch:
                next_batch = tree_util.tree_map(lambda v: jax.device_put_sharded(list(v), devices), next_batch)

            if prev_batch is not None:
                yield prev_batch, prev_size, prev_is_remainder

            prev_batch = next_batch
            prev_size = actual_size
            prev_is_remainder = is_remainder

        if batch_idx + 1 == max_length:
            break

    if prev_batch is not None:
        yield prev_batch, prev_size, prev_is_remainder


@jax.jit
def _accumulate_scalars(
    accum_scalars: dict[str, tuple[chex.Array, chex.Array]],
    new_scalars: dict[str, chex.Array],
    weight: float = 1,
) -> dict[str, tuple[chex.Array, chex.Array]]:
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


def train_loop(
    train_fun: TrainFun,
    prefix: tp.Optional[str] = None,
    mode: str = "none",
    prefetch: bool = False,
    replicate: bool = True,
    axis_name: tp.Optional[str] = None,
    devices: tp.Optional[list[chex.Device]] = None,
) -> TrainLoop:
    """
    Args:
        train_fun: training function.
        prefix: prefix string of the logging values.
        mode: compile mode. (none | jit | pmap)
        prefetch: enable prefetching batches on devices. Only used when mode is `pmap`.
        replicate: if True, automatically replicate and unreplicate train_state
            for multi-device training.
        axis_name: train_epoch uses multiple devices.
        devices: JAX devices.
    """
    assert mode in ["none", "jit", "pmap"], f"mode should be none, jit or pmap, but given {mode}"

    prefix = prefix or ""
    if mode in "pmap":
        train_fun = jax.pmap(train_fun, axis_name=axis_name, devices=devices)

        def train_epoch(train_state, iterable, max_length: tp.Optional[int] = None):
            if replicate:
                train_state = jax_utils.replicate(train_state, devices)

            if max_length is None:
                max_length = -1

            accum_scalars = {}
            for batch, actual_size, is_remainder in _split_batch(iterable, max_length, prefetch, devices):
                if is_remainder:
                    raise RuntimeError("Batch size should be divisible by number of devices for training.")
                train_state, scalars = train_fun(train_state, batch)
                accum_scalars = _accumulate_scalars(accum_scalars, scalars, actual_size)

            summary = _summarize_scalars(prefix, accum_scalars)
            if replicate:
                train_state = jax_utils.unreplicate(train_state, devices)

            return train_state, summary

    else:
        if isinstance(devices, tp.Sequence):
            if len(devices) > 1:
                warnings.warn(f"Multiple devices are specified, but only use the first device: {devices[0]}.")
            devices = devices[0]

        if mode == "jit":
            train_fun = jax.jit(train_fun, device=devices)

        def train_epoch(train_state, iterable, max_length: tp.Optional[int] = None):
            if max_length is None:
                max_length = -1

            accum_scalars = {}
            for batch_idx, batch in enumerate(iterable):
                batch_size = min(map(len, tree_util.tree_leaves(batch)))
                train_state, scalars = train_fun(train_state, batch)
                accum_scalars = _accumulate_scalars(accum_scalars, scalars, batch_size)
                if batch_idx + 1 == max_length:
                    break

            summary = _summarize_scalars(prefix, accum_scalars)
            return train_state, summary

    return train_epoch


def eval_loop(
    eval_fun: EvalFun,
    prefix: tp.Optional[str] = None,
    mode: str = "none",
    prefetch: bool = False,
    replicate: bool = True,
    axis_name: tp.Optional[str] = None,
    devices: tp.Optional[list[chex.Device] | chex.Device] = None,
) -> EvalLoop:
    """
    Args:
        eval_fun: evaluation function.
        prefix: prefix string of the logging values.
        mode: compile mode. (none | jit | pmap)
        prefetch: enable prefetching batches on devices. Only used when mode is `pmap`.
        replicate: if True, automatically replicate and unreplicate train_state
            for multi-device training.
        axis_name: train_epoch uses multiple devices.
        devices: JAX devices.
    """
    assert mode in ["none", "jit", "pmap"], f"mode should be none, jit or pmap, but given {mode}"

    prefix = prefix or ""
    if mode == "pmap":
        eval_fun = jax.pmap(eval_fun, axis_name=axis_name, devices=devices)

        def eval_epoch(train_state, iterable, max_length=None):
            if replicate:
                train_state = jax_utils.replicate(train_state, devices)

            if max_length is None:
                max_length = -1

            accum_scalars = {}
            remainder_count = 0
            for batch, actual_size, is_remainder in _split_batch(iterable, max_length, prefetch, devices):
                if is_remainder:
                    remainder_count += 1

                if remainder_count == 2:
                    warnings.warn(
                        """You use multi-GPUs to evaluate your model, but it seems that the batch size
                        is not divisble by the number of GPUs. This configuration also generates the
                        correct summary, but results in inefficient evaluation. You are recommended
                        to set batch size a mupltiple of number of GPUs for efficient evaluation."""
                    )

                scalars = eval_fun(train_state, batch)
                accum_scalars = _accumulate_scalars(accum_scalars, scalars, actual_size)

            summary = _summarize_scalars(prefix, accum_scalars)
            return summary

    else:
        if isinstance(devices, tp.Sequence):
            if len(devices) > 1:
                warnings.warn(f"Multiple devices are specified, but use {devices[0]}.")
            devices = devices[0]

        if mode == "jit":
            eval_fun = jax.jit(eval_fun, device=devices)

        def eval_epoch(train_state, iterable, max_length=None):
            if max_length is None:
                max_length = -1

            accum_scalars = {}
            for batch_idx, batch in enumerate(iterable):
                batch_size = min(map(len, tree_util.tree_leaves(batch)))
                scalars = eval_fun(train_state, batch)
                accum_scalars = _accumulate_scalars(accum_scalars, scalars, batch_size)
                if batch_idx + 1 == max_length:
                    break

            summary = _summarize_scalars(prefix, accum_scalars)
            return summary

    return eval_epoch

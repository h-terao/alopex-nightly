"""Training and evaluation loops."""
from __future__ import annotations
import typing as tp
import functools
import itertools
import warnings

import jax
import jax.random as jr
import jax.numpy as jnp
from jax import tree_util
from flax import struct, jax_utils
from einops import rearrange, repeat
import chex

__all__ = ["train_loop", "eval_loop", "DynamicScale", "accumulate_gradients"]


TrainState = chex.ArrayTree
TrainFun = tp.Callable[[TrainState, chex.ArrayTree], tuple[TrainState, dict[str, chex.Array]]]
TrainLoop = tp.Callable[
    [TrainState, tp.Iterable[chex.ArrayTree], tp.Optional[int]], tuple[TrainState, dict[str, chex.Array]]
]
EvalFun = tp.Callable[[TrainState, chex.ArrayTree], dict[str, chex.Array]]
EvalLoop = tp.Callable[[TrainState, tp.Iterable[chex.ArrayTree], tp.Optional[int]], dict[str, chex.Array]]


def _split_batch(iterable: tp.Iterable, max_length: tp.Optional[int] = None, prefetch: bool = False, devices=None):
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

    if max_length is not None:
        iterable = itertools.islice(iterable, max_length)

    prev_batch, prev_size, prev_is_remainder = None, None, None
    for batch in iterable:
        for next_batch, actual_size, is_remainder in split(batch):
            if prefetch:
                next_batch = tree_util.tree_map(lambda v: jax.device_put_sharded(list(v), devices), next_batch)

            if prev_batch is not None:
                yield prev_batch, prev_size, prev_is_remainder

            prev_batch = next_batch
            prev_size = actual_size
            prev_is_remainder = is_remainder

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

    FIXME:
        Currently, mode=jit is very slow. Use mode=none for debugging and pmap for training and evaluation.
    """
    assert mode in ["none", "jit", "pmap"], f"mode should be none, jit or pmap, but given {mode}"

    if prefix is None:
        prefix = ""

    if mode == "pmap":
        train_fun = jax.pmap(train_fun, axis_name=axis_name)

        def train_epoch(train_state, iterable, max_length: tp.Optional[int] = None):
            if replicate:
                train_state = jax_utils.replicate(train_state)

            accum_scalars = {}
            for batch, actual_size, is_remainder in _split_batch(iterable, max_length, prefetch):
                if is_remainder:
                    raise RuntimeError("Batch size should be divisible by number of devices for training.")
                train_state, scalars = train_fun(train_state, batch)
                accum_scalars = _accumulate_scalars(accum_scalars, scalars, actual_size)

            summary = _summarize_scalars(prefix, accum_scalars)
            if replicate:
                train_state = jax_utils.unreplicate(train_state)

            return train_state, summary

    else:
        if mode == "jit":
            warnings.warn(
                (
                    "JIT mode may leads very slow evaluation, because of bugs. "
                    "Consider to use none or pmap mode, instead of jit."
                )
            )
            train_fun = jax.jit(train_fun)

        def train_epoch(train_state, iterable, max_length: tp.Optional[int] = None):
            if max_length is not None:
                iterable = itertools.islice(iterable, max_length)

            accum_scalars = {}
            for batch in iterable:
                batch_size = min(map(len, tree_util.tree_leaves(batch)))
                train_state, scalars = train_fun(train_state, batch)
                accum_scalars = _accumulate_scalars(accum_scalars, scalars, batch_size)

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
    disallow_remainder: bool = False,
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
        disallow_remainder: if True, raise error when the batch size is not divisible by the number of devices.
    """
    assert mode in ["none", "jit", "pmap"], f"mode should be none, jit or pmap, but given {mode}"

    if prefix is None:
        prefix = ""

    if mode == "pmap":
        eval_fun = jax.pmap(eval_fun, axis_name=axis_name)

        def eval_epoch(train_state, iterable, max_length=None):
            if replicate:
                train_state = jax_utils.replicate(train_state)

            accum_scalars = {}
            num_remainders = 0
            for batch, actual_size, is_remainder in _split_batch(iterable, max_length, prefetch):
                if is_remainder:
                    num_remainders += 1

                if num_remainders == 2:
                    warnings.warn(
                        """You use multi-GPUs to evaluate your model, but it seems that the batch size
                        is not divisble by the number of GPUs. This configuration also generates the
                        correct summary, but results in inefficient evaluation. You are recommended
                        to set batch size a mupltiple of number of GPUs for efficient evaluation."""
                    )

                if disallow_remainder and is_remainder:
                    raise RuntimeError("Batch size should be divisible by number of devices for evaluation.")

                scalars = eval_fun(train_state, batch)
                accum_scalars = _accumulate_scalars(accum_scalars, scalars, actual_size)

            summary = _summarize_scalars(prefix, accum_scalars)
            return summary

    else:
        if mode == "jit":
            warnings.warn(
                "JIT mode may leads very slow evaluation, because of bugs. Consider to use none or pmap mode, instead of jit."  # noqa
            )
            eval_fun = jax.jit(eval_fun)

        def eval_epoch(train_state, iterable, max_length=None):
            if max_length is not None:
                iterable = itertools.islice(iterable, max_length)

            accum_scalars = {}
            for batch in iterable:
                batch_size = min(map(len, tree_util.tree_leaves(batch)))
                scalars = eval_fun(train_state, batch)
                accum_scalars = _accumulate_scalars(accum_scalars, scalars, batch_size)

            summary = _summarize_scalars(prefix, accum_scalars)
            return summary

    return eval_epoch


class DynamicScale(struct.PyTreeNode):
    """Dynamic loss scaling for mixed precision gradients.

    A forked version of `flax.training.dynamic_scale.DynamicScale`.
    This implementation separates `gradient scaling` and `scaling factor update.`
    The former is applied by `value_and_grad` or `grad`, and the later is applied by
    `update`. See Example section for the specific example.

    Attributes:
        growth_factor
        backoff_factor
        growth_interval
        fin_steps
        scale
        minimum_scale

    Example:
        ::
            dyn_scale = DynamicScale()
            grad_fn = dyn_scale.grad(loss_fn)
            grads = grad_fn(params, ...)  # grads are scaled by `dyn_scale.scale`.
            new_dyn_scale, is_fin = dyn_scale.update(grads)  # update scale factor.
    """

    growth_factor: float = struct.field(pytree_node=False, default=2.0)
    backoff_factor: float = struct.field(pytree_node=False, default=0.5)
    growth_interval: int = struct.field(pytree_node=False, default=2000)
    fin_steps: chex.Array = 0
    scale: chex.Array = 65536.0
    minimum_scale: tp.Optional[float] = struct.field(pytree_node=False, default=jnp.finfo(jnp.float32).tiny)

    def value_and_grad(
        self,
        fun: tp.Callable[..., tp.Any],
        argnums: tp.Union[int, tp.Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> tp.Any:
        @functools.wraps(fun)
        def loss_wrapper(*args):
            aux = fun(*args)
            if has_aux:
                return (self.scale * aux[0], aux[1])
            else:
                return self.scale * aux

        grad_fn = jax.value_and_grad(loss_wrapper, argnums, has_aux)

        def grad_fn_wrapper(*args):
            aux, grad = grad_fn(*args)
            aux = (aux[0] / self.scale, aux[1]) if has_aux else aux / self.scale
            grad = jax.tree_util.tree_map(lambda g: jnp.asarray(g, jnp.float32) / self.scale, grad)
            return aux, grad

        return grad_fn_wrapper

    def grad(
        self,
        fun: tp.Callable[..., tp.Any],
        argnums: tp.Union[int, tp.Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> tp.Any:
        grad_fn = self.value_and_grad(fun, argnums, has_aux)

        def grad_fn_wrapper(*args):
            aux, grad = grad_fn(*args)
            if has_aux:
                return grad, aux[1]
            else:
                return grad

        return grad_fn_wrapper

    def update(self, grads: chex.ArrayTree) -> tuple[DynamicScale, chex.Array]:
        """Update `DynamicScale`.

        Args:
            grads: a tree that holds gradients.

        Returns:
            A tuple of new `DynamicScale` and a bool array that represents
            whether all grads are finite.
        """
        finite = jnp.array(True)
        for g in jax.tree_util.tree_leaves(grads):
            finite &= jnp.all(jax.lax.is_finite(g))

        grow = self.fin_steps == self.growth_interval
        fin_scale = jnp.where(
            grow & finite,
            jnp.minimum(self.scale * self.growth_factor, jnp.finfo(jnp.float32).max),
            self.scale,
        )
        inf_scale = self.scale * self.backoff_factor
        if self.minimum_scale is not None:
            inf_scale = jnp.maximum(inf_scale, self.minimum_scale)
        new_scale = jnp.where(finite, fin_scale, inf_scale)
        new_fin_steps = jnp.where(grow | (~finite), 0, self.fin_steps + 1)

        new_self = self.replace(fin_steps=new_fin_steps, scale=new_scale)
        return new_self, finite


def accumulate_gradients(
    grad_fun: tp.Callable,
    accumulates: int,
    rng_axis: tp.Optional[int] = 1,
    batch_axes: tp.Optional[int | tp.Sequence[int]] = 2,
    aggregator: tp.Optional[tp.Callable] = None,
) -> tp.Callable:
    """Transforms gradient function to accumulate gradients.

    Applies gradient accumulation. Unlike optax.MultiSteps, this method receives all examples to accumulate
    at the same time, split them into the `accumulates` chunks, and aggregate outputs of `grad_fun` from all chunks.
    This approach needs more RAM to have all batches, but useful to implement exact the same training when the
    gradient accumulation is not used. (e.g., compute batch stats, uptate EMA model, update loss scaler for mixed
    precision training...)

    Args:
        grad_fun: a callable that compute gradients.
        accumulates: number of steps to accumulate gradients.
        rng_axis: axis name that represents the PRNG key. If given, split the PRNG key every calls.
        batch_axes: list of axis name that represents batches.
        aggregator: a callable that receives stacked outputs of grad_fun, and returns the aggregated outputs.
            If not given, all outputs are averaged.

    Returns:
        Averaged outputs of grad_fun.

    NOTE:
        The transformed grad_fun only receives position-only arguments.

    Example:
        ::

    """
    if batch_axes is None:
        batch_axes = []
    elif isinstance(batch_axes, int):
        batch_axes = [batch_axes]

    if aggregator is None:
        aggregator = functools.partial(tree_util.tree_map, functools.partial(jnp.mean, axis=0))

    @functools.wraps(grad_fun)
    def wrapped(*args):
        def scan_fn(rng, batch):
            new_rng = None
            if rng is not None:
                rng, new_rng = jr.split(rng)

            _args = list(args)
            if rng_axis is not None:
                _args[rng_axis] = rng

            for axis, x in zip(batch_axes, batch):
                _args[axis] = x

            outputs = grad_fun(*_args)
            return new_rng, outputs

        rng = None
        if rng_axis is not None:
            rng = args[rng_axis]

        batches = [args[axis] for axis in batch_axes]
        batches = tree_util.tree_map(lambda arr: jnp.reshape(arr, (accumulates, -1, *jnp.shape(arr)[1:])), batches)
        _, outputs = jax.lax.scan(scan_fn, init=rng, xs=batches)
        outputs = aggregator(outputs)
        return outputs

    return wrapped

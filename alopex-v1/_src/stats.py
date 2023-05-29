from __future__ import annotations
import typing as tp
import math
import time

import jax
from jax import tree_util
import chex


def _convert_size(v: chex.Scalar, unit: str | None = None, base: int = 1000) -> chex.Scalar:
    units = [None, "K", "M", "G", "T", "P", "E", "Z"]
    unit = None if unit is None else unit.upper()
    assert unit in units, f"Invalid unit is specified. Use {units}."

    idx = units.index(unit)
    return v / math.pow(base, idx)


def flop(
    fun: tp.Callable,
    unit: str | None = None,
    static_argnums: int | tp.Iterable[int] = (),
    backend: str | None = None,
    donate_argnums: int | tp.Iterable[int] = (),
) -> tp.Callable[..., chex.Scalar]:
    """Creates a function that count floating operations (FLOPs) of fun.

    Args:
        fun: Function to count FLOPs.
        unit: Unit of FLOPs. None, K, M, G, T, P, E or Z.
        static_argnums: An optional int or collection of ints that specify which positional
            arguments to treat as static (compile-time constant).
        backend: A string representing the XLA backend. cpu, gpu or tpu.
        donate_argnums: Specify which positional argument buffers are "donated" to the computation.

    Returns:
        A wrapped version of fun.

    NOTE:
        Modify from
        https://github.com/google-research/scenic/blob/main/scenic/common_lib/debug_utils.py
    """

    def wrapped(*args, **kwargs) -> chex.Scalar:
        computation = jax.xla_computation(
            fun,
            static_argnums=static_argnums,
            backend=backend,
            donate_argnums=donate_argnums,
        )(*args, **kwargs)
        module = computation.as_hlo_module()
        client = jax.lib.xla_bridge.get_backend()
        analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, module)
        return _convert_size(analysis["flops"], unit)

    return wrapped


def mac(
    fun: tp.Callable,
    unit: str | None = None,
    static_argnums: int | tp.Iterable[int] = (),
    backend: str | None = None,
    donate_argnums: int | tp.Iterable[int] = (),
) -> tp.Callable[..., chex.Scalar]:
    """Creates a function that count multiply-accumulate operation (MACs) of fun.
        MAC is more commonly used than FLOP in literature.

    Args:
        fun: Function to count MACs.
        unit: Unit of MACs. None, K, M, G, T, P, E or Z.
        static_argnums: An optional int or collection of ints that specify which positional
            arguments to treat as static (compile-time constant).
        backend: A string representing the XLA backend. cpu, gpu or tpu.
        donate_argnums: Specify which positional argument buffers are "donated" to the computation.

    Returns:
        A wrapped version of fun.
    """

    def wrapped(*args, **kwargs) -> chex.Scalar:
        flops = flop(fun, unit, static_argnums, backend, donate_argnums)(*args, **kwargs)
        return flops / 2

    return wrapped


def memory_access(
    fun: tp.Callable,
    unit: str | None = None,
    static_argnums: int | tp.Iterable[int] = (),
    backend: str | None = None,
    donate_argnums: int | tp.Iterable[int] = (),
) -> tp.Callable[..., chex.Scalar]:
    """Creates a function that count the total memory access cost (bytes) of fun.

    Args:
        fun: Function to count memory access cost.
        unit: Unit of memory access cost. None, K, M, G, T, P, E or Z.
        static_argnums: An optional int or collection of ints that specify which positional
            arguments to treat as static (compile-time constant).
        backend: A string representing the XLA backend. cpu, gpu or tpu.
        donate_argnums: Specify which positional argument buffers are "donated" to the computation.

    Returns:
        A wrapped version of fun.
    """

    def wrapped(*args, **kwargs) -> chex.Scalar:
        computation = jax.xla_computation(
            fun,
            static_argnums=static_argnums,
            backend=backend,
            donate_argnums=donate_argnums,
        )(*args, **kwargs)
        module = computation.as_hlo_module()
        client = jax.lib.xla_bridge.get_backend()
        analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, module)
        return _convert_size(analysis["bytes accessed"], unit, base=1024)

    return wrapped


def latency(
    fun: tp.Callable,
    num_iters: int = 100,
    warmup_iters: int = 0,
) -> tp.Callable[..., float]:
    """Creates a function that computes average latency (sec) of fun.

    Args:
        fun: Function to time.
        num_iters: Number of iterations used to compute average runtime of fun.
        warmup_iters: Number of iterations used to warmup fun.

    Returns:
        A wrapped version of fun.
    """

    def call(*args, **kwargs) -> None:
        fun(*args, **kwargs)
        jax.random.uniform(jax.random.PRNGKey(0)).block_until_ready()

    def wrapped(*args, **kwargs):
        [call(*args, **kwargs) for _ in range(warmup_iters)]

        start_time = time.time()
        [call(*args, **kwargs) for _ in range(num_iters)]
        return (time.time() - start_time) / num_iters

    return wrapped


def count_params(tree: chex.ArrayTree, unit: str | None = None) -> chex.Scalar:
    """Count number of elements stored in PyTree.

    Args:
        tree: A PyTree to count elements.
        unit: Unit of number of parameters. None, K, M, G, T, P, E or Z.

    Returns:
        Number of elements stored in tree.
    """
    tree_size = sum([x.size for x in tree_util.tree_leaves(tree)])
    return _convert_size(tree_size, unit)

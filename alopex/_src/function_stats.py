from __future__ import annotations
import typing as tp
import time

import jax
import chex


def flop(
    fun: tp.Callable,
    static_argnums: int | tp.Iterable[int] = (),
    backend: str | None = None,
    donate_argnums: int | tp.Iterable[int] = (),
) -> tp.Callable[..., chex.Scalar]:
    """Creates a function that count floating operations (FLOPs) of fun.

    Args:
        fun: Function to count FLOPs.
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
        return analysis["flops"]

    return wrapped


def mac(
    fun: tp.Callable,
    static_argnums: int | tp.Iterable[int] = (),
    backend: str | None = None,
    donate_argnums: int | tp.Iterable[int] = (),
) -> tp.Callable[..., chex.Scalar]:
    """Creates a function that count multiply-accumulate operation (MACs) of fun.
        MAC is more commonly used than FLOP in literature.

    Args:
        fun: Function to count MACs.
        static_argnums: An optional int or collection of ints that specify which positional
            arguments to treat as static (compile-time constant).
        backend: A string representing the XLA backend. cpu, gpu or tpu.
        donate_argnums: Specify which positional argument buffers are "donated" to the computation.

    Returns:
        A wrapped version of fun.
    """

    def wrapped(*args, **kwargs) -> chex.Scalar:
        flops = flop(fun, static_argnums, backend, donate_argnums)(*args, **kwargs)
        return flops / 2

    return wrapped


def memory_access(
    fun: tp.Callable,
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
        return analysis["bytes accessed"]

    return wrapped


def timeit(
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

    def call(*args, n, **kwargs) -> None:
        for _ in range(n):
            fun(*args, **kwargs)
            jax.random.uniform(jax.random.PRNGKey(0)).block_until_ready()

    def wrapped(*args, **kwargs):
        call(*args, **kwargs, n=warmup_iters)

        start_time = time.time()
        call(*args, **kwargs, n=num_iters)
        return (time.time() - start_time) / num_iters

    return wrapped

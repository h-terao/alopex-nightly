"""Reimplementation of the harvest transformation from `oryx.core`"""
from __future__ import annotations
import typing as tp
from functools import wraps
from collections import deque
import threading

import jax
import chex

__all__ = ["sow", "sow_grad", "harvest", "plant", "call_and_reap", "reap"]

_thread_local = threading.local()


def _get_dynamic_context(name: str) -> dict:
    return getattr(_thread_local, name, dict())


def sow(value: chex.Array, *, col: str, name: str, mode: str = "strict", reverse: bool = False) -> chex.Array:
    """Marks a variable with a name and col.

    This method works as an identity function, but also marks a variable to be tagged and named.
    Harvest transformation collects or rewrites the marked variables.

    Args:
        value: Variable to be tagged and named.
        col: Collection name where value is sown.
        name: Name of value.
        mode: How to sow the value.
            If `strict`, the same pair of col and name is available once. If another value is sown with
            the same col and name, raise an error. If `clobber`, the same pair of col and name is available
            any number of times, and the latest value is kept. If `append`, all values sown with the same
            col and name are reaped as a list of objects.
        reverse: Reverse the output of harvest transformation.
            If mode is clobber, the oldest value is returned. If mode is append, the order of the reaped values
            is reversed; i.e., the first element becomes the latest value.

    Returns:
        The original value.
    """
    ctx_reaps = _get_dynamic_context("reaps")
    if col in ctx_reaps:
        if mode in ["strict", "clobber"]:
            if mode == "strict" and name in ctx_reaps[col]:
                msg = f"strict mode is specified but (col, name)=({col}, {name}) is already registered."
                raise RuntimeError(msg)
            ctx_reaps[col].setdefault(name, {})
            if not reverse or name not in ctx_reaps[col][name]:
                # If reverse, keep the first sow-ed value.
                ctx_reaps[col][name] = value

        elif mode == "append":
            ctx_reaps[col].setdefault(name, deque())
            if reverse:
                ctx_reaps[col][name].appendleft(value)
            else:
                ctx_reaps[col][name].append(value)
        else:
            raise ValueError(f"Unknown mode ({mode}) is specified.")

    ctx_plants = _get_dynamic_context("plants")
    if col in ctx_plants:
        if name in ctx_plants[col]:
            value = ctx_plants[col][name]

    return value


def sow_grad(
    value: chex.Array, col: str = "grads", *, name: str, mode: str = "strict", reverse: bool = False
) -> chex.Array:
    """Marks grads of a variable with a name and col.

    This method works as an identity function, but also marks grads of a variable to be tagged and named.
    Harvest transformation collects grads of the marked variables.

    Args:
        value: Variable to be tagged and named.
        col: Collection name where grad is sown.
        name: Name of value.
        mode: How to sow the grad of value.
            If `strict`, the same pair of col and name is available once. If another value is sown with
            the same col and name, raise an error. If `clobber`, the same pair of col and name is available
            any number of times, and the latest value is kept. If `append`, all values sown with the same
            col and name are reaped as a list of objects.
        reverse: Reverse the output of harvest transformation.
            If mode is clobber, the oldest value is returned. If mode is append, the order of the reaped values
            is reversed; i.e., the first element becomes the latest value.

    Returns:
        The original value.

    Notes:
        To reap grads, transform the grad function using reap method. If call sow_grad with append mode,
        the order of reaped grads is as same as the sowing order of variables, i.e., the first element of
        reaped grads come from the earliest sown value.


    Example::

        fun = lambda x, y: x + sow_grad(y, name="y")
        grad_fun = jax.grad(lambda x, y, t: jax.numpy.mean((fun(x, y) - t)**2))
        assert reap(grad_fun, col="grads")(1, 2, 3) = {"y": 0}
    """

    @jax.custom_vjp
    def identity(x):
        return x

    def forward(x):
        return x, ()

    def backward(_, g):
        g = sow(g, col=col, name=name, mode=mode, reverse=not reverse)
        return (g,)

    identity.defvjp(forward, backward)
    return identity(value)


def harvest(fun: tp.Callable, *, col: str) -> tp.Callable:
    """Creates a function that harvest sown values in fun.

    This is the core method of harvest transformation. `plant`, `call_and_reap` and `reap`
    are based on `harvest`.

    Args:
        fun: Function to transform.
        col: Name of the variable collection.

    Returns:
        A wraped version of fun.
    """

    def wrapped(plants: dict[str, tp.Any], *args, **kwargs):
        ctx_reaps = _thread_local.reaps = _get_dynamic_context("reaps")
        ctx_plants = _thread_local.plants = _get_dynamic_context("plants")

        if col in ctx_reaps or col in ctx_plants:
            raise RuntimeError(f"{col} is already used. Use different name.")

        ctx_reaps[col] = {}
        ctx_plants[col] = plants

        value = fun(*args, **kwargs)

        reaped = ctx_reaps.pop(col)
        ctx_plants.pop(col)

        # Deque -> List.
        reaped = {key: list(val) if isinstance(val, deque) else val for key, val in reaped.items()}
        return value, reaped

    return wrapped


def plant(fun: tp.Callable, *, col: str | tp.Sequence[str]) -> tp.Callable:
    """Creates a function that replaces sow-ed values in fun to the specified `plants`.

    Args:
        fun: Function to plant values.
        col: A name of the variable collection to plant values.

    Returns:
        A wrapped version of fun.

    Note:
        If you sow multiple variables with the same pair of (col, name) using
        clobber or append mode, all sown variables are replaced with the `plants`.

    Example::

        fun = lambda x, y: x + sow(y, col="values", name="y")
        assert fun(1, 2) == 3  # compute 1 + 2

        fun = plant(fun, col="values")
        assert fun({"y": 100}, 1, 2) == 101  # compute 1 + 101
    """

    def wrapped(plants: dict[str, chex.ArrayTree], *args, **kwargs):
        value, _ = harvest(fun, col=col)(plants, *args, **kwargs)
        return value

    return wrapped


def call_and_reap(fun: tp.Callable, *, col: str) -> tp.Callable:
    """Creates a function that returns outputs and collection of sown variables from fun.

    Args:
        fun: Function to collect the sown values.
        col: A name of the variable collection to collect values.

    Returns:
        A wrapped version of fun.

    Example::

        fun = lambda x, y: x + sow(y, col="values", name="y")
        fun = call_and_reap(fun, col="values")
        output, reaped = fun(1, 2)
        assert output == 3
        assert reaped == {"y": 2}
    """

    @wraps(fun)
    def wrapped(*args, **kwargs):
        return harvest(fun, col=col)({}, *args, **kwargs)

    return wrapped


def reap(fun: tp.Callable, *, col: str) -> tp.Callable:
    """Creates a function that returns the collection of sown variables from fun.

    Args:
        fun: Function to collect the sown values.
        col: A name of the variable collection to collect values.

    Returns:
        A wrapped version of fun.

    Example::

        fun = lambda x, y: x + sow(y, col="values", name="y")
        fun = call_and_reap(fun, col="values")
        reaped = fun(1, 2)
        assert reaped == {"y": 2}
    """

    @wraps(fun)
    def wrapped(*args, **kwargs):
        _, reaped = call_and_reap(fun, col=col)(*args, **kwargs)
        return reaped

    return wrapped

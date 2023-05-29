"""Configuration utility."""
from __future__ import annotations
import typing as tp
import functools
import contextlib
import threading

_local_config = threading.local()


def set_default_config(**configs) -> None:
    """Set default configuration values."""
    strict = configs.pop("strict", True)
    for name, value in configs.items():
        if hasattr(_local_config, name) and strict:
            msg = (
                f"{name} is already registered in configurations. "
                "In order to overwrite default values of already registered configurations, "
                "pass `strict=False` as configs."
            )
            raise RuntimeError(msg)
        setattr(_local_config, name, value)


@contextlib.contextmanager
def using_config(**configs) -> None:
    """Context manager to temporarily change the configuration.

    Args:
        **configs: Named values to configure the inner processing.
            Note that the below keys are reserved for some special behaviours.
        strict (bool): If True, raise RuntimeError if some keys are not configured yet.
            Default is True.
    """
    strict = configs.pop("strict", True)

    prev_config = {}
    for name, value in configs.items():
        if hasattr(_local_config, name):
            prev_config[name] = getattr(_local_config, name)
        elif strict:
            msg = (
                f"Unknown configuration {name} is found. "
                "In order to set any configurations, you must first set them with default values "
                "using `set_default_config`. To disable this error, pass `strict=False` as configs."
            )
            raise RuntimeError(msg)
        setattr(_local_config, name, value)

    try:
        yield

    finally:
        # Restore the previous local config
        for name in configs:
            if name in prev_config:
                setattr(_local_config, name, prev_config[name])
            else:
                delattr(_local_config, name)


def configure(fun: tp.Callable, **configs) -> tp.Callable:
    """Creates a function that configure `fun` using the specified arguments.
    In the configured function, configs are available via `get_config`.

    Args:
        fun: A function to transform.

    Returns:
        A wrapped version of `fun`.
    """

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        with using_config(**configs):
            return fun(*args, **kwargs)

    return wrapped


def get_config(name: str) -> tp.Any:
    """Access the configured value.

    Args:
        name: Configuration name to access.

    Returns:
        Configured value.
    """
    if hasattr(_local_config, name):
        return getattr(_local_config, name)
    else:
        raise KeyError(f"{name} is not configured.")

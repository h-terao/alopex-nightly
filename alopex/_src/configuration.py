"""Configuration utility."""
from __future__ import annotations
import typing as tp
import functools
import contextlib
import threading


_thread_local = threading.local()
_thread_local.global_configs = {}
_thread_local.local_configs = {}


class Unspecified:
    pass


unspecified = Unspecified()


@contextlib.contextmanager
def using_config(**configs):
    """

    Example:
        ::
        with using_config(is_training=True):
            assert get_config("is_training")
    """
    prev_local_configs = {}

    for key, value in configs.items():
        if key in _thread_local.local_configs:
            prev_local_configs[key] = _thread_local.local_configs[key]
        _thread_local.local_configs[key] = value

    yield

    for key, value in prev_local_configs.items():
        _thread_local.local_configs[key] = value


def configure(fun: tp.Callable, **configs) -> tp.Callable:
    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        with using_config(**configs):
            return fun(*args, **kwargs)

    return wrapped


def set_config(**configs) -> None:
    """Set global configuration.

    Args:
        _strict_: special argument. If true, already configured values are not allowed.

    Note:
        If this method is called after any values are locally configured by `configure` or `using_config`,
        the local configurations are used.
    """
    strict = configs.pop("_strict_", True)
    for key, value in configs.items():
        if strict:
            assert key not in _thread_local.global_configs
        _thread_local.global_configs[key] = value


def get_config(key: str, default=unspecified) -> tp.Any:
    configs = dict(_thread_local.global_configs, **_thread_local.local_configs)
    if key in configs:
        return configs[key]
    else:
        if isinstance(default, Unspecified):
            raise KeyError(f"{key} is not configured yet.")
        return default

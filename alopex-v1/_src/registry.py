from __future__ import annotations
import typing as tp
import functools


class Registry(dict):
    """A registry class for callable objects."""

    def register(self, name: str, fun_or_class: tp.Callable | None = None, **kwargs) -> tp.Callable:
        """Register a callable object in this registry.

        Args:
            name: Name of a callable object to register.
            fun_or_class: A callable object to register.
                If None, this method works as a decorator.
            **kwargs: Argument parameters to set using `functools.partial`.

        Raises:
            If name is already registered, raise an error.
        """
        if name in self:
            raise RuntimeError(f"{name} is already registered.")

        if fun_or_class is None:

            def deco(fun_or_class):
                self[name] = functools.partial(fun_or_class, **kwargs)
                return fun_or_class

            return deco
        else:
            fun_or_class = functools.partial(fun_or_class, **kwargs)
            self[name] = fun_or_class
            return fun_or_class


# Create a global registry for the simple usage.
registry = Registry()
register = registry.register

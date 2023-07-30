from __future__ import annotations
import typing as tp
import functools

__all__ = ["easydict", "Registry", "registry", "register", "add_prefix_to_dict"]


class easydict(dict):
    """dict class that supports dot access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class Registry(easydict):
    """
    Registry class that registers functions and classes with the specified name and default parameters.
    This is a subclass of easydict so that all dict methods are available.

    Examples:
        ::
        registry = Register()

        @registry.register("add")
        @registry.register("add1", y=1)
        def add(x, y):
            return x + y

        assert registry.add(2, 1) == 3
        assert registry.add1(2) == 3
    """

    def register(self, name: str, /, **defaults):
        if name in self:
            raise ValueError(f"{name} is already registered. Use different name to register function or class.")

        def deco(fun_or_class):
            if defaults:
                self[name] = functools.partial(fun_or_class, **defaults)
            else:
                self[name] = fun_or_class
            return fun_or_class

        return deco


# common register.
registry = Registry()
register = registry.register


def add_prefix_to_dict(inputs: tp.Mapping, prefix: str) -> tp.Mapping:
    cast_to = type(dict)
    output = {f"{prefix}{k}": v for k, v in inputs.items()}
    return cast_to(output)

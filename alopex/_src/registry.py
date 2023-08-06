import functools


class Registry(dict):
    """
    Registry class that registers functions and classes with the specified name and default parameters.

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

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

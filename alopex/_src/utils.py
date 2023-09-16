from __future__ import annotations
import typing as tp
import math

__all__ = ["easydict", "add_prefix_to_dict"]


class easydict(dict):
    """dict class that supports dot access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def add_prefix_to_dict(inputs: tp.Mapping, prefix: str) -> tp.Mapping:
    cast_to = type(inputs)
    outputs = {f"{prefix}{k}": v for k, v in inputs.items()}
    return cast_to(outputs)

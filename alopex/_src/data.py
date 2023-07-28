"""Dataset utilities."""
from __future__ import annotations
import typing as tp
import itertools
import math


def _cycle(iterable):
    while True:
        yield from iter(iterable)


def count_steps_per_epoch(total: int, batch_size: int, drop_remainder: bool = False) -> int:
    """Count epoch length from total number of examples and batch size.

    NOTE:
        This function counts epoch length in the PyTorch manner. It means that if drop_remainder=True,
        the last few examples are not used in the epoch and ignore them.
    """
    epoch_length = total / batch_size
    if not drop_remainder:
        epoch_length = math.ceil(epoch_length)
    return epoch_length


class DataLoader:
    """A convenient object that use tensorflow dataset with __len__ support."""

    def __init__(self, iterable: tp.Iterable, length: tp.Optional[int] = None):
        self.iterable = _cycle(iterable)
        self.length = length if length is not None else len(iterable)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> tp.Iterable:
        yield from itertools.islice(self.iterable, len(self))

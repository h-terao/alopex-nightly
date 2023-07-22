import typing as tp
import functools

from flax import linen
import chex


def call_norm_layer(norm_layer: linen.Module, x: chex.Array, is_training: bool = False) -> chex.Array:
    """Calling any types of norm layer with the same API.

    In Flax, BatchNorm needs `use_running_average` argument, while other normalization layers
    raise errors if the argument is given. This function

    Args:
        norm_layer: Instantiated normalization layer such as BatchNorm and LayerNorm.
        x: Input array.
        is_training:

    Returns:
        The output of `norm_layer`.
    """
    if isinstance(norm_layer, linen.BatchNorm):
        norm_layer = functools.partial(norm_layer, use_running_average=not is_training)
    return norm_layer(x)

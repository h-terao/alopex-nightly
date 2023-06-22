from __future__ import annotations
import typing as tp
import functools
import math

import jax.numpy as jnp
from flax import linen
import chex

ModuleDef = tp.Any


class SqueezeExcite(linen.Module):
    features: int
    rd_ratio: float = 1.0 / 16
    rd_features: tp.Optional[int] = None
    rd_divisor: int = 8
    use_bias: bool = True
    act_layer: ModuleDef = linen.relu
    norm_layer: ModuleDef = linen.BatchNorm
    gate_layer: ModuleDef = linen.sigmoid
    dtype: chex.ArrayDType = jnp.float32
    norm_dtype: chex.ArrayDType = jnp.float32
    param_dtype: chex.ArrayDType = jnp.float32
    axis_name: tp.Optional[str] = None

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False) -> chex.Array:
        conv_layer = functools.partial(
            linen.Conv,
            kernel_size=(1, 1),
            padding="VALID",
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        norm_layer = functools.partial(
            self.norm_layer,
            dtype=self.norm_dtype,
            param_dtype=self.param_dtype,
            axis_name=self.axis_name,
        )

        features = jnp.shape(x)[-1]
        rd_features = self.rd_features
        if rd_features is None:
            rd_features = 128  # TODO: make_divisible

        h = jnp.mean(x, axis=(-2, -3), keepdims=True)
        h = conv_layer(rd_features, name="fc1")(h)
        h = norm_layer(name="bn")(h)
        h = self.act_layer(name="act")(h)
        h = conv_layer(features, name="fc2")(h)
        h = x * self.gate_layer(name="gate")(h)
        return h

from __future__ import annotations
import typing as tp
from functools import partial

import jax.numpy as jnp
from flax import linen
import chex


ModuleDef = tp.Any


class BasicBlock(linen.Module):
    features: int
    stride: int = 1
    groups: int = 1
    base_width: int = 64
    expansion: int = 1
    dilation: int = 1

    conv: tp.Type[linen.Conv] = None
    norm: tp.Type[linen.BatchNorm] = None

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        # TODO: Add error message.
        assert self.groups == 1
        assert self.base_width == 64
        assert self.expansion == 1
        assert self.dilation == 1, "dilation > 1 is not supported in BasicBlock."

        h = self.conv(
            self.features,
            kernel_size=(3, 3),
            strides=self.stride,
            padding=self.dilation,
            kernel_dilation=self.dilation,
            feature_group_count=self.groups,
            name="conv1",
        )(x)
        h = self.norm(name="bn1")(h)
        h = linen.relu(h)

        h = self.conv(
            self.features,
            kernel_size=(3, 3),
            padding=self.dilation,
            kernel_dilation=self.dilation,
            feature_group_count=self.groups,
            name="conv2",
        )(h)
        h = self.norm(scale_init=linen.initializers.constant(0), name="bn2")(h)

        if x.shape != h.shape:
            x = self.conv(
                self.features,
                kernel_size=(1, 1),
                strides=self.stride,
                name="downsample.0",
            )(x)
            x = self.norm(name="downsample.1")(x)

        return linen.relu(x + h)


class Bottleneck(linen.Module):
    features: int
    stride: int = 1
    groups: int = 1
    base_width: int = 64
    expansion: int = 4
    dilation: int = 1

    conv: tp.Type[linen.Conv] = None
    norm: tp.Type[linen.BatchNorm] = None

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        width = int(self.features * self.base_width / 64.0) * self.groups

        h = self.conv(width, kernel_size=(1, 1), name="conv1")(x)
        h = self.norm(name="bn1")(h)
        h = linen.relu(h)

        h = self.conv(
            width,
            kernel_size=(3, 3),
            strides=self.stride,
            padding=self.dilation,
            kernel_dilation=self.dilation,
            feature_group_count=self.groups,
            name="conv2",
        )(h)
        h = self.norm(name="bn2")(h)
        h = linen.relu(h)

        h = self.conv(self.features * self.expansion, kernel_size=(1, 1), name="conv3")(h)
        h = self.norm(scale_init=linen.initializers.constant(0), name="bn3")(h)

        if x.shape != h.shape:
            x = self.conv(
                self.features * self.expansion,
                kernel_size=(1, 1),
                strides=self.stride,
                padding="VALID",
                name="downsample.0",
            )(x)
            x = self.norm(name="downsample.1")(x)

        return linen.relu(x + h)


class ResNet(linen.Module):
    block: tp.Type[BasicBlock | Bottleneck]
    stage_sizes: tp.Sequence[int]

    num_classes: int = 1000
    drop_rate: float = 0
    groups: int = 1
    width_per_group: int = 64
    replace_stride_with_dilation: tp.Sequence[bool] | None = None
    dtype: chex.ArrayDType = jnp.float32
    norm_dtype: chex.ArrayDType | None = None
    axis_name: str | None = None

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False) -> chex.Array:
        conv: tp.Type[linen.Conv] = partial(linen.Conv, padding="VALID", dtype=self.dtype, use_bias=False)
        norm: tp.Type[linen.BatchNorm] = partial(
            linen.BatchNorm,
            use_running_average=not is_training,
            dtype=self.norm_dtype or self.dtype,
            axis_name=self.axis_name,
        )

        replace_stride_with_dilation = self.replace_stride_with_dilation
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False] * len(self.stage_sizes)

        x = conv(64, (7, 7), strides=2, padding=3, use_bias=False, name="conv1")(x)
        x = norm(name="bn1")(x)
        x = linen.relu(x)
        x = linen.max_pool(x, (3, 3), (2, 2), ((1, 1), (1, 1)))

        dilation = 1
        for i, block_size in enumerate(self.stage_sizes):
            stride = 1 if i == 0 else 2
            if replace_stride_with_dilation[i]:
                dilation *= stride
                stride = 1

            for j in range(block_size):
                x = self.block(
                    64 * (2**i),
                    stride=stride if j == 0 else 1,
                    groups=self.groups,
                    base_width=self.width_per_group,
                    dilation=dilation,
                    conv=conv,
                    norm=norm,
                    name=f"layer{i+1}.{j}",
                )(x)

        x = jnp.mean(x, axis=(-2, -3))  # global average pooling.
        if self.num_classes > 0:
            x = linen.Dropout(self.drop_rate, deterministic=not is_training)(x)
            x = linen.Dense(self.num_classes, dtype=self.dtype, name="fc")(x)
        return x

    @property
    def rng_keys(self) -> tp.Sequence[str]:
        return ["dropout"]


def resnet18(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, **kwargs)


def resnext101_32x8d(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, **kwargs)


def resnext101_64x4d(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], groups=64, width_per_group=4, **kwargs)


def wide_resnet50_2(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=64 * 2, **kwargs)


def wide_resnet101_2(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], width_per_group=64 * 2, **kwargs)

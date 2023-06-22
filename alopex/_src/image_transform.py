from __future__ import annotations
import typing as tp
import functools

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import tree_util
import jax.scipy.ndimage as ndi
from einops import rearrange, repeat
import chex


def _pair(x):
    if hasattr(x, "__getitem__"):
        return tuple(x)
    return x, x


def affine_transform(
    inputs: chex.Array,
    matrix: chex.Array,
    *,
    size: tp.Optional[int | tuple[int, int]] = None,
    method: str = "linear",
    padding_mode: str = "nearest",
    cval: float = 0.0,
) -> chex.Array:
    """Apply an affine transformation given by matrix.

    Args:
        inputs: image array with shape (..., height, width, channels).
        matrix: the inverse coordinate transformation matrix with shape (3, 3).
        size: desired size of the output images. If None, (height, width) of `image` is used.
        method: the resizing method to use. (nearest | linear).
        padding_mode: padding mode for the outside of boundaries.
        cval: value to fill past edges.

    Returns:
        The transformed images.

    Notes:
        If multiple images are given, they are transformed given the same matrix.
        To transform images with different transformations, use `jax.vmap` or `jax.lax.scan`.
    """
    if method in ["nearest"]:
        order = 0
    elif method in ["linear", "bilinear", "trilinear"]:
        order = 1
    else:
        raise RuntimeError

    *batch_dims, height, width, channel = jnp.shape(inputs)
    inputs = rearrange(inputs, "... h w c -> (... c) h w")

    if size is None:
        size = (height, width)
    size = (*_pair(size), jnp.shape(inputs)[-1])

    x_t, y_t = jnp.meshgrid(
        jnp.linspace(0, width - 1, size[1]),
        jnp.linspace(0, height - 1, size[0]),
    )
    pixel_coords = jnp.stack([x_t, y_t, jnp.ones_like(x_t)])
    x_coords, y_coords, _ = jnp.einsum("ij,jkl->ikl", matrix, pixel_coords)
    coordinates = [
        repeat(jnp.arange(len(inputs)), "n -> n h w", h=size[0], w=size[1]),
        repeat(y_coords, "h w -> n h w", n=len(inputs)),
        repeat(x_coords, "h w -> n h w", n=len(inputs)),
    ]

    image = ndi.map_coordinates(inputs, jnp.stack(coordinates), order, padding_mode, cval)
    image = rearrange(image, "(b c) h w -> b h w c", c=channel)
    image = jnp.reshape(image, (*batch_dims, *jnp.shape(image)[1:]))
    return image


def flip_left_right(inputs: chex.Array) -> chex.Array:
    return inputs[..., :, ::-1, :]


def flip_up_down(inputs: chex.Array) -> chex.Array:
    return inputs[..., ::-1, :, :]


def resized_crop(
    inputs: chex.Array,
    top: int,
    left: int,
    height: int,
    width: int,
    size: int | tuple[int, int],
    method="linear",
    antialias: bool = True,
) -> chex.Array:
    *batch_dims, _, _, channel = jnp.shape(inputs)
    size = _pair(size)
    shape = jnp.array([*batch_dims, *size, channel])
    scale = jnp.array((size[0] / height, size[1] / width))
    translation = -scale * jnp.array((top, left))
    transformed = jax.image.scale_and_translate(
        inputs,
        shape=shape,
        spatial_dims=(-3, -2),
        scale=scale,
        translation=translation,
        method=method,
        antialias=antialias,
    )
    return transformed


def crop(inputs: chex.Array, top: int, left: int, height: int, width: int) -> chex.Array:
    *batch_dims, _, _, channel = jnp.shape(inputs)
    start_indices = [0] * len(batch_dims) + [top, left, 0]
    slice_sizes = [*batch_dims, height, width, channel]
    transformed = jax.lax.dynamic_slice(inputs, start_indices, slice_sizes)
    return transformed


def center_crop(inputs: chex.Array, *, size: int | tuple[int, int]) -> chex.Array:
    *_, img_h, img_w, _ = jnp.shape(inputs)
    height, width = _pair(size)
    top = int((img_h - height) // 2)
    left = int((img_w - width) // 2)
    return crop(inputs, top, left, height, width)


def three_crop(
    inputs: chex.Array,
    *,
    size: int | tuple[int, int],
    method: str = "linear",
    antialias: bool = True,
) -> tp.Sequence[chex.Array]:
    """Cropping three patches from images.

    Args:
        inputs: image array with shape (..., H, W, C).
        size: desired size of output images.
        method: interpolation method.
        antialias: apply antialias.

    Returns:
        A tuple of upper left, center and lower right patches.
    """
    *batch_dims, img_h, img_w, channel = jnp.shape(inputs)
    base_size = min(img_h, img_w)
    upper_left = crop(inputs, 0, 0, base_size, base_size)
    center = center_crop(inputs, size=base_size)
    lower_right = crop(inputs, img_h - base_size, img_w - base_size, base_size, base_size)
    output_shape = (*batch_dims, *_pair(size), channel)
    return tree_util.tree_map(
        lambda x: jax.image.resize(x, output_shape, method=method, antialias=antialias),
        (upper_left, center, lower_right),
    )


def five_crop(inputs: chex.Array, *, size: int | tuple[int, int]) -> tp.Sequence[chex.Array]:
    """Cropping five patches from images.

    Args:
        inputs: image array with shape (..., H, W, C).
        size: desired size of output images.

    Returns:
        A tuple of upper left, upper right, center, lower left and lower right patches.
    """
    *_, img_h, img_w, _ = jnp.shape(inputs)
    size = _pair(size)

    upper_left = crop(inputs, 0, 0, *size)
    upper_right = crop(inputs, 0, img_w - size[1], *size)
    center = center_crop(inputs, size=size)
    lower_left = crop(inputs, img_h - size[0], 0, *size)
    lower_right = crop(inputs, img_h - size[0], img_w - size[1], *size)
    return upper_left, upper_right, center, lower_left, lower_right


def ten_crop(inputs: chex.Array, *, size: int | tuple[int, int], vertical: bool = False) -> tp.Sequence[chex.Array]:
    """Cropping ten patches from images.

    Args:
        inputs: image array with shape (..., H, W, C).
        size: desired size of output images.
        vertical: if True, vertically flipping the cropped patches, instead of horizontal.

    Returns:
        A tuple of upper left, upper right, center, lower left and lower right patches,
        and flipped version of them.
    """
    cropped = five_crop(inputs, size=size)
    flipped = tree_util.tree_map(flip_up_down if vertical else flip_left_right, cropped)
    return (*cropped, *flipped)


def random_crop(key: chex.PRNGKey, inputs: chex.Array, *, size: int | tuple[int, int]) -> chex.Array:
    *_, img_h, img_w, _ = jnp.shape(inputs)
    height, width = _pair(size)
    top_key, left_key = jr.split(key)
    top = jr.randint(top_key, (), 0, img_h - height + 1)
    left = jr.randint(left_key, (), 0, img_w - width + 1)
    return crop(inputs, top, left, height, width)


def random_resized_crop(
    key: chex.PRNGKey,
    inputs: chex.Array,
    *,
    size: int | tuple[int, int],
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    method: str = "linear",
    antialias: bool = True,
) -> chex.Array:
    """Random resized crop.

    Args:
        key: pseudo-random generator.
        inputs: image array with shape (..., h, w, c).
        size: desired size.
        scale:
        ratio:
        method: interpolation method.
        antialias: antialias.
    """
    size = _pair(size)
    *_, img_h, img_w, _ = jnp.shape(inputs)

    area = img_h * img_w
    log_ratio = jnp.log(jnp.array(ratio))
    in_ratio = img_w / img_h
    crop_h = jnp.where(in_ratio < min(ratio), jnp.round(img_w / min(ratio)), img_h)
    crop_w = jnp.where(in_ratio > max(ratio), jnp.round(img_w / max(ratio)), img_w)
    top = (img_h - size[0]) // 2
    left = (img_w - size[1]) // 2

    for next_key in jr.split(key, 10):
        target_area_key, aspect_ratio_key, top_key, left_key = jr.split(next_key, 4)
        target_area = area * jr.uniform(target_area_key, (), minval=scale[0], maxval=scale[1])
        aspect_ratio = jr.uniform(aspect_ratio_key, (), minval=log_ratio[0], maxval=log_ratio[1])
        aspect_ratio = jnp.exp(aspect_ratio)

        w = jnp.round(jnp.sqrt(target_area * aspect_ratio))
        h = jnp.round(jnp.sqrt(target_area / aspect_ratio))
        i = jr.randint(top_key, (), 0, min(img_h - h + 1, 1))
        j = jr.randint(left_key, (), 0, min(img_w - w + 1, 1))

        condition = jnp.logical_and(0 < w < img_w, 0 < h < img_h)
        top = jnp.where(condition, i, top)
        left = jnp.where(condition, j, left)
        crop_h = jnp.where(condition, h, crop_h)
        crop_w = jnp.where(condition, w, crop_w)

    return resized_crop(inputs, top, left, crop_h, crop_w, size=size, method=method, antialias=antialias)


def rotate(
    inputs: chex.Array,
    *,
    angle: float,
    center: tp.Optional[float | tuple[float, float]],
    method: str = "linear",
    padding_mode: str = "nearest",
    cval: float = 0.0,
) -> chex.Array:
    # TODO: add argument support.
    if center is None:
        *_, img_h, img_w, _ = jnp.shape(inputs)
        center = ((img_h - 1) / 2, (img_w - 1) / 2)

    angle = angle * jnp.pi / 180
    sin, cos = jnp.sin(angle), jnp.cos(angle)

    center_y, center_x = center
    offset_x = center_x - center_x * cos + center_y * sin
    offset_y = center_y - center_x * sin - center_y * cos
    matrix = jnp.array([[cos, -sin, offset_x], [sin, cos, offset_y], [0, 0, 1]], dtype=inputs.dtype)
    return affine_transform(inputs, matrix, method=method, padding_mode=padding_mode, cval=cval)


def rot90(inputs: chex.Array, n: int) -> chex.Array:
    branches = [functools.partial(jnp.rot90, k=k, axes=(-3, -2)) for k in range(4)]
    transformed = jax.lax.switch(n % 4, branches, inputs)
    return transformed

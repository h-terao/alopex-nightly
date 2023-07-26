from __future__ import annotations
import typing as tp
import functools
import inspect

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


def _is_random_transform(fun: tp.Callable) -> bool:
    """Check the given is random operation or not.

    If the first argument name of `fun` is rng, key or rng_key or
    `fun` has two positional-only arguments, returns True.

    """

    if isinstance(fun, functools.partial):
        return _is_random_transform(fun.func)

    signature = inspect.signature(fun)
    random_keys = ["rng", "key", "rng_key"]
    for random_key in random_keys:
        if random_key in signature.parameters:
            return True

    positional_only_args = filter(lambda p: p.kind == p.POSITIONAL_ONLY, signature.parameters.values())
    if len(list(positional_only_args)) == 2:
        return True

    return False


# Utility.
def random_apply(transform: tp.Callable, rate: float = 0.5) -> tp.Callable:
    def wrapped(rng: chex.PRNGKey, img: chex.Array, /) -> chex.Array:
        f = transform
        if _is_random_transform(transform):
            rng, transform_rng = jr.split(rng)
            f = functools.partial(f, transform_rng)
        return jax.lax.cond(jr.uniform(rng) < rate, f, lambda x: x, img)

    return wrapped


def random_choice(*transforms, p: tp.Sequence[float] | None = None) -> tp.Callable:
    """Randomly choices one of the `transforms`."""
    n = len(transforms)
    if p is None:
        p = jnp.ones((n,))
    else:
        p = jnp.asarray(p)

    assert jnp.shape(p) == (n,)
    p = jnp.asarray(p) / sum(p)
    branches = [f if _is_random_transform(f) else lambda rng, img: f(img) for f in transforms]

    def wrapped(rng, img, /):
        choice_rng, rng = jr.split(rng)
        chosen = jr.choice(choice_rng, n, p=p)
        return jax.lax.switch(chosen, branches, rng, img)

    return wrapped


def chain(
    *transforms, axes: int | tp.Sequence[int] | None = None
) -> tp.Callable[[chex.PRNGKey | None, chex.Array], chex.Array]:
    """Creates a callable that sequentially transforms images.

    Args:
        transforms: list of transforms.
        axes: axis or list of axes to apply transformations with different PRNG keys.

    Returns:
        Transformed image.
    """
    if isinstance(axes, int):
        axes = (axes,)

    def wrapped(rng: chex.PRNGKey | None, img: chex.Array, /) -> chex.Array:
        for transform in transforms:
            if _is_random_transform(transform):
                if rng is None:
                    raise RuntimeError("Any stochastic transformations are given, but rng is None.")

                rng, next_rng = jr.split(rng)
                if axes is None:
                    img = transform(next_rng, img)
                else:
                    dim = jnp.size(next_rng, axis=-1)
                    sizes = jnp.asarray([jnp.size(img, axis=axis) for axis in axes])
                    next_rng = jnp.reshape(jr.split(next_rng, jnp.prod(sizes)), (*sizes, dim))
                    img = jax.vmap(transform, in_axes=axes, out_axes=axes)(next_rng, img)

            else:
                img = transform(img)
        return img

    return wrapped


# Color augmentation.
def grayscale(img: chex.Array, /) -> chex.Array:
    assert jnp.size(img, axis=-1) == 3
    red, green, blue = rearrange(img, "... c -> c ...")
    gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    return jnp.stack([gray, gray, gray], axis=-1)


def blend(x1: chex.Array, x2: chex.Array, factor: float) -> chex.Array:
    """Return factor * x1 + (1.-factor) * x2."""
    return jnp.clip(factor * (x1 - x2) + x2, 0, 1)


def solarize(img: chex.Array, /, threshold: float = 0.5) -> chex.Array:
    return jnp.where(img < threshold, img, 1 - img)


def solarize_add(img: chex.Array, /, threshold: float = 0.5, addition: float = 0) -> chex.Array:
    img = jnp.where(img < threshold, img + addition, img)
    return jnp.clip(img, 0, 1)


def adjust_color(img: chex.Array, /, color_factor: float) -> chex.Array:
    return blend(img, grayscale(img), color_factor)


def adjust_contrast(img: chex.Array, /, contrast_factor: float) -> chex.Array:
    degenerate = grayscale(img)
    degenerate = jnp.mean(degenerate, axis=(-1, -2, -3), keepdims=True)
    return blend(img, degenerate, contrast_factor)


def adjust_brightness(img: chex.Array, /, brightness_factor: float) -> chex.Array:
    degenerate = jnp.zeros_like(img)
    return blend(img, degenerate, brightness_factor)


def invert(img: chex.Array, /) -> chex.Array:
    return 1.0 - img


def posterize(img: chex.Array, /, bits: int) -> chex.Array:
    shift = 8 - bits
    degenerate = jnp.asarray(255 * img, dtype=jnp.uint8)
    degenerate = jnp.left_shift(jnp.right_shift(degenerate, shift), shift)
    return jnp.asarray(degenerate / 255, dtype=img.dtype)


def autocontrast(img: chex.Array, /) -> chex.Array:
    def scale_channel(_, xi):
        low = jnp.min(xi)
        high = jnp.max(xi)

        def _scale_values(v: chex.Array) -> chex.Array:
            scale = 1.0 / (high - low)
            offset = -low * scale
            v = v * scale + offset
            return jnp.clip(v, 0, 1)

        xi = jax.lax.cond(high > low, _scale_values, lambda v: v, xi)
        return None, xi

    *batch_dims, img_h, img_w, img_c = jnp.shape(img)
    img = rearrange(img, "... h w c -> (... c) h w")
    _, img = jax.lax.scan(scale_channel, None, img)
    img = jnp.reshape(img, (*batch_dims, img_c, img_h, img_w))
    return rearrange(img, "... c h w -> ... h w c")


def equalize(img: chex.Array, /) -> chex.Array:
    """Equalize images."""

    def build_lut(histo, step):
        lut = (jnp.cumsum(histo) + (step // 2)) // step
        lut = jnp.concatenate([jnp.array([0]), lut[:-1]], axis=0)
        return jnp.clip(lut, 0, 255)

    def scale_channel(carry, xi):
        new_xi = (xi * 255).astype(jnp.int32)
        histo = jnp.histogram(new_xi, bins=255, range=(0, 255))[0]
        last_nonzero = jnp.argmax(histo[::-1] > 0)
        step = (jnp.sum(histo) - jnp.take(histo[::-1], last_nonzero)) // 255

        new_xi = jax.lax.cond(
            step == 0,
            lambda x: x.astype("uint8"),
            lambda x: jnp.take(build_lut(histo, step), x).astype("uint8"),
            new_xi,
        )

        new_xi = (new_xi / 255).astype(xi.dtype)
        return carry, new_xi

    *batch_dims, img_h, img_w, img_c = jnp.shape(img)
    img = rearrange(img, "... h w c -> (... c) h w")
    _, img = jax.lax.scan(scale_channel, None, img)
    img = jnp.reshape(img, (*batch_dims, img_c, img_h, img_w))
    return rearrange(img, "... c h w -> ... h w c")


# Filter.
def convolve(
    img: chex.Array, kernel: chex.Array | tp.Callable, kernel_size: int | tuple[int, int] | None = None
) -> chex.Array:
    *batch_dims, img_h, img_w, img_c = jnp.shape(img)
    img = rearrange(img, "... h w c -> (... c) h w")

    if isinstance(kernel, chex.Array):
        kernel_size = jnp.shape(kernel)
    else:
        kernel_size = _pair(kernel_size)

    col = jax.lax.conv_general_dilated_patches(
        rearrange(img, "n h w c -> n c h w"),
        kernel_size,
        window_strides=(1, 1),
        padding="SAME",
    )

    col = rearrange(col, "n (c kh kw) h w -> (n h w c) kh kw", kh=kernel_size[0], kw=kernel_size[1])
    if isinstance(kernel, chex.Array):
        img = jnp.sum(col * kernel, axis=(-1, -2))
    else:
        img = kernel(col)
    img = jnp.reshape(img, (*batch_dims, img_h, img_w, img_c))
    return img


def sharpness(img: chex.Array, /, sharpness_factor: float) -> chex.Array:
    kernel = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1],
    ]
    kernel = jnp.array(kernel) / 13.0
    degenerate = convolve(img, kernel)
    return blend(degenerate, img, sharpness_factor)


def mean_blur(img: chex.Array, /, blur_factor: float = 1.0, kernel_size: int | tuple[int, int] = 9) -> chex.Array:
    degenerate = convolve(img, kernel=functools.partial(jnp.mean, axis=(1, 2)), kernel_size=kernel_size)
    return blend(degenerate, img, blur_factor)


def median_blur(img: chex.Array, /, blur_factor: float = 1.0, kernel_size: int | tuple[int, int] = 9) -> chex.Array:
    degenerate = convolve(img, kernel=functools.partial(jnp.median, axis=(1, 2)), kernel_size=kernel_size)
    return blend(degenerate, img, blur_factor)


# Affine transformation.
def affine_transform(
    img: chex.Array,
    matrix: chex.Array,
    /,
    size: tp.Optional[int | tuple[int, int]] = None,
    method: str = "linear",
    padding_mode: str = "nearest",
    cval: float = 0.0,
) -> chex.Array:
    """Apply an affine transformation given by matrix.

    Args:
        img: image array with shape (..., height, width, channels).
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

    *batch_dims, img_h, img_w, img_c = jnp.shape(img)
    img = rearrange(img, "... h w c -> (... c) h w")

    if size is None:
        size = (img_h, img_w)
    size = _pair(size)

    x_t, y_t = jnp.meshgrid(jnp.linspace(0, img_w - 1, size[1]), jnp.linspace(0, img_h - 1, size[0]))
    pixel_coords = jnp.stack([x_t, y_t, jnp.ones_like(x_t)])
    x_coords, y_coords, _ = jnp.einsum("ij,jkl->jkl", matrix, pixel_coords)

    coordinates = [
        repeat(jnp.arange(len(img)), "n -> n h w", h=size[0], w=size[1]),
        repeat(y_coords, "h w -> n h w", n=len(img)),
        repeat(x_coords, "h w -> n h w", n=len(img)),
    ]

    img = ndi.map_coordinates(img, jnp.stack(coordinates), order=order, mode=padding_mode, cval=cval)
    img = rearrange(img, "(b c) h w -> b h w c", c=img_c)
    img = jnp.reshape(img, (*batch_dims, size[0], size[1], img_c))
    return img


def translate(
    img: chex.Array,
    /,
    translates: tuple[float, float],
    method: str = "linear",
    padding_mode: str = "nearest",
    cval: float = 0,
) -> chex.Array:
    """Translates image.

    Args:
        img: image array.
        translate: tuple of (x-axis shift, y-axis shift).
    """
    matrix = [
        [1, 0, -translates[1]],
        [0, 1, -translates[0]],
        [0, 0, 1],
    ]
    return affine_transform(img, jnp.array(matrix), method=method, padding_mode=padding_mode, cval=cval)


def shear(
    img: chex.Array,
    /,
    angles: tuple[float, float] = (0, 0),
    method: str = "linear",
    padding_mode: str = "nearest",
    cval: float = 0,
) -> chex.Array:
    """Shear image.

    Args:
        img: image array.
        angles: tuple of x-axis and y-axis angles.
    """
    angle_y, angle_x = angles
    angle_x = angle_x * jnp.pi / 180
    angle_y = angle_y * jnp.pi / 180
    matrix = [
        [1, jnp.tan(angle_x), 0],
        [jnp.tan(angle_y), 1, 0],
        [0, 0, 1],
    ]
    return affine_transform(img, jnp.array(matrix), method=method, padding_mode=padding_mode, cval=cval)


def rotate(
    img: chex.Array,
    /,
    angle: float = 0,
    center: tuple[float, float] | None = None,
    method: str = "linear",
    padding_mode: str = "nearest",
    cval: float = 0,
) -> chex.Array:
    if center is None:
        *_, img_h, img_w, _ = jnp.shape(img)
        center = (int(round(img_h - 1) / 2), int(round(img_w - 1) / 2))

    center_y, center_x = center
    angle = angle * jnp.pi / 180
    shift_x = center_x - center_x * jnp.cos(angle) + center_y * jnp.sin(angle)
    shift_y = center_y - center_x * jnp.sin(angle) - center_y * jnp.cos(angle)
    matrix = [
        [jnp.cos(angle), -jnp.sin(angle), shift_x],
        [jnp.sin(angle), jnp.cos(angle), shift_y],
        [0, 0, 1],
    ]
    return affine_transform(img, jnp.array(matrix), method=method, padding_mode=padding_mode, cval=cval)


def rot90(img: chex.Array, /, n: int = 0) -> chex.Array:
    branches = [functools.partial(jnp.rot90, k=k, axes=(-3, -2)) for k in range(4)]
    return jax.lax.switch(n % 4, branches, img)


def flip_left_right(img: chex.Array, /) -> chex.Array:
    return img[..., :, ::-1, :]


def flip_up_down(img: chex.Array, /) -> chex.Array:
    return img[..., ::-1, :, :]


def crop(img: chex.Array, top: int, left: int, height: int, width: int) -> chex.Array:
    *batch_dims, _, _, img_c = jnp.shape(img)
    start_indices = [0] * len(batch_dims) + [top, left, 0]
    slice_sizes = [*batch_dims, height, width, img_c]
    return jax.lax.dynamic_slice(img, start_indices, slice_sizes)


def resized_crop(
    img: chex.Array,
    top: int,
    left: int,
    height: int,
    width: int,
    size: int | tuple[int, int],
    method: str = "linear",
    antialias: bool = True,
) -> chex.Array:
    """
    Args:
        img: image array.
        top:
        left:
        height:
        width:
        size: desired size of the cropped image.
        method: interpolation method.
        antialias:
    """
    *batch_dims, _, _, img_c = jnp.shape(img)
    size = _pair(size)
    shape = jnp.array([*batch_dims, *size, img_c])
    scale = jnp.array((size[0] / height, size[1] / width))
    translation = -scale * jnp.array((top, left))
    return jax.image.scale_and_translate(
        img,
        shape=shape,
        spatial_dims=(-3, -2),
        scale=scale,
        translation=translation,
        method=method,
        antialias=antialias,
    )


def center_crop(img: chex.Array, /, size: int | tuple[int, int]) -> chex.Array:
    *_, img_h, img_w, _ = jnp.shape(img)
    height, width = _pair(size)
    top = int(round((img_h - height) / 2))
    left = int(round((img_w - width) // 2))
    return crop(img, top, left, height, width)


def three_crop(
    img: chex.Array, /, size: int | tuple[int, int], method: str = "linear", antialias: bool = True
) -> tp.Sequence[chex.Array]:
    """
    Returns:
        a tuple of three cropped patches.
    """
    *_, img_h, img_w, _ = jnp.shape(img)
    base_size = min(img_h, img_w)
    f = functools.partial(
        resized_crop, height=base_size, width=base_size, size=size, method=method, antialias=antialias
    )
    upper_left = f(img, top=0, left=0)
    center = f(img, top=int(round((img_h - base_size) / 2)), left=int(round((img_w - base_size) / 2)))
    lower_right = f(img, top=img_h - base_size, left=img_w - base_size)
    return (upper_left, center, lower_right)


def five_crop(img: chex.Array, /, size: int | tuple[int, int]) -> tp.Sequence[chex.Array]:
    """
    Returns:
        a tuple of three cropped patches.
    """
    *_, img_h, img_w, _ = jnp.shape(img)
    size = _pair(size)
    f = functools.partial(crop, height=size[0], width=size[1])
    upper_left = f(img, top=0, left=0)
    upper_right = f(img, top=0, left=img_w - size[1])
    center = center_crop(img, size=size)
    lower_left = f(img, top=img_h - size[0], left=0)
    lower_right = f(img, top=img_h - size[0], left=img_w - size[1])
    return (upper_left, upper_right, center, lower_left, lower_right)


def ten_crop(img: chex.Array, /, size: int | tuple[int, int], vertical: bool = False) -> tp.Sequence[chex.Array]:
    cropped = five_crop(img, size=size)
    flipped = tree_util.tree_map(flip_up_down if vertical else flip_left_right, cropped)
    return tuple(*cropped, *flipped)


def random_flip_left_right(rng: chex.PRNGKey, img: chex.Array, /, rate: float = 0.5) -> chex.Array:
    return random_apply(flip_left_right, rate=rate)(rng, img)


def random_flip_up_down(rng: chex.PRNGKey, img: chex.Array, /, rate: float = 0.5) -> chex.Array:
    return random_apply(flip_up_down, rate=rate)(rng, img)


def random_crop(rng: chex.PRNGKey, img: chex.Array, /, size: int | tuple[int, int]) -> chex.Array:
    *_, img_h, img_w, _ = jnp.shape(img)
    height, width = _pair(size)
    top_rng, left_rng = jr.split(rng)
    top = jr.randint(top_rng, (), 0, img_h - height + 1)
    left = jr.randint(left_rng, (), 0, img_w - width + 1)
    return crop(img, top, left, height, width)


def random_resized_crop(
    rng: chex.PRNGKey,
    img: chex.Array,
    /,
    size: int | tuple[int, int],
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    method: str = "linear",
    antialias: bool = True,
) -> chex.Array:
    """Random resized crop.

    Args:
        rng: pseudo-random generator.
        img: image array with shape (..., h, w, c).
        size: desired size.
        scale:
        ratio:
        method: interpolation method.
        antialias: antialias.
    """
    size = _pair(size)
    *_, img_h, img_w, _ = jnp.shape(img)

    area = img_h * img_w
    log_ratio = jnp.log(jnp.array(ratio))
    in_ratio = img_w / img_h
    crop_h = jnp.where(in_ratio < min(ratio), jnp.round(img_w / min(ratio)), img_h)
    crop_w = jnp.where(in_ratio > max(ratio), jnp.round(img_w / max(ratio)), img_w)
    top = (img_h - size[0]) // 2
    left = (img_w - size[1]) // 2

    # Follow PyTorch implementation, but always loop 10 times to avoid errors.
    for next_rng in jr.split(rng, 10):
        target_area_rng, aspect_ratio_rng, top_rng, left_rng = jr.split(next_rng, 4)
        target_area = area * jr.uniform(target_area_rng, (), minval=scale[0], maxval=scale[1])
        aspect_ratio = jr.uniform(aspect_ratio_rng, (), minval=log_ratio[0], maxval=log_ratio[1])
        aspect_ratio = jnp.exp(aspect_ratio)

        w = jnp.round(jnp.sqrt(target_area * aspect_ratio))
        h = jnp.round(jnp.sqrt(target_area / aspect_ratio))
        i = jr.randint(top_rng, (), 0, jnp.clip(img_h - h + 1, 1, None))
        j = jr.randint(left_rng, (), 0, jnp.clip(img_w - w + 1, 1, None))

        top, left, crop_h, crop_w = tree_util.tree_map(
            functools.partial(jnp.where, jnp.logical_and(0 < w < img_w, 0 < h < img_h)),
            (i, j, h, w),
            (top, left, crop_h, crop_w),
        )

    return resized_crop(img, top, left, crop_h, crop_w, size=size, method=method, antialias=antialias)


def random_cutout_mask(rng: chex.PRNGKey, img: chex.Array, mask_size: int | tuple[int, int]):
    """Creates a boolean mask for cutout and cutmix."""
    mask_size = _pair(mask_size)
    half_mask_size = tuple(map(lambda x: int(round(x / 2)), mask_size))

    *_, img_h, img_w, _ = jnp.shape(img)
    mask = jnp.ones((img_h + mask_size[0], img_w + mask_size[1]))

    y_rng, x_rng = jr.split(rng)
    start_indices = [jr.randint(y_rng, (), 0, img_h + 1), jr.randint(x_rng, (), 0, img_w + 1)]
    mask = jax.lax.dynamic_update_slice(mask, update=jnp.zeros(mask_size), start_indices=start_indices)

    mask = jax.lax.dynamic_slice(mask, half_mask_size, (img_h, img_w))
    return jnp.reshape(mask, (img_h, img_w, 1))


def random_cutout(
    rng: chex.PRNGKey, img: chex.Array, /, mask_size: int | tuple[int, int], cval: float = 0.5
) -> chex.Array:
    mask = random_cutout_mask(rng, img, mask_size=mask_size)
    return jnp.where(mask, img, jnp.full_like(img, cval))

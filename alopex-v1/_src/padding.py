from __future__ import annotations
import typing as tp


def make_padding(
    kernel_size: int | tp.Sequence[int],
    stride: int | tp.Sequence[int] = 1,
    dilation: int | tp.Sequence[int] = 1,
    num_spatial_dims: int | None = None,
) -> tp.Sequence[tuple[int, int]]:
    """Creates a PyTorch-like padding parameter from kernel_size.

    Args:
        kernel_size: Kernel size of convolution.
        stride: Stride parameter.
        dilation: Dilation parameter.
        num_spatial_dims: Number of kernel dimensions. This argument is required when
            kernel_size is an integer. If kernel_size is a sequence of int, this argument
            is ignored and len(kernel_size) is used as num_spatial_dims.

    Returns:
        Padding parameter.
    """
    if isinstance(kernel_size, int):
        msg = "If kernel_size is an integer, specify num_spatial_dims."
        assert num_spatial_dims is not None, msg
        kernel_size = [kernel_size] * num_spatial_dims

    num_spatial_dims = num_spatial_dims or len(kernel_size)
    assert len(kernel_size) == num_spatial_dims

    if isinstance(stride, int):
        stride = (stride,) * num_spatial_dims
    assert len(stride) == num_spatial_dims

    if isinstance(dilation, int):
        dilation = (dilation,) * num_spatial_dims
    assert len(dilation) == num_spatial_dims

    padding = []
    for k, s, d in zip(kernel_size, stride, dilation):
        pad_size = ((s - 1) + d * (k - 1)) // 2
        padding.append((pad_size, pad_size))

    return padding

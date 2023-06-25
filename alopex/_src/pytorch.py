"""PyTorch utilities."""

from __future__ import annotations
import typing as tp
import warnings
from collections import defaultdict

import numpy as np
from jax import tree_util
import chex
from einops import rearrange

import torch
import torch.nn as nn


__all__ = ["assert_allclose_array_tensor", "register_torch_module", "convert_torch_model"]
REGISTRY = {}


def assert_allclose_array_tensor(
    array: chex.Array,
    tensor: torch.Tensor,
    array_format: str = "...",
    torch_format: str = "...",
    *,
    rtol: float = 0,
    atol: float = 1e-5,
    err_msg: tp.Optional[str] = None,
    verbose: tp.Optional[bool] = None,
) -> None:
    """Assert whether JAX array and PyTorch tensor are close.

        This function validates whether JAX array and PyTorch tensor are
        close or not, and useful for debugging of re-implementation
        from PyTorch to JAX and JAX to PyTorch.

    Args:
        array: JAX array.
        tensor: desired PyTorch tensor.
        array_format: format of JAX array.
        torch_format: format of PyTorch tensor. If `torch_format` is not equal
            to `array_format`, the dimensions of `tensor` are automatically
            moved according to `array_format`.
        rtol, atol, err_msg, verbose: corresponding to arguments of
            `numpy.testing.assert_allclose.`

    Returns:
        None

    Raises:
        AssertionError: raised when some values are not close
            between JAX array and PyTorch tensor.
    """
    tensor = rearrange(tensor, f"{torch_format} -> {array_format}")
    tensor = tensor.detach().cpu().numpy()  # Torch -> NumPy.
    np.testing.assert_allclose(
        array,
        tensor,
        rtol=rtol,
        atol=atol,
        err_msg=err_msg,
        verbose=verbose,
    )


def register_torch_module(*torch_modules):
    def deco(f):
        for m in torch_modules:
            REGISTRY[m] = f
        return f

    return deco


def convert_torch_model(torch_model: nn.Module, buffer_col: str = "buffers") -> dict:
    """Convert the parameters contained in the PyTorch model
        into a format that can be loaded by `utils.load_variables`.

    Args:
        torch_model: PyTorch model.
        buffer_col: column name of unknown buffers.

    Returns:
        A dict that can be loaded by `utils.load_variables`.
    """

    def convert_tensor(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        else:
            return v

    for module_type, convert in REGISTRY.items():
        if isinstance(torch_model, module_type):
            state_dict = tree_util.tree_map(convert_tensor, torch_model.state_dict())
            variables, named_modules = convert(torch_model, state_dict)
            break
    else:
        variables = defaultdict(dict)
        named_modules = torch_model._modules

    # NOTE:
    # Store unconverted parameters and buffers in variables to support the simply added
    # parameters and buffers (e.g., `class_token` in transformers.) Currently, converted or
    # unconverted is detected by comparing the converted names, so that duplicate parameters and
    # buffers are also stored in variables.
    for name, param in torch_model.named_parameters(recurse=False):
        if name not in variables["params"]:
            variables["params"][name] = convert_tensor(param)

    variables.setdefault(buffer_col, dict())
    for name, buffer in torch_model.named_buffers(recurse=False):
        if name not in variables[buffer_col]:
            variables[buffer_col][name] = convert_tensor(buffer)

    for name, module in named_modules.items():
        child_variables = convert_torch_model(module)
        for col, arrays in child_variables.items():
            assert name not in variables[col]
            variables[col][name] = arrays

    return dict(variables)


#
#  Register PyTorch modules for the conversion.
#
@register_torch_module(nn.Linear)
def convert_linear(m, state_dict):
    params = {"kernel": rearrange(state_dict["weight"], "outC inC -> inC outC")}
    if "bias" in state_dict:
        params["bias"] = state_dict["bias"]
    return {"params": params}, {}


@register_torch_module(nn.Conv1d, nn.Conv2d, nn.Conv3d)
def convert_conv(m, state_dict):
    params = {"kernel": rearrange(state_dict["weight"], "outC inC ... -> ... inC outC")}
    if "bias" in state_dict:
        params["bias"] = state_dict["bias"]
    return {"params": params}, {}


@register_torch_module(nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
def convert_conv_transpose(m, state_dict):
    msg = (
        "`ConvTransposeNd` layer is found in the PyTorch model. "
        "The flax model can output different values than expected "
        "because `ConvTranspose` implemented in Flax is not compatible with that of PyTorch. "
        "See details in https://flax.readthedocs.io/en/latest/guides/convert_pytorch_to_flax.html"
    )

    warnings.warn(msg)
    params = {"kernel": rearrange(state_dict["weight"], "outC inC ... -> ... inC outC")}
    if "bias" in state_dict:
        params["bias"] = state_dict["bias"]
    return {"params": params}, {}


@register_torch_module(nn.Embedding)
def convert_embedding(m, state_dict):
    params = {"embedding": state_dict["weight"]}
    return {"params": params}, {}


@register_torch_module(nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
def convert_batch_norm(m, state_dict):
    variables = defaultdict(dict)
    if "weight" in state_dict:
        variables["params"]["scale"] = state_dict["weight"]
    if "bias" in state_dict:
        variables["params"]["bias"] = state_dict["bias"]
    if "running_mean" in state_dict:
        variables["batch_stats"]["mean"] = state_dict["running_mean"]
    if "running_var" in state_dict:
        variables["batch_stats"]["var"] = state_dict["running_var"]
    return dict(variables), {}


@register_torch_module(nn.LayerNorm)
def convert_layer_norm(m, state_dict):
    params = {}
    if "weight" in state_dict:
        params["scale"] = state_dict["weight"]
    if "bias" in state_dict:
        params["bias"] = state_dict["bias"]
    return {"params": params}, {}

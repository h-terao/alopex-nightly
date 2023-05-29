"""PyTorch utilities."""

from __future__ import annotations
import warnings
from collections import defaultdict

from jax import tree_util
from einops import rearrange

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = ["register_torch_module", "convert_torch_model"]
REGISTRY = {}


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
    if not TORCH_AVAILABLE:
        msg = "PyTorch is not available. Install `torch` to use the `convert_torch_model` method"
        raise RuntimeError(msg)

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

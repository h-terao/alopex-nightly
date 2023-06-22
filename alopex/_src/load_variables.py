"""Utilities to test."""

from __future__ import annotations
import typing as tp
import warnings

from flax import core
from flax import traverse_util
import chex


def load_variables(
    variables: chex.ArrayTree,
    to_load: chex.ArrayTree,
    module_name: tp.Optional[str] = None,
) -> tuple[chex.ArrayTree, chex.ArrayTree]:
    """Load variables as much as possible.

    Args:
        variables: initialized variables to be overwritten.
        to_load: variables to be loaded.
        module_name: module name for partial loading.

    Returns:
        A tuple of overwritten variables and mask.
        The mask is a PyTree with the same structure of variables,
        and all leaves are boolean, where True means the element is
        loaded correctly.

    """

    def load(variables, to_load):
        to_load = traverse_util.flatten_dict(to_load, sep=".")

        new_variables, new_mask = {}, {}
        for keys, array in traverse_util.flatten_dict(variables).items():
            join_name = ".".join(keys)
            if join_name not in to_load:
                warnings.warn(f"{join_name} is not found in PyTorch model.")
            elif array.shape != to_load[join_name].shape:
                msg = f"Mismatch the shape of {join_name}. " f"({array.shape} != {to_load[join_name].shape})."
                warnings.warn(msg)
                new_mask[keys] = False
            else:
                array = to_load[join_name]
                new_mask[keys] = True
            new_variables[keys] = array

        new_variables = traverse_util.unflatten_dict(new_variables)
        new_mask = traverse_util.unflatten_dict(new_mask)

        return new_variables, new_mask

    cast_to = type(variables)
    if isinstance(variables, core.FrozenDict):
        variables = variables.unfreeze()

    if module_name is None:
        new_variables, new_mask = load(variables, to_load)
    else:
        if module_name[-1] == ".":
            module_name = module_name[:-1]

        tmp_dict = {}
        for col, arrays in variables.items():
            tmp_dict.setdefault(col, dict())
            flatten_arrays = traverse_util.flatten_dict(arrays, sep=".")
            for key, array in flatten_arrays.items():
                if key.startswith(module_name):
                    tmp_dict[col][key.removeprefix(module_name + ".")] = array

        tmp_dict, tmp_mask = load(tmp_dict, to_load)

        new_variables, new_mask = {}, {}
        for col, arrays in variables.items():
            flatten_arrays = traverse_util.flatten_dict(arrays)
            for keys, array in flatten_arrays.items():
                key, name = (col, *keys), ".".join(keys)
                if name.startswith(module_name):
                    tmp_name = name.removeprefix(module_name + ".")
                    new_variables[key] = tmp_dict[col][tmp_name]
                    new_mask[key] = tmp_mask[col][tmp_name]
                else:
                    new_variables[key] = array
                    new_mask[key] = False  # unoverwritten array is False.

        new_variables = traverse_util.unflatten_dict(new_variables)

    new_variables, new_mask = map(cast_to, (new_variables, new_mask))
    return new_variables, new_mask

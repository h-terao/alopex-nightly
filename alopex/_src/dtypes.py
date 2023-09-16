import typing as tp
import jax
import jax.numpy as jnp
import chex


class DTypes(tp.NamedTuple):
    dtype: chex.ArrayDType
    norm_dtype: chex.ArrayDType
    param_dtype: chex.ArrayDType


def get_dtypes(precision: str | int = "fp32") -> DTypes:
    """Returns dtypes for computation, normalization and parameters.

    Args:
        precision:

    Returns:
        A namedtuple of (dtype, norm_dtype, param_dtype).
    """
    if jax.default_backend() == "tpu" and precision == 16:
        precision = "bf16"

    param_dtype = jnp.float32
    if precision in ["float32", "fp32", 32]:
        dtype = norm_dtype = jnp.float32
    elif precision in ["float16", "fp16", 16]:
        dtype = jnp.float16
        norm_dtype = jnp.float32
    elif precision in ["bfloat16", "bf16"]:
        dtype = norm_dtype = jnp.bfloat16
    else:
        raise ValueError(f"Unknown precision {precision} is specified.")

    return DTypes(dtype=dtype, norm_dtype=norm_dtype, param_dtype=param_dtype)

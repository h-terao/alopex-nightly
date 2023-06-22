import typing as tp
import jax
import jax.numpy as jnp
import chex


class DTypes(tp.NamedTuple):
    dtype: chex.ArrayDType
    norm_dtype: chex.ArrayDType
    param_dtype: chex.ArrayDType


def get_dtypes(precision: str | int = "fp32") -> DTypes:
    """Get dtypes for computation, normalization and parameters.

    Args:
        precision: determine precision.

    Returns:
        A namedtuple of (dtype, norm_dtype, param_dtype).
    """
    if isinstance(precision, int):
        assert precision in [16, 32]
        if precision == 32:
            precision = "fp32"
        elif jax.default_backend() == "tpu":
            precision = "bf16"
        else:
            precision = "fp16"

    assert precision in ["fp32", "fp16", "bf16"]
    match precision:
        case "fp32":
            dtype = norm_dtype = param_dtype = jnp.float32
        case "fp16":
            dtype = jnp.float16
            norm_dtype = param_dtype = jnp.float32
        case "bf16":
            dtype = norm_dtype = param_dtype = jnp.bfloat16
        case _:
            raise RuntimeError(f"Unknwon precision {precision} is specified.")

    dtypes = DTypes(dtype=dtype, norm_dtype=norm_dtype, param_dtype=param_dtype)
    return dtypes

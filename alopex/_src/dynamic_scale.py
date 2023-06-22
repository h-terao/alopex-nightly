from __future__ import annotations
import typing as tp
import functools

import jax
import jax.numpy as jnp
from flax import struct
import chex


class DynamicScale(struct.PyTreeNode):
    """Dynamic loss scaling for mixed precision gradients.

    A forked version of `flax.training.dynamic_scale.DynamicScale`.
    This implementation separates `gradient scaling` and `scaling factor update.`
    The former is applied by `value_and_grad` or `grad`, and the later is applied by
    `update`. See Example section for the specific example.

    Attributes:
        growth_factor
        backoff_factor
        growth_interval
        fin_steps
        scale
        minimum_scale

    Example:
        ::
            dyn_scale = DynamicScale()
            grad_fn = dyn_scale.grad(loss_fn)
            grads = grad_fn(params, ...)  # grads are scaled by `dyn_scale.scale`.
            new_dyn_scale, is_fin = dyn_scale.update(grads)  # update scale factor.
    """

    growth_factor: float = struct.field(pytree_node=False, default=2.0)
    backoff_factor: float = struct.field(pytree_node=False, default=0.5)
    growth_interval: int = struct.field(pytree_node=False, default=2000)
    fin_steps: chex.Array = 0
    scale: chex.Array = 65536.0
    minimum_scale: tp.Optional[float] = struct.field(pytree_node=False, default=jnp.finfo(jnp.float32).tiny)

    def value_and_grad(
        self,
        fun: tp.Callable[..., tp.Any],
        argnums: tp.Union[int, tp.Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> tp.Any:
        @functools.wraps(fun)
        def loss_wrapper(*args):
            aux = fun(*args)
            if has_aux:
                return (self.scale * aux[0], aux[1])
            else:
                return self.scale * aux

        grad_fn = jax.value_and_grad(loss_wrapper, argnums, has_aux)

        def grad_fn_wrapper(*args):
            aux, grad = grad_fn(*args)
            aux = (aux[0] / self.scale, aux[1]) if has_aux else aux / self.scale
            grad = jax.tree_util.tree_map(lambda g: jnp.asarray(g, jnp.float32) / self.scale, grad)
            return aux, grad

        return grad_fn_wrapper

    def grad(
        self,
        fun: tp.Callable[..., tp.Any],
        argnums: tp.Union[int, tp.Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> tp.Any:
        grad_fn = self.value_and_grad(fun, argnums, has_aux)

        def grad_fn_wrapper(*args):
            aux, grad = grad_fn(*args)
            if has_aux:
                return grad, aux[1]
            else:
                return grad

        return grad_fn_wrapper

    def update(self, grads: chex.ArrayTree) -> tuple[DynamicScale, chex.Array]:
        """Update `DynamicScale`.

        Args:
            grads: a tree that holds gradients.

        Returns:
            A tuple of new `DynamicScale` and a bool array that represents
            whether all grads are finite.
        """
        finite = jnp.array(True)
        for g in jax.tree_util.tree_leaves(grads):
            finite &= jnp.all(jax.lax.is_finite(g))

        grow = self.fin_steps == self.growth_interval
        fin_scale = jnp.where(
            grow & finite,
            jnp.minimum(self.scale * self.growth_factor, jnp.finfo(jnp.float32).max),
            self.scale,
        )
        inf_scale = self.scale * self.backoff_factor
        if self.minimum_scale is not None:
            inf_scale = jnp.maximum(inf_scale, self.minimum_scale)
        new_scale = jnp.where(finite, fin_scale, inf_scale)
        new_fin_steps = jnp.where(grow | (~finite), 0, self.fin_steps + 1)

        new_self = self.replace(fin_steps=new_fin_steps, scale=new_scale)
        return new_self, finite

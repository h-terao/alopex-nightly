from __future__ import annotations
import typing as tp

import jax.random as jr
import chex


class Transformed(tp.NamedTuple):
    """Holds a pair of pure functions.

    Attributes:
        init_with_output: A pure function.
        init: A pure function that initializes variables.
        apply: A pure function that transforms inputs.
    """

    init_with_output: tp.Callable
    init: tp.Callable
    apply: tp.Callable


def hk_like(flax_model: tp.Any, rng_keys: str | tp.Sequence[str] | None) -> Transformed:
    """Creates Haiku-like model API from flax model.

    Args:
        flax_model: Flax model.
        rng_keys: Names of PRNG keys to split.

    Returns:
        Namedtuple of (init_with_output, init, apply).
    """

    if isinstance(rng_keys, str):
        rng_keys = [rng_keys]
    elif rng_keys is None:
        rng_keys = []

    def make_rngs(rng: chex.PRNGKey | None):
        if rng is None or len(rng_keys) == 0:
            return dict()  # empty dict.
        else:
            return dict(zip(rng_keys, jr.split(rng, len(rng_keys))))

    def init_with_output(rng: chex.PRNGKey | None, *args, **kwargs):
        param_rng, rng = jr.split(rng)
        rngs = dict(params=param_rng, **make_rngs(rng))
        output, variables = flax_model.init_with_output(rngs=rngs, *args, **kwargs)
        state, params = map(lambda x: x.unfreeze(), variables.pop("params"))
        return output, (params, state)

    def apply(
        params: chex.ArrayTree, state: chex.ArrayTree, rng: chex.PRNGKey | None, *args, **kwargs
    ):
        assert "mutable" not in kwargs, "Cannot specify `mutable` argument."
        variables = {"params": params, **state}
        rngs = make_rngs(rng)
        mutable = list(state)  # batch_stats, for example
        outputs, new_state = flax_model.apply(variables, *args, rngs=rngs, mutable=mutable, **kwargs)
        return outputs, new_state

    return Transformed(
        init_with_output=init_with_output,
        init=lambda rng, *args, **kwargs: init_with_output(rng, *args, **kwargs)[1],
        apply=apply,
    )

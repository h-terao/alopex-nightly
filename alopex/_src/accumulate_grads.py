import optax


def accumulate_grads(
    opt: optax.GradientTransformation, num_accumulate_steps: int = 1
) -> optax.GradientTransformation:
    """Wraps optax optimizer and performs the gradient accumulation.

    Args:
        opt: optax optimizer.
        num_accumulate_steps: number of accumulation steps.

    Returns:
        A wrapped version of `opt`.
    """
    return optax.MultiSteps(opt, every_k_schedule=num_accumulate_steps)

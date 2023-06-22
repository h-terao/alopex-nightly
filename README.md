<h1 align='center'>Alopex</h1>

Alopex provides useful tools to accelerate the prototyping of deep learning projects with JAX.

## Installation

1. Install JAX for your environment. See details in [installation section of JAX project](https://github.com/google/jax#installation).
2. Install Alopex via pip:
```bash
$ pip install git+https://github.com/h-terao/alopex-nightly
```

## Modules overview

- training: This module provides training utilities, including abstracted training loop, mixed precision, and finetuning.
- interpreters: Useful tools to evaluate functions.
- pytorch: Utilities to integrate the PyTorch models into JAX/Flax using the name-based conversion.
- serialization: Simple read/write functions.
- nets: Off-the-shelf neural network models with pretrained parameters. All pretraied parameters are originally provided by the PyTorch libraries, and converted by `alopex.pytorch`.
- transforms: Off-the-shelf transformation for the computer vision.

## JAX Tips

This section provide JAX and Flax tips.

### Gradient accumulation

To perform the gradient accumulation, `optax.MultiSteps` is easily available.
Loss scaling is not required.

```python
import optax

num_accumulate_batches = 10

tx = ...  # define Optax optimizer (e.g., SGD).
tx = optax.MultiSteps(tx, every_k_schedule=num_accumulate_batches)
```
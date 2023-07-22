<h1 align='center'>Alopex</h1>

Alopex provides useful tools to accelerate the prototyping of deep learning projects with JAX and Flax.

## Installation

1. Install JAX for your environment. See details in [installation section of JAX project](https://github.com/google/jax#installation).
2. Install Alopex via pip:
```bash
$ pip install git+https://github.com/h-terao/alopex-nightly
```

## Modules overview

- training: This module provides training utilities, including abstracted training loop, mixed precision, and finetuning.
- layers (future): Flax layer modules. Includes activations and minor modules.
- functions: Loss and other functions. Losses are mainly imported from optax, but some losses are newly implemented.
- interpreters: Useful tools to evaluate functions.
- pytorch: Utilities to integrate the PyTorch models into JAX/Flax using the name-based conversion.
- pytypes: Alopex-specific types for annotation.
- serialization: Simple read/write functions.
- vision: Off-the-shelf networks, transformations and utilities for computer vision.
- nlp (future): Off-the-shelf networks, transformations and utilities for natural language processing.

## JAX Tips

This section provide JAX and Flax tips.

## Current problem

- mode=jit is very slow. Use mode=pmap for acceleration.
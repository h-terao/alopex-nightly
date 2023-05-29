<h1 align='center'>Alopex</h1>

Alopex is a library that accelerates the prototyping of deep learning projects with JAX.

## Installation

1. Install JAX for your environment. See details in [installation section of JAX project](https://github.com/google/jax#installation).
2. Install Alopex via pip:
```bash
$ pip install git+https://github.com/h-terao/Alopex
```

## Modules overview

### training

Utilities for training. `train_loop` and `eval_loop` applied functions. `DynamicScale` is implemented for mixed precision training. This is a forked version of `flax.training.dynamic_scale.DynamicScale`, but split `grad` and `update`

### interpreters

### pytorch

`alopex.pytorch` is a utility to integrate the PyTorch models into JAX/Flax using the name-based conversion.

### serialization

`alopex.serialization` provides simple read/write functions.

### nets

```python
from torchvision import models
import alopex

torch_model = models.resnet18(pretrained=True)
torch_vars = alopex.pytorch.convert_torch_model(torch_model)

flax_model = alopex.nets.resnet18(...)
variables = flax_model.init(key, ...)  # init.
variables, masks = alopex.interpreters.load_variables(variables, torch_vars)
```

### transforms

`alopex.transforms` mainly provides transformations for image and video domains.
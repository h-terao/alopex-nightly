# flake8: noqa
__version__ = "0.1.0"

from alopex import training
from alopex import functions
from alopex import interpreters
from alopex import pytorch
from alopex import serialization
from alopex import pytypes
from alopex import data
from alopex import utils

from alopex import vision

# from alopex import layers  # TODO
# from alopex import nlp  # TODO: future work

from alopex._src.configuration import using_config
from alopex._src.configuration import configure
from alopex._src.configuration import set_config
from alopex._src.configuration import get_config

from alopex._src.registry import Registry
from alopex._src.registry import registry
from alopex._src.registry import register

# Utilities.
from alopex._src.dtypes import get_dtypes

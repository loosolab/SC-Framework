"""Provides a range of different utilitis."""

import importlib as _importlib
import os
import sys

# Set base R home directory to prevent R_HOME not found error
os.environ['R_HOME'] = os.path.join(sys.executable.split('/bin/')[0], 'lib/R')

# define what is exported in this module
__all__ = [
    "adata",
    "assemblers",
    "bioutils",
    "checker",
    "creators",
    "decorator",
    "general",
    "io",
    "jupyter",
    "multiprocessing",
    "tables"
]


def __dir__():
    """Return the defined submodules."""
    return __all__


def __getattr__(name):
    """Lazyload modules (inspired by scipy)."""
    if name in __all__:
        # return import to make it directly available
        return _importlib.import_module(f"sctoolbox.utils.{name}")
    else:
        raise AttributeError(f"Module 'sctoolbox.utils' does not contain a module named '{name}'.")

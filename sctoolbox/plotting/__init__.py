"""sctoolbox plotting functions."""

import importlib as _importlib

# define what is exported in this module
__all__ = [
    "clustering",
    "embedding",
    "general",
    "genometracks",
    "highly_variable",
    "marker_genes",
    "qc_filter",
    "velocity"
]


def __dir__():
    """Returns the defined submodules."""
    return __all__


def __getattr__(name):
    """Lazyload modules. (Inspired by scipy)"""
    if name in __all__:
        _importlib.import_module(f"sctoolbox.plotting.{name}")
    else:
        raise AttributeError(f"Module 'sctoolbox.plotting' does not contain a module named '{name}'.")

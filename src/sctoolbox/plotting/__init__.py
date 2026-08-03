"""sctoolbox plotting functions."""

import importlib as _importlib
from types import ModuleType

# define what is exported in this module
__all__ = [
    "clustering",
    "embedding",
    "general",
    "genometracks",
    "gsea",
    "highly_variable",
    "marker_genes",
    "multiomics",
    "qc_filter",
    "velocity",
    "planet_plot"
]


def __dir__() -> list:
    """Return the defined submodules."""
    return __all__


def __getattr__(name: str) -> ModuleType:
    """Lazyload modules (inspired by scipy)."""
    if name in __all__:
        # return import to make it directly available
        return _importlib.import_module(f"sctoolbox.plotting.{name}")
    else:
        raise AttributeError(f"Module 'sctoolbox.plotting' does not contain a module named '{name}'.")

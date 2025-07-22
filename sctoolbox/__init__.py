"""Single Cell Toolbox (sctoolbox)."""

# get all loaded modules
from sys import modules as _modules
__cached_modules = set(_modules.keys())
__cached_modules.remove("sctoolbox")  # to include the sctoolbox as a newly loaded package

# import with prefix _ to hide them
from ._version import __version__
from ._settings import settings
import importlib as _importlib

submodules = [
    "plotting",
    "tools",
    "utils"
]

# define what is exported in this module
__all__ = submodules + [
    "__version__",
    "settings",
    "__cached_modules"
]


def __dir__():
    """Return the defined submodules."""
    return __all__


def __getattr__(name):
    """Lazyload modules (inspired by scipy)."""
    if name in submodules:
        # return import to make it directly available
        return _importlib.import_module(f"sctoolbox.{name}")
    else:
        raise AttributeError(f"Module 'sctoolbox' does not contain a module named '{name}'.")

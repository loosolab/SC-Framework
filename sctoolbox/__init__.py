"""Single Cell Toolbox (sctoolbox)."""

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
    "settings"
]


def __dir__():
    """Return the defined submodules."""
    return __all__


def __getattr__(name):
    """Lazyload modules (inspired by scipy)."""
    if name in submodules:
        _importlib.import_module(f"sctoolbox.{name}")
    else:
        raise AttributeError(f"Module 'sctoolbox' does not contain a module named '{name}'.")

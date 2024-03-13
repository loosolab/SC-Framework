"""Provides a range of different utilitis."""

from importlib import import_module
from sctoolbox._modules import import_submodules
import os
import sys

# Set base R home directory to prevent R_HOME not found error
os.environ['R_HOME'] = os.path.join(sys.executable.split('/bin/')[0], 'lib/R')

function_dict = import_submodules(globals())

# Add settings_from_config to utils (it is now located in _settings)
settings_module = import_module("sctoolbox._settings")
globals()["settings_from_config"] = getattr(settings_module, "settings_from_config")

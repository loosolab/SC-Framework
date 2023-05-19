from importlib import import_module
from sctoolbox._modules import import_submodules
function_dict = import_submodules(globals())

# Add settings_from_config to utils (it is now located in _settings)
settings_module = import_module("sctoolbox._settings")
globals()["settings_from_config"] = getattr(settings_module, "settings_from_config")

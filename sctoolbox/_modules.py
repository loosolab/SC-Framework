from importlib import import_module
import glob
import os


def import_submodules(globals):
    """Import all functions from submodules into the upper module namespace.

    This enables usage with `sctoolbox.plotting.plot_umap()` as well as `sctoolbox.plotting.embedding.plot_umap()`.

    Parameters
    ----------
    globals : dict
        globals() of the module
    """

    # Find name of module
    module_dir = os.path.dirname(globals["__file__"])  # path to init file
    module_name = os.path.basename(module_dir)

    # Get all submodules
    submodules = glob.glob(module_dir + "/*.py")
    submodules = [module for module in submodules if "__init__.py" not in module]
    submodule_names = [os.path.basename(module).replace(".py", "") for module in submodules]

    # Import each submodule to find functions
    for submodule_name in submodule_names:
        module = import_module("sctoolbox." + module_name + "." + submodule_name)
        function_names = [function for function in dir(module) if not function.startswith("__")]  # do not add internal functions

        # Add each function to globals for this module
        for function_name in function_names:
            function = getattr(module, function_name)
            globals[function_name] = function

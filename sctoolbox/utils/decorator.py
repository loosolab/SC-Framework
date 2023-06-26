import anndata
import functools
import pandas as pd

import sctoolbox.utils.general as utils


def log_anndata(func):
    """
    Decorator to log function call inside adata.

    Parameters
    ----------
    func : function
        Function to decorate.

    Returns
    -------
    function :
        Decorated function
    """
    # TODO store datatypes not supported by scanpy.write as string representation (repr())

    @functools.wraps(func)  # preserve information of the decorated func
    def wrapper(*args, **kwargs):

        # find anndata object within parameters (if there are more use the first one)
        adata = None
        for param in list(args) + list(kwargs.values()):
            if isinstance(param, anndata.AnnData):
                adata = param
                break

        if adata is None:
            raise ValueError("Can only log functions that receive an AnnData object as parameter.")

        # init adata if necessary
        if "sctoolbox" not in adata.uns.keys():
            adata.uns["sctoolbox"] = dict()

        funcname = func.__name__
        if funcname not in adata.uns["sctoolbox"].keys():
            adata.uns["sctoolbox"][funcname] = []

        # Convert anndata objects to string representation
        args_repr = [repr(element) for element in args if isinstance(element, anndata.AnnData)]
        kwargs_repr = {key: repr(value) if isinstance(value, anndata.AnnData) else value for key, value in kwargs.items()}

        # log information on run
        d = {}
        d["timestamp"] = utils.get_datetime()
        d["user"] = utils.get_user()
        d["func"] = func.__name__
        d["args"] = args_repr
        d["kwargs"] = kwargs_repr

        adata.uns["sctoolbox"][funcname].append(d)

        return func(*args, **kwargs)

    return wrapper


def get_parameter_table(adata):
    """
    Get a table of all function calls with their parameters from the adata.uns["sctoolbox"] dictionary.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with logged function calls.

    Returns
    -------
    pd.DataFrame
        Table with all function calls and their parameters.
    """

    if "sctoolbox" not in adata.uns.keys():
        raise ValueError("No sctoolbox function calls logged in adata.")

    # Create an overview table for each function
    function_tables = []
    for function in adata.uns["sctoolbox"].keys():
        table = pd.DataFrame(adata.uns["sctoolbox"][function])
        table.sort_values("timestamp", inplace=True)
        table.insert(3, "func_count", range(1, len(table) + 1))
        function_tables.append(table)

    # Concatenate all tables and sort by timestamp
    complete_table = pd.concat(function_tables)
    complete_table.reset_index(drop=True, inplace=True)
    complete_table.sort_values("timestamp", inplace=True)

    return complete_table


def debug_func_log(func):
    """ Decorator to print function call with arguments and keyword arguments.

    In progress.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"DEBUG: {func.__name__} called with args: {args} and kwargs: {kwargs}")
        return func(*args, **kwargs)

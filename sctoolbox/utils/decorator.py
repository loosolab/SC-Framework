"""Decorators and related functions."""

import anndata
import functools
import pandas as pd
import matplotlib

import sctoolbox.utils.general as utils


def log_anndata(func):
    """
    Decorate function to log adata inside function call.

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

        if "log" not in adata.uns["sctoolbox"].keys():
            adata.uns["sctoolbox"]["log"] = dict()

        funcname = func.__name__
        if funcname not in adata.uns["sctoolbox"]["log"].keys():
            adata.uns["sctoolbox"]["log"][funcname] = {}

        # Convert objects to safe representations, e.g. anndata objects to string representation and tuple to list
        args_repr = {f"arg{i+1}": element for i, element in enumerate(args)}  # create dict with arg1, arg2, ... as keys instead of list to prevent errors with wrongly shaped arrays
        kwargs_repr = kwargs
        convert = {anndata.AnnData: repr, tuple: list, matplotlib.axes._axes.Axes: str}
        for typ, convfunc in convert.items():
            args_repr = {param: convfunc(element) if isinstance(element, typ) else element for param, element in args_repr.items()}
            kwargs_repr = {param: convfunc(element) if isinstance(element, typ) else element for param, element in kwargs_repr.items()}

        # log information on run
        d = {}
        d["timestamp"] = utils.get_datetime()
        d["user"] = utils.get_user()
        d["func"] = funcname
        d["args"] = args_repr
        d["kwargs"] = kwargs_repr

        run_n = len(adata.uns["sctoolbox"]["log"][funcname]) + 1
        adata.uns["sctoolbox"]["log"][funcname][f"run_{run_n}"] = d

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

    Raises
    ------
    ValueError
        If no logs are found.
    """
    if "sctoolbox" not in adata.uns.keys() or "log" not in adata.uns["sctoolbox"].keys():
        raise ValueError("No sctoolbox function calls logged in adata.")

    # Create an overview table for each function
    function_tables = []
    for function in adata.uns["sctoolbox"]["log"].keys():
        table = pd.DataFrame.from_dict(adata.uns["sctoolbox"]["log"][function], orient='index')
        table.sort_values("timestamp", inplace=True)
        table.insert(len(table.columns), "func_count", table.index)
        function_tables.append(table)

    # Concatenate all tables and sort by timestamp
    complete_table = pd.concat(function_tables)
    complete_table.sort_values("timestamp", inplace=True)
    complete_table.reset_index(drop=True, inplace=True)

    # reorder columns
    first_cols = ["func", "args", "kwargs"]
    complete_table = complete_table.reindex(columns=first_cols + list(set(complete_table.columns) - set(first_cols)))

    return complete_table


def debug_func_log(func):
    """
    Decorate function to print function call with arguments and keyword arguments.

    In progress.

    Parameters
    ----------
    func : function
        Function to decorate.

    Returns
    -------
    function :
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"DEBUG: {func.__name__} called with args: {args} and kwargs: {kwargs}")
        return func(*args, **kwargs)

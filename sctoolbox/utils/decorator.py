import anndata
import functools

import sctoolbox.utils.logging as log


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
        # TODO use setup function?
        if "sctoolbox" not in adata.uns.keys():
            adata.uns["sctoolbox"] = dict()
        if "func_log" not in adata.uns["sctoolbox"].keys():
            adata.uns["sctoolbox"]["func_log"] = dict()

        # log stuff
        d = adata.uns["sctoolbox"]["func_log"]

        d.setdefault("timestamp", []).append(log.get_datetime())
        d.setdefault("user", []).append(log.get_user())
        d.setdefault("func", []).append(func.__name__)
        d.setdefault("args", []).append(args)
        d.setdefault("kwargs", []).append(kwargs)

        adata.uns["sctoolbox"]["func_log"] = d

        return func(*args, **kwargs)
    return wrapper

import getpass
from datetime import datetime
import anndata


def build_legend(adata, key, value, inplace=True):
    """
    Adding info anndata.uns["legend"]
    :param adata:
    :param key:
    :param value:
    :param inplace:
    :return:
    """
    if type(adata) != anndata.AnnData:
        raise TypeError("Invalid data type. AnnData object is required.")

    m_adata = adata if inplace else adata.copy()

    if "legend" not in m_adata.uns:
        m_adata.uns["legend"] = {}
    m_adata.uns["legend"][key] = value

    if not inplace:
        return m_adata


def vprint(verbose=True):
    """
    Generates a function with given verbosity. Either hides or prints all messages.

    Parameters
    ----------
    verbose : boolean, default True
        Set to False to disable the verbose message.

    Returns
    -------
        function :
            Function that expects a single str argument. Will print string depending on verbosity.
    """
    return lambda message: print(message) if verbose is True else None


def get_user():
    """ Get the name of the current user.

    Returns
    -------
    str
        The name of the current user.
    """

    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown"

    return username


def get_datetime():
    """ Get a string with the current date and time for logging.

    Returns
    -------
    str
        A string with the current date and time in the format dd/mm/YY H:M:S
    """

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")  # dd/mm/YY H:M:S

    return dt_string


def add_uns_info(adata, key, value, how="overwrite"):
    """ Add information to adata.uns['sctoolbox']. This is used for logging the parameters and options of different steps in the analysis.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object.
    key : str or list
        The key to add to adata.uns['sctoolbox']. If the key is a list, it represents a path within a nested dictionary.
    value : any
        The value to add to adata.uns['sctoolbox'].
    """

    if "sctoolbox" not in adata.uns:
        adata.uns["sctoolbox"] = {}

    if isinstance(key, str):
        key = [key]

    d = adata.uns["sctoolbox"]
    for k in key[:-1]:  # iterate over all keys except the last one
        if k not in d:
            d[k] = d.get(k, {})
        d = d[k]

    # Add value to last key
    last_key = key[-1]
    if how == "overwrite":
        d[last_key] = value  # last key contains value

    elif how == "append":
        if key[-1] not in d:
            d[last_key] = value  # initialize with a value if key does not exist

        else:  # append to existing key
            if not isinstance(d[last_key], list):
                d[last_key] = [d[last_key]]

            # Remove value if it is already in list; this prevents duplicates and ensures that value is now last in list
            if value in d[last_key]:
                d[last_key].remove(value)
            d[last_key].append(value)


def initialize_uns(adata, keys=[]):
    """ Initialize the sctoolbox keys in adata.uns.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object.
    keys : str or list of str, optional
        Additional keys to be initialized in adata.uns['sctoolbox'].

    Returns
    -------
    None
        keys are initialized in adata.uns['sctoolbox'].
    """
    if "sctoolbox" not in adata.uns:
        adata.uns["sctoolbox"] = {}

    # Add additional keys if needed
    if isinstance(keys, str):
        keys = [keys]

    for key in keys:
        if key not in adata.uns["sctoolbox"]:
            adata.uns["sctoolbox"][key] = {}

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

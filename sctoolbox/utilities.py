import pandas as pd
import sys
import os
import scanpy as sc
import importlib
import matplotlib.pyplot as plt

import sctoolbox.checker as ch
import sctoolbox.creators as cr


# ----------------- String functions ---------------- #

def longest_common_suffix(list_of_strings):
    """
    Find the longest common suffix of a list of strings.

    Parameters
    -----------
    list_of_strings : list of str
        List of strings.

    Returns
    -------
    str
        Longest common suffix of the list of strings.
    """

    reversed_strings = [s[::-1] for s in list_of_strings]
    reversed_lcs = os.path.commonprefix(reversed_strings)
    lcs = reversed_lcs[::-1]

    return lcs


def remove_prefix(s, prefix):
    """ Remove prefix from a string. """
    return s[len(prefix):] if s.startswith(prefix) else s


def remove_suffix(s, suffix):
    """ Remove suffix from a string. """
    return s[:-len(suffix)] if s.endswith(suffix) else s


def _is_notebook():
    """ Utility to check if function is being run from a notebook or a script """
    try:
        _ = get_ipython()
        return(True)
    except NameError:
        return(False)


# ------------------ I/O functions ----------------- #

def create_dir(path):
    """ Create a directory if it is not existing yet.

    Parameters
    -----------
    path : str
        Path to the directory to be created.
    """

    dirname = os.path.dirname(path)  # the last dir of the path
    if dirname != "":  # if dirname is "", file is in current dir
        os.makedirs(dirname, exist_ok=True)


def is_str_numeric(ans):
    try:
        float(ans)
        return True
    except ValueError:
        return False


def save_figure(path):
    """
    Save the current figure to a file.

    Parameters
    ----------
    path : str
        Path to the file to be saved.
        Add the extension (e.g. .tiff) you wanna save your figure in the end of path, e.g., /mnt/*/note2_violin.tiff
        The lack of extension indicates the figure will be saved as .png
    """

    if path is not None:
        create_dir(path)  # recursively create parent dir if needed
        plt.savefig(path, dpi=600, bbox_inches="tight")


def vprint(verbose=True):
    """ Print the verbose message.

    Parameters
    -----------
    verbose : Boolean, optional
        Set to False to disable the verbose message. Default: True
    """
    return lambda message: print(message) if verbose is True else None


# Requirement for installed tools
def check_module(module):
    """ Check if <module> can be imported without error.

    Parameters
    -----------
    module : str
        Name of the module to check.

    Raises
    ------
    ImportError
        If the module is not available for import.
    """

    error = 0
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        error = 1
    except Exception:
        raise  # unexpected error loading module

    # Write out error if module was not found
    if error == 1:
        s = f"ERROR: Could not find the '{module}' module on path, but the module is needed for this functionality. Please install this package to proceed."
        raise ImportError(s)


# Loading adata file and adding the information to be evaluated and color list
def load_anndata(is_from_previous_note=True, which_notebook=None, data_to_evaluate=None):
    '''
    Load anndata object
    ==========
    Parameters
    ==========
    is_from_previous_note : Boolean
        Set to False if you wanna load an anndata object from other source rather than scRNAseq autom workflow.
    which_notebook : Int.
        The number of the notebook that generated the anndata object you want to load
        If is_from_previous_note=False, this parameter will be ignored
    data_to_evaluate : String
        This is the anndata.obs[STRING] to be used for analysis, e.g. "condition"
    '''
    # Author : Guilherme Valente
    def loading_adata(NUM):
        pathway = ch.fetch_info_txt()
        files = os.listdir(''.join(pathway))
        loading = "anndata_" + str(NUM)
        if any(loading in items for items in files):
            for file in files:
                if loading in file:
                    anndata_file = file
        else:  # In case the user provided an inexistent anndata number
            sys.exit(loading + " was not found in " + pathway)

        return(''.join(pathway) + "/" + anndata_file)

    # Messages and others
    m1 = "You choose is_from_previous_note=True. Then, set an which_notebook=[INT], which INT is the number of the notebook that generated the anndata object you want to load."
    m2 = "Set the data_to_evaluate=[STRING], which STRING is anndata.obs[STRING] to be used for analysis, e.g. condition."
    m3 = "Paste the pathway and filename where your anndata object deposited."
    m4 = "Correct the pathway or filename or type q to quit."
    opt1 = ["q", "quit"]

    if isinstance(data_to_evaluate, str) is False:  # Close if the anndata.obs is not correct
        sys.exit(m2)
    if is_from_previous_note is True:  # Load anndata object from previous notebook
        try:
            ch.check_notebook(which_notebook)
        except TypeError:
            sys.exit(m1)
        file_path = loading_adata(which_notebook)
        data = sc.read_h5ad(filename=file_path)  # Loading the anndata
        cr.build_infor(data, "data_to_evaluate", data_to_evaluate)  # Annotating the anndata data to evaluate
        return(data)

    elif is_from_previous_note is False:  # Load anndata object from other source
        answer = input(m3)
        while os.path.isfile(answer) is False:  # False if pathway is wrong
            if answer.lower() in opt1:
                sys.exit("You quit and lost all modifications :(")
            print(m4)
            answer = input(m4)
        data = sc.read_h5ad(filename=answer)  # Loading the anndata
        cr.build_infor(data, "data_to_evaluate", data_to_evaluate)  # Annotating the anndata data to evaluate
        cr.build_infor(data, "Anndata_path", answer.rsplit('/', 1)[0])  # Annotating the anndata path
        return(data)


def saving_anndata(ANNDATA, current_notebook=None):
    '''
    Save your anndata object

    Parameters
    ===========
    ANNDATA : anndata object
        adata object
    current_notebook : int
        The number of the current notebook.
    '''
    # Author : Guilherme Valente
    # Messages and others
    m1 = "Set an current_notebook=[INT], which INT is the number of current notebook."
    m2 = "Your new anndata object is saved here: "

    try:
        ch.check_notebook(current_notebook)
    except TypeError:
        sys.exit(m1)  # Close if the notebook number is not an integer
    adata_output = ANNDATA.uns["infoprocess"]["Anndata_path"] + "anndata_" + str(current_notebook) + "_" + ANNDATA.uns["infoprocess"]["Test_number"] + ".h5ad"
    ANNDATA.write(filename=adata_output)
    print(m2 + adata_output)


def pseudobulk_table(adata, groupby, how="mean"):
    """ Get a pseudobulk table of values per cluster.

    Parameters
    -----------
    adata : anndata.AnnData
        An annotated data matrix containing counts in .X.
    groupby : str
        Name of a column in adata.obs to cluster the pseudobulks by.
    how : str, optional
        How to calculate the value per cluster. Can be one of "mean" or "sum". Default: "mean"
    """

    adata = adata.copy()
    adata.obs[groupby] = adata.obs[groupby].astype('category')

    # Fetch the mean/sum counts across each category in cluster_by
    res = pd.DataFrame(columns=adata.var_names, index=adata.obs[groupby].cat.categories)
    for clust in adata.obs[groupby].cat.categories:

        if how == "mean":
            res.loc[clust] = adata[adata.obs[groupby].isin([clust]), :].X.mean(0)
        elif how == "sum":
            res.loc[clust] = adata[adata.obs[groupby].isin([clust]), :].X.sum(0)

    res = res.T  # transform to genes x clusters
    return(res)


def split_list(lst, n):
    """ Split list into n chunks.

    Parameters
    -----------
    lst : list
        List to be chunked
    n : int
        Number of chunks.

    """
    chunks = []
    for i in range(0, n):
        chunks.append(lst[i::n])

    return chunks

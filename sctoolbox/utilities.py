import pandas as pd
import numpy as np
import sys
import os
import re
import scanpy as sc
import importlib
import sctoolbox.checker as ch
import sctoolbox.creators as cr
import matplotlib.pyplot as plt
import matplotlib
from os.path import join, dirname, exists
from pathlib import Path
from IPython.core.magic import register_cell_magic
from IPython.display import HTML, display


def get_package_versions():
    """
    Utility to get a dictionary of currently installed python packages and versions.

    Returns
    --------
    A dict in the form:
    {"package1": "1.2.1", "package2":"4.0.1", (...)}

    """

    # Import freeze
    try:
        from pip._internal.operations import freeze
    except ImportError:  # pip < 10.0
        from pip.operations import freeze

    # Get list of packages and versions with freeze
    package_list = freeze.freeze()
    package_dict = {}  # dict for collecting versions
    for s in package_list:
        try:
            name, version = re.split("==| @ ", s)
            package_dict[name] = version
        except Exception:
            print(f"Error reading version for package: {s}")

    return package_dict


# ------------------ Type checking ------------------ #

def is_integer_array(arr):
    """
    Check if all values of arr are integers.

    Parameters
    ----------
    x : numpy.array
        Array of values to be checked.

    Returns
    -------
    boolean :
        True if all values are integers, False otherwise.
    """

    # https://stackoverflow.com/a/7236784
    boolean = np.equal(np.mod(arr, 1), 0)

    return np.all(boolean)


# ----------------- String functions ---------------- #

def clean_flanking_strings(list_of_strings):
    """
    Remove common suffix and prefix from a list of strings, e.g. running the function on
    ['path/a.txt', 'path/b.txt', 'path/c.txt'] would yield ['a', 'b', 'c'].

    Parameters
    -----------
    list_of_strings : list of str
        List of strings.

    Returns
    ---------
    List of strings without common suffix and prefix
    """

    suffix = longest_common_suffix(list_of_strings)
    prefix = os.path.commonprefix(list_of_strings)

    list_of_strings_clean = [remove_prefix(s, prefix) for s in list_of_strings]
    list_of_strings_clean = [remove_suffix(s, suffix) for s in list_of_strings_clean]

    return list_of_strings_clean


def longest_common_suffix(list_of_strings):
    """
    Find the longest common suffix of a list of strings.

    Parameters
    ----------
    list_of_strings : list of str
        List of strings.

    Returns
    -------
    str :
        Longest common suffix of the list of strings.
    """
    reversed_strings = [s[::-1] for s in list_of_strings]
    reversed_lcs = os.path.commonprefix(reversed_strings)
    lcs = reversed_lcs[::-1]

    return lcs


def remove_prefix(s, prefix):
    """
    Remove prefix from a string.

    Parameters
    ----------
    s : str
        String to be processed.
    prefix : str
        Prefix to be removed.

    Returns
    -------
    str :
        String without prefix.
    """
    return s[len(prefix):] if s.startswith(prefix) else s


def remove_suffix(s, suffix):
    """
    Remove suffix from a string.

    Parameters
    ----------
    s : str
        String to be processed.
    suffix : str
        Suffix to be removed.

    Returns
    -------
    str :
        String without suffix.
    """
    return s[:-len(suffix)] if s.endswith(suffix) else s


def _is_interactive():
    """
    Check if matplotlib backend is interactive.

    Returns
    -------
    boolean :
        True if interactive, False otherwise.
    """

    backend = matplotlib.get_backend()

    if backend == 'module://ipympl.backend_nbagg':
        return True
    else:
        return False


# ---------------- jupyter functions --------------- #

def _is_notebook():
    """
    Utility to check if function is being run from a notebook or a script.

    Returns
    -------
    boolean :
        True if running from a notebook, False otherwise.
    """
    try:
        _ = get_ipython()
        return True
    except NameError:
        return False


if _is_notebook():
    @register_cell_magic
    def bgcolor(color, cell=None):
        """
        Set background color of current jupyter cell. Adapted from https://stackoverflow.com/a/53746904.
        Note: Jupyter notebook v6+ needed

        Change color of the cell by either calling the function
        `bgcolor("yellow")`
        or with magic (has to be first line in cell!)
        `%%bgcolor yellow`

        Parameters
        ----------
        color : str
            Background color of the cell. A valid CSS color e.g.:
                - red
                - rgb(255,0,0)
                - #FF0000
            See https://www.rapidtables.com/web/css/css-color.html
        cell : str, default None
            Code of the cell that will be evaluated.

        Returns
        -------
        varying or None :
            Output is the executed cell code.

        """
        script = f"""
                var cell = this.closest('.code_cell');
                var editor = cell.querySelector('.CodeMirror-sizer');
                editor.style.background='{color}';
                this.parentNode.removeChild(this)
                """

        display(HTML(f'<img src onerror="{script}">'))

        if cell:
            return exec(cell)


# ------------------ I/O functions ----------------- #

def create_dir(path):
    """
    Create a directory if it is not existing yet.
    'path' can be either a direct path of the directory, or a path to a file for which the upper directory should be created.

    Parameters
    ----------
    path : str
        Path to the directory to be created.
    """

    base = os.path.basename(path)
    if "." in base:  # path is expected to be a file
        dirname = os.path.dirname(path)  # the last dir of the path
        if dirname != "":  # if dirname is "", file is in current dir
            os.makedirs(dirname, exist_ok=True)

    else:
        os.makedirs(path, exist_ok=True)


def is_str_numeric(ans):
    """ Check if string can be converted to number. """
    try:
        float(ans)
        return True
    except ValueError:
        return False


def save_figure(path, dpi=600):
    """
    Save the current figure to a file.

    Parameters
    ----------
    path : str
        Path to the file to be saved.
        Add the extension (e.g. .tiff) you want save your figure in to the end of the path, e.g., /some/path/plot.tiff
        The lack of extension indicates the figure will be saved as .png.
    dpi : int, default 600
        Dots per inch. Higher value increases resolution.
    """
    if path is not None:
        create_dir(path)  # recursively create parent dir if needed
        plt.savefig(path, dpi=dpi, bbox_inches="tight")


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


def check_module(module):
    """
    Check if <module> can be imported without error.

    Parameters
    ----------
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


def load_anndata(is_from_previous_note=True, which_notebook=None, data_to_evaluate=None):
    """
    Load anndata from a previous notebook.

    Parameters
    ----------
    is_from_previous_note : boolean, default True
        Set to False if you want to load an anndata object from other source rather than scRNAseq autom workflow.
    which_notebook : int, default None
        The number of the notebook that generated the anndata object you want to load
        If is_from_previous_note=False, this parameter will be ignored
    data_to_evaluate : str, default None
        This is the anndata.obs column (`anndata.obs[data_to_evaluate]`) to be used for analysis, e.g. "condition"

    Returns
    -------
    anndata.AnnData :
        Loaded anndata object.
    """
    def loading_adata(NUM):
        """ TODO add documentation """
        pathway = ch.fetch_info_txt()
        files = os.listdir(''.join(pathway))
        loading = "anndata_" + str(NUM)
        if any(loading in items for items in files):
            for file in files:
                if loading in file:
                    anndata_file = file
        else:  # In case the user provided an inexistent anndata number
            sys.exit(loading + " was not found in " + pathway)

        return ''.join(pathway) + "/" + anndata_file

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
        return data

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
        return data


def saving_anndata(anndata, current_notebook):
    """
    Save your anndata object

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata object to save.
    current_notebook : int
        The number of the current notebook.
    """
    if not isinstance(current_notebook, int):
        raise TypeError(f"Invalid type! Current_notebook has to be int got {current_notebook} of type {type(current_notebook)}.")

    adata_output = os.path.join(anndata.uns["infoprocess"]["Anndata_path"], "anndata_" + str(current_notebook) + "_" + anndata.uns["infoprocess"]["Test_number"] + ".h5ad")
    anndata.write(filename=adata_output)

    print(f"Your new anndata object is saved here: {adata_output}")


def pseudobulk_table(adata, groupby, how="mean"):
    """
    Get a pseudobulk table of values per cluster.

    TODO avoid adata.copy()

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object with counts in .X.
    groupby : str
        Column name in adata.obs from which the pseudobulks are created.
    how : str, default "mean"
        How to calculate the value per group (psuedobulk). Can be one of "mean" or "sum".

    Returns
    -------
    pandas.DataFrame :
        DataFrame with aggregated counts (adata.X). With groups as columns and genes as rows.
    """
    adata = adata.copy()
    adata.obs[groupby] = adata.obs[groupby].astype('category')

    # Fetch the mean/ sum counts across each category in cluster_by
    res = pd.DataFrame(columns=adata.var_names, index=adata.obs[groupby].cat.categories)
    for clust in adata.obs[groupby].cat.categories:

        if how == "mean":
            res.loc[clust] = adata[adata.obs[groupby].isin([clust]), :].X.mean(0)
        elif how == "sum":
            res.loc[clust] = adata[adata.obs[groupby].isin([clust]), :].X.sum(0)

    res = res.T  # transpose to genes x clusters (switch columns with rows)
    return res


def split_list(lst, n):
    """
    Split list into n chunks.

    Parameters
    -----------
    lst : list
        List to be chunked
    n : int
        Number of chunks.

    Returns
    -------
    list :
        List of lists (chunks).
    """
    chunks = []
    for i in range(0, n):
        chunks.append(lst[i::n])

    return chunks


def write_list_file(lst, path):
    """
    Write a list to a file with one element per line.

    Parameters
    -----------
    lst : list
        A list of values/strings to write to file
    path : str
        Path to output file.
    """

    lst = [str(s) for s in lst]
    s = "\n".join(lst)

    with open(path, "w") as f:
        f.write(s)


def read_list_file(path):
    """
    Read a list from a file with one element per line.

    Parameters
    ----------
    path : str
        Path to read file from.

    Returns
    -------
    list :
        List of strings read from file.
    """

    f = open(path)
    lst = f.read().splitlines()  # get lines without "\n"
    f.close()

    return lst


def clear():
    """
    Clear stout of console or jupyter notebook.
    https://stackoverflow.com/questions/37071230/clear-overwrite-standard-output-in-python
    """
    import platform

    if _is_notebook():
        check_module("IPython")
        from IPython.display import clear_output

        clear_output(wait=True)
    elif platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


def setup_R(r_home=None):
    """
    Setup R installation for rpy2 use.

    Parameters:
    -----------
    r_home : str, default None
        Path to the R home directory. If None will construct path based on location of python executable.
        E.g for ".conda/scanpy/bin/python" will look at ".conda/scanpy/lib/R"

    """
    # Set R installation path
    if not r_home:
        # https://stackoverflow.com/a/54845971
        r_home = join(dirname(dirname(Path(sys.executable).as_posix())), "lib", "R")

    if not exists(r_home):
        raise Exception(f'Path to R installation does not exist! Make sure R is installed. {r_home}')

    os.environ['R_HOME'] = r_home


def _none2null(none_obj):
    """ rpy2 converter that translates python 'None' to R 'NULL' """
    # See https://stackoverflow.com/questions/65783033/how-to-convert-none-to-r-null
    from rpy2.robjects import r

    return r("NULL")

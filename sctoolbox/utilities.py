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
import time
import shutil
import tempfile
import warnings
from scipy.sparse import issparse
import apybiomart
import requests
import subprocess

from os.path import join, dirname, exists
from pathlib import Path
from IPython.core.magic import register_line_magic
from IPython.display import HTML, display


# ------------------ Packages and tools ----------------- #

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


def get_binary_path(tool):
    """ Get path to a binary commandline tool. Looks either in the local dir, on path or in the dir of the executing python binary.

    Parameters
    ----------
    tool : str
        Name of the commandline tool to be found.

    Returns
    -------
    str :
        Full path to the tool.
    """

    python_dir = os.path.dirname(sys.executable)
    if os.path.exists(tool):
        tool_path = f"./{tool}"

    else:

        # Check if tool is available on path
        tool_path = shutil.which(tool)
        if tool_path is None:

            # Search for tool within same folder as python (e.g. in an environment)
            python_dir = os.path.dirname(sys.executable)
            tool_path = shutil.which(tool, path=python_dir)

    # Check that tool is executable
    if tool_path is None or shutil.which(tool_path) is None:
        raise ValueError(f"Could not find an executable for {tool} on path.")

    return tool_path


def run_cmd(cmd):
    """
    Run a commandline command.

    Parameters
    ----------
    cmd : str
        Command to be run.
    """
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"Command '{cmd}' ran successfully!")
    except subprocess.CalledProcessError as e:
        # print(f"Error running command '{cmd}': {e}")
        if e.output is not None:
            print(f"Command standard output: {e.output.decode('utf-8')}")
        if e.stderr is not None:
            print(f"Command standard error: {e.stderr.decode('utf-8')}")
        raise e


# ------------------- Multiprocessing ------------------- #

def get_pbar(total, description):
    """
    Get a progress bar depending on whether the user is using a notebook or not.

    Parameters
    ----------
    total : int
        Total number elements to be shown in the progress bar.
    description : str
        Description to be shown in the progress bar.

    Returns
    -------
    tqdm
        A progress bar object.
    """

    if _is_notebook() is True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    pbar = tqdm(total=total, desc=description)
    return pbar


def monitor_jobs(jobs, description="Progress"):
    """
    Monitor the status of jobs submitted to a pool.

    Parameters
    ----------
    jobs : list of job objects
        List of job objects, e.g. as returned by pool.map_async().
    description : str, default "Progress"
        Description to be shown in the progress bar.
    """

    if isinstance(jobs, dict):
        jobs = list(jobs.values())

    # Wait for all jobs to finish
    n_ready = sum([job.ready() for job in jobs])
    pbar = get_pbar(len(jobs), description)
    while n_ready != len(jobs):
        if n_ready != pbar.n:
            pbar.n = n_ready
            pbar.refresh()
        time.sleep(1)
        n_ready = sum([job.ready() for job in jobs])

    pbar.n = n_ready  # update progress bar to 100%
    pbar.refresh()
    pbar.close()


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


def check_columns(df, columns, name="dataframe"):
    """
    Utility to check whether columns are found within a pandas dataframe.

    Parameters
    ------------
    df : pandas.DataFrame
        A pandas dataframe to check.
    columns : list
        A list of column names to check for within 'df'.

    Raises
    --------
    KeyError
        If any of the columns are not in 'df'.
    """

    df_columns = df.columns

    not_found = []
    for column in columns:  # for each column to be checked
        if column is not None:
            if column not in df_columns:
                not_found.append(column)

    if len(not_found) > 0:
        error_str = f"Columns '{not_found}' are not found in {name}. Available columns are: {list(df_columns)}"
        raise KeyError(error_str)


def check_file_ending(file, pattern="gtf"):
    """
    Check if a file has a certain file ending.

    Parameters
    ----------
    file : str
        Path to the file.
    pattern : str or regex
        File ending to be checked for. If regex, the regex must match the entire string.

    Raises
    ------
    ValueError
        If file does not have the expected file ending.
    """

    valid = False
    if is_regex:
        if re.match(pattern, file):
            valid = True

    else:
        if file.endswith(pattern):
            valid = True

    if not valid:
        raise ValueError(f"File '{file}' does not have the expected file ending '{pattern}'")


def is_regex(regex):
    """
    Check if a string is a valid regex.

    Parameters
    ----------
    regex : str
        String to be checked.

    Returns
    -------
    boolean :
        True if string is a valid regex, False otherwise.
    """

    try:
        re.compile(regex)
        return True

    except re.error:
        return False


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
    --------
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


def sanitize_string(s, char_list, replace="_"):
    """
    Replace every occurrence of given substrings.

    Parameters
    ----------
    s : str
        String to sanitize
    char_list : list of str
        Strings that should be replaced.
    replace : str, default "_"
        Replacement of substrings.

    Returns
    -------
    str :
        Sanitized string.
    """

    for char in char_list:
        s = s.replace(char, replace)

    return s


def sanitize_sheetname(s, replace="_"):
    """
    Alters given string to produce a valid excel sheetname.
    https://www.excelcodex.com/2012/06/worksheets-naming-conventions/

    Parameters
    ----------
    s : str
        String to sanitize
    replace : str, default "_"
        Replacement of substrings.

    Returns
    -------
    str :
        Valid excel sheetname
    """

    return sanitize_string(s, char_list=["\\", "/", "*", "?", ":", "[", "]"], replace=replace)[0:31]


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
    @register_line_magic
    def bgcolor(color, cell=None):
        """
        Set background color of current jupyter cell. Adapted from https://stackoverflow.com/a/53746904.
        Note: Jupyter notebook v6+ needed

        Change color of the cell by either calling the function
        `bgcolor("yellow")`
        or with magic (has to be first line in cell!)
        `%bgcolor yellow`

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
        """
        script = f"""
                var cell = this.closest('.code_cell');
                var editor = cell.querySelector('.CodeMirror-sizer');
                editor.style.background='{color}';
                this.parentNode.removeChild(this)
                """

        display(HTML(f'<img src onerror="{script}">'))


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
        if path != "":
            os.makedirs(path, exist_ok=True)


def get_temporary_filename(tempdir="."):
    """ Get a writeable temporary filename by creating a temporary file and closing it again. """

    filehandle = tempfile.NamedTemporaryFile(mode="w", dir=tempdir, delete=True)
    filename = filehandle.name
    filehandle.close()  # remove the file again

    return filename


def remove_files(file_list):
    """ Delete all files in a file list. Prints a warning if deletion was not possible. """

    for f in file_list:
        try:
            os.remove(f)
        except Exception as e:
            warnings.warn(f"Could not remove file {f}. Exception was: {e}")


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
        """
        Loading information of pathway where is stored the anndata object.

        Parameters
        ----------
        NUM = int
            The number of a particular notebook that created the latest anndata object.

        Returns
        -------
        Str : The name of the latest anndata object with its pathway.
        """
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

    if data_to_evaluate is not None:
        if isinstance(data_to_evaluate, str) is False:  # Close if the anndata.obs is not correct
            sys.exit(m2)

    if is_from_previous_note is True:  # Load anndata object from previous notebook
        try:
            ch.check_notebook(which_notebook)
        except TypeError:
            sys.exit(m1)

        file_path = loading_adata(which_notebook)
        data = sc.read_h5ad(filename=file_path)  # Loading the anndata

        print(f"Source: {file_path}")

        if data_to_evaluate is not None:
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

        print(f"Source: {file_path}")

        if data_to_evaluate is not None:
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

    adata_output = os.path.join(anndata.uns["infoprocess"]["Anndata_path"], "anndata_" + str(current_notebook) + "_" + anndata.uns["infoprocess"]["Run_id"] + ".h5ad")
    anndata.write(filename=adata_output)

    print(f"Your new anndata object is saved here: {adata_output}")


def pseudobulk_table(adata, groupby, how="mean", layer=None,
                     percentile_range=(0, 100), chunk_size=1000):
    """
    Get a pseudobulk table of values per cluster.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object with counts in .X.
    groupby : str
        Column name in adata.obs from which the pseudobulks are created.
    how : str, default "mean"
        How to calculate the value per group (psuedobulk). Can be one of "mean" or "sum".
    percentile_range : tuple of 2 values, default (0,100)
        The percentile of cells used to calculate the mean/sum for each feature.
        Is used to limit the effect of individual cell outliers, e.g. by setting (0,95) to exclude high values in the calculation.
    chunk_size : int, default 1000
        If percentile_range is not default, chunk_size controls the number of features to process at once. This is used to avoid memory issues.

    Returns
    -------
    pandas.DataFrame :
        DataFrame with aggregated counts (adata.X). With groups as columns and genes as rows.
    """

    groupby_categories = adata.obs[groupby].astype('category').cat.categories

    if isinstance(percentile_range, tuple) is False:
        raise TypeError("percentile_range has to be a tuple of two values.")

    if layer is not None:
        mat = adata.layers[layer]
    else:
        mat = adata.X

    # Fetch the mean/ sum counts across each category in cluster_by
    res = pd.DataFrame(index=adata.var_names, columns=groupby_categories)
    for column_i, clust in enumerate(groupby_categories):

        cluster_values = mat[adata.obs[groupby].isin([clust]), :]

        if percentile_range == (0, 100):  # uses all cells
            if how == "mean":
                vals = cluster_values.mean(0)
                res[clust] = vals.A1 if issparse(cluster_values) else vals
            elif how == "sum":
                vals = cluster_values.sum(0)
                res[clust] = vals.A1 if issparse(cluster_values) else vals
        else:

            n_features = cluster_values.shape[1]

            # Calculate mean individually per gene/feature
            for i in range(0, n_features, chunk_size):

                chunk_values = cluster_values[:, i:i + chunk_size]
                chunk_values = chunk_values.A if issparse(chunk_values) else chunk_values
                chunk_values = chunk_values.astype(float)

                # Calculate the lower and upper limits for each feature
                limits = np.percentile(chunk_values, percentile_range, axis=0, method="lower")
                lower_limits = limits[0]
                upper_limits = limits[1]

                # Set values outside the limits to nan and calculate mean/sum
                bool_filter = (chunk_values < lower_limits) | (chunk_values > upper_limits)
                chunk_values[bool_filter] = np.nan

                if how == "mean":
                    vals = np.nanmean(chunk_values, axis=0)
                elif how == "sum":
                    vals = np.nansum(chunk_values, axis=0)

                res.iloc[i:i + chunk_size, column_i] = vals

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


def split_list_size(lst, max_size):
    """
    Split list into chunks of max_size.

    Parameters
    -----------
    lst : list
        List to be chunked
    max_size : int
        Max size of chunks.

    Returns
    -------
    list :
        List of lists (chunks).
    """

    chunks = []
    for i in range(0, len(lst), max_size):
        chunks.append(lst[i:i + max_size])

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


def fill_na(df, inplace=True, replace={"bool": False, "str": "-", "float": 0, "int": 0, "category": ""}):
    """
    Fill all NA values in pandas depending on the column data type

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame object with NA values over multiple columns
    inplace : boolean, default True
        Whether the DataFrame object is modified inplace.
    replace :  dict, default {"bool": False, "str": "-", "float": 0, "int": 0, "category": ""}
        dict that contains default values to replace nas depedning on data type
    """

    if not inplace:
        df = df.copy()

    # Set default of missing replace value
    replace_def = {"bool": False, "str": "-", "float": 0, "int": 0, "category": ""}
    for t in ["bool", "str", "float", "int", "category"]:
        if t not in replace:
            warnings.warn(f"Value for replace key '{t}' not given. Set to default value: '{replace_def[t]}'")
            replace[t] = replace_def[t]

    for nan_col in df.columns[df.isna().any()]:
        col_type = df[nan_col].dtype.name
        if col_type == "category":
            df[nan_col] = df[nan_col].cat.add_categories(replace[col_type])
            df[nan_col].fillna(replace[col_type], inplace=True)
        elif col_type.startswith("float"):
            df[nan_col].fillna(replace["float"], inplace=True)
        elif col_type.startswith("int"):
            df[nan_col].fillna(replace["int"], inplace=True)
        elif col_type == "object":
            value_set = list({x for x in set(df[nan_col]) if x == x})
            o_type = type(value_set[0]).__name__ if value_set else "str"
            df[nan_col].fillna(replace[o_type], inplace=True)
    if not inplace:
        return df


def get_adata_subsets(adata, groupby):
    """
    Split an anndata object into a dict of sub-anndata objects based on a grouping column.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object to split.
    groupby : str
        Column name in adata.obs to split by.

    Returns
    -------
    dict :
        Dictionary of anndata objects in the format {<group1>: anndata, <group2>: anndata, (...)}.
    """

    group_names = adata.obs[groupby].astype("category").cat.categories.tolist()
    adata_subsets = {name: adata[adata.obs[groupby] == name] for name in group_names}

    return adata_subsets


def write_excel(table_dict, filename, index=False):
    """
    Write a dictionary of tables to a single excel file with one table per sheet.

    Parameters
    ----------
    table_dict : dict
        Dictionary of tables in the format {<sheet_name1>: table, <sheet_name2>: table, (...)}.
    filename : str
        Path to output file.
    index : bool, default False
        Whether to include the index of the tables in file.
    """

    # Check if tables are pandas dataframes
    for name, table in table_dict.items():
        if not isinstance(table, pd.DataFrame):
            raise Exception(f"Table {name} is not a pandas DataFrame!")

    # Write to excel
    with pd.ExcelWriter(filename) as writer:
        for name, table in table_dict.items():
            table.to_excel(writer, sheet_name=sanitize_sheetname(f'{name}'), index=index)


def add_expr_to_obs(adata, gene):
    """
    Add expression of a gene from adata.X to adata.obs as a new column.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object to add expression to.
    gene : str
        Gene name to add expression of.
    """

    boolean = adata.var.index == gene
    if sum(boolean) == 0:
        raise Exception(f"Gene {gene} not found in adata.var.index")

    else:
        idx = np.argwhere(boolean)[0][0]
        adata.obs[gene] = adata.X[:, idx].todense().A1


# -------------------- bio utils ------------------- #
# section could be its own file

def get_organism(ensembl_id, host="http://www.ensembl.org/id/"):
    """
    Get the organism name to the given Ensembl ID.

    Parameters
    ----------
    ensembl_id : str
    Any Ensembl ID. E.g. ENSG00000164690
    host : str
    Ensembl server address.

    Returns
    -------
    str :
        Organism assigned to the Ensembl ID
    """
    # this will redirect
    url = f"{host}{ensembl_id}"
    response = requests.get(url)

    if response.status_code != 200:
        raise ConnectionError(f"Server response: {response.status_code}.\n With link {url}. Is the host/ path correct?")

    # get redirect url
    # e.g. http://www.ensembl.org/Homo_sapiens/Gene/...
    # get species name from url
    species = response.url.split("/")[3]

    # invalid id
    if species == "Multi":
        raise ValueError(f"Organism returned as '{species}' ({response.url}).\n Usually due to invalid Ensembl ID. Make sure to use an Ensembl ID as described in http://www.ensembl.org/info/genome/stable_ids/index.html")

    return species


def gene_id_to_name(ids, species):
    """
    Get Ensembl gene names to Ensembl gene id.

    Parameters
    ----------
    ids : list of str
        List of gene ids. Set to `None` to return all ids.
    species : str
        Species matching the gene ids. Set to `None` for list of available species.

    Returns
    -------
    pandas.DataFrame :
        DataFrame with gene ids and matching gene names.
    """
    if not all(id.startswith("ENS") for id in ids):
        raise ValueError("Invalid Ensembl IDs detected. A valid ID starts with 'ENS'.")

    avail_species = sorted([s.split("_gene_ensembl")[0] for s in apybiomart.find_datasets()["Dataset_ID"]])

    if species is None or species not in avail_species:
        raise ValueError("Invalid species. Available species are: ", avail_species)

    id_name_mapping = apybiomart.query(
        attributes=["ensembl_gene_id", "external_gene_name"],
        dataset=f"{species}_gene_ensembl",
        filters={}
    )

    if ids:
        # subset to given ids
        return id_name_mapping[id_name_mapping["Gene stable ID"].isin(ids)]

    return id_name_mapping


def convert_id(adata, id_col_name=None, index=False, name_col="Gene name", species="auto", inplace=True):
    """
    Add gene names to adata.var.

    Parameters
    ----------
    adata : scanpy.AnnData
        AnnData with gene ids.
    id_col_name : str, default None
        Name of the column in `adata.var` that stores the gene ids.
    index : boolean, default False
        Use index of `adata.var` instead of column name speciefied in `id_col_name`.
    name_col : str, default "Gene name"
        Name of the column added to `adata.var`.
    species : str, default "auto"
        Species of the dataset. On default, species is inferred based on gene ids.
    inplace : boolean, default True
        Whether to modify adata inplace.

    Returns
    -------
    scanpy.AnnData or None :
        AnnData object with gene names.
    """
    if not id_col_name and not index:
        raise ValueError("Either set parameter id_col_name or index.")
    elif not index and id_col_name not in adata.var.columns:
        raise ValueError("Invalid id column name. Name has to be a column found in adata.var.")

    if not inplace:
        adata = adata.copy()

    # get gene ids
    if index:
        gene_ids = list(adata.var.index)
    else:
        gene_ids = list(adata.var[id_col_name])

    # infer species from gene id
    if species == "auto":
        ensid = gene_ids[0]

        species = get_organism(ensid)

        # bring into biomart format
        # first letter of all words but complete last word e.g. hsapiens, mmusculus
        spl_name = species.lower().split("_")
        species = "".join(map(lambda x: x[0], spl_name[:-1])) + spl_name[-1]

        print(f"Identified species as {species}")

    # get id <-> name table
    id_name_table = gene_id_to_name(ids=gene_ids, species=species)

    # create new .var and replace in adata
    new_var = pd.merge(
        left=adata.var,
        right=id_name_table.set_index("Gene stable ID"),
        left_on=id_col_name,
        left_index=index,
        right_index=True,
        how="left"
    )
    new_var["Gene name"].fillna('', inplace=True)
    new_var.rename(columns={"Gene name": name_col}, inplace=True)

    adata.var = new_var

    if not inplace:
        return adata


def unify_genes_column(adata, column, unified_column="unified_names", species="auto", inplace=True):
    """
    Given an adata.var column with mixed Ensembl IDs and Ensembl names, this function creates a new column where Ensembl IDs are replaced with their respective Ensembl names.

    Parameters
    ----------
    adata: scanpy.AnnData
        AnnData object
    column: str
        Column name in adata.var
    unified_names: str, default "unified_names"
        Defines the column in which unified gene names are saved. Set same as parameter 'column' to overwrite original column.
    species : str, default "auto"
        Species of the dataset. On default, species is inferred based on gene ids.
    inplace: boolean, default True
        Whether to modify adata or return a copy.

    Returns
    -------
    scanpy.AnnData or None :
        AnnData object with modified gene column.
    """
    if column not in adata.var.columns:
        raise ValueError(f"Invalid column name. Name has to be a column found in adata.var. Available names are: {adata.var.columns}.")

    if not inplace:
        adata = adata.copy()

    # check for ensembl ids
    ensids = [el for el in adata.var[column] if el.startswith("ENS")]

    if not ensids:
        raise ValueError(f"No Ensembl IDs in adata.var['{column}'] found.")

    # infer species from gene id
    if species == "auto":
        ensid = ensids[0]

        species = get_organism(ensid)

        # bring into biomart format
        # first letter of all words but complete last word e.g. hsapiens, mmusculus
        spl_name = species.lower().split("_")
        species = "".join(map(lambda x: x[0], spl_name[:-1])) + spl_name[-1]

        print(f"Identified species as {species}")

    # get id <-> name table
    id_name_table = gene_id_to_name(ids=ensids, species=species)

    count = 0
    for index, row in adata.var.iterrows():
        if row[column] in id_name_table['Gene stable ID'].values:
            count += 1

            # replace gene id with name
            adata.var.at[index, unified_column] = id_name_table.at[id_name_table.index[id_name_table["Gene stable ID"] == row[column]][0], "Gene name"]
        else:
            adata.var.at[index, unified_column] = adata.var.at[index, column]
    print(f'{count} ensembl gene ids have been replaced with gene names')

    if not inplace:
        return adata

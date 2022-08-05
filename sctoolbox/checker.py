"""
Modules for checking the existence of directories and files
"""
# Importing modules
import os
import sys
import re


def check_notebook(notebook_num):
    """
    Check if the notebook number is int.

    TODO do we need this?

    Parameters
    ----------
    notebook_num : int
       The number of the notebook assigned for the user for both load or save an anndata object
    """
    if not isinstance(notebook_num, int):
        raise TypeError('Only integer allowed')


def write_info_txt(path_value, file_path="./"):
    """
    Write path to info.txt

    Parameters
    ----------
    path_value : str
        Path that is written to the info.yml.
        Adds info.txt to end if no filename is given.
    file_path : str, default ./
        Path where the info.yml is stored
    """
    pattern = re.compile(r'[<>:"\\\|\?\*]')
    if re.search(pattern, path_value):
        raise ValueError("Invalid character in directory string.")

    if os.path.isdir(file_path):
        file_path = os.path.join(file_path, "info.txt")
    else:
        raise ValueError("Invalid directory given.")

    with open(file_path, "w") as file:
        file.write(path_value)


def fetch_info_txt(file_path="./info.txt"):
    """
    Get path stored in the info.txt file

    Parameters
    ----------
    file_path : str
        Path to info.txt file.

    Returns
    -------
    str :
        Path that was stored in the first line of info.txt.
    """
    with open(file_path, "r") as file:
        return file.readline()


def check_cuts(ans, limit1, limit2):  # Checking cutoffs validity
    """
    Checking if a given value range into acceptable limits

    Parameters
    ----------
    ans : str of int or int
        The number to check validity described as a string.
    limit1 : int or float
        The lower limit number.
    limit2 : int or float
        The upper limit number.

    Returns
    -------
    str :
        Returns "valid" if in bounds and "invalid" if out of given bounds.

    Notes
    -----
    Author: Guilherme Valente
    """
    quiters = ["q", "quit"]

    # check if the first input is string or not
    # in the context of pipeline the ans is always coming as STRING,
    # however it could be provided also as an integer.
    if isinstance(ans, str):
        ans = ans.replace('.', "", 1)
        if not ans.isdigit():
            if ans in quiters:
                sys.exit("You quit and lost all modifications")
            else:
                sys.exit("You must provide string or number!")

    # Check the range of provided integer input
    x = float(ans)
    if x >= limit1 and x <= limit2:
        return "valid"
    else:
        return "invalid"


def check_options(answer, options=["q", "quit", "y", "yes", "n", "no"]):
    """
    Check if answers from input() command were properly replied

    TODO is this needed?

    Parameters
    ----------
    answer : str
        The answer provided by the user.
    options : list, default ["q", "quit", "y", "yes", "n", "no"]
        Options allowed.

    Returns
    -------
        boolean :
            Returns True if ans is a string and in options. Exits python if "q" or "quit".
    """
    if type(answer) is str:
        answer = answer.lower()

        if answer in options:
            check_quit(answer)

            return True

    return False


def check_quit(answer):
    """
    Exit python if the answer is q or quit.

    TODO is this needed?

    Parameters
    ----------
    answer : str
        String with answer.
    """
    if answer in ["q", "quit"]:
        sys.exit("You quit and lost all modifications :(")


def check_requirements(anndata, current_notebook, check_previous=True):
    """
    Check if the current anndata object has all requirements to be analysed by the current notebook.

    Parameters
    ----------
    anndata : anndata.Anndata
        Anndata object to check for requirements.
    current_notebook : int
        The number of the current notebook. It is important to set the correct number because each notebook has its
        own mandatory anndata requirements.
    check_previous : boolean
        If True will check all dependencies of previous notebooks.

    Requirements of each notebook
    -----------------------------
    Notebook 3 :
        Requires filtered total_counts.
    """
    # Check if the current_notebook is int. Then, check if the anndata fits the requirements.
    check_notebook(current_notebook)

    # set True if any check was done
    class Checked:
        any_check = False

    def do_check(c, n, p):
        """ Do check if notebook is a previous one or is equal to current. """
        if p and c >= n or c == n:
            Checked.any_check = True
            return True
        return False

    # TODO add missing checks
    if do_check(current_notebook, 1, check_previous):
        # assembling anndata
        print("Check 1 to be implemented")

    if do_check(current_notebook, 2, check_previous):
        # qc and filtering
        print("Check 2 to be implemented")

    if do_check(current_notebook, 3, check_previous):
        # normalization, correction and comparison
        if "total_counts" not in str(anndata.uns["infoprocess"]["Cell filter"]):
            raise ValueError("This notebook demands total_counts filtered. Run notebook 2.")

    if do_check(current_notebook, 4, check_previous):
        # clustering
        print("Check 4 to be implemented")

    if do_check(current_notebook, 5, check_previous):
        # annotation
        print("Check 5 to be implemented")

    if do_check(current_notebook, 6, check_previous):
        # differential expression
        print("Check 6 to be implemented")

    if do_check(current_notebook, 7, check_previous):
        # general plots
        print("Check 7 to be implemented")

    if do_check(current_notebook, 8, check_previous):
        # cell counting
        print("Check 8 to be implemented")

    if do_check(current_notebook, 9, check_previous):
        # velocity
        print("Check 9 to be implemented")

    if do_check(current_notebook, 10, check_previous):
        # trajectory
        print("Check 10 to be implemented")

    if do_check(current_notebook, 11, check_previous):
        # receptor-ligand
        print("Check 11 to be implemented")

    if do_check(current_notebook, 12, check_previous):
        # cyber
        print("Check 12 to be implemented")

    if not Checked.any_check:
        raise ValueError(f"Invalid notebook number detected. Got current_notebook={current_notebook}.")

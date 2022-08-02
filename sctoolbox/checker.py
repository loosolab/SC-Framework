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


def check_requirements(anndata, current_notebook=None):
    """
    Check if the current anndata object has all requirements to be analysed by the current notebook.

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata to check for requirements.
    current_notebook : int, default None
        The number of the current notebook. It is important to set the correct number because each notebook has its
        own mandatory anndata requirements.

    Requirements of each notebook
    -----------------------------
    Notebook 3 :
        Requires filtered total_counts.
    """
    # Check if the current_notebook is int. Then, check if the anndata fits the requirements.
    check_notebook(current_notebook)
    if current_notebook == 3:
        if "total_counts" not in str(anndata.uns["infoprocess"]["Cell filter"]):
            raise ValueError("Notebook 3 demands total_counts filtered. Run notebook 2 before the 3rd notebook.")
    # TODO : add other notebooks as elif
    else:
        raise ValueError("Set the current_notebook properly.")
    print(anndata)

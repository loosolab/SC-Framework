"""
Modules for checking the existence of directories and files
"""
# Importing modules
import os
import sys
import re


def check_notebook(notebook_num):
    '''Check if the notebook number is int.
    Parameters
    ----------
    notebook_num : Int
       The number of the notebook assigned for the user for both load or save an anndata object
    '''
    if isinstance(notebook_num, int) is False:
        raise TypeError('Only integer allowed')


def write_info_txt(path_value, file_path="./"):
    ''' Write path to info.txt

    Parameters:
    ===========
    path_value : String
        path that is written to the info.yml.
        Adds info.txt to end if no filename is given.
    file_path : String
        path where the info.yml is stored
    '''

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
    ''' Get path stored in the info.txt file
    Parameters:
    ===========
    file_path : String
        full path of info.txt file

    Returns:
    ========
    path as string that was stored in the first line of info.txt
    '''

    with open(file_path, "r") as file:
        return file.readline()


def check_cuts(ANS, LIMIT1, LIMIT2):  # Checking cutoffs validity
    '''
    Checking if a given value range into acceptable limits

    Parameter
    ----------
    ANS : String (integer) or Integer
        The number to check validity described as an string.
    LIMIT1 : Int or float
        The lower limit number
    LIMIT2 : Int or float
        The uper limit number

    Return
    ----------
    The True or False
    '''
    # Author: Guilherme Valente
    quiters = ["q", "quit"]

    # check if the first input is string or not
    # in the context of pipeline the ANS is always coming as STRING,
    # however it could be provided also as an integer.
    if isinstance(ANS, str):
        ANS = ANS.replace('.', "", 1)
        if not ANS.isdigit():
            if ANS in quiters:
                sys.exit("You quit and lost all modifications")
            else:
                sys.exit("You must provide string or number!")

    # Check the range of provided integer input
    x = float(ANS)
    if x >= LIMIT1 and x <= LIMIT2:
        return("valid")
    else:
        return("invalid")


def check_options(ANS, OPTS1=["q", "quit", "y", "yes", "n", "no"]):
    '''
    Check if answers from input() command were properly replied
    Parameters
    ------------
    ANS : String
        The answer provided by the user.
    OPTS1 : List
        Options allowed. Default : ["q", "quit", "y", "yes", "n", "no"]
    Returns
    -----------
        Return True or False
    '''
    if type(ANS) is str:
        ANS = ANS.lower()
        if ANS in OPTS1:
            check_quit(ANS)
            return True
    return False


def check_quit(answer):
    '''Quit the functions if the answer is q or quite.
    Parameters
    ----------
    answer: String
        The answer of a user
    Return
    ----------
        Quit the process
    '''
    if answer in ["q", "quit"]:
        sys.exit("You quit and lost all modifications :(")


def chek_requirements(anndata, current_notebook=None):
    '''Check if the current anndata object has all requirements to be analysed by the current notebook.
    Parameters
    ----------
    anndata : Anndata object
    current_notebook : Int
        The number of the current notebook. It is important to set the correct number because each notebook has its
        own mandatory anndata requirements.

    Requirements of each notebook
    ----------
        Notebook 3 demands total_counts filtered

    Return
    ---------
        Report the information is missing in the anndata object.
    '''
# Messages and others
    m1 = "Set the current_notebook[INT]"
    m2 = "Notebook 3 demands total_counts filtered. Run notebook 2 before the 3rd notebook."

# Check if the current_notebook is int
    try:
        ch.check_notebook(current_notebook)
        if current_notebook == 3:
            if "total_counts" not in str(anndata.uns["infoprocess"]["Cell filter"]):
                sys.exit(m2)
        display(anndata)
    except:
        sys.exit(m1)

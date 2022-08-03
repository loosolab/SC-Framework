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
    '''
    Get path stored in the info.txt file
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
    '''
    Quit the functions if the answer is q or quite.
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


def check_requirements(anndata, current_notebook):
    """
    Check if the current anndata object has all requirements to be analysed by the current notebook.

    Parameters
    ----------
    anndata : anndata.Anndata
        Anndata object to check for requirements.
    current_notebook : int
        The number of the current notebook. It is important to set the correct number because each notebook has its
        own mandatory anndata requirements.

    Requirements of each notebook
    -----------------------------
        Notebook 3 demands total_counts filtered
    """
    # Check if the current_notebook is int. Then, check if the anndata fits the requirements.
    check_notebook(current_notebook)

    # TODO add missing checks
    if current_notebook >= 1:
        # assembling anndata
        print("Check 1 to be implemented")
    
    if current_notebook >= 2:
        # qc and filtering
        print("Check 2 to be implemented")
    
    if current_notebook >= 3:
        # normalization, correction and comparison
        if "total_counts" not in str(anndata.uns["infoprocess"]["Cell filter"]):
            raise ValueError("This notebook demands total_counts filtered. Run notebook 2.")
    
    if current_notebook >= 4:
        # clustering
        print("Check 4 to be implemented")
    
    if current_notebook >= 5:
        # annotation
        print("Check 5 to be implemented")
    
    if current_notebook >= 6:
        # differential expression
        print("Check 6 to be implemented")
    
    if current_notebook >= 7:
        # general plots
        print("Check 7 to be implemented")
    
    if current_notebook >= 8:
        # cell counting
        print("Check 8 to be implemented")
    
    if current_notebook >= 9:
        # velocity
        print("Check 9 to be implemented")
    
    if current_notebook >= 10:
        # trajectory
        print("Check 10 to be implemented")
    
    if current_notebook >= 11:
        # receptor-ligand
        print("Check 11 to be implemented")
    
    if current_notebook >= 12:
        # cyber
        print("Check 12 to be implemented")
    
    else:
        raise ValueError(f"Invalid notebook number detected. Got current_notebook={current_notebook}.")

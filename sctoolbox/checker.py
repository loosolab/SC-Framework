"""
Modules for checking the existence of directories and files
"""
#Importing modules
from multiprocessing.sharedctypes import Value
import sctoolbox
import os
from os import path
import sys
import sctoolbox.creators as cr
import sctoolbox.utilities as ut
import warnings
import re

def check_notebook(notebook_num):
    '''Check if the notebook number is int.
    Parameters
    ----------
    notebook_num : Int
       The number of the notebook assigned for the user for both load or save an anndata object
    '''
    if isinstance(notebook_num, int) == False:
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

    pattern = re.compile('[<>:"\\\|\?\*]')
    if re.search(pattern,path_value):
        raise ValueError("Invalid character in directory string.")

    if os.path.isdir(file_path):
        file_path = os.path.join(file_path,"info.txt")
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

def check_cuts(ANS, LIMIT1, LIMIT2): #Checking cutoffs validity
    '''
    Checking if a given value range into acceptable limits
    
    Parameter
    ----------
    ANS : String
        The number to check validity described as an string.
    LIMIT1 : Int or float
        The lower limit number
    LIMIT2 : Int or float
        The uper limit number

    Return
    ----------
    The True or False
    '''
    #Author: Guilherme Valente
    quiters=["q", "quit"]
    if ut.is_str_numeric(ANS) == True:
        x=float(ANS)
        if x >=LIMIT1 and x <= LIMIT2:
            return(True)
        else:
            return(False)
    else:
        if ANS in quiters:
            sys.exit("You quit and lost all modifications :(")
        else:
            return(False)

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
        ANS=ANS.lower()
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

import pytest
import sys

def check_options(ANS, OPTS1, OPTS2):
    '''
    Check if answers from input() command were properly replied
    Parameters
    ------------
    ANS : String
        The answer provided by the user.
    OPTS1 : List
        Options to quit the execution: ["q", "quit"]
    OPTS2 : List
        Options to consider the answer valid, e.g. ["y", "yes", "n", "no"]
    '''
    #Author : Guilherme Valente
    #Editor: Noah Knoppik
    ANS=ANS.lower()
    if ANS in OPTS1:
        #input for quit
        #maybe sys.exit("You quit and lost all modifications :(")
        return 0
    elif ANS in OPTS2:
        #valid answer
        return 1
    else: 
        #invalid input
        #sys.exit or re-run input function?
        return 2



def test_check_options():
    #user input (input, exit code)
    user_inputs = [("q", 0),("y",1),("", 2)]
    opts1 = ["q", "quit"]
    opts2 = ["y", "yes", "n", "no"]
    for input in user_inputs:
        assert check_options(input[0],opts1 ,opts2 ) == input[1]
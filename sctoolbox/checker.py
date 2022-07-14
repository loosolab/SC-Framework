"""
Modules for checking the existence of directories and files
"""
#Importing modules
import sctoolbox
import os
from os import path
import sys
import sctoolbox.creators as cr
##################################
def check_infoyml(VALUE=None, TASK=None):
    '''
    This is to create load or modify the info.txt file that store the main output path
    Parameters:
    ===========
    VALUE : String.
        The pathway to be intored in the info.txt
    TASKE : String.
        Activity to be performed using the info.txt
    '''
    #Author : Guilherme Valente
    if TASK == "create":
        print(VALUE, file=open("./info.txt", "w")) #Create a new key and add an information.
    elif TASK == "give_path":
        with open('./info.txt') as a:
            return next(a).split(":")[1].strip()

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
    if ANS.replace('.', "", 1).isdigit() == True:
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

def check_input_path_velocity(path_QUANT, tenX, assembling_10_velocity, dtype="filtered"): #Check if the main directory of solo (MAINPATH) and files exist to assembling the anndata object to make velocyte analysis. tenX is the configuration of samples in the 10X.yml.
    '''
    Checking if the paths are proper for assembling 10X for velocity.
    
    Parameters
    =============
    path_QUANT : String.
        The directory where the quant folder from snakemake preprocessing is located.
    tenX : List.
        Configurations to setup the samples for anndata assembling. It must containg the sample, the word used in snakemake to assign the condition, and the condition, e.g., sample1:condition:room_air
    assembling_10_velocity : Boolean
        If True, the anndata 10X assembling for velocity will be executed.
    dtype : String.
        The type of Solo data choose, which default is filtered. The options are raw or filtered.
    '''
    #Author : Guilherme Valente
    #Tracking is pathways exist.
    def checking_paths(CHECK_PATH, MES):
        if path.exists(path_QUANT):
            return("valid")
        else:
            sys.exit(MES)
            
    #Messages and others
    go_assembling=False
    closed_gene_path="/solo/Gene/" + dtype
    closed_velocito_path="/solo/Velocyto/" + dtype
    genes_path_files=['barcodes.tsv', 'genes.tsv', 'matrix.mtx']
    velocyto_path_files=["ambiguous.mtx", "barcodes.tsv", "genes.tsv", "spliced.mtx", "unspliced.mtx"]
    m1="Set dtype as raw or filtered."
    m2=path_QUANT + "\nis wrong or not found.\n"
    m3="\nis wrong or not found.\n"

    #Checking if the anndata 10X velocity should be assembled.
    if assembling_10_velocity == True:
        if dtype == "filtered" or dtype == "raw":
            go_assembling=True
        else:
            sys.exit(m1)
    #Checking if the files are appropriated
    if go_assembling == True:
        #Check if */quant exist
        if checking_paths(path_QUANT, m2) == "valid":
            path_QUANT=path_QUANT.replace("//", "/")
            if path_QUANT.endswith('/'):
                path_QUANT=path_QUANT[:-1]
            return(path_QUANT)
            list_quant_folders=[b for b in os.listdir(path_QUANT)] #List the folders inside the quant folder.
        #Check if the */quant/* files exist. These files are stored at genes_path_files and velocyto_path_files lists
        for a in list_quant_folders:
            path_solo_gene=path_QUANT + "/" + a + closed_gene_path
            path_solo_velocyto=path_QUANT + "/" + a + closed_velocito_path
            if checking_paths(path_solo_gene, m2) == "valid": #Checking if *sample*/solo/Gene/filtered exist
                for b in genes_path_files:  #Checking if *sample*/solo/Gene/filtered/* files exist
                    if b not in os.listdir(path_solo_gene):
                        sys.exit(path_solo_gene + "/" + b + m3)
            if checking_paths(path_solo_velocyto, m2) == "valid": #Checking if *sample*/solo/Gene/filtered exist
                for b in velocyto_path_files:  #Checking if *sample*/solo/Gene/filtered/* files exist
                    if b not in os.listdir(path_solo_velocyto):
                        sys.exit(path_solo_velocyto + "/" + b + m3)


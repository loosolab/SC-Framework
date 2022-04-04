"""
Modules for checking the existence of directories and files
"""
#Importing modules
import sctoolbox
import os
from os import path
import sys
import yaml
import sctoolbox.creators as cr
##################################
#Lists
list_velocyto=["barcodes.tsv", "genes.tsv", "spliced.mtx", "unspliced.mtx", "ambiguous.mtx"]
list_gene=["matrix.mtx"]
list_solo=["quant"]
##################################

#Functions
def error_message(DIRE): #Print error messages and close pipeline
    sys.exit("The " + DIRE + "\nis wrong or not found.\n")

def filesDir_for_check(DIR): #List files in a given directory for check the existence of specific directories
    global list_velocyto
    global list_gene
    global list_solo
    list_data=[b for b in os.listdir(DIR)]
    if "Gene" in DIR:
        filesDir_for_check2(list_gene, list_data) #Gene and Velocyto folders have the proper data (.tsv and .mtx)
    elif "Velocyto" in DIR:
        filesDir_for_check2(list_velocyto, list_data) #Gene and Velocyto folders have the proper data (.tsv and .mtx)
    else:
        filesDir_for_check2(list_solo, list_data) #Check if the quant folder is in the input directory

def filesDir_for_check2(LIST, LISTDATA): #Check the existence of specific directories
    for c in LIST:
        if c not in LISTDATA:
            list_data_not_found.append(c)

def input_path_velocity(MAINPATH, tenX, DTYPE): #Check if the main directory of solo (MAINPATH) and files exist to assembling the anndata object to make velocyte analysis. tenX is the configuration of samples in the 10X.yml, the DTYPE is the type of Solo data choose (raw or filtered)
    global list_data_not_found
    global list_velocyto
    global list_gene
    if path.exists(MAINPATH):
        list_data_not_found=[]
        filesDir_for_check(MAINPATH) #Check if the quant folder exist
        if len(list_data_not_found) != 0: #Stop the script if quant folder is not found
            error_message(MAINPATH + "/quant")
        else:
           cr.infoyml("Input_solo_path", MAINPATH + "/quant") #Printing the input in yaml
        del list_data_not_found
        for a in tenX: #Checking the presence of all sample directories
            sample=a.split(":")[0]
            list_data_not_found=[]
            if path.exists(MAINPATH + "/quant/" + sample + "/solo/Gene/" + DTYPE) != True or path.exists(MAINPATH + "/quant/" + sample + "/solo/Velocyto/" + DTYPE) != True: #Checking the existence of quant/sampleX/solo/Gene/ or Velocyto
                error_message(MAINPATH + "/quant/" + sample + "/solo/Gene/" + DTYPE)
                print("OR\n")
                error_message(MAINPATH + "/quant/" + sample + "/solo/Velocyto/" + DTYPE)
            else: #Check if .mtx and .tsv files exist in Gene and Velocyto folders
                filesDir_for_check(MAINPATH + "/quant/" + sample + "/solo/Gene/" + DTYPE) 
                filesDir_for_check(MAINPATH + "/quant/" + sample + "/solo/Velocyto/" + DTYPE)
                if len(list_data_not_found) == 0:
                    print("Sample " + sample + " have the proper data for assembling an anndata for velocity analysis.")
                else:
                    print("Missing data in Gene or Velocyto child directories to assembling anndata object for velocity: " + str(list_data_not_found))
    else:
        error_message(MAINPATH)

def output_path(OUTPATH, TEST): #Check if the directory for output exist.
    path1=OUTPATH + "/results"
    path2=OUTPATH + "/results/" + TEST
    if path.exists(path1) != True: #Check if result dir exist
        cr.directory(path1)
    if path.exists(path2) != True: #Check if result dir exist
        cr.directory(path2)
    cr.infoyml("Output_path", path2) #Printing the output dir in yaml

def check_infoyml(KEY): #Check the existence of a given key in info.yml and load the choose key.
    if path.exists("info.yml"):
        yml=yaml.load(open("info.yml", "r"), Loader=yaml.FullLoader)
        if KEY in yml: #Check if the key exist.
            return(yml[KEY]) #Loading the choose key.
        else:
            sys.exit("The " + KEY + " is absent in info.yml.")
    else:
        error_message("./info.yml")

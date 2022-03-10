"""
Modules for creating files or directories
"""
#Importing modules
import os
from os import path
import yaml

##################################
#Functions
def infoyml(KEY, INFORMATION): #Save infor in yml file. The KEY will be the key of yaml file. The INFORMATION must be an string.
    information1=KEY + ":" + "\n - " + INFORMATION
    information2="\n - " + INFORMATION
    if path.exists("info.yml"):
        yml=yaml.load(open("info.yml", "r"), Loader=yaml.FullLoader)
        if KEY not in yml: #Check if the key exist.
            print("The info.yml has a new key.")
            print(information1, file=open("./info.yml", "a")) #Create a new key and add an information.
        else:
            if INFORMATION not in yml[KEY]: #Check if the information exist in the key.
                print("The info.yml information was updated.")
                print(information2, file=open("./info.yml", "a")) #Create a new information in the key.
    else:
        print("The info.yml was created.")
        print(information1, file=open("./info.yml", "a")) #Create a key and add an information.

def directory(DIREC): #Directory creator.
    print("Creating the " + DIREC + " directory.")
    print("The " + DIREC + " directory is ready.")
    os.mkdir(DIREC)

"""
Modules for creating files or directories
"""
#Importing modules
import os
from os import path
import yaml

##################################
color_list=['red', 'blue', 'green', 'pink', 'chartreuse', 'gray', 'yellow', 'brown', 'purple', 'orange', 'wheat', 'lightseagreen', 'cyan', 'khaki', 'cornflowerblue', 'olive', 'gainsboro', 'darkmagenta', 'slategray', 'ivory', 'darkorchid', 'papayawhip', 'paleturquoise', 'oldlace', 'orangered', 'lavenderblush', 'gold', 'seagreen', 'deepskyblue', 'lavender', 'peru', 'silver', 'midnightblue', 'antiquewhite', 'blanchedalmond', 'firebrick', 'greenyellow', 'thistle', 'powderblue', 'darkseagreen', 'darkolivegreen', 'moccasin', 'olivedrab', 'mediumseagreen', 'lightgray', 'darkgreen', 'tan', 'yellowgreen', 'peachpuff', 'cornsilk', 'darkblue', 'violet', 'cadetblue', 'palegoldenrod', 'darkturquoise', 'sienna', 'mediumorchid', 'springgreen', 'darkgoldenrod', 'magenta', 'steelblue', 'navy', 'lightgoldenrodyellow', 'saddlebrown', 'aliceblue', 'beige', 'hotpink', 'aquamarine', 'tomato', 'darksalmon', 'navajowhite', 'lawngreen', 'lightsteelblue', 'crimson', 'mediumturquoise', 'mistyrose', 'lightcoral', 'mediumaquamarine', 'mediumblue', 'darkred', 'lightskyblue', 'mediumspringgreen', 'darkviolet', 'royalblue', 'seashell', 'azure', 'lightgreen', 'fuchsia', 'floralwhite', 'mintcream', 'lightcyan', 'bisque', 'deeppink', 'limegreen', 'lightblue', 'darkkhaki', 'maroon', 'aqua', 'lightyellow', 'plum', 'indianred', 'linen', 'honeydew', 'burlywood', 'goldenrod', 'mediumslateblue', 'lime', 'lightslategray', 'forestgreen', 'dimgray', 'lemonchiffon', 'darkgray', 'dodgerblue', 'darkcyan', 'orchid', 'blueviolet', 'mediumpurple', 'darkslategray', 'turquoise', 'salmon', 'lightsalmon', 'coral', 'lightpink', 'slateblue', 'darkslateblue', 'white', 'sandybrown', 'chocolate', 'teal', 'mediumvioletred', 'skyblue', 'snow', 'palegreen', 'ghostwhite', 'indigo', 'rosybrown', 'palevioletred', 'darkorange', 'whitesmoke']
##################################
#Functions
def infoyml(KEY, INFORMATION): #Save infor in yml file. The KEY will be the key of yaml file. The INFORMATION must be an string.
    information1=KEY + ":" + "\n - " + INFORMATION
    information2="- " + INFORMATION
    if path.exists("info.yml"):
        yml=yaml.load(open("info.yml", "r"), Loader=yaml.FullLoader)
        if KEY not in yml: #Check if the key exist.
            print("The info.yml has a new key.")
            print(information1, file=open("./info.yml", "a")) #Create a new key and add an information.
        else:
            if INFORMATION in yml[KEY]: #Check if the information exist in the key.
                pass
            else:
                print("The info.yml information was updated.")
                yml[KEY]=information2
                for k, v in yml.items():
                    print(k + ":\n" + v, file=open("./info.yml", "w")) #Create a new information in the key.
    else:
        print("The info.yml was created.")
        print(information1, file=open("./info.yml", "a")) #Create a key and add an information.

def directory(DIREC): #Directory creator.
    print("Creating the " + DIREC + " directory.")
    print("The " + DIREC + " directory is ready.")
    os.mkdir(DIREC)

def build_infor(ANNDATA, KEY, VALUE):
    '''
    Adding info anndata.uns["infoprocess"]
    Parameters
    ------------
    ANNDATA : anndata object
        adata object
    KEY : String
        The name of key to be added
    VALUE : String, list, int, float, boolean, dict
	Information to be added for a given key
    '''
    #Author: Guilherme Valente
    if "infoprocess" not in ANNDATA.uns:
        ANNDATA.uns["infoprocess"]={}
        ANNDATA.uns["infoprocess"][KEY]=VALUE
    else:
        ANNDATA.uns["infoprocess"][KEY]=VALUE
    if "color_set" not in ANNDATA.uns:
        ANNDATA.uns["color_set"]=color_list
    return ANNDATA.copy()

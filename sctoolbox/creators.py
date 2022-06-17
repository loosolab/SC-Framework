"""
Modules for creating files or directories
"""
import os
from os import path
import sys
import sctoolbox
import sctoolbox.checker as ch

##################################
color_list=['red', 'blue', 'green', 'pink', 'chartreuse', 'gray', 'yellow', 'brown', 'purple', 'orange', 'wheat', 'lightseagreen', 'cyan', 'khaki', 'cornflowerblue', 'olive', 'gainsboro', 'darkmagenta', 'slategray', 'ivory', 'darkorchid', 'papayawhip', 'paleturquoise', 'oldlace', 'orangered', 'lavenderblush', 'gold', 'seagreen', 'deepskyblue', 'lavender', 'peru', 'silver', 'midnightblue', 'antiquewhite', 'blanchedalmond', 'firebrick', 'greenyellow', 'thistle', 'powderblue', 'darkseagreen', 'darkolivegreen', 'moccasin', 'olivedrab', 'mediumseagreen', 'lightgray', 'darkgreen', 'tan', 'yellowgreen', 'peachpuff', 'cornsilk', 'darkblue', 'violet', 'cadetblue', 'palegoldenrod', 'darkturquoise', 'sienna', 'mediumorchid', 'springgreen', 'darkgoldenrod', 'magenta', 'steelblue', 'navy', 'lightgoldenrodyellow', 'saddlebrown', 'aliceblue', 'beige', 'hotpink', 'aquamarine', 'tomato', 'darksalmon', 'navajowhite', 'lawngreen', 'lightsteelblue', 'crimson', 'mediumturquoise', 'mistyrose', 'lightcoral', 'mediumaquamarine', 'mediumblue', 'darkred', 'lightskyblue', 'mediumspringgreen', 'darkviolet', 'royalblue', 'seashell', 'azure', 'lightgreen', 'fuchsia', 'floralwhite', 'mintcream', 'lightcyan', 'bisque', 'deeppink', 'limegreen', 'lightblue', 'darkkhaki', 'maroon', 'aqua', 'lightyellow', 'plum', 'indianred', 'linen', 'honeydew', 'burlywood', 'goldenrod', 'mediumslateblue', 'lime', 'lightslategray', 'forestgreen', 'dimgray', 'lemonchiffon', 'darkgray', 'dodgerblue', 'darkcyan', 'orchid', 'blueviolet', 'mediumpurple', 'darkslategray', 'turquoise', 'salmon', 'lightsalmon', 'coral', 'lightpink', 'slateblue', 'darkslateblue', 'white', 'sandybrown', 'chocolate', 'teal', 'mediumvioletred', 'skyblue', 'snow', 'palegreen', 'ghostwhite', 'indigo', 'rosybrown', 'palevioletred', 'darkorange', 'whitesmoke']
##################################

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

def output_path(OUTPATH, TEST): #Check if the directory for output exist.
    '''
    This will create the directory to store the results of scRNAseq autom pipeline
    Parameters
    ==========
    OUTPATH : String.
        The pathway where the user wanna to store the data
    TEST : String.
        The name of the user wanna use to define the analysis of this pipeline, e.g., Test1
    '''
    #Author : Guilherme Valente
    m1="Define an appropriate path_out to stablish a place to save your results, e.g., /mnt/workspace/YOUR_NAME"
    m2="Output directory is ready: "

    #Checking if the first directory determined exists
    directories=list(filter(None, OUTPATH.split("/")))[0]
    if path.exists("/" + directories) != True:
        sys.exit(m1)

    #Checking if the other directories exist
    OUTPATH2=OUTPATH + "/results/" + TEST
    directories=list(filter(None, OUTPATH2.split("/")))
    dire="/"
    for a in directories:
        dire=dire + a + "/"
        if path.exists(dire) != True: #Create folder if it does not exist
            os.mkdir(dire)
    print(m2 + dire)

    #Creating storing information for next
    to_info_txt="Output_path:" + dire
    ch.check_infoyml(VALUE=to_info_txt, TASK="create") #Printing the output dir detailed in the info.txt

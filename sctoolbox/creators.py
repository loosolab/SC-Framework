"""
Modules for creating files or directories
"""
import os
from os import path
import sys
import sctoolbox
import sctoolbox.checker as ch
import anndata

#Do we need this?
def add_color_set(adata, inplace = True):
    """ Add color set to adata object

    Parameter:
    ----------
    adata : AnnData object
        AnnData object from scanpy
    inplace : boolean
        Add color set inplace

    Returns:
    -------
        AnnData object with color set
    """

    color_list=['red', 'blue', 'green', 'pink', 'chartreuse',
    'gray', 'yellow', 'brown', 'purple', 'orange', 'wheat',
    'lightseagreen', 'cyan', 'khaki', 'cornflowerblue', 'olive',
    'gainsboro', 'darkmagenta', 'slategray', 'ivory', 'darkorchid',
    'papayawhip', 'paleturquoise', 'oldlace', 'orangered',
    'lavenderblush', 'gold', 'seagreen', 'deepskyblue', 'lavender',
    'peru', 'silver', 'midnightblue', 'antiquewhite', 'blanchedalmond',
    'firebrick', 'greenyellow', 'thistle', 'powderblue', 'darkseagreen',
    'darkolivegreen', 'moccasin', 'olivedrab', 'mediumseagreen',
    'lightgray', 'darkgreen', 'tan', 'yellowgreen', 'peachpuff',
    'cornsilk', 'darkblue', 'violet', 'cadetblue', 'palegoldenrod',
    'darkturquoise', 'sienna', 'mediumorchid', 'springgreen',
    'darkgoldenrod', 'magenta', 'steelblue', 'navy', 'lightgoldenrodyellow',
    'saddlebrown', 'aliceblue', 'beige', 'hotpink', 'aquamarine', 'tomato',
    'darksalmon', 'navajowhite', 'lawngreen', 'lightsteelblue', 'crimson',
    'mediumturquoise', 'mistyrose', 'lightcoral', 'mediumaquamarine',
    'mediumblue', 'darkred', 'lightskyblue', 'mediumspringgreen',
    'darkviolet', 'royalblue', 'seashell', 'azure', 'lightgreen', 'fuchsia',
    'floralwhite', 'mintcream', 'lightcyan', 'bisque', 'deeppink',
    'limegreen', 'lightblue', 'darkkhaki', 'maroon', 'aqua', 'lightyellow',
    'plum', 'indianred', 'linen', 'honeydew', 'burlywood', 'goldenrod',
    'mediumslateblue', 'lime', 'lightslategray', 'forestgreen', 'dimgray',
    'lemonchiffon', 'darkgray', 'dodgerblue', 'darkcyan', 'orchid',
    'blueviolet', 'mediumpurple', 'darkslategray', 'turquoise', 'salmon',
    'lightsalmon', 'coral', 'lightpink', 'slateblue', 'darkslateblue',
    'white', 'sandybrown', 'chocolate', 'teal', 'mediumvioletred', 'skyblue',
    'snow', 'palegreen', 'ghostwhite', 'indigo', 'rosybrown', 'palevioletred',
    'darkorange', 'whitesmoke']

    if  type(adata) != anndata.AnnData:
        raise TypeError("Invalid data type. AnnData object is required.")

    m_adata = adata if inplace else adata.copy()
    if "color_set" not in m_adata.uns:
        m_adata.uns["color_set"] = color_list
    if not inplace:
        return m_adata


def build_infor(adata, key, value, inplace = True):
    """ Adding info anndata.uns["infoprocess"]

    Parameters
    ------------
    adata : anndata object
        adata object
    key : String
        The name of key to be added
    value : String, list, int, float, boolean, dict
	    Information to be added for a given key
    inplace : boolean
        Add info inplace
    """
    #Author: Guilherme Valente

    if  type(adata) != anndata.AnnData:
        raise TypeError("Invalid data type. AnnData object is required.")

    m_adata = adata if inplace else adata.copy()

    if "infoprocess" not in m_adata.uns:
        m_adata.uns["infoprocess"]={}
    m_adata.uns["infoprocess"][key]=value
    add_color_set(m_adata)

    if not inplace:
        return m_adata


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
    ch.write_info_txt(path_value=to_info_txt) #Printing the output dir detailed in the info.txt

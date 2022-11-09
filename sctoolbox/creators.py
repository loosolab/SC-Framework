"""
Modules for creating files or directories
"""
import os
import sctoolbox.checker as ch
import anndata


def add_color_set(adata, inplace=True):
    """
    Add color set to adata object

    TODO Do we need this?

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to add the colors to.
    inplace : boolean
        Whether the anndata object is modified in place.

    Returns
    -------
    anndata.AnnData or None :
        AnnData object with color set.
    """
    color_list = ['red', 'blue', 'green', 'pink', 'chartreuse',
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

    if type(adata) != anndata.AnnData:
        raise TypeError("Invalid data type. AnnData object is required.")

    m_adata = adata if inplace else adata.copy()
    if "color_set" not in m_adata.uns:
        m_adata.uns["color_set"] = color_list

    if not inplace:
        return m_adata


def build_infor(adata, key, value, inplace=True):
    """
    Adding info anndata.uns["infoprocess"]

    Parameters
    ----------
    adata : anndata.AnnData
        adata object
    key : String
        The name of key to be added
    value : String, list, int, float, boolean, dict
        Information to be added for a given key
    inplace : boolean
        Add info inplace

    Returns
    -------
    anndata.AnnData or None :
        AnnData object with added info in .uns["infoprocess"].
    """
    if type(adata) != anndata.AnnData:
        raise TypeError("Invalid data type. AnnData object is required.")

    m_adata = adata if inplace else adata.copy()

    if "infoprocess" not in m_adata.uns:
        m_adata.uns["infoprocess"] = {}
    m_adata.uns["infoprocess"][key] = value
    add_color_set(m_adata)

    if not inplace:
        return m_adata


def create_dir(outpath, test):
    """
    This will create the directory to store the results of scRNAseq autom pipeline.
    Constructed path has following scheme: /<outpath>/results/<test>/

    Parameters
    ----------
    outpath : str
        Path to where the data is stored.
    test : str
        Name of the specific folder the output will be stored in. Is appended to outpath.
    """
    output_dir = os.path.join(outpath, "results", test)

    # Check if the directory exist and create if not
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory is ready: {output_dir}")

    # Creating storing information for next
    ch.write_info_txt(path_value=output_dir)  # Printing the output dir detailed in the info.txt
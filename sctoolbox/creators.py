"""
Modules for creating files or directories
"""
import os
import sctoolbox.checker as ch
import anndata
import pathlib


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


def gitlab_download(repo, internal_path, branch="main", commit="latest", out_path="./", cred=None):
    """
    Download file or dir from gitlab

    Parameters
    ----------
    repo : str
        Link to repository
    internal_path :  str
        Dir or file in repository to download
    branch :  str, default 'main'
        What branch to use
    commit : str, default 'latest'
        What commit to use
    out_path : str, default './'
        Where the fike/dir should be downloaded to
    cred : str, default None
        Credentials in case of private repository

    Returns
    -------
    None
    """
    pass


def setup_experiment(dest, dirs=["raw", "preprocessing", "Analysis"]):
    """
    Create initial folder structure

    Parameters
    ----------
    dest :  str
        Path to new experiment
    dir : list, default ['raw', 'preprocessing']
        Internal folders to create

    Returns
    -------
    None
    """
    print("Setting up experiment:")
    if pathlib.Path(dest).exists():
        raise Exception(f"Directory '{dest}' already exists. "
                        + "Please make sure you are not going to "
                        + "overwrite an existing project. Exiting..")

    for dir in dirs:
        path_to_build = pathlib.Path(dest) / dir
        path_to_build.mkdir(parents=True, exist_ok=True)
        print(f"Build: {path_to_build}")


def add_analysis(dest, analysis_name,
                 dirs=['figures', 'data', 'notebooks', 'logs'],
                 starts_with=1, **kwargs):
    """
    Create and add a new analysis

    Parameter
    ---------
    dest : str
        Path to experiment
    analysis_name : str
        Name of the new analysis run
    dirs : list, default ['figures', 'data', 'notebooks', 'logs']
        Internal folders to create
    start_with : int, default 1
        Notebook the analysis will start with
    kwargs : kwargs
        forwarded to gitlab_download

    Returns
    -------
    None
    """
    analysis_path = pathlib.Path(dest) / "Analysis"
    if not analysis_path.exists():
        raise FileNotFoundError("Analysis directory not found."
                                + "Please check if you entered the right "
                                + "directory or if it was setup correctly.")
    run_path = analysis_path / analysis_name

    setup_experiment(run_path, dirs=dirs)

    #ToDo Download Notebook to notebook directory

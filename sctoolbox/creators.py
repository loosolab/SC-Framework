"""
Modules for creating files or directories
"""
import os
import sctoolbox.checker as ch
import anndata
import pathlib
import gitlab
from getpass import getpass
import warnings


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


def gitlab_download(internal_path, host="https://gitlab.gwdg.de/",
                    repo="loosolab_SC_RNA_framework", branch="main",
                    commit=None, out_path="./", private=False,
                    load_token=pathlib.Path.home() / ".gitlab_token",
                    save_token=pathlib.Path.home() / ".gitlab_token", overwrite=False):
    """
    Download file from gitlab.
    Requires project access token if the project is private.
    The token needs following scopes:
    - read_api
    - read_repository
    - read registry

    Parameters
    ----------
    internal_path :  str
        Dir or file in repository to download
    host : str, default 'https://gitlab.gwdg.de/'
        Link to host
    repo : str, default 'loosolab_SC_RNA_framework'
        Name of the repository
    branch :  str, default 'main'
        What branch to use
    commit : str, default None
        What commit to use, overwrites branch
    out_path : str, default './'
        Where the fike/dir should be downloaded to
    private, boolean, default False
        Set true if repo is private
    load_token : str, default 'pathlib.Path.home() / ".gitlab_token"'
        Load token from file. Set to None for new token
    save_token : str, default 'pathlib.Path.home() / ".gitlab_token"'
        Save token to file
    overwrite : boolean, default False
        Overwrite file if it exsits in the directory

    Returns
    -------
    None
    """
    token = None
    if commit:
        branch = commit

    if private:
        load_token_file = pathlib.Path(load_token).is_file() if load_token else False
        if load_token_file:
            with open(load_token, 'r') as token_file:
                token = token_file.readline().strip()
        else:
            token = getpass("Please enter your token: ")
            if save_token:
                with open(save_token, 'w') as token_file:
                    token_file.write(token)

    try:
        gl = gitlab.Gitlab(host, private_token=token)
        pl = gl.projects.list(search=repo)
        for p in pl:
            if p.name == repo:
                project = p
                break
        if project:
            with open(out_path, 'wb') as f:
                if not pathlib.Path(out_path).is_file() or overwrite:
                    project.files.raw(file_path=internal_path, ref=branch, streamed=True, action=f.write)
                else:
                    warnings.warn("File already existis. Use overwrite parameter to overwrite file.")
        else:
            raise ValueError("Reposiotry not found")
    except Exception as e:
        print("Error:", e)


def setup_experiment(dest, dirs=["raw", "preprocessing", "Analysis"]):
    """
    Create initial folder structure

    Parameters
    ----------
    dest :  str
        Path to new experiment
    dir : list, default ['raw', 'preprocessing', 'Analysis']
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

"""Modules for creating files or directories."""

import pathlib
import gitlab
from getpass import getpass
import warnings
import re
from ratelimiter import RateLimiter
import time


def gitlab_download(internal_path, file_regex, host="https://gitlab.gwdg.de/",
                    repo="sc_framework", branch="main",
                    commit=None, out_path="./", private=False,
                    load_token=pathlib.Path.home() / ".gitlab_token",
                    save_token=pathlib.Path.home() / ".gitlab_token",
                    overwrite=False, max_calls=5, period=60) -> None:
    """
    Download file(s) from gitlab.

    Parameters
    ----------
    internal_path : str
        path to directory in repository
    file_regex : str
        regex for target file(s)
    host : str, default 'https://gitlab.gwdg.de/'
        Link to host
    repo : str, default 'sc_framework'
        Name of the repository
    branch :  str, default 'main'
        What branch to use
    commit : str, default None
        What commit to use, overwrites branch
    out_path : str, default './'
        Where the fike/dir should be downloaded to
    private :  boolean, default False
        Set true if repo is private
    load_token : str, default 'pathlib.Path.home() / ".gitlab_token"'
        Load token from file. Set to None for new token
    save_token : str, default 'pathlib.Path.home() / ".gitlab_token"'
        Save token to file
    overwrite : boolean, default False
        Overwrite file if it exsits in the directory
    max_calls : int, default 5
        limit file download rate per period
    period : int, deafult 60
        period length in seconds

    Raises
    ------
    ValueError
        If repository is inaccesible.
    """

    def limited(until):
        duration = int(round(until - time.time()))
        print('Rate limited, sleeping for {:d} seconds'.format(duration))

    rate_limiter = RateLimiter(max_calls=max_calls, period=period, callback=limited)
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
        if not pl:
            raise ValueError("Could not find repository. Is the repository private?")
        for p in pl:
            if p.name == repo:
                project = p
                break

        if not project:
            raise ValueError("Repository not found")

        items = project.repository_tree(path=internal_path, ref=branch)
        for item in items:
            if item["type"] != "blob" or not re.search(file_regex, item["name"]):
                continue
            out = pathlib.Path(out_path) / item["name"]
            if not out.is_file() or overwrite:
                print(f"Downloading: {item['name']}")
                with rate_limiter:
                    with open(out, 'wb') as f:
                        project.files.raw(file_path=item["path"], ref=branch, streamed=True, action=f.write)
            else:
                warnings.warn("File already exists. Use overwrite parameter to overwrite file.")
    except Exception as e:
        print("Error:", e)


def setup_experiment(dest, dirs=["raw", "preprocessing", "Analysis"]) -> None:
    """
    Create initial folder structure.

    Parameters
    ----------
    dest : str
        Path to new experiment
    dirs : list, default ['raw', 'preprocessing', 'Analysis']
        Internal folders to create

    Raises
    ------
    Exception
        If directory exists.
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


def add_analysis(dest, analysis_name, method="rna",
                 dirs=['figures', 'data', 'logs'],
                 starts_with=1, **kwargs) -> None:
    """
    Create and add a new analysis/run.

    Note: Only works for Notebooks until number 99.
    Needs to be adjusted if we exceed 89 notebooks.

    Parameters
    ----------
    dest : str
        Path to experiment.
    analysis_name : str
        Name of the new analysis run.
    method : {'rna', 'atac'}
        Type of notebooks to download.
    dirs : list[str], default ['figures', 'data', 'logs']
        Internal folders to create besides 'notebooks' directory.
    starts_with : int, default 1
        Notebook the analysis will start with.
    **kwargs : dict
        Forwarded to `gitlab_download`.

    Raises
    ------
    FileNotFoundError
        If path to experiment does not exist.
    ValueError
        If `method` is invalid.
    """

    analysis_path = pathlib.Path(dest) / "Analysis"
    if not analysis_path.exists():
        raise FileNotFoundError("Analysis directory not found."
                                + "Please check if you entered the right "
                                + "directory or if it was setup correctly.")
    run_path = analysis_path / analysis_name
    method = method.lower()
    if method not in ['rna', 'atac']:
        raise ValueError("Invalid method type. Valid options: 'rna', 'atac'")

    # Setup run directorys
    setup_experiment(run_path, dirs=dirs + ["notebooks"])
    # Build notebook regex
    regex = build_notebooks_regex(starts_with)

    # Download notebooks
    print("Downloading notebooks..")
    gitlab_download(f"{method}-notebooks", file_regex=regex, out_path=run_path / "notebooks", **kwargs)
    gitlab_download(f"{method}-notebooks", file_regex="config.yaml", out_path=run_path / "notebooks", **kwargs)


def build_notebooks_regex(starts_with) -> str:
    """
    Build regex for notebooks starting with given number.

    Note: Only works up to 89. If we reach notebook 90 this function needs to be adjusted.

    Parameters
    ----------
    starts_with : int
        Starting number

    Returns
    -------
    str
        notebook regex

    Raises
    ------
    ValueError
        If `starts_with` is < 1 or > 89.
    """

    if starts_with < 1:
        raise ValueError("starts_with needs to be at least 1")
    elif 1 <= starts_with < 10:
        regex = f"[0]*[1-9]?[{starts_with}-9].*.ipynb"
    elif 10 <= starts_with < 90:
        regex = f"[0]*([{str(starts_with)[0]}][{str(starts_with)[1]}-9]|[{str(starts_with+10)[0]}-9][0-9]).*.ipynb"
    else:
        # Needs change if we ever reach 90+ Notebooks
        raise ValueError("starts_with needs to be lower than 90")
    return regex

"""Modules for creating files or directories."""

import pathlib
import gitlab
from getpass import getpass
import warnings
import re
from throttler import Throttler
import time
from github import Github, Auth
from pathlib import Path
import tqdm
import requests
from deprecated import deprecated

from beartype import beartype
from beartype.typing import Optional, Any, Literal

from sctoolbox._settings import settings
logger = settings.logger


@deprecated(version='0.15.0', reason="Superseeded by 'sctoolbox.utils.creators.github_download'.")
@beartype
def gitlab_download(internal_path: str,  # noqa: C901
                    file_regex: str,
                    host: str = "https://gitlab.gwdg.de/",
                    repo: str = "sc_framework",
                    branch: str = "main",
                    commit: Optional[str] = None,
                    out_path: str = "./",
                    private: bool = False,
                    load_token: str = str(pathlib.Path.home() / ".gitlab_token"),
                    save_token: str = str(pathlib.Path.home() / ".gitlab_token"),
                    overwrite: bool = False,
                    max_calls: int = 5,
                    period: int = 60) -> None:
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
    commit : Optional[str], default None
        What commit to use, overwrites branch
    out_path : str, default './'
        Where the fike/dir should be downloaded to
    private :  bool, default False
        Set true if repo is private
    load_token : str, default 'pathlib.Path.home() / ".gitlab_token"'
        Load token from file. Set to None for new token
    save_token : str, default 'pathlib.Path.home() / ".gitlab_token"'
        Save token to file
    overwrite : bool, default False
        Overwrite file if it exsits in the directory
    max_calls : int, default 5
        limit file download rate per period
    period : int, default 60
        period length in seconds

    Raises
    ------
    ValueError
        If repository is inaccesible.
    """

    def limited(until: float) -> None:
        duration = int(round(until - time.time()))
        print('Rate limited, sleeping for {:d} seconds'.format(duration))

    rate_limiter = Throttler(max_calls=max_calls, period=period, callback=limited)
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


@deprecated(version='0.15.0', reason="Use `sctoolbox.utils.creators.add_analysis` instead.")
@beartype
def setup_experiment(dest: str,
                     dirs: list[str] = ["raw", "preprocessing", "Analysis"]) -> None:
    """
    Create initial folder structure.

    Parameters
    ----------
    dest : str
        Path to new experiment
    dirs : list[str], default ['raw', 'preprocessing', 'Analysis']
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


@beartype
def add_analysis(dest: str,
                 analysis_name: str,
                 method: Literal["rna", "atac"] = "rna",
                 general: bool = True,
                 **kwargs: Any) -> None:
    """
    Create and add a new analysis/run.

    This function will create a directory in `dest` with the name `analysis_name` and populate it with the selected analysis notebooks.

    Parameters
    ----------
    dest : str
        Path to experiment.
    analysis_name : str
        Name of the new analysis run.
    method : Literal["rna", "atac"], default "rna"
        Type of notebooks to download.
    general : bool, default True
        Whether to download the notebooks independent of the method.
    **kwargs : Any
        Forwarded to `github_download`.

    Raises
    ------
    FileNotFoundError
        If path to experiment does not exist.
    FileExistsError
        If the `analysis_name` already exists.
    """

    analysis_path = pathlib.Path(dest) / "Analysis"
    if not analysis_path.exists():
        raise FileNotFoundError("Analysis directory not found."
                                + "Please check if you entered the right "
                                + "directory or if it was setup correctly.")

    run_path = analysis_path / analysis_name

    if run_path.exists():
        raise FileExistsError(f"The analysis name {run_path} already exists suggesting a preexisting analysis."
                              + "Please use another name or manually delete the folder before trying again.")

    # Download notebooks
    logger.info("Downloading notebooks...")

    # create a dict of default parameters
    ghd_params = {
        "outpath": run_path,
        "match": ".ipynb|.yaml|.pptx",
    }
    ghd_params.update(kwargs)

    # method specific notebooks
    if method == "rna":
        nb_path = "rna_analysis/notebooks"
    elif method == "atac":
        nb_path = "atac_analysis/notebooks"

    github_download(path=nb_path, **ghd_params)

    # general notebooks
    if general:
        # update the parameters to download the general notebooks to the notebooks dir
        ghd_params.update({
            "outpath": str(Path(run_path) / nb_path),
            "keep_repo_structure": False
        })

        github_download(path="general_notebooks", **ghd_params)


@deprecated(version='0.15.0', reason="No longer required.")
@beartype
def build_notebooks_regex(starts_with: int) -> str:
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
        regex = f"[0]*([{str(starts_with)[0]}][{str(starts_with)[1]}-9]|[{str(starts_with + 10)[0]}-9][0-9]).*.ipynb"
    else:
        # Needs change if we ever reach 90+ Notebooks
        raise ValueError("starts_with needs to be lower than 90")
    return regex


@beartype
def github_download(path: str,
                    repo: str = "loosolab/SC-Framework",
                    outpath: str = ".",
                    keep_repo_structure: bool = True,
                    match: Optional[str] = None,
                    access_token: Optional[str] = None,
                    overwrite: bool = False,
                    reference: Optional[str] = None) -> None:
    """
    Download the a file or directory from the given repository.

    Parameters
    ----------
    path : str
        Path to a file or folder within the repository. This is not a recursive downloader, meaning that directories in the specified path will be ignored.
    repo : str, default loosolab/SC-Framework
        The repository name.
    outpath : str, default .
        Local path where the file(s) will be saved.
    keep_repo_structure : bool, default True
        If True, will create the path given by `path` in the location given by `outpath`.
    match : Optional[str], default None
        Regex; matching files will be downloaded.
    access_token : Optional[str], default None
        GitHub personal access token. May be supplied in case of throttling.
        See `here <https://github.com/settings/tokens>`_ to create one (you have to be logged in).
    overwrite : bool, default False
        Skip or overwrite preexisting files.
    reference : Optional[str], default None
        Download the files from a specific branch/tag/commit. Can be a branch name, tag name or a commit-sha.
    """
    # authenticate
    if access_token:
        auth = Auth.Token(access_token)
    else:
        auth = None

    # initialize GitHub
    with Github(auth=auth) as github:
        # connect to repository
        repo_ = github.get_repo(repo)

        # get the contents of the path
        content = repo_.get_contents(**{"path": path, "ref": reference} if reference else {"path": path})

        if not isinstance(content, list):
            content = [content]

        # filter based on the match regex
        if match:
            content = [f for f in content if re.search(match, f.name)]

        # save files to the output folder
        for f in tqdm.tqdm(content, desc="Downloading"):
            # construct the path + filename
            out_path = Path(outpath) / f.path if keep_repo_structure else f.name

            # ignore directories
            if f.type == "file":
                logger.debug(f"Downloading {f.name}")

                # warn and skip instead of overwriting a file
                if not overwrite:
                    logger.warning(f"{out_path} already exists. Skipping...")
                    continue

                # create the directories if necessary
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # write the files
                try:
                    content_data = f.decoded_content

                    # Write based on content type/encoding
                    if isinstance(content_data, bytes):
                        out_path.write_bytes(content_data)
                    else:
                        # try to write as string if no encoding is given
                        out_path.write_text(str(content_data), encoding='utf-8')

                except Exception:
                    # If there's an issue, fall back to direct download
                    try:
                        response = requests.get(f.download_url)
                        out_path.write_bytes(response.content)
                    except Exception as e2:
                        logger.error(f"Failed to download {f.name}: {e2}")

"""Functions related to multiprocessing."""

import time
import sctoolbox.utils as utils

# type hint imports
from typing import TYPE_CHECKING, Any, Tuple
from beartype import beartype

if TYPE_CHECKING:
    import tqdm


@beartype
def get_pbar(total: int, description: str) -> "tqdm.tqdm":
    """
    Get a progress bar depending on whether the user is using a notebook or not.

    Parameters
    ----------
    total : int
        Total number elements to be shown in the progress bar.
    description : str
        Description to be shown in the progress bar.

    Returns
    -------
    tqdm.tqdm
        A progress bar object.
    """

    if utils._is_notebook() is True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    pbar = tqdm(total=total, desc=description)
    return pbar


@beartype
def monitor_jobs(jobs: dict[Tuple[int,int], Any] | list[Any], description: str = "Progress") -> None:
    """
    Monitor the status of jobs submitted to a pool.

    Parameters
    ----------
    jobs : list[Any]
        List of job objects, e.g. as returned by pool.map_async().
    description : str, default "Progress"
        Description to be shown in the progress bar.
    """

    if isinstance(jobs, dict):
        jobs = list(jobs.values())

    # Wait for all jobs to finish
    n_ready = sum([job.ready() for job in jobs])
    pbar = get_pbar(len(jobs), description)
    while n_ready != len(jobs):
        if n_ready != pbar.n:
            pbar.n = n_ready
            pbar.refresh()
        time.sleep(1)
        n_ready = sum([job.ready() for job in jobs])

    pbar.n = n_ready  # update progress bar to 100%
    pbar.refresh()
    pbar.close()

"""Functions related to multiprocessing."""
import multiprocessing as mp
import sctoolbox.utils as utils
from sctoolbox._settings import settings
from functools import partial
import scanpy as sc
import time

# type hint imports
from beartype.typing import TYPE_CHECKING, Any, Tuple, Callable, Iterable, Optional
from beartype import beartype

if TYPE_CHECKING:
    import tqdm


@beartype
def get_pbar(total: int, description: str, **kwargs: Any) -> "tqdm.tqdm":
    """
    Get a progress bar depending on whether the user is using a notebook or not.

    Parameters
    ----------
    total : int
        Total number elements to be shown in the progress bar.
    description : str
        Description to be shown in the progress bar.
    **kwargs : Any
        Keyword arguments to be passed to tqdm.

    Returns
    -------
    tqdm.tqdm
        A progress bar object.
    """

    if utils.jupyter._is_notebook() is True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    pbar = tqdm(total=total, desc=description, **kwargs)
    return pbar


@beartype
def monitor_jobs(jobs: dict[Tuple[int, int | str], Any] | list[Any], description: str = "Progress") -> None:
    """
    Monitor the status of jobs submitted to a pool.

    Parameters
    ----------
    jobs : dict[Tuple[int, int | str], Any] | list[Any]
        List or dict of job objects, e.g. as returned by pool.map_async().
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


@beartype
def mp_first_position(func: Callable,
                      iterable: Iterable,
                      threads: Optional[int] = None,
                      **kwargs: Any):
    """
    Run a function in multiple processes.

    Parameters
    ----------
    func: Callable
        The function to be executed in parallel.
    iterable: Iterable
        An iterable of arguments to pass to the function.
    threads: Optional[int]
        Number of processes to spawn. Defaults to the settings.threads.
    **kwargs: Any
        Additional keyword arguments to pass to the function.

    Returns
    -------
    Results of the given function
    """

    # Determine number of processes
    if threads is None:
        threads = settings.get_threads()

    pool = mp.Pool(threads)

    # Create a pool of processes
    jobs = [pool.apply_async(func, args=(items,), kwds=kwargs) for items in iterable]
    pool.close()

    # Monitor the jobs
    utils.multiprocessing.monitor_jobs(jobs)

    # Wait for all processes to finish
    pool.join()

    # Collect results
    results = [job.get() for job in jobs]

    return results


@beartype
def mp_adata_first_arg(func: Callable,
                       adata: sc.AnnData,
                       iterable: Iterable,
                       threads: Optional[int] = None,
                       **kwargs: Any):
    """
    Run a function in multiple processes, with anndata object as the first argument.

    Parameters
    ----------
    func: Callable
        The function to be executed in parallel.
    adata: sc.AnnData
        The anndata object to be passed as the first argument to the function.
    iterable: Iterable
        An iterable of arguments to pass to the function.
    threads: Optional[int]
        Number of processes to spawn. Defaults to the settings.threads.
    **kwargs: Any
        Additional keyword arguments to pass to the function.

    Returns
    -------
    Results of the given function
    """

    # Set adata as the first argument
    func_wrapper = partial(func, adata)

    # Determine number of processes
    if threads is None:
        threads = settings.get_threads()

    pool = mp.Pool(threads)

    # Create a pool of processes
    jobs = [pool.apply_async(func_wrapper, args=(items,), kwds=kwargs) for items in iterable]
    pool.close()

    # Monitor the jobs
    utils.multiprocessing.monitor_jobs(jobs)

    # Wait for all processes to finish
    pool.join()

    # Collect results
    results = [job.get() for job in jobs]

    return results
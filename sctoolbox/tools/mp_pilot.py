import multiprocessing as mp
import sctoolbox.utils as utils

from sctoolbox._settings import settings
from beartype.typing import Any, Optional
from beartype import beartype
from functools import partial


@beartype
def mp_first_position(func: Any, iterable: Any, threads: Optional[int] = None, **kwargs: Any) -> None:

    '''
    Decorator to run a function in multiple processes.

    Parameters:
    -----------
    func:
        The function to be executed in parallel.
    iterable (iterable):
        An iterable of arguments to pass to the function.
    threads (int, optional):
        Number of processes to spawn. Defaults to the number of CPU.
    **kwargs:
        Additional keyword arguments to pass to the function.
    '''

    # Determine number of processes
    if threads is None:
        threads = settings.get_threads()

    pool = mp.Pool(threads)

    # Create a pool of processes
    jobs = []
    job = [pool.apply_async(func, args=(items,), kwds=kwargs) for items in iterable]
    jobs.append(job)
    pool.close()

    # Monitor the jobs
    results = utils.multiprocessing.monitor_jobs(jobs)

    # Wait for all processes to finish
    pool.join()

    return results


@beartype
def adata_first_arg(func: Any, adata: Any, iterable: Any, threads: Optional[int] = None, **kwargs: Any) -> None:

    '''
    Decorator to run a function in multiple processes, with anndata object as the first argument.

    Parameters:
    -----------
    adata:
        The anndata object to be passed as the first argument.
    func:
        The function to be executed in parallel.
    iterable (iterable):
        An iterable of arguments to pass to the function.
    threads (int, optional):
        Number of processes to spawn. Defaults to the number of CPU.
    **kwargs:
        Additional keyword arguments to pass to the function.
    '''

    # Set adata as the first argument
    func_wrapper = partial(func, adata)

    # Determine number of processes
    if threads is None:
        threads = settings.get_threads()

    pool = mp.Pool(threads)

    # Create a pool of processes
    jobs = []
    job = [pool.apply_async(func_wrapper, args=(items,), kwds=kwargs) for items in iterable]
    jobs.append(job)
    pool.close()

    # Monitor the jobs
    results = utils.multiprocessing.monitor_jobs(jobs)

    # Wait for all processes to finish
    pool.join()

    return results

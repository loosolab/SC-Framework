"""Multiprocessing utilities for parallel execution of functions."""
import multiprocessing as mp
import sctoolbox.utils as utils
import scanpy as sc

from sctoolbox._settings import settings
from beartype import beartype
from beartype.typing import Any,  Callable, Iterable, Optional
from functools import partial


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

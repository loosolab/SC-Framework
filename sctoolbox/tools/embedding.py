"""Embedding tools."""
import scanpy as sc
import multiprocessing as mp

from beartype.typing import Iterable
from beartype import beartype

import sctoolbox.utils as utils


@beartype
def wrap_umap(adatas: Iterable[sc.AnnData], threads: int = 4, **kwargs: Any) -> None:
    """
    Compute umap for a list of adatas in parallel.

    Parameters
    ----------
    adatas : Iterable[sc.AnnData]
        List of anndata objects to compute umap on.
    threads : int, default 4
        Number of threads to use.
    **kwargs : Any
        Additional arguments to be passed to sc.tl.umap.
    """

    pool = mp.Pool(threads)

    kwargs["copy"] = True  # always copy

    jobs = []
    for i, adata in enumerate(adatas):
        adata_minimal = utils.get_minimal_adata(adata)
        job = pool.apply_async(sc.tl.umap, args=(adata_minimal, ), kwds=kwargs)
        jobs.append(job)
    pool.close()

    utils.monitor_jobs(jobs, "Computing UMAPs ")
    pool.join()

    # Get results and add to adatas
    for i, adata in enumerate(adatas):
        adata_return = jobs[i].get()
        adata.obsm["X_umap"] = adata_return.obsm["X_umap"]

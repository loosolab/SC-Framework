"""Embedding tools."""
import scanpy as sc
import multiprocessing as mp

import sctoolbox.utils as utils


def wrap_umap(adatas, threads=4) -> None:
    """
    Compute umap for a list of adatas in parallel.

    TODO Check that adatas is a list of anndata objects

    Parameters
    ----------
    adatas : list of anndata.AnnData
        List of anndata objects to compute umap on.
    threads : int, default 4
        Number of threads to use.
    """

    pool = mp.Pool(threads)

    jobs = []
    for i, adata in enumerate(adatas):
        adata_minimal = utils.get_minimal_adata(adata)
        job = pool.apply_async(sc.tl.umap, args=(adata_minimal, ), kwds={"copy": True})
        jobs.append(job)
    pool.close()

    utils.monitor_jobs(jobs, "Computing UMAPs ")
    pool.join()

    # Get results and add to adatas
    for i, adata in enumerate(adatas):
        adata_return = jobs[i].get()
        adata.obsm["X_umap"] = adata_return.obsm["X_umap"]

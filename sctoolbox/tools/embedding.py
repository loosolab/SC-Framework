"""Embedding tools."""
import scanpy as sc
import multiprocessing as mp

import sctoolbox.utils as utils


def wrap_umap(adatas, threads=4):
    """
    Compute umap for a list of adatas in parallel.

    Parameters
    ----------
    adatas : list of anndata.AnnData
        List of anndata objects to compute umap on.
    threads : int, default 4
        Number of threads to use.

    Returns
    -------
    None
        UMAP coordinates are added to each adata.obsm["X_umap"].

    TODO Check that adatas is a list of anndata objects
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

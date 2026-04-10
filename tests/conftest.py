"""Fixtures available to all tests."""

import pytest
import scanpy as sc
import numpy as np


# ---------------------------- Script variables --------------------------- #
# global variables for this script

__rank_key = "rank_genes_groups"


# ------------------------------ FIXTURES --------------------------------- #


def _make_adata():
    """Load and returns an anndata object.

    Returns
    -------
    anndata.AnnData
        AnnData object with processed data and clustering results.
    """

    np.random.seed(1)  # set seed for reproducibility

    adata = sc.datasets.pbmc68k_reduced()
    adata.raw = None

    # create nonsense layers to enable velocity testing
    adata.layers["spliced"] = adata.X * 2
    adata.layers["unspliced"] = adata.X * 3

    adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
    adata.obs["cat"] = adata.obs["condition"].astype("category")

    adata.obs["LISI_score_pca"] = np.random.normal(size=adata.shape[0])
    adata.obs["qc_float"] = np.random.uniform(0, 1, size=adata.shape[0])
    adata.var["qc_float_var"] = np.random.uniform(0, 1, size=adata.shape[1])

    adata.obs["qcvar1"] = np.random.normal(size=adata.shape[0])
    adata.obs["qcvar2"] = np.random.normal(size=adata.shape[0])

    sc.tl.umap(adata, n_components=3)  # to have more than two components available
    sc.tl.tsne(adata)

    sc.tl.rank_genes_groups(adata, groupby='louvain', key_added=__rank_key)

    return adata


@pytest.fixture(scope="session")  # reuse the fixture for all tests
def adata():
    """Create a fixture of the adata with session scope.

    Returns
    -------
    anndata.AnnData
        AnnData object with session scope.
    """
    return _make_adata()


@pytest.fixture(scope="function")  # create a new fixture for each test
def adata_fun_scope():
    """Create a fixture of the adata with function scope.

    Returns
    -------
    anndata.AnnData
        AnnData object with function scope.
    """
    return _make_adata()

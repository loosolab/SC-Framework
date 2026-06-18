"""Fixtures available to all tests."""

import pytest
import scanpy as sc
import numpy as np
import os
import tempfile

# Redirect scanpy dataset cache to a temp directory to avoid writing to the repo
sc.settings.datasetdir = tempfile.mkdtemp()

# ---------------------------- Script variables --------------------------- #
# global variables for this script

__rank_key = "rank_genes_groups"

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ATAC_DATA_DIR = os.path.join(DATA_DIR, 'atac')


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


@pytest.fixture(scope="session")
def adata_raw():
    """Load and return the raw PBMC3k dataset.

    Returns
    -------
    anndata.AnnData
        AnnData object with raw counts.
    """
    return sc.datasets.pbmc3k()


def _load_adata_h5ad():
    """Load the shared ``adata.h5ad`` test fixture from the data directory.

    Returns
    -------
    anndata.AnnData
        AnnData object read from ``DATA_DIR/adata.h5ad``.
    """
    return sc.read_h5ad(os.path.join(DATA_DIR, "adata.h5ad"))


@pytest.fixture(scope="function")
def adata_h5ad():
    """Provide the shared ``adata.h5ad`` object with function scope.

    Function scope is required because consumers mutate ``var`` index/names; a
    shared session-scoped object would leak mutations across tests.

    Returns
    -------
    anndata.AnnData
        A freshly loaded AnnData object from ``DATA_DIR/adata.h5ad``.
    """
    return _load_adata_h5ad()


@pytest.fixture(scope="function")
def pbmc3k_processed():
    """Provide the shared processed PBMC3k dataset with function scope.

    Function scope is used so that consumers cannot leak mutations across
    tests; the equivalent (read-only) consumers depend on this fixture, while
    mutating consumers build on their own copies.

    Returns
    -------
    anndata.AnnData
        A freshly loaded processed PBMC3k dataset from
        ``scanpy.datasets.pbmc3k_processed``.
    """
    return sc.datasets.pbmc3k_processed()


@pytest.fixture
def adata_atac():
    """Load and return an ATAC-seq AnnData object.

    Returns
    -------
    anndata.AnnData
        ATAC-seq AnnData object from mm10_atac.h5ad.
    """
    return sc.read_h5ad(os.path.join(ATAC_DATA_DIR, 'mm10_atac.h5ad'))

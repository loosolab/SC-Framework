"""Fixtures available to all tests."""

import pytest
import scanpy as sc
import numpy as np
import os
import tempfile
import filelock
from typing import Callable

# ---------------------------- Script variables --------------------------- #
# global variables for this script

__rank_key = "rank_genes_groups"

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ATAC_DATA_DIR = os.path.join(DATA_DIR, 'atac')


# ------------------------------ FIXTURES --------------------------------- #


@pytest.fixture(scope="session", autouse=True)
def scanpy_datasetdir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Point scanpy's dataset cache at a directory shared across xdist workers.

    Also works with xdist disabled.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Session-scoped factory whose ``getbasetemp().parent`` is the directory
        xdist shares across the controller and every ``gw<N>`` worker.

    Returns
    -------
    str
        The directory scanpy writes downloaded datasets into.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    if worker_id == "master":
        datasetdir = os.path.join(tempfile.gettempdir(), "sctoolbox_scanpy_data")
    else:
        datasetdir = os.path.join(tmp_path_factory.getbasetemp().parent, "scanpy_data")

    os.makedirs(datasetdir, exist_ok=True)
    sc.settings.datasetdir = datasetdir
    return datasetdir


def _download_once(datasetdir: str, name: str, loader: Callable[[], sc.AnnData]) -> sc.AnnData:
    """Fetch a scanpy dataset under a per-dataset lock so workers never race.

    Parameters
    ----------
    datasetdir : str
        The shared scanpy dataset cache directory.
    name : str
        Dataset name, used to name the lock file.
    loader : Callable[[], anndata.AnnData]
        The ``sc.datasets.*`` loader to call inside the lock.

    Returns
    -------
    anndata.AnnData
        The loaded dataset.
    """
    lock_path = os.path.join(datasetdir, f"{name}.lock")
    with filelock.FileLock(lock_path):
        return loader()


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
def adata_fun_scope(adata):
    """Provide a function-scoped, mutable copy of the shared session ``adata``.

    Depends on the session-scoped ``adata`` fixture and returns a deep copy
    (``AnnData.copy()``), so the expensive build runs once per session while
    each consuming test still gets an isolated object it can mutate in place
    without leaking changes into the shared session build.

    The shared session ``adata`` can accumulate ``@log_anndata`` entries under
    ``uns["sctoolbox"]`` whenever another test passes it to a logged function
    (the decorator records the call even when the function later raises). A
    fresh ``_make_adata()`` build carries no such log, so the copy strips it to
    give each consumer the same clean slate the fresh build used to provide.

    Returns
    -------
    anndata.AnnData
        A deep copy of the session ``adata`` fixture, with any accumulated
        ``uns["sctoolbox"]`` log removed.
    """
    obj = adata.copy()
    obj.uns.pop("sctoolbox", None)
    return obj


@pytest.fixture(scope="session")
def adata_raw(scanpy_datasetdir: str) -> sc.AnnData:
    """Load and return the raw PBMC3k dataset.

    Parameters
    ----------
    scanpy_datasetdir : str
        The shared scanpy dataset cache directory (autouse fixture).

    Returns
    -------
    anndata.AnnData
        AnnData object with raw counts.
    """
    return _download_once(scanpy_datasetdir, "pbmc3k_raw", sc.datasets.pbmc3k)


@pytest.fixture(scope="function")
def adata_raw_small(adata_raw):
    """Provide a downsized, function-scoped copy of the raw PBMC3k dataset.

    Returns
    -------
    anndata.AnnData
        A ``300 x 3000`` copy of the raw PBMC3k dataset.
    """
    return adata_raw[:300, :3000].copy()


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


def _load_adata_scsa_h5ad():
    """Load the shared ``scsa/adata_scsa.h5ad`` test fixture from the data directory.

    Returns
    -------
    anndata.AnnData
        AnnData object read from ``DATA_DIR/scsa/adata_scsa.h5ad``.
    """
    return sc.read_h5ad(os.path.join(DATA_DIR, "scsa", "adata_scsa.h5ad"))


@pytest.fixture(scope="function")
def pbmc3k_processed(scanpy_datasetdir: str) -> sc.AnnData:
    """Provide the shared processed PBMC3k dataset with function scope.

    Function scope is used so that consumers cannot leak mutations across
    tests; the equivalent (read-only) consumers depend on this fixture, while
    mutating consumers build on their own copies.

    Parameters
    ----------
    scanpy_datasetdir : str
        The shared scanpy dataset cache directory (autouse fixture).

    Returns
    -------
    anndata.AnnData
        A freshly loaded processed PBMC3k dataset from
        ``scanpy.datasets.pbmc3k_processed``.
    """
    return _download_once(scanpy_datasetdir, "pbmc3k_processed", sc.datasets.pbmc3k_processed)


@pytest.fixture(scope="function")
def random_adata():
    """Provide a trivial synthetic AnnData object for tests.

    Function-scoped so each test receives a fresh object; consumers that
    augment it in place therefore do not leak mutations into one another.

    Returns
    -------
    anndata.AnnData
        A new ``100 x 100`` integer-count AnnData object.
    """
    return sc.AnnData(np.random.randint(0, 100, (100, 100)))


@pytest.fixture
def adata_atac():
    """Load and return an ATAC-seq AnnData object.

    Returns
    -------
    anndata.AnnData
        ATAC-seq AnnData object from mm10_atac.h5ad.
    """
    return sc.read_h5ad(os.path.join(ATAC_DATA_DIR, 'mm10_atac.h5ad'))


@pytest.fixture
def adata_atac_emptyvar(adata_atac):
    """Create adata with empty adata.var.

    Returns
    -------
    anndata.AnnData
        AnnData object with empty var table.
    """
    adata = adata_atac.copy()
    adata.var = adata.var.drop(columns=adata.var.columns)
    return adata


@pytest.fixture
def atac_fragments():
    """Path to ATAC-seq fragments BED file.

    Returns
    -------
    str
        Path to mm10_atac_fragments.bed.
    """
    return os.path.join(ATAC_DATA_DIR, 'mm10_atac_fragments.bed')


@pytest.fixture
def sorted_fragments():
    """Path to sorted ATAC-seq fragments BED file.

    Returns
    -------
    str
        Path to mm10_sorted_fragments.bed.
    """
    return os.path.join(ATAC_DATA_DIR, 'mm10_sorted_fragments.bed')

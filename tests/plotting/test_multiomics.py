"""Tests for sctoolbox.plotting.multiomics."""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import sctoolbox.plotting.multiomics as pl_multi
import sctoolbox.tools.multiomics as tools_multi

plt.switch_backend("Agg")


# ------------------------------ FIXTURES --------------------------------- #


@pytest.fixture(scope="module")
def mdata():
    """Build a MuData object with RNA and ATAC modalities for testing.

    Returns
    -------
    mu.MuData
        MuData object with 30 cells, leiden clustering, X_umap, PCA and neighbors.
    """
    np.random.seed(42)
    n_cells = 30
    barcodes = [f"cell{i}" for i in range(n_cells)]

    # RNA modality
    X_rna = np.random.rand(n_cells, 50)
    obs_rna = pd.DataFrame(
        {"leiden": pd.Categorical(np.random.choice(["0", "1", "2"], n_cells))},
        index=barcodes,
    )
    var_rna = pd.DataFrame(index=[f"gene{i}" for i in range(50)])
    adata_rna = sc.AnnData(X=X_rna, obs=obs_rna, var=var_rna)
    sc.pp.pca(adata_rna)
    sc.pp.neighbors(adata_rna)
    sc.tl.umap(adata_rna)  # sets X_umap in adata_rna.obsm

    # ATAC modality
    X_atac = np.random.rand(n_cells, 30)
    obs_atac = pd.DataFrame(
        {"leiden": pd.Categorical(np.random.choice(["0", "1", "2"], n_cells))},
        index=barcodes,
    )
    var_atac = pd.DataFrame(index=[f"peak{i}" for i in range(30)])
    adata_atac = sc.AnnData(X=X_atac, obs=obs_atac, var=var_atac)
    sc.pp.pca(adata_atac)
    sc.pp.neighbors(adata_atac)
    sc.tl.umap(adata_atac)  # sets X_umap in adata_atac.obsm

    mdata_obj = mu.MuData({"RNA": adata_rna, "ATAC": adata_atac})

    # Set up joint neighbors for umap_parameter_sweep using RNA PCA
    mdata_obj.obsm["X_pca"] = adata_rna.obsm["X_pca"]
    sc.pp.neighbors(mdata_obj, use_rep="X_pca")

    return mdata_obj


@pytest.fixture(scope="module")
def mdata_with_comparison(mdata):
    """MuData with cluster_comparison_best_matches set in uns.

    Returns
    -------
    mu.MuData
        Copy of mdata with compare_clusters already called.
    """
    mdata_copy = mdata.copy()
    tools_multi.compare_clusters(mdata_copy, "leiden", "leiden")
    return mdata_copy


@pytest.fixture(scope="module")
def sankey_df(mdata):
    """DataFrame for Sankey diagram returned by compare_clusters.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns RNA:leiden, ATAC:leiden, Cells_per_cluster.
    """
    mdata_copy = mdata.copy()
    _, _, dfs_sankey = tools_multi.compare_clusters(mdata_copy, "leiden", "leiden")
    return dfs_sankey[0]


# ------------------------------ TESTS --------------------------------- #


def test_check_modality_is_valid(mdata):
    """Valid modality name should not raise any exception."""
    pl_multi.check_modality_is_valid(mdata, "RNA")
    pl_multi.check_modality_is_valid(mdata, "ATAC")


def test_check_modality_is_valid_raises(mdata):
    """Invalid modality name should raise KeyError."""
    with pytest.raises(KeyError):
        pl_multi.check_modality_is_valid(mdata, "INVALID_MODALITY")


def test_visualize_cluster_comparison(mdata):
    """Function should return a matplotlib Figure and axes array."""
    fig, axes = pl_multi.visualize_cluster_comparison(
        mdata,
        clusters_mod1="leiden",
        clusters_mod2="leiden",
    )
    plt.close("all")

    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(axes, np.ndarray)


def test_compare_cluster_heatmap(mdata):
    """Function should return a matplotlib Figure and axes array."""
    # Use a fresh copy of heatmap DataFrames so reset_index called inside the
    # function does not affect other tests sharing the module-scoped fixture.
    mdata_copy = mdata.copy()
    heatmaps, _, _ = tools_multi.compare_clusters(mdata_copy, "leiden", "leiden")

    fig, axes = pl_multi.compare_cluster_heatmap(mdata_copy, heatmaps)
    plt.close("all")

    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (2,)


def test_plot_sankey(sankey_df):
    """Function should return a plotly Figure."""
    fig = pl_multi.plot_sankey(
        sankey_df,
        modalities=["RNA", "ATAC"],
        clustercols=["leiden", "leiden"],
    )

    assert isinstance(fig, go.Figure)


def test_visualize_modality_grid(mdata_with_comparison):
    """Function should return a matplotlib Figure and axes array."""
    fig, axes = pl_multi.visualize_modality_grid(
        mdata_with_comparison,
        clustercols=["leiden", "leiden"],
    )
    plt.close("all")

    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(axes, np.ndarray)


def test_umap_parameter_sweep(mdata):
    """Function should return a matplotlib Figure and a numpy array of axes."""
    fig, axes = pl_multi.umap_parameter_sweep(
        mdata,
        min_dist_range=(0.1, 0.3, 0.1),
        spread_range=(0.5, 0.7, 0.1),
    )
    plt.close("all")

    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (6,)  # ceil(4 combos / 3 cols) * 3 cols = 2x3 flattened

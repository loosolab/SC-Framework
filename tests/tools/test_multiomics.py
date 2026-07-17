"""Test multiomics functions."""

import pytest
import os
import anndata
import scanpy as sc
import numpy as np
import muon as mu
import pandas as pd

import sctoolbox.tools.multiomics as multi


# ------------------------- FIXTURES -------------------------#


@pytest.fixture
def adata():
    """Load and returns an anndata object.

    Returns
    -------
    anndata.AnnData
        RNA-seq AnnData object.
    """
    f = os.path.join(os.path.dirname(__file__), '../data', "adata.h5ad")

    return sc.read_h5ad(f)


@pytest.fixture
def adata2(adata):
    """Build second adata from first.

    Returns
    -------
    anndata.AnnData
        Copy of AnnData object with mock PCA and UMAP embeddings.
    """
    adata2 = adata.copy()
    adata2.obsm['X_pca'] = np.random.uniform(low=-3, high=3, size=(200, 50))
    adata2.obsm['X_umap'] = np.random.uniform(low=-30, high=70, size=(200, 3))
    return adata2


@pytest.fixture
def mdata():
    """MuData with two modalities and categorical leiden clustering.

    Returns
    -------
    mu.MuData
        MuData object with RNA and ATAC modalities.
    """
    np.random.seed(42)
    n_cells = 30
    barcodes = [f"cell{i}" for i in range(n_cells)]

    obs_rna = pd.DataFrame({"leiden": pd.Categorical(np.random.choice(["0", "1", "2"], n_cells))}, index=barcodes)
    adata_rna = sc.AnnData(X=np.random.rand(n_cells, 20), obs=obs_rna,
                           var=pd.DataFrame(index=[f"gene{i}" for i in range(20)]))

    obs_atac = pd.DataFrame({"leiden": pd.Categorical(np.random.choice(["0", "1", "2"], n_cells))}, index=barcodes)
    adata_atac = sc.AnnData(X=np.random.rand(n_cells, 15), obs=obs_atac,
                            var=pd.DataFrame(index=[f"peak{i}" for i in range(15)]))

    return mu.MuData({"RNA": adata_rna, "ATAC": adata_atac})


@pytest.fixture
def adata_mod1():
    """AnnData with mod1 barcodes.

    Returns
    -------
    sc.AnnData
        AnnData object with mod1 barcodes and Sample column.
    """
    obs = pd.DataFrame({"Sample": ["s1", "s1"]}, index=["AAAA-s1", "BBBB-s1"])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(5)])
    return sc.AnnData(X=np.random.rand(2, 5), obs=obs, var=var)


@pytest.fixture
def adata_mod2():
    """AnnData with mod2 barcodes (different from mod1).

    Returns
    -------
    sc.AnnData
        AnnData object with mod2 barcodes and Sample column.
    """
    obs = pd.DataFrame({"Sample": ["s1", "s1"]}, index=["CCCC-s1", "DDDD-s1"])
    var = pd.DataFrame(index=[f"peak{i}" for i in range(5)])
    return sc.AnnData(X=np.random.rand(2, 5), obs=obs, var=var)


@pytest.fixture
def barcode_map_file(tmp_path):
    """TSV file mapping mod1 barcodes to mod2 barcodes.

    Returns
    -------
    str
        Path to the barcode map TSV file.
    """
    f = tmp_path / "barcode_map.tsv"
    f.write_text("AAAA\tCCCC\nBBBB\tDDDD\n")
    return str(f)


# ------------------------------ TESTS --------------------------------- #


def test_merge_anndata(adata, adata2):
    """Test if anndata are merged correctly."""
    adata_to_merge = {"1": adata, "2": adata2}
    merged_adata = multi.merge_anndata(adata_to_merge)

    new_obs_index = list(set(adata.obs.index) & set(adata2.obs.index))
    new_obs_cols, new_obsm_entries, new_var_index = list(), list(), list()
    new_var_cols = ["source"]
    for key, value in adata_to_merge.items():
        new_var_cols += [f"{key}:{i}" for i in value.var.columns]
        new_obs_cols += [f"{key}:{i}" for i in value.obs.columns]
        new_obsm_entries += [f"X_{key}_{i.split('_')[-1]}" for i in list(value.obsm)]
        new_var_index += [f"{key}:{i}" for i in value.var.index]

    # Check if merged object is an AnnData object
    assert isinstance(merged_adata, anndata.AnnData)
    # Check if merged obsm entries are merged properly
    assert all(elem in new_obsm_entries for elem in list(merged_adata.obsm))
    # Check if merged var columns are merged properly
    assert all(elem in new_var_cols for elem in merged_adata.var.columns)
    # Check if merged obs columns are merged properly
    assert all(elem in new_obs_cols for elem in merged_adata.obs.columns)
    # Check if merged var index is merged properly
    assert all(elem in new_var_index for elem in merged_adata.var.index)
    # Check if merged obs index is merged properly
    assert all(elem in new_obs_index for elem in merged_adata.obs.index)


def test_deep_merge_anndata(adata, adata2):
    """Test if anndata are merged correctly by checking obsm coordinates and matrix values for each cell."""
    adata_to_merge = {"1": adata, "2": adata2}
    merged_adata = multi.merge_anndata(adata_to_merge)

    for cell_id in merged_adata.obs.index:
        m_index = list(merged_adata.obs.index).index(cell_id)
        r_index = list(adata.obs.index).index(cell_id)
        c_index = list(adata2.obs.index).index(cell_id)
        m = merged_adata.X.tocsr()[m_index, :].todense().tolist()[0]
        r = adata.X.tocsr()[r_index, :].todense().tolist()[0]
        c = adata2.X.tocsr()[c_index, :].todense().tolist()[0]

        adata2.obsm["X_umap"][c_index]

        assert (list(merged_adata.obsm["X_1_umap"][m_index]) == list(adata.obsm["X_umap"][r_index]))
        assert (list(merged_adata.obsm["X_2_umap"][m_index]) == list(adata2.obsm["X_umap"][c_index]))
        assert merged_adata.obs.index[m_index] == adata.obs.index[r_index]
        assert merged_adata.obs.index[m_index] == adata2.obs.index[c_index]
        assert (m[:len(r)] == r) and (m[-len(c):] == c)


def test_add_multiome_prefix_inplace(adata):
    """Test that prefix is added to obs columns, var index, and obsm keys in place."""
    a = adata.copy()
    a.obsm["X_pca"] = np.random.rand(a.shape[0], 10)
    original_obs_cols = list(a.obs.columns)
    original_var_index = list(a.var.index)

    multi.add_multiome_prefix(a, "RNA")

    assert all(col.startswith("RNA_") for col in a.obs.columns)
    assert all(idx.startswith("RNA_") for idx in a.var.index)
    assert "X_RNA_pca" in a.obsm
    assert not any(col in a.obs.columns for col in original_obs_cols)
    assert not any(idx in a.var.index for idx in original_var_index)


def test_add_multiome_prefix_not_inplace(adata):
    """Test that not-inplace returns a copy with prefix and leaves original unchanged."""
    a = adata.copy()
    original_obs_cols = list(a.obs.columns)

    result = multi.add_multiome_prefix(a, "RNA", inplace=False)

    assert result is not None
    assert all(col.startswith("RNA_") for col in result.obs.columns)
    assert list(a.obs.columns) == original_obs_cols  # original unchanged


def test_add_multiome_prefix_ignore_obs_col(adata):
    """Test that columns in ignore_obs_col are not prefixed."""
    a = adata.copy()
    a.obs["test_col"] = "value"
    ignore_col = "test_col"

    multi.add_multiome_prefix(a, "RNA", ignore_obs_col=[ignore_col])

    assert ignore_col in a.obs.columns
    assert all(col == ignore_col or col.startswith("RNA_") for col in a.obs.columns)


def test_join_modalities(adata):
    """Test that two AnnData objects are joined into a MuData object."""
    adata1 = adata.copy()
    adata2 = adata.copy()

    mdata = multi.join_modalities([adata1, adata2], ["RNA", "ATAC"])

    assert isinstance(mdata, mu.MuData)
    assert "RNA" in mdata.mod
    assert "ATAC" in mdata.mod
    assert mdata.n_obs == adata.n_obs


def test_join_modalities_keep_outer(adata):
    """Test that keep_outer=True retains cells not in both modalities."""
    mdata_inner = multi.join_modalities([adata.copy(), adata[:100].copy()], ["RNA", "ATAC"], keep_outer=False)
    mdata_outer = multi.join_modalities([adata.copy(), adata[:100].copy()], ["RNA", "ATAC"], keep_outer=True)

    assert mdata_inner.mod["RNA"].n_obs == 100
    assert mdata_outer.mod["RNA"].n_obs == adata.n_obs


def test_match_barcodes_not_inplace(adata_mod1, adata_mod2, barcode_map_file):
    """Test that non-matching barcodes are remapped and returned as a tuple."""
    result = multi.match_barcodes(adata_mod1, adata_mod2, barcode_map_file, inplace=False)

    assert result is not None
    assert isinstance(result, tuple)
    result_mod1, result_mod2 = result
    assert list(result_mod1.obs.index) == list(result_mod2.obs.index)


def test_cluster_comparison_data_frames(mdata):
    """Test that three data frames are returned with correct structure."""
    df_heatmap, df_final, df_sankey = multi.cluster_comparison_data_frames(
        mdata.obs, modalities=("RNA", "ATAC"), clustercols=("RNA:leiden", "ATAC:leiden")
    )

    assert isinstance(df_heatmap, pd.DataFrame)
    assert isinstance(df_final, pd.DataFrame)
    assert isinstance(df_sankey, pd.DataFrame)
    assert df_heatmap.index.name == "RNA_clusters"
    assert "Cells_per_cluster_pct" in df_heatmap.columns
    assert "Best_match" in df_final.columns


def test_mean_percent_data_frame(mdata):
    """Test that mean percent data frame is a Styler with correct shape."""
    from pandas.io.formats.style import Styler

    df_heatmap_rna, _, _ = multi.cluster_comparison_data_frames(
        mdata.obs, modalities=("RNA", "ATAC"), clustercols=("RNA:leiden", "ATAC:leiden")
    )
    df_heatmap_atac, _, _ = multi.cluster_comparison_data_frames(
        mdata.obs, modalities=("ATAC", "RNA"), clustercols=("ATAC:leiden", "RNA:leiden")
    )

    result = multi.mean_percent_data_frame([df_heatmap_rna, df_heatmap_atac], ["RNA", "ATAC"])

    assert isinstance(result, Styler)
    assert result.data.shape[0] > 0
    assert result.data.shape[1] > 0


def test_compare_clusters(mdata):
    """Test that compare_clusters returns three arrays and populates mdata.uns."""
    dfs_heatmaps, dfs_mods, dfs_sankey = multi.compare_clusters(mdata, "leiden", "leiden")

    assert len(dfs_heatmaps) == 2
    assert len(dfs_mods) == 4
    assert len(dfs_sankey) == 2
    assert isinstance(dfs_heatmaps[0], pd.DataFrame)
    assert "cluster_comparison_best_matches" in mdata.uns


def test_export_modality_adatas(mdata, tmp_path, monkeypatch):
    """Test that modality AnnData files are saved for each modality."""
    monkeypatch.chdir(tmp_path)

    rna_file = tmp_path / "anndata_multiome_RNA.h5ad"
    atac_file = tmp_path / "anndata_multiome_ATAC.h5ad"

    try:
        multi.export_modality_adatas(mdata, obs_cols=None, obsm_keys=[])

        assert rna_file.exists()
        assert atac_file.exists()
    finally:
        rna_file.unlink(missing_ok=True)
        atac_file.unlink(missing_ok=True)

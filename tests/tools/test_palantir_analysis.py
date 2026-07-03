"""Tests for Palantir analysis helper functions."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

import sctoolbox.tools.palantir_analysis as pla


# ------------------------- FIXTURES -------------------------#


@pytest.fixture
def adata_rna():
    """Load the small RNA AnnData object used in the test suite."""
    adata_f = os.path.join(os.path.dirname(__file__), "../data", "adata.h5ad")
    return sc.read_h5ad(adata_f)


@pytest.fixture
def simple_adata():
    """Create a small AnnData object with cell type annotations."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 50

    X = np.random.poisson(lam=5, size=(n_obs, n_vars)).astype(np.float32)
    X = np.log1p(X)

    adata = AnnData(X=X)
    adata.obs["celltype"] = np.random.choice(["TypeA", "TypeB", "TypeC"], n_obs)
    adata.obs["batch"] = np.random.choice(["Batch1", "Batch2"], n_obs)
    adata.var["gene_id"] = [f"GENE_{i}" for i in range(n_vars)]

    return adata


@pytest.fixture
def preprocessed_adata(simple_adata):
    """Create a small AnnData object with PCA, UMAP and neighbors."""
    adata = simple_adata.copy()

    np.random.seed(42)
    adata.obsm["X_pca"] = np.random.randn(adata.n_obs, 10)
    adata.obsm["X_umap"] = np.random.randn(adata.n_obs, 2)
    adata.obsp["connectivities"] = sp.csr_matrix((adata.n_obs, adata.n_obs))
    adata.obsp["distances"] = sp.csr_matrix((adata.n_obs, adata.n_obs))
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
    }

    return adata


@pytest.fixture
def raw_integer_adata():
    """Create an AnnData object with raw integer counts."""
    np.random.seed(42)
    n_obs = 80
    n_vars = 40

    X = np.random.poisson(lam=10, size=(n_obs, n_vars)).astype(np.uint32)
    adata = AnnData(X=X)
    adata.obs["sample"] = np.random.choice(["S1", "S2"], n_obs)
    adata.var["gene_name"] = [f"GENE_{i}" for i in range(n_vars)]

    return adata


@pytest.fixture
def sparse_adata():
    """Create an AnnData object with sparse expression data."""
    np.random.seed(42)
    n_obs = 120
    n_vars = 60

    X = sp.random(n_obs, n_vars, density=0.1, format="csr")
    X.data = np.exp(X.data * 2)

    adata = AnnData(X=X)
    adata.obs["condition"] = np.random.choice(["CtrlA", "TreatedB"], n_obs)
    adata.var["gene_symbol"] = [f"G{i}" for i in range(n_vars)]

    return adata


@pytest.fixture
def branch_adata(preprocessed_adata):
    """Create an AnnData object with pseudotime and branch masks."""
    adata = preprocessed_adata.copy()
    n_obs = adata.n_obs

    adata.obs["palantir_pseudotime"] = np.linspace(0, 1, n_obs)
    adata.obsm["branch_masks"] = pd.DataFrame(
        {
            "BranchA": np.random.rand(n_obs) > 0.6,
            "BranchB": np.random.rand(n_obs) > 0.5,
            "BranchC": np.random.rand(n_obs) > 0.7,
        },
        index=adata.obs_names,
    )

    return adata


# ------------------------- is_preprocessed -------------------------#


def test_is_preprocessed_accepts_project_test_data(adata_rna):
    """Check that the project RNA test object contains the required preprocessing."""
    result = pla.is_preprocessed(
        adata_rna,
        normalize=False,
        log1p=False,
        pca="error",
        embedding="error",
        neighbors="error",
    )

    assert result is True


def test_is_preprocessed_missing_pca_warns_without_failing(simple_adata):
    """A missing PCA should not fail when the severity is set to warn."""
    result = pla.is_preprocessed(
        simple_adata,
        normalize=False,
        log1p=False,
        pca="warn",
        embedding=False,
        neighbors=False,
    )

    assert result is True


def test_is_preprocessed_missing_pca_fails_with_error(simple_adata):
    """A missing PCA should fail when the severity is set to error."""
    result = pla.is_preprocessed(
        simple_adata,
        normalize=False,
        log1p=False,
        pca="error",
        embedding=False,
        neighbors=False,
    )

    assert result is False


def test_is_preprocessed_missing_embedding_fails(simple_adata):
    """The embedding check should fail if no embedding is present."""
    adata = simple_adata.copy()
    adata.obsm["X_pca"] = np.random.randn(adata.n_obs, 10)

    result = pla.is_preprocessed(
        adata,
        normalize=False,
        log1p=False,
        pca=False,
        embedding="error",
        neighbors=False,
    )

    assert result is False


def test_is_preprocessed_detects_raw_counts(raw_integer_adata):
    """Raw integer counts should fail the normalization check."""
    result = pla.is_preprocessed(
        raw_integer_adata,
        normalize="error",
        log1p=False,
        pca=False,
        embedding=False,
        neighbors=False,
    )

    assert result is False


def test_is_preprocessed_all_checks_can_be_disabled(simple_adata):
    """Disabled checks should not block the input object."""
    result = pla.is_preprocessed(
        simple_adata,
        normalize=False,
        log1p=False,
        pca=False,
        embedding=False,
        neighbors=False,
    )

    assert result is True


def test_is_preprocessed_accepts_sparse_matrix(sparse_adata):
    """Sparse AnnData objects should pass when preprocessing keys are present."""
    sparse_adata.obsm["X_pca"] = np.random.randn(sparse_adata.n_obs, 10)
    sparse_adata.obsm["X_umap"] = np.random.randn(sparse_adata.n_obs, 2)
    sparse_adata.obsp["connectivities"] = sp.csr_matrix(
        (sparse_adata.n_obs, sparse_adata.n_obs)
    )
    sparse_adata.obsp["distances"] = sp.csr_matrix(
        (sparse_adata.n_obs, sparse_adata.n_obs)
    )
    sparse_adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
    }

    result = pla.is_preprocessed(
        sparse_adata,
        normalize=False,
        log1p=False,
        pca="error",
        embedding="error",
        neighbors="error",
    )

    assert result is True


# ------------------------- remove_cells_by_label -------------------------#


def test_remove_cells_by_label_removes_single_label(simple_adata):
    """Cells with one selected label should be removed."""
    result = pla.remove_cells_by_label(
        simple_adata,
        column_name="celltype",
        labels_to_remove="TypeA",
    )

    assert result.n_obs < simple_adata.n_obs
    assert "TypeA" not in result.obs["celltype"].values
    assert result.n_vars == simple_adata.n_vars


def test_remove_cells_by_label_removes_multiple_labels(simple_adata):
    """Cells with several selected labels should be removed."""
    result = pla.remove_cells_by_label(
        simple_adata,
        column_name="celltype",
        labels_to_remove=["TypeA", "TypeB"],
    )

    assert "TypeA" not in result.obs["celltype"].values
    assert "TypeB" not in result.obs["celltype"].values
    assert result.n_vars == simple_adata.n_vars


def test_remove_cells_by_label_missing_column_raises(simple_adata):
    """A missing annotation column should raise a ValueError."""
    with pytest.raises(ValueError, match="does not exist"):
        pla.remove_cells_by_label(
            simple_adata,
            column_name="missing_column",
            labels_to_remove="TypeA",
        )


def test_remove_cells_by_label_missing_label_raises(simple_adata):
    """Removing labels that are not present should raise a ValueError."""
    with pytest.raises(ValueError, match="None of the requested labels"):
        pla.remove_cells_by_label(
            simple_adata,
            column_name="celltype",
            labels_to_remove="MissingType",
        )


def test_remove_cells_by_label_preserves_gene_dimension(simple_adata):
    """Cell filtering should not change the number of genes."""
    result = pla.remove_cells_by_label(
        simple_adata,
        column_name="celltype",
        labels_to_remove="TypeA",
    )

    assert result.n_vars == simple_adata.n_vars


# ------------------------- branch_qc -------------------------#


def test_branch_qc_returns_expected_columns(branch_adata):
    """Branch QC should return one row per branch with expected metrics."""
    qc = pla.branch_qc(
        branch_adata,
        masks_key="branch_masks",
        pseudotime_key="palantir_pseudotime",
    )

    assert isinstance(qc, pd.DataFrame)
    assert qc.shape[0] == 3
    assert set(qc.columns) == {
        "branch",
        "n_cells",
        "pt_min",
        "pt_max",
        "coverage",
        "keep",
    }


def test_branch_qc_cell_counts_match_masks(branch_adata):
    """The reported cell counts should match the branch mask sums."""
    qc = pla.branch_qc(branch_adata)
    masks = branch_adata.obsm["branch_masks"]

    for _, row in qc.iterrows():
        assert row["n_cells"] == masks[row["branch"]].sum()


def test_branch_qc_missing_masks_key_returns_empty_table(branch_adata):
    """Missing branch masks should result in an empty QC table."""
    qc = pla.branch_qc(
        branch_adata,
        masks_key="missing_masks",
        pseudotime_key="palantir_pseudotime",
    )

    assert qc.empty


def test_branch_qc_missing_pseudotime_key_returns_empty_table(branch_adata):
    """Missing pseudotime values should result in an empty QC table."""
    qc = pla.branch_qc(
        branch_adata,
        masks_key="branch_masks",
        pseudotime_key="missing_pseudotime",
    )

    assert qc.empty


def test_branch_qc_min_cells_threshold_marks_branches_as_not_kept(branch_adata):
    """Branches below the minimum cell threshold should not be kept."""
    qc = pla.branch_qc(
        branch_adata,
        masks_key="branch_masks",
        pseudotime_key="palantir_pseudotime",
        min_cells=1000,
    )

    assert qc["keep"].sum() == 0


def test_branch_qc_handles_empty_branch(branch_adata):
    """An empty branch should be reported with zero cells and keep=False."""
    branch_adata.obsm["branch_masks"]["EmptyBranch"] = False

    qc = pla.branch_qc(branch_adata)
    empty_row = qc[qc["branch"] == "EmptyBranch"]

    assert not empty_row.empty
    assert empty_row["n_cells"].iloc[0] == 0
    assert not bool(empty_row["keep"].iloc[0])


# ------------------------- _classify_trend_pattern -------------------------#


def test_classify_trend_pattern_stable():
    """Constant curves should be classified as stable."""
    curve = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])

    assert pla._classify_trend_pattern(curve) == "stable"


def test_classify_trend_pattern_transient_peak():
    """Curves peaking in the middle should be classified as transient peak."""
    pt = np.linspace(0, 1, 50)
    curve = 1.0 - 4.0 * (pt - 0.5) ** 2

    assert pla._classify_trend_pattern(pd.Series(curve)) == "transient peak"


def test_classify_trend_pattern_transient_dip():
    """Curves dipping in the middle should be classified as transient dip."""
    pt = np.linspace(0, 1, 50)
    curve = 4.0 * (pt - 0.5) ** 2

    assert pla._classify_trend_pattern(pd.Series(curve)) == "transient dip"


def test_classify_trend_pattern_late_increase():
    """Curves with a late maximum should be classified as late increase."""
    pt = np.linspace(0, 1, 50)
    curve = pt**2

    assert pla._classify_trend_pattern(pd.Series(curve)) == "late increase"


def test_classify_trend_pattern_early_high_then_decrease():
    """Curves starting high and decreasing should be classified accordingly."""
    pt = np.linspace(0, 1, 50)
    curve = 2.0 - 2.0 * pt

    assert (
        pla._classify_trend_pattern(pd.Series(curve))
        == "early high, then decrease"
    )


def test_classify_trend_pattern_broad_increase():
    """Broadly increasing curves should be classified as broad increase."""
    curve = np.concatenate(
        [
            np.full(10, 0.4),
            np.linspace(0.45, 1.1, 20),
            np.linspace(1.05, 1.0, 20),
        ]
    )

    assert pla._classify_trend_pattern(pd.Series(curve)) == "broad increase"


def test_classify_trend_pattern_broad_decrease():
    """Broadly decreasing curves should be classified as broad decrease."""
    curve = np.concatenate(
        [
            np.full(10, 0.9),
            np.linspace(1.0, 1.2, 10),
            np.linspace(1.1, 0.2, 30),
        ]
    )

    assert pla._classify_trend_pattern(pd.Series(curve)) == "broad decrease"


def test_classify_trend_pattern_handles_nan_values():
    """Curves with NaN values should not make the classifier crash."""
    pt = np.linspace(0, 1, 50)
    curve = pd.Series(np.sin(2 * np.pi * pt))
    curve.iloc[10:15] = np.nan

    pattern = pla._classify_trend_pattern(curve)

    assert pattern in {
        "stable",
        "transient peak",
        "transient dip",
        "late increase",
        "early high, then decrease",
        "broad increase",
        "broad decrease",
        "complex",
    }


# ------------------------- compare_all_layers -------------------------#


def test_compare_all_layers_returns_figure_for_project_test_data(adata_rna):
    """Layer comparison should create a matplotlib figure."""
    fig, axs = pla.compare_all_layers(adata_rna)

    assert fig is not None
    assert axs is not None
    assert len(np.atleast_1d(axs)) >= 1
    plt.close(fig)


def test_compare_all_layers_can_exclude_layer(adata_rna):
    """Excluded layers should not be plotted."""
    fig, axs = pla.compare_all_layers(adata_rna, exclude_layers=["spliced"])
    axs_array = np.atleast_1d(axs)

    assert len(axs_array) == 3
    plt.close(fig)


# ------------------------- small integration checks -------------------------#


def test_preprocessing_then_remove_cells(preprocessed_adata):
    """Check validation before filtering and basic structure after filtering."""
    is_valid = pla.is_preprocessed(
        preprocessed_adata,
        normalize=False,
        log1p=False,
        pca="warn",
        embedding="warn",
        neighbors="warn",
    )
    assert is_valid

    filtered = pla.remove_cells_by_label(
        preprocessed_adata,
        column_name="celltype",
        labels_to_remove="TypeA",
    )

    assert filtered.n_obs < preprocessed_adata.n_obs
    assert filtered.n_vars == preprocessed_adata.n_vars
    assert "TypeA" not in filtered.obs["celltype"].values
    assert "X_pca" in filtered.obsm
    assert "X_umap" in filtered.obsm


def test_branch_qc_after_filtering(branch_adata):
    """Branch QC should still run after filtering cells by annotation."""
    filtered = pla.remove_cells_by_label(
        branch_adata,
        column_name="celltype",
        labels_to_remove="TypeA",
    )

    qc = pla.branch_qc(
        filtered,
        masks_key="branch_masks",
        pseudotime_key="palantir_pseudotime",
    )

    assert isinstance(qc, pd.DataFrame)
    assert qc.shape[0] > 0


# ------------------------- plot_branch_qc_distributions -------------------------#


def test_plot_branch_qc_distributions_returns_figure(branch_adata):
    """Branch QC distributions should create a matplotlib figure."""
    qc = pla.branch_qc(
        branch_adata,
        masks_key="branch_masks",
        pseudotime_key="palantir_pseudotime",
    )

    fig, axes = pla.plot_branch_qc_distributions(
        branch_adata,
        qc,
        masks_key="branch_masks",
        pseudotime_key="palantir_pseudotime",
        max_branches=2,
        bins=10,
        ncols=2,
    )

    assert fig is not None
    assert isinstance(axes, np.ndarray)
    assert len(axes) >= 2
    plt.close(fig)


def test_plot_branch_qc_distributions_missing_keys_returns_empty(branch_adata):
    """Missing branch masks should return an empty plot without crashing."""
    qc = pla.branch_qc(
        branch_adata,
        masks_key="branch_masks",
        pseudotime_key="palantir_pseudotime",
    )

    fig, axes = pla.plot_branch_qc_distributions(
        branch_adata,
        qc,
        masks_key="missing_masks",
        pseudotime_key="palantir_pseudotime",
    )

    assert fig is not None
    assert isinstance(axes, np.ndarray)
    assert len(axes) == 0
    plt.close(fig)


# ------------------------- plot_interactive_embedding_cell_ids -------------------------#


def test_plot_interactive_embedding_cell_ids_returns_figure(
    monkeypatch, preprocessed_adata
):
    """Interactive embedding plots should return a Plotly figure."""
    monkeypatch.setattr(pla.PlotlyFigure, "show", lambda self: None)

    fig = pla.plot_interactive_embedding_cell_ids(
        preprocessed_adata,
        embedding_key="X_umap",
        color_by="celltype",
        renderer=None,
    )

    assert fig is not None
    assert len(fig.data) > 0


def test_plot_interactive_embedding_cell_ids_missing_embedding(simple_adata):
    """A missing embedding should raise a ValueError."""
    with pytest.raises(ValueError, match="not found"):
        pla.plot_interactive_embedding_cell_ids(
            simple_adata,
            embedding_key="missing_embedding",
            renderer=None,
        )


def test_plot_interactive_embedding_cell_ids_rejects_non_2d_embedding(simple_adata):
    """Embeddings with more than two dimensions should raise a ValueError."""
    simple_adata.obsm["X_bad"] = np.random.randn(simple_adata.n_obs, 3)

    with pytest.raises(ValueError, match="must be 2D"):
        pla.plot_interactive_embedding_cell_ids(
            simple_adata,
            embedding_key="X_bad",
            renderer=None,
        )


def test_plot_interactive_embedding_cell_ids_missing_color_column(preprocessed_adata):
    """A missing color column should raise a ValueError."""
    with pytest.raises(ValueError, match="not found"):
        pla.plot_interactive_embedding_cell_ids(
            preprocessed_adata,
            embedding_key="X_umap",
            color_by="missing_column",
            renderer=None,
        )


# ------------------------- run_diffusion_and_magic -------------------------#


def test_run_diffusion_and_magic_stores_expected_outputs(
    monkeypatch, preprocessed_adata, tmp_path
):
    """Diffusion and MAGIC results should be stored in the AnnData object."""
    dm_res = {
        "EigenVectors": pd.DataFrame(
            np.ones((preprocessed_adata.n_obs, 3)),
            index=preprocessed_adata.obs_names,
        ),
        "EigenValues": pd.Series([1.0, 0.5, 0.25]),
        "T": sp.csr_matrix(np.eye(preprocessed_adata.n_obs)),
    }

    def fake_run_diffusion_maps(adata, **kwargs):
        adata.obsp["DM_Similarity"] = sp.csr_matrix(np.eye(adata.n_obs))
        adata.uns["DM_EigenValues"] = dm_res["EigenValues"]
        adata.obsm["DM_EigenVectors"] = dm_res["EigenVectors"].values
        return dm_res

    def fake_determine_multiscale_space(dm_res_arg):
        return pd.DataFrame(
            np.ones((preprocessed_adata.n_obs, 3)),
            index=preprocessed_adata.obs_names,
        )

    def fake_run_magic_imputation(
        data, imputation_key="MAGIC_imputed_data", **kwargs
    ):
        data.layers[imputation_key] = data.X.copy()
        return data.layers[imputation_key]

    monkeypatch.setattr(
        pla.palantir.utils,
        "run_diffusion_maps",
        fake_run_diffusion_maps,
    )
    monkeypatch.setattr(
        pla.palantir.utils,
        "determine_multiscale_space",
        fake_determine_multiscale_space,
    )
    monkeypatch.setattr(
        pla.palantir.utils,
        "run_magic_imputation",
        fake_run_magic_imputation,
    )
    monkeypatch.setattr(
        pla.utils.adata,
        "save_h5ad",
        lambda *args, **kwargs: None,
    )

    result = pla.run_diffusion_and_magic(
        preprocessed_adata,
        save_path=str(tmp_path / "palantir_magic.h5ad"),
        n_components=3,
        n_jobs=1,
    )

    assert result is preprocessed_adata
    assert "DM_EigenVectors_multiscaled" in result.obsm
    assert result.obsm["DM_EigenVectors_multiscaled"].shape == (
        preprocessed_adata.n_obs,
        3,
    )
    assert "MAGIC_imputed_data" in result.layers


    # ------------------------- tests -------------------------#


@pytest.fixture
def gene_trend_adata(preprocessed_adata):
    """Create AnnData with small branch-specific gene trend tables."""
    adata = preprocessed_adata.copy()

    genes = [f"Gene{i}" for i in range(8)]
    adata = adata[:, :8].copy()
    adata.var_names = genes

    pt = np.linspace(0.0, 1.0, 12)

    trends = pd.DataFrame(
        {
            t: [
                t,
                1.0 - t,
                np.sin(np.pi * t),
                np.cos(np.pi * t) + 1.0,
                t**2,
                1.0 - t**2,
                0.5 + t,
                1.5 - t,
            ]
            for t in pt
        },
        index=adata.var_names,
    )

    branch_b_trends = trends.copy()
    branch_b_trends.iloc[:, :] = trends.iloc[:, ::-1].to_numpy()

    adata.varm["gene_trends_BranchA"] = trends
    adata.varm["gene_trends_BranchB"] = branch_b_trends

    adata.var["highly_variable"] = [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    adata.obs["palantir_pseudotime"] = np.linspace(0, 1, adata.n_obs)
    adata.obs["palantir_entropy"] = np.linspace(1, 0, adata.n_obs)
    adata.obsm["DM_EigenVectors_multiscaled"] = np.random.randn(adata.n_obs, 3)

    return adata


def test_run_diffusion_and_magic_uses_complete_cache(preprocessed_adata, tmp_path):
    """A complete cached AnnData object should be loaded without recomputation."""
    cached = preprocessed_adata.copy()
    cached.layers["MAGIC_imputed_data"] = cached.X.copy()

    cache_path = tmp_path / "cached_magic.h5ad"
    cached.write_h5ad(cache_path)

    result = pla.run_diffusion_and_magic(
        preprocessed_adata,
        save_path=str(cache_path),
        recompute_magic=False,
    )

    assert "MAGIC_imputed_data" in result.layers
    assert result.n_obs == cached.n_obs
    assert result.n_vars == cached.n_vars


def test_detect_and_plot_terminal_states_manual(monkeypatch, gene_trend_adata):
    """Manual terminal cells should be converted into a labeled Series."""
    monkeypatch.setattr(
        pla.palantir.plot,
        "highlight_cells_on_umap",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    early_cell = gene_trend_adata.obs_names[0]
    terminal_cells = [
        gene_trend_adata.obs_names[1],
        gene_trend_adata.obs_names[2],
    ]

    result = pla.detect_and_plot_terminal_states(
        gene_trend_adata,
        early_cell=early_cell,
        terminal_states=terminal_cells,
        clusters_to_analyze="celltype",
        embedding_key="X_umap",
        eigvec_key="DM_EigenVectors_multiscaled",
    )

    assert isinstance(result, pd.Series)
    assert list(result.index) == terminal_cells
    assert result.shape[0] == 2


def test_detect_and_plot_terminal_states_auto(monkeypatch, gene_trend_adata):
    """Automatic terminal-state detection should use Palantir output."""
    terminal_cells = [
        gene_trend_adata.obs_names[1],
        gene_trend_adata.obs_names[2],
    ]
    excluded_cells = [gene_trend_adata.obs_names[3]]

    monkeypatch.setattr(
        pla.palantir.core,
        "identify_terminal_states",
        lambda *args, **kwargs: (terminal_cells, excluded_cells),
    )
    monkeypatch.setattr(
        pla.palantir.plot,
        "highlight_cells_on_umap",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    result = pla.detect_and_plot_terminal_states(
        gene_trend_adata,
        early_cell=gene_trend_adata.obs_names[0],
        terminal_states=None,
        clusters_to_analyze=None,
        embedding_key="X_umap",
        eigvec_key="DM_EigenVectors_multiscaled",
        show_excluded_boundaries=True,
    )

    assert isinstance(result, pd.Series)
    assert list(result.index) == terminal_cells
    assert list(result.values) == ["terminal_state_1", "terminal_state_2"]


def test_analyze_top_genes_communities_hvg_smoke(monkeypatch, gene_trend_adata):
    """HVG-based community analysis should return selected genes and communities."""
    def fake_cluster_gene_trends(adata, branch, genes, n_neighbors=30):
        return pd.Series(
            ["0" if i % 2 == 0 else "1" for i, _ in enumerate(genes)],
            index=genes,
            name="community",
        )

    monkeypatch.setattr(
        pla.palantir.presults,
        "cluster_gene_trends",
        fake_cluster_gene_trends,
    )
    monkeypatch.setattr(
        pla.palantir.plot,
        "plot_gene_trend_clusters",
        lambda *args, **kwargs: plt.figure(),
    )
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    selected_genes, communities = pla.analyze_top_genes_communities(
        gene_trend_adata,
        trend_branches="BranchA",
        gene_source="hvg",
        top_n=4,
        n_neighbors=2,
    )

    assert selected_genes == ["Gene0", "Gene1", "Gene2", "Gene3"]
    assert "BranchA" in communities
    assert isinstance(communities["BranchA"], pd.Series)


def test_plot_branch_gene_programs_smoke(monkeypatch, gene_trend_adata):
    """Branch gene programs should be generated from small synthetic trends."""
    def fake_cluster_gene_trends(adata, branch, genes, n_neighbors=30):
        labels = ["0" if i % 2 == 0 else "1" for i, _ in enumerate(genes)]
        return pd.Series(labels, index=genes)

    def fake_plot_gene_trend_clusters(*args, **kwargs):
        fig, axes = plt.subplots(1, 2)
        axes[0].plot([0, 1], [0, 1])
        axes[0].set_title("Cluster 0")
        axes[1].plot([0, 1], [1, 0])
        axes[1].set_title("Cluster 1")
        return fig

    monkeypatch.setattr(
        pla.palantir.presults,
        "cluster_gene_trends",
        fake_cluster_gene_trends,
    )
    monkeypatch.setattr(
        pla.palantir.plot,
        "plot_gene_trend_clusters",
        fake_plot_gene_trend_clusters,
    )
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    result = pla.plot_branch_gene_programs(
        gene_trend_adata,
        branch="BranchA",
        n_dynamic_genes=4,
        gene_ranking_mode="union",
        trend_knn_neighbors=2,
        plot_representative_genes=False,
    )

    assert "BranchA" in result
    assert "selected_genes" in result["BranchA"]
    assert "trend_clusters" in result["BranchA"]
    assert "centroid_z" in result["BranchA"]
    assert "cluster_summary" in result["BranchA"]


def test_plot_branch_early_late_genes_smoke(monkeypatch, gene_trend_adata):
    """Early/late gene plotting should return trend and UMAP result handles."""
    def fake_plot_embedding(*args, **kwargs):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig

    monkeypatch.setattr(pla.pl, "plot_embedding", fake_plot_embedding)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    result = pla.plot_branch_early_late_genes(
        gene_trend_adata,
        branches=["BranchA"],
        top_n=2,
        quantile_cutoff=0.25,
        umap_ncols=2,
    )

    assert "BranchA" in result
    assert "trend_fig" in result["BranchA"]
    assert "trend_axes" in result["BranchA"]
    assert "top_genes" in result["BranchA"]
    assert len(result["BranchA"]["top_genes"]["early"]) == 2
    assert len(result["BranchA"]["top_genes"]["late"]) == 2
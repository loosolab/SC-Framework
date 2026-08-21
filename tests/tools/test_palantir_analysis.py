"""Tests for Palantir analysis helper functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from plotly.graph_objects import Figure as PlotlyFigure
import sctoolbox.tools.palantir_analysis as pla

plt.switch_backend("Agg")


# ------------------------- FIXTURES -------------------------#


# Use the shared `adata_h5ad` fixture from tests/conftest.py instead of
# defining a local duplicate fixture here.


@pytest.fixture
def simple_adata(adata_h5ad):
    """Derive a lightweight test `AnnData` from the shared `adata_h5ad`.

    The fixture reuses the project-provided `adata_h5ad` to avoid rebuilding a
    synthetic AnnData from scratch. Only the minimal columns required by the
    tests are (re)created deterministically so test runs are stable.

    Returns
    -------
    anndata.AnnData
        A fresh AnnData object without preprocessing keys.
    """
    adata = adata_h5ad.copy()

    adata.obsm.pop("X_pca", None)
    adata.obsm.pop("X_umap", None)
    adata.obsm.pop("X_tsne", None)
    adata.obsp.pop("connectivities", None)
    adata.obsp.pop("distances", None)
    adata.uns.pop("neighbors", None)

    # Use a fixed RNG so tests remain deterministic
    rng = np.random.default_rng(42)
    n_obs = adata.n_obs

    # Ensure expected obs columns exist and are reproducible
    adata.obs["celltype"] = rng.choice(["TypeA", "TypeB", "TypeC"], size=n_obs)
    adata.obs["batch"] = rng.choice(["Batch1", "Batch2"], size=n_obs)

    # Provide a stable gene identifier column if missing
    if "gene_id" not in adata.var.columns:
        adata.var["gene_id"] = [f"GENE_{i}" for i in range(adata.n_vars)]

    return adata


@pytest.fixture
def preprocessed_adata(simple_adata):
    """Return a copy of `simple_adata` augmented with minimal preprocessing keys.

    The fixture derives from `simple_adata` to keep test setup small while
    providing PCA/UMAP/neighbors keys required by functions under test.

    Returns
    -------
    anndata.AnnData
        A copy of simple_adata with preprocessing keys added.
    """
    adata = simple_adata.copy()

    rng = np.random.default_rng(42)
    adata.obsm["X_pca"] = rng.standard_normal((adata.n_obs, 10))
    adata.obsm["X_umap"] = rng.standard_normal((adata.n_obs, 2))
    adata.obsp["connectivities"] = sp.csr_matrix((adata.n_obs, adata.n_obs))
    adata.obsp["distances"] = sp.csr_matrix((adata.n_obs, adata.n_obs))
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
    }

    return adata


@pytest.fixture
def branch_adata(preprocessed_adata):
    """Create an AnnData object with pseudotime and branch masks.

    Parameters
    ----------
    preprocessed_adata : anndata.AnnData
        AnnData object containing the preprocessing keys required by the tests.

    Returns
    -------
    anndata.AnnData
        Copy of the input AnnData with pseudotime values and deterministic
        branch masks.
    """
    adata = preprocessed_adata.copy()
    n_obs = adata.n_obs
    rng = np.random.default_rng(42)

    adata.obs["palantir_pseudotime"] = np.linspace(0, 1, n_obs)
    adata.obsm["branch_masks"] = pd.DataFrame(
        {
            "BranchA": rng.random(n_obs) > 0.6,
            "BranchB": rng.random(n_obs) > 0.5,
            "BranchC": rng.random(n_obs) > 0.7,
        },
        index=adata.obs_names,
    )

    return adata


def _patch_diffusion_magic(monkeypatch, n_obs, multiscale_components):
    """Patch Palantir diffusion and MAGIC calls with deterministic test doubles."""

    def fake_run_diffusion_maps(adata, **kwargs):
        adata.obsp["DM_Similarity"] = sp.csr_matrix(np.eye(adata.n_obs))
        return {
            "EigenVectors": pd.DataFrame(
                np.ones((adata.n_obs, multiscale_components)),
                index=adata.obs_names,
            ),
            "EigenValues": pd.Series(np.linspace(1.0, 0.5, multiscale_components)),
            "T": sp.csr_matrix(np.eye(adata.n_obs)),
        }

    def fake_determine_multiscale_space(dm_res):
        return pd.DataFrame(
            np.ones((n_obs, multiscale_components)),
        )

    def fake_run_magic_imputation(
        data,
        imputation_key="MAGIC_imputed_data",
        **kwargs,
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


# ------------------------- is_preprocessed -------------------------#


@pytest.mark.parametrize(
    ("check_kwargs", "expected"),
    [
        ({"pca": "warn"}, True),
        ({"pca": "error"}, False),
        ({"embedding": "error"}, False),
        ({}, True),
    ],
    ids=[
        "missing-pca-warn",
        "missing-pca-error",
        "missing-embedding-error",
        "all-checks-disabled",
    ],
)
def test_is_preprocessed_configurable_checks(simple_adata, check_kwargs, expected):
    """Configurable preprocessing checks should respect their severity."""
    kwargs = {
        "normalize": False,
        "log1p": False,
        "pca": False,
        "embedding": False,
        "neighbors": False,
    }
    kwargs.update(check_kwargs)

    result = pla.is_preprocessed(simple_adata, **kwargs)

    assert result is expected


def test_is_preprocessed_accepts_project_test_data(adata_h5ad):
    """Check that the project RNA test object contains the required preprocessing."""
    result = pla.is_preprocessed(
        adata_h5ad,
        normalize=False,
        log1p=False,
        pca="error",
        embedding="error",
        neighbors="error",
    )

    assert result is True


def test_is_preprocessed_detects_raw_counts(adata_raw_small):
    """Raw integer counts should fail the normalization check."""
    result = pla.is_preprocessed(
        adata_raw_small,
        normalize="error",
        log1p=False,
        pca=False,
        embedding=False,
        neighbors=False,
    )

    assert result is False


def test_is_preprocessed_accepts_sparse_matrix(adata_h5ad):
    """Sparse expression matrices should be handled without failing."""
    assert sp.issparse(adata_h5ad.X)

    result = pla.is_preprocessed(
        adata_h5ad,
        normalize=False,
        log1p="warn",
        pca=False,
        embedding=False,
        neighbors=False,
    )

    assert result is True


# ------------------------- remove_cells_by_label -------------------------#


@pytest.mark.parametrize(
    ("labels_to_remove", "removed_labels"),
    [
        ("TypeA", ["TypeA"]),
        (["TypeA", "TypeB"], ["TypeA", "TypeB"]),
    ],
    ids=["single-label", "multiple-labels"],
)
def test_remove_cells_by_label_removes_labels(
    simple_adata,
    labels_to_remove,
    removed_labels,
):
    """Selected cell labels should be removed."""
    result = pla.remove_cells_by_label(
        simple_adata,
        column_name="celltype",
        labels_to_remove=labels_to_remove,
    )

    assert result.n_obs < simple_adata.n_obs
    assert all(label not in result.obs["celltype"].values for label in removed_labels)
    assert result.n_vars == simple_adata.n_vars


@pytest.mark.parametrize(
    ("column_name", "labels_to_remove", "match"),
    [
        ("missing_column", "TypeA", "does not exist"),
        ("celltype", "MissingType", "None of the requested labels"),
    ],
    ids=["missing-column", "missing-label"],
)
def test_remove_cells_by_label_invalid_input_raises(
    simple_adata,
    column_name,
    labels_to_remove,
    match,
):
    """Invalid removal inputs should raise a ValueError."""
    with pytest.raises(ValueError, match=match):
        pla.remove_cells_by_label(
            simple_adata,
            column_name=column_name,
            labels_to_remove=labels_to_remove,
        )


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


@pytest.mark.parametrize(
    ("masks_key", "pseudotime_key"),
    [
        ("missing_masks", "palantir_pseudotime"),
        ("branch_masks", "missing_pseudotime"),
    ],
    ids=["missing-masks", "missing-pseudotime"],
)
def test_branch_qc_missing_input_returns_empty_table(
    branch_adata,
    masks_key,
    pseudotime_key,
):
    """Missing required branch QC input should return an empty table."""
    qc = pla.branch_qc(
        branch_adata,
        masks_key=masks_key,
        pseudotime_key=pseudotime_key,
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


@pytest.mark.parametrize(
    ("curve", "expected"),
    [
        (
            pd.Series([2.0, 2.0, 2.0, 2.0, 2.0]),
            "stable",
        ),
        (
            pd.Series(1.0 - 4.0 * (np.linspace(0, 1, 50) - 0.5) ** 2),
            "transient peak",
        ),
        (
            pd.Series(4.0 * (np.linspace(0, 1, 50) - 0.5) ** 2),
            "transient dip",
        ),
        (
            pd.Series(np.linspace(0, 1, 50) ** 2),
            "late increase",
        ),
        (
            pd.Series(2.0 - 2.0 * np.linspace(0, 1, 50)),
            "early high, then decrease",
        ),
        (
            pd.Series(
                np.concatenate(
                    [
                        np.full(10, 0.4),
                        np.linspace(0.45, 1.1, 20),
                        np.linspace(1.05, 1.0, 20),
                    ]
                )
            ),
            "broad increase",
        ),
        (
            pd.Series(
                np.concatenate(
                    [
                        np.full(10, 0.9),
                        np.linspace(1.0, 1.2, 10),
                        np.linspace(1.1, 0.2, 30),
                    ]
                )
            ),
            "broad decrease",
        ),
    ],
    ids=[
        "stable",
        "transient-peak",
        "transient-dip",
        "late-increase",
        "early-high-then-decrease",
        "broad-increase",
        "broad-decrease",
    ],
)
def test_classify_trend_pattern(curve, expected):
    """Trend curves should be assigned to their expected pattern."""
    assert pla._classify_trend_pattern(curve) == expected


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


def test_compare_all_layers_returns_figure_for_project_test_data(adata_h5ad):
    """Layer comparison should create one subplot for adata.X and each layer."""
    fig, axs = pla.compare_all_layers(adata_h5ad)

    try:
        axs_array = np.atleast_1d(axs)
        expected_axes = 1 + sum(
            layer_name is not None for layer_name in adata_h5ad.layers.keys()
        )

        assert fig is not None
        assert axs is not None
        assert len(axs_array) == expected_axes
    finally:
        plt.close(fig)


def test_compare_all_layers_can_exclude_layer(adata_h5ad):
    """Excluded layers should not be plotted."""
    excluded_layers = ["spliced"]

    fig, axs = pla.compare_all_layers(
        adata_h5ad,
        exclude_layers=excluded_layers,
    )

    try:
        axs_array = np.atleast_1d(axs)
        expected_axes = 1 + sum(
            layer_name is not None and layer_name not in excluded_layers
            for layer_name in adata_h5ad.layers.keys()
        )

        titles = [ax.get_title() for ax in axs_array]

        assert len(axs_array) == expected_axes
        assert not any("layers['spliced']" in title for title in titles)
    finally:
        plt.close(fig)


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

    assert fig is None
    assert isinstance(axes, np.ndarray)
    assert len(axes) == 0


# ------------------------- plot_interactive_embedding_cell_ids -------------------------#


def test_plot_interactive_embedding_cell_ids_returns_figure(
    preprocessed_adata,
):
    """Interactive embedding plots should return a Plotly figure."""
    fig = pla.plot_interactive_embedding_cell_ids(
        preprocessed_adata,
        embedding_key="X_umap",
        color_by="celltype",
        renderer=None,
        show=False,
    )

    assert isinstance(fig, PlotlyFigure)
    assert len(fig.data) > 0


def test_plot_interactive_embedding_cell_ids_missing_embedding(simple_adata):
    """A missing embedding should raise a ValueError."""
    with pytest.raises(ValueError, match="not found"):
        pla.plot_interactive_embedding_cell_ids(
            simple_adata,
            embedding_key="missing_embedding",
            renderer=None,
        )


def test_plot_interactive_embedding_cell_ids_supports_3d_embedding(simple_adata):
    """Three-dimensional embeddings should create a Plotly 3D scatter plot."""
    simple_adata.obsm["X_3d"] = np.random.randn(simple_adata.n_obs, 3)

    fig = pla.plot_interactive_embedding_cell_ids(
        simple_adata,
        embedding_key="X_3d",
        renderer=None,
        show=False,
    )

    assert fig is not None
    assert len(fig.data) > 0
    assert all(trace.type == "scatter3d" for trace in fig.data)


def test_plot_interactive_embedding_cell_ids_rejects_one_dimension(simple_adata):
    """Embeddings with fewer than two dimensions should raise a ValueError."""
    simple_adata.obsm["X_1d"] = np.random.randn(simple_adata.n_obs, 1)

    with pytest.raises(ValueError, match="at least two dimensions"):
        pla.plot_interactive_embedding_cell_ids(
            simple_adata,
            embedding_key="X_1d",
            renderer=None,
            show=False,
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
    monkeypatch,
    preprocessed_adata,
    tmp_path,
):
    """Diffusion and MAGIC results should be stored in the AnnData object."""
    _patch_diffusion_magic(
        monkeypatch,
        n_obs=preprocessed_adata.n_obs,
        multiscale_components=3,
    )

    result = pla.run_diffusion_and_magic(
        preprocessed_adata,
        save_path=str(tmp_path / "palantir_magic.h5ad"),
        n_components=3,
        knn=30,
        kernel_backend="scanpy",
        n_jobs=1,
    )

    assert result is preprocessed_adata
    assert "DM_EigenVectors_multiscaled" in result.obsm
    assert result.obsm["DM_EigenVectors_multiscaled"].shape == (
        preprocessed_adata.n_obs,
        3,
    )
    assert "MAGIC_imputed_data" in result.layers


def test_run_diffusion_and_magic_recomputes_when_cache_is_incomplete(
    monkeypatch,
    preprocessed_adata,
    tmp_path,
):
    """An incomplete cache missing the MAGIC layer should recompute imputation."""
    cached = preprocessed_adata.copy()
    cache_path = tmp_path / "cached_magic_incomplete.h5ad"
    cached.write_h5ad(cache_path)

    _patch_diffusion_magic(
        monkeypatch,
        n_obs=preprocessed_adata.n_obs,
        multiscale_components=2,
    )

    result = pla.run_diffusion_and_magic(
        preprocessed_adata,
        save_path=str(cache_path),
        recompute_magic=False,
    )

    assert result is not preprocessed_adata
    assert "DM_EigenVectors_multiscaled" in result.obsm
    assert result.obsm["DM_EigenVectors_multiscaled"].shape == (
        preprocessed_adata.n_obs,
        2,
    )
    assert "MAGIC_imputed_data" in result.layers


@pytest.mark.parametrize(
    "terminal_labels,expected",
    [
        (["Macrophage", "Macrophage"], ["Macrophage", "Macrophage_2"]),
        (["TypeA", "TypeA"], ["TypeA", "TypeA_2"]),
    ],
)
def test_detect_and_plot_terminal_states_manual_duplicate_cluster_labels(
    monkeypatch, gene_trend_adata, terminal_labels, expected
):
    """Duplicate cluster labels on manual terminals should be uniquified."""
    monkeypatch.setattr(
        pla.palantir.plot,
        "highlight_cells_on_umap",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    gene_trend_adata.obs["celltype"] = "Macrophage"
    gene_trend_adata.obs.loc[
        [gene_trend_adata.obs_names[1], gene_trend_adata.obs_names[2]],
        "celltype",
    ] = terminal_labels

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

    assert list(result.index) == terminal_cells
    assert list(result.values) == expected


@pytest.mark.parametrize(
    ("parameter", "value", "match"),
    [
        ("eigvec_key", "missing_key", "Diffusion embedding"),
        ("embedding_key", "missing_embedding", "Embedding"),
        ("early_cell", "missing_cell", "Early cell"),
    ],
    ids=[
        "missing-diffusion-embedding",
        "missing-plot-embedding",
        "missing-early-cell",
    ],
)
def test_detect_and_plot_terminal_states_missing_inputs_raise(
    gene_trend_adata,
    parameter,
    value,
    match,
):
    """Missing required terminal-state inputs should raise a ValueError."""
    kwargs = {
        "early_cell": gene_trend_adata.obs_names[0],
        "terminal_states": None,
        "embedding_key": "X_umap",
        "eigvec_key": "DM_EigenVectors_multiscaled",
    }
    kwargs[parameter] = value

    with pytest.raises(ValueError, match=match):
        pla.detect_and_plot_terminal_states(
            gene_trend_adata,
            **kwargs,
        )


def test_analyze_top_genes_communities_marker_selection_and_pvals_adj(
    monkeypatch, gene_trend_adata
):
    """Marker-based selection should respect pvals_adj filtering and ranking."""
    captured = {}

    def fake_cluster_gene_trends(adata, branch, genes, n_neighbors=30):
        captured["branch"] = branch
        captured["genes"] = list(genes)
        captured["n_neighbors"] = n_neighbors
        return pd.Series(["0" for _ in genes], index=genes, name="community")

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

    marker_table = {
        "GroupA": pd.DataFrame(
            {
                "names": ["Gene0", "Gene1", "Gene2"],
                "logfoldchanges": [2.0, 1.0, 0.5],
                "scores": [10.0, 5.0, 3.0],
                "pvals_adj": [0.01, 0.04, 0.20],
            }
        ),
        "GroupB": pd.DataFrame(
            {
                "names": ["Gene3", "Gene4"],
                "logfoldchanges": [1.5, 0.3],
                "scores": [8.0, 1.0],
                "pvals_adj": [0.02, 0.03],
            }
        ),
    }

    selected_genes, communities = pla.analyze_top_genes_communities(
        gene_trend_adata,
        trend_branches=["BranchA"],
        gene_source="markers",
        marker_table=marker_table,
        top_n=4,
        n_neighbors=5,
        store_key="marker_store",
    )

    assert "Gene0" in selected_genes
    assert "Gene1" in selected_genes
    assert "Gene4" in selected_genes
    assert "BranchA" in communities
    assert captured["branch"] == "BranchA"
    assert captured["n_neighbors"] == 5
    assert set(captured["genes"]).issubset(set(gene_trend_adata.var_names))
    assert (
        "marker_store" in gene_trend_adata.uns["sctoolbox"]["palantir_gene_communities"]
    )


def test_analyze_top_genes_communities_marker_table_missing_columns_raises(
    monkeypatch, gene_trend_adata
):
    """Markers without required ranking columns should raise a ValueError."""
    marker_table = {
        "GroupA": pd.DataFrame(
            {
                "names": ["Gene0", "Gene1"],
                "logfoldchanges": [1.0, 2.0],
            }
        )
    }

    with pytest.raises(ValueError, match="missing required columns"):
        pla.analyze_top_genes_communities(
            gene_trend_adata,
            trend_branches="BranchA",
            gene_source="markers",
            marker_table=marker_table,
            top_n=2,
        )


@pytest.mark.parametrize(
    "mode",
    ["range", "var", "union"],
)
def test_plot_branch_gene_programs_ranking_modes(monkeypatch, gene_trend_adata, mode):
    """Gene program ranking modes should forward parameters to Palantir."""
    captured = {}

    def fake_cluster_gene_trends(adata, branch, genes, n_neighbors=30):
        captured[branch] = {
            "genes": genes,
            "n_neighbors": n_neighbors,
        }
        return pd.Series(
            ["0" if i % 2 == 0 else "1" for i, _ in enumerate(genes)],
            index=genes,
        )

    def fake_plot_gene_trend_clusters(*args, **kwargs):
        fig, axes = plt.subplots(1, 1)
        axes.plot([0, 1], [0, 1])
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
        gene_ranking_mode=mode,
        trend_knn_neighbors=4,
        plot_representative_genes=False,
    )

    assert "BranchA" in result
    assert captured["BranchA"]["n_neighbors"] == 4
    assert len(captured["BranchA"]["genes"]) <= 4


def test_plot_branch_gene_programs_excludes_flat_genes(monkeypatch, gene_trend_adata):
    """Flat gene trends should be filtered before clustering."""
    adata = gene_trend_adata.copy()
    adata.varm["gene_trends_BranchA"].loc["Gene0"] = 0.5

    captured = {}

    def fake_cluster_gene_trends(adata_obj, branch, genes, n_neighbors=30):
        captured[branch] = set(genes)
        return pd.Series(
            ["0" if i % 2 == 0 else "1" for i, _ in enumerate(genes)],
            index=genes,
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

    result = pla.plot_branch_gene_programs(
        adata,
        branch="BranchA",
        n_dynamic_genes=8,
        gene_ranking_mode="union",
        trend_knn_neighbors=2,
        plot_representative_genes=False,
    )

    assert "BranchA" in result
    assert "Gene0" not in captured["BranchA"]


def test_plot_branch_gene_programs_limits_n_dynamic_genes(
    monkeypatch, gene_trend_adata
):
    """The function should not select more genes than n_dynamic_genes."""
    captured = {}

    def fake_cluster_gene_trends(adata_obj, branch, genes, n_neighbors=30):
        captured[branch] = list(genes)
        return pd.Series(
            ["0" if i % 2 == 0 else "1" for i, _ in enumerate(genes)],
            index=genes,
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

    result = pla.plot_branch_gene_programs(
        gene_trend_adata,
        branch="BranchA",
        n_dynamic_genes=3,
        gene_ranking_mode="range",
        trend_knn_neighbors=2,
        plot_representative_genes=False,
    )

    assert len(result["BranchA"]["selected_genes"]) == 3
    assert len(captured["BranchA"]) == 3


def test_plot_branch_gene_programs_unknown_branch_returns_empty(
    monkeypatch, gene_trend_adata
):
    """Requesting an unknown branch should return an empty result."""
    assert (
        pla.plot_branch_gene_programs(
            gene_trend_adata,
            branch="MissingBranch",
            n_dynamic_genes=4,
            gene_ranking_mode="union",
            trend_knn_neighbors=2,
            plot_representative_genes=False,
        )
        == {}
    )


def test_plot_branch_gene_programs_invalid_mode_returns_empty(
    monkeypatch, gene_trend_adata
):
    """An unsupported ranking mode should be skipped gracefully."""
    monkeypatch.setattr(
        pla.palantir.presults,
        "cluster_gene_trends",
        lambda *args, **kwargs: pd.Series([], index=[]),
    )
    monkeypatch.setattr(
        pla.palantir.plot,
        "plot_gene_trend_clusters",
        lambda *args, **kwargs: plt.figure(),
    )
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    result = pla.plot_branch_gene_programs(
        gene_trend_adata,
        gene_ranking_mode="invalid_mode",
        plot_representative_genes=False,
    )

    assert result == {}


def test_plot_branch_early_late_genes_skips_umap_when_missing(
    monkeypatch, gene_trend_adata
):
    """Missing UMAP data should skip the UMAP panel but still produce trends."""
    gene_trend_adata = gene_trend_adata.copy()
    gene_trend_adata.obsm.pop("X_umap", None)

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    result = pla.plot_branch_early_late_genes(
        gene_trend_adata,
        branches=["BranchA"],
        top_n=2,
        quantile_cutoff=0.25,
        umap_ncols=2,
    )

    assert result["BranchA"]["trend_fig"] is not None
    assert result["BranchA"]["umap_fig"] is None
    assert result["BranchA"]["umap_axes"].size == 0


@pytest.fixture
def gene_trend_adata(preprocessed_adata):
    """Create AnnData with small branch-specific gene trend tables.

    Returns
    -------
    anndata.AnnData
        Synthetic AnnData with branch-specific trend tables and Palantir metadata.
    """
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

    gene_trend_adata.obs["celltype"] = "TypeA"

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
        "missing_cell",
        gene_trend_adata.obs_names[1],
        gene_trend_adata.obs_names[2],
    ]
    excluded_cells = [gene_trend_adata.obs_names[3]]
    captured = {}

    def fake_identify_terminal_states(diffusion_df, **kwargs):
        captured["early_cell"] = kwargs.get("early_cell")
        captured["knn"] = kwargs.get("knn")
        captured["num_waypoints"] = kwargs.get("num_waypoints")
        return terminal_cells, excluded_cells

    monkeypatch.setattr(
        pla.palantir.core,
        "identify_terminal_states",
        fake_identify_terminal_states,
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
        knn=10,
        num_waypoints=20,
        show_excluded_boundaries=True,
    )

    assert isinstance(result, pd.Series)
    assert list(result.index) == [
        gene_trend_adata.obs_names[1],
        gene_trend_adata.obs_names[2],
    ]
    assert list(result.values) == ["terminal_state_1", "terminal_state_2"]
    assert captured["early_cell"] == gene_trend_adata.obs_names[0]
    assert captured["knn"] == 10
    assert captured["num_waypoints"] == 20


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
        store_key="test_store",
    )

    assert selected_genes == ["Gene0", "Gene1", "Gene2", "Gene3"]
    assert "BranchA" in communities
    assert isinstance(communities["BranchA"], pd.Series)
    assert "sctoolbox" in gene_trend_adata.uns
    assert (
        "test_store" in gene_trend_adata.uns["sctoolbox"]["palantir_gene_communities"]
    )


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
        branch=None,
        n_dynamic_genes=4,
        gene_ranking_mode="union",
        trend_knn_neighbors=2,
        plot_representative_genes=False,
    )

    assert set(result) == {"BranchA", "BranchB"}
    assert "selected_genes" in result["BranchA"]
    assert len(result["BranchA"]["selected_genes"]) == 4
    assert "trend_clusters" in result["BranchA"]
    assert "centroid_z" in result["BranchA"]
    assert "cluster_summary" in result["BranchA"]


def test_detect_and_plot_terminal_states_supports_3d_embedding(
    monkeypatch, gene_trend_adata
):
    """Three-dimensional embeddings should use the first two dimensions for plotting."""
    adata = gene_trend_adata.copy()
    adata.obsm["X_umap"] = np.random.randn(adata.n_obs, 3)

    captured = {}

    def fake_highlight_cells_on_umap(adata_obj, cells, embedding_basis):
        captured["embedding_shape"] = adata_obj.obsm[embedding_basis].shape
        captured["cells"] = list(cells)

    monkeypatch.setattr(
        pla.palantir.plot,
        "highlight_cells_on_umap",
        fake_highlight_cells_on_umap,
    )
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    early_cell = adata.obs_names[0]
    terminal_cells = [
        adata.obs_names[1],
        adata.obs_names[2],
    ]

    result = pla.detect_and_plot_terminal_states(
        adata,
        early_cell=early_cell,
        terminal_states=terminal_cells,
        clusters_to_analyze=None,
        embedding_key="X_umap",
        eigvec_key="DM_EigenVectors_multiscaled",
    )

    assert isinstance(result, pd.Series)
    assert captured["embedding_shape"] == (adata.n_obs, 2)
    assert captured["cells"] == terminal_cells
    assert adata.obsm["X_umap"].shape == (adata.n_obs, 3)


def test_plot_branch_gene_programs_continues_after_branch_error(
    monkeypatch,
    gene_trend_adata,
):
    """A failing branch should not stop later branches when branch=None."""
    processed_branches = []

    def fake_cluster_gene_trends(adata, branch, genes, n_neighbors=30):
        if branch == "BranchA":
            raise ValueError("simulated branch failure")

        processed_branches.append(branch)

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
    monkeypatch.setattr(
        plt,
        "show",
        lambda *args, **kwargs: None,
    )

    result = pla.plot_branch_gene_programs(
        gene_trend_adata,
        branch=None,
        n_dynamic_genes=4,
        gene_ranking_mode="union",
        trend_knn_neighbors=2,
        plot_representative_genes=False,
    )

    assert "BranchA" not in result
    assert "BranchB" in result
    assert processed_branches == ["BranchB"]


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
    assert result["BranchA"]["top_genes"]["early"][0] == "Gene3"
    assert result["BranchA"]["top_genes"]["late"][0] == "Gene0"


@pytest.mark.parametrize(
    ("top_n", "quantile_cutoff", "match"),
    [
        (2, 0.5, "quantile_cutoff"),
        (0, 0.25, "top_n"),
    ],
    ids=["invalid-quantile-cutoff", "invalid-top-n"],
)
def test_plot_branch_early_late_genes_invalid_parameters(
    gene_trend_adata,
    top_n,
    quantile_cutoff,
    match,
):
    """Invalid early/late gene parameters should raise ValueError."""
    with pytest.raises(ValueError, match=match):
        pla.plot_branch_early_late_genes(
            gene_trend_adata,
            branches=["BranchA"],
            top_n=top_n,
            quantile_cutoff=quantile_cutoff,
        )

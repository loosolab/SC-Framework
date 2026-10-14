"""Tests for Palantir analysis helper functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly.graph_objects import Figure as PlotlyFigure
import sctoolbox.tools.palantir_analysis as pla

plt.switch_backend("Agg")


# ------------------------- FIXTURES -------------------------#


# Use the shared `adata_h5ad` fixture from tests/conftest.py instead of
# defining a local duplicate fixture here.


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all Matplotlib figures after each test.

    The plotting functions return their figures and leave them registered with
    pyplot, so without this they pile up across the file.
    """
    yield
    plt.close("all")


@pytest.fixture
def preprocessed_adata(adata_h5ad):
    """Return a copy of `adata_h5ad` with a 2D UMAP and cell types.

    The shared `adata_h5ad` already carries PCA and a neighbor graph. Its UMAP
    is three-dimensional, so only the first two dimensions are kept.

    Returns
    -------
    anndata.AnnData
        A copy of adata_h5ad with a 2D `X_umap` and `obs["celltype"]`.
    """
    adata = adata_h5ad.copy()

    rng = np.random.default_rng(42)
    adata.obs["celltype"] = rng.choice(["TypeA", "TypeB", "TypeC"], size=adata.n_obs)
    adata.obsm["X_umap"] = adata.obsm["X_umap"][:, :2].copy()

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
    adata.obs["palantir_entropy"] = np.linspace(1, 0, n_obs)
    adata.obsm["branch_masks"] = pd.DataFrame(
        {
            "BranchA": rng.random(n_obs) > 0.6,
            "BranchB": rng.random(n_obs) > 0.5,
            "BranchC": rng.random(n_obs) > 0.7,
        },
        index=adata.obs_names,
    )

    return adata


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


@pytest.fixture
def fate_probability_adata(preprocessed_adata):
    """Create AnnData with a three-column Palantir fate-probability table.

    Returns
    -------
    anndata.AnnData
        Copy of the input AnnData with row-normalized fate probabilities in
        ``adata.obsm["palantir_fate_probabilities"]``.
    """
    adata = preprocessed_adata.copy()

    rng = np.random.default_rng(42)
    probabilities = rng.random((adata.n_obs, 3))
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    adata.obsm["palantir_fate_probabilities"] = pd.DataFrame(
        probabilities,
        index=adata.obs_names,
        columns=["FateA", "FateB", "FateC"],
    )

    return adata


# ------------------------- SHARED PATCH HELPERS -------------------------#


def _patch_diffusion_magic(monkeypatch, n_obs):
    """Patch Palantir diffusion and MAGIC calls.

    The fakes derive their output from the arguments they receive, so a test can
    assert on the AnnData the function produces instead of on captured keyword
    arguments: the multiscale array is ``n_components`` wide and the MAGIC layer
    is the expression matrix scaled by ``n_steps``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to apply the patches.
    n_obs : int
        Number of observations in the AnnData under test.
    """

    def fake_run_diffusion_maps(adata, **kwargs):
        n_components = kwargs["n_components"]
        adata.obsp["DM_Similarity"] = sp.csr_matrix(np.eye(adata.n_obs))
        return {
            "EigenVectors": pd.DataFrame(
                np.ones((adata.n_obs, n_components)),
                index=adata.obs_names,
            ),
            "EigenValues": pd.Series(np.linspace(1.0, 0.5, n_components)),
            "T": sp.csr_matrix(np.eye(adata.n_obs)),
        }

    def fake_determine_multiscale_space(dm_res):
        return pd.DataFrame(
            np.ones((n_obs, dm_res["EigenVectors"].shape[1])),
        )

    def fake_run_magic_imputation(
        data,
        imputation_key="MAGIC_imputed_data",
        **kwargs,
    ):
        data.layers[imputation_key] = data.X.copy() * kwargs["n_steps"]
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


def _patch_gene_program_clustering(monkeypatch, failing_branch=None, failing_step=None):
    """Patch gene-trend clustering and plotting for gene-program tests.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to apply the patches.
    failing_branch : str | None, default=None
        Branch for which ``failing_step`` raises a ValueError.
    failing_step : str | None, default=None
        Step that fails for ``failing_branch``, either ``"clustering"`` or
        ``"plotting"``.

    Returns
    -------
    list[str]
        Branches that reached the patched plotting step, in call order.
    """
    plotted_branches = []

    def fake_cluster_gene_trends(adata, branch, genes, n_neighbors=30):
        if failing_step == "clustering" and branch == failing_branch:
            raise ValueError("simulated branch failure")

        return pd.Series(
            ["0" if i % 2 == 0 else "1" for i, _ in enumerate(genes)],
            index=list(genes),
            name="community",
        )

    def fake_plot_gene_trend_clusters(adata, branch):
        if failing_step == "plotting" and branch == failing_branch:
            raise ValueError("simulated plot failure")

        plotted_branches.append(branch)

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

    return plotted_branches


def _patch_plot_trend(monkeypatch):
    """Patch Palantir's trend plot so it draws into the axes it is given.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to apply the patch.

    Returns
    -------
    list[tuple[str, str]]
        Branch and gene pairs passed to the patched call, in call order.
    """
    plotted = []

    def fake_plot_trend(adata, branch, gene, ax=None, **kwargs):
        plotted.append((branch, gene))

        if ax is None:
            _, ax = plt.subplots()

        ax.plot([0, 1], [0, 1])
        cell_ax = ax.twinx()
        cell_ax.scatter([0, 1], [0, 1])

    monkeypatch.setattr(pla.palantir.plot, "plot_trend", fake_plot_trend)

    return plotted


def _record_saves(monkeypatch):
    """Patch `_save_figure` and record the paths it is asked to write.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to apply the patch.

    Returns
    -------
    list[tuple[str, bool]]
        Path and report-flag pairs, in call order.
    """
    saved = []

    monkeypatch.setattr(
        pla,
        "_save_figure",
        lambda path, report=False, **kwargs: saved.append((path, report)),
    )

    return saved


# ------------------------- is_preprocessed -------------------------#


@pytest.mark.parametrize(
    ("check_kwargs", "missing_obsm", "expected"),
    [
        ({"pca": "warn"}, ["X_pca"], True),
        ({"pca": "error"}, ["X_pca"], False),
        ({"embedding": "error"}, ["X_umap", "X_tsne"], False),
        ({}, ["X_pca", "X_umap", "X_tsne"], True),
    ],
    ids=[
        "missing-pca-warn",
        "missing-pca-error",
        "missing-embedding-error",
        "all-checks-disabled",
    ],
)
def test_is_preprocessed_configurable_checks(
    preprocessed_adata, check_kwargs, missing_obsm, expected
):
    """Configurable preprocessing checks should respect their severity."""
    adata = preprocessed_adata.copy()
    for key in missing_obsm:
        adata.obsm.pop(key, None)

    kwargs = {
        "normalize": False,
        "log1p": False,
        "pca": False,
        "embedding": False,
        "neighbors": False,
    }
    kwargs.update(check_kwargs)

    result = pla.is_preprocessed(adata, **kwargs)

    assert result is expected


@pytest.mark.parametrize(
    ("check_kwargs", "expected"),
    [
        ({"pca": "error", "embedding": "error", "neighbors": "error"}, True),
        ({"log1p": "warn"}, True),
    ],
    ids=["project-test-data", "sparse-matrix"],
)
def test_is_preprocessed_accepts_project_test_data(adata_h5ad, check_kwargs, expected):
    """The project RNA test object should pass the relevant preprocessing checks."""
    assert sp.issparse(adata_h5ad.X)

    kwargs = {
        "normalize": False,
        "log1p": False,
        "pca": False,
        "embedding": False,
        "neighbors": False,
    }
    kwargs.update(check_kwargs)

    assert pla.is_preprocessed(adata_h5ad, **kwargs) is expected


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
    preprocessed_adata,
    labels_to_remove,
    removed_labels,
):
    """Selected cell labels should be removed."""
    result = pla.remove_cells_by_label(
        preprocessed_adata,
        column_name="celltype",
        labels_to_remove=labels_to_remove,
    )

    assert result.n_obs < preprocessed_adata.n_obs
    assert all(label not in result.obs["celltype"].values for label in removed_labels)
    assert result.n_vars == preprocessed_adata.n_vars


@pytest.mark.parametrize(
    ("column_name", "labels_to_remove", "match"),
    [
        ("missing_column", "TypeA", "does not exist"),
        ("celltype", "MissingType", "None of the requested labels"),
    ],
    ids=["missing-column", "missing-label"],
)
def test_remove_cells_by_label_invalid_input_raises(
    preprocessed_adata,
    column_name,
    labels_to_remove,
    match,
):
    """Invalid removal inputs should raise a ValueError."""
    with pytest.raises(ValueError, match=match):
        pla.remove_cells_by_label(
            preprocessed_adata,
            column_name=column_name,
            labels_to_remove=labels_to_remove,
        )


# ------------------------- branch_qc -------------------------#


def test_branch_qc_returns_one_row_per_branch(branch_adata):
    """Branch QC should return one row per branch with the expected metrics."""
    qc = pla.branch_qc(branch_adata)
    masks = branch_adata.obsm["branch_masks"]

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
    # Work on a copy so the fixture object is not altered for other assertions.
    adata = branch_adata.copy()
    adata.obsm["branch_masks"]["EmptyBranch"] = False

    qc = pla.branch_qc(adata)
    empty_row = qc[qc["branch"] == "EmptyBranch"]

    assert not empty_row.empty
    assert empty_row["n_cells"].iloc[0] == 0
    assert not bool(empty_row["keep"].iloc[0])


# ------------------------- _classify_trend_pattern -------------------------#


pt = np.linspace(0, 1, 50)

broad_increase_curve = pd.Series(
    np.concatenate(
        [
            np.full(10, 0.4),
            np.linspace(0.45, 1.1, 20),
            np.linspace(1.05, 1.0, 20),
        ]
    )
)

broad_decrease_curve = pd.Series(
    np.concatenate(
        [
            np.full(10, 0.9),
            np.linspace(1.0, 1.2, 10),
            np.linspace(1.1, 0.2, 30),
        ]
    )
)


@pytest.mark.parametrize(
    ("curve", "expected"),
    [
        (pd.Series([2.0] * 5), "stable"),
        (pd.Series(1.0 - 4.0 * (pt - 0.5) ** 2), "transient peak"),
        (pd.Series(4.0 * (pt - 0.5) ** 2), "transient dip"),
        (pd.Series(pt**2), "late increase"),
        (pd.Series(2.0 - 2.0 * pt), "early high, then decrease"),
        (broad_increase_curve, "broad increase"),
        (broad_decrease_curve, "broad decrease"),
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


@pytest.mark.parametrize(
    "excluded_layers",
    [None, ["spliced"]],
    ids=["all-layers", "excluded-layer"],
)
def test_compare_all_layers_plots_one_panel_per_layer(adata_h5ad, excluded_layers):
    """Layer comparison should create one subplot for adata.X and each kept layer."""
    fig = None

    try:
        fig, axs = pla.compare_all_layers(adata_h5ad, exclude_layers=excluded_layers)

        axs_array = np.atleast_1d(axs)
        expected_axes = 1 + sum(
            layer_name is not None
            and (excluded_layers is None or layer_name not in excluded_layers)
            for layer_name in adata_h5ad.layers.keys()
        )
        titles = [ax.get_title() for ax in axs_array]

        assert fig is not None
        assert len(axs_array) == expected_axes

        if excluded_layers is not None:
            assert not any("layers['spliced']" in title for title in titles)
    finally:
        if fig is not None:
            plt.close(fig)


# ------------------------- plot_branch_qc_distributions -------------------------#


@pytest.mark.parametrize(
    ("masks_key", "expect_figure"),
    [("branch_masks", True), ("missing_masks", False)],
    ids=["present-masks", "missing-masks"],
)
def test_plot_branch_qc_distributions(branch_adata, masks_key, expect_figure):
    """Distributions should plot for valid masks and return None otherwise."""
    qc = pla.branch_qc(branch_adata)

    result = pla.plot_branch_qc_distributions(
        branch_adata,
        qc,
        masks_key=masks_key,
        pseudotime_key="palantir_pseudotime",
        max_branches=2,
        bins=10,
        ncols=2,
    )

    if not expect_figure:
        assert result is None
        return

    fig, axes = result
    assert isinstance(axes, np.ndarray)
    assert len(axes) >= 2
    plt.close(fig)


# ------------------------- select_early_cell -------------------------#


@pytest.mark.parametrize("fallback_seed", [7, None])
def test_select_early_cell_auto(monkeypatch, preprocessed_adata, fallback_seed):
    """Automatic early-cell selection should delegate to Palantir."""
    expected = preprocessed_adata.obs_names[0]

    def fake_early_cell(adata, celltype, celltype_column, eigvec_key, fallback_seed):
        # Echo the forwarded arguments back through the return value, so the
        # assertion below checks the function's own result rather than a patch.
        return f"{expected}|{celltype}|{celltype_column}|{eigvec_key}|{fallback_seed}"

    monkeypatch.setattr(pla.palantir.utils, "early_cell", fake_early_cell)

    result = pla.select_early_cell(
        preprocessed_adata,
        celltype="TypeA",
        celltype_column="celltype",
        fallback_seed=fallback_seed,
        eigvec_key="custom_key",
    )

    assert result == f"{expected}|TypeA|celltype|custom_key|{fallback_seed}"


def test_select_early_cell_manual(preprocessed_adata):
    """A valid manually provided early cell should be returned unchanged."""
    early_cell = preprocessed_adata.obs_names[0]

    result = pla.select_early_cell(
        preprocessed_adata,
        celltype="TypeA",
        celltype_column="celltype",
        early_cell=early_cell,
    )

    assert result == early_cell


def test_select_early_cell_invalid_manual_cell_raises(preprocessed_adata):
    """An unknown manual early cell should raise a ValueError."""
    with pytest.raises(ValueError, match="not found"):
        pla.select_early_cell(
            preprocessed_adata,
            celltype="TypeA",
            celltype_column="celltype",
            early_cell="missing_cell",
        )


def test_plot_early_cell(monkeypatch, preprocessed_adata):
    """Early-cell plotting should highlight, save, and return the figure."""
    early_cell = preprocessed_adata.obs_names[0]
    fig, ax = plt.subplots()

    def fake_highlight_cells_on_umap(adata, cells, **kwargs):
        # Echo the arguments onto the axes so the assertions below read the
        # function's own result rather than a captured dict.
        assert adata is preprocessed_adata  # a 2D embedding is used as it is
        ax.set_xlabel(kwargs["embedding_basis"])
        ax.set_ylabel(",".join(cells))
        return fig, ax

    monkeypatch.setattr(
        pla.palantir.plot,
        "highlight_cells_on_umap",
        fake_highlight_cells_on_umap,
    )
    saved = _record_saves(monkeypatch)

    result_fig, result_ax = pla.plot_early_cell(
        preprocessed_adata,
        early_cell=early_cell,
        save="early_cell.pdf",
        report_save="01_early_cell.png",
        show=False,
    )

    assert (result_fig, result_ax) == (fig, ax)
    assert result_ax.get_ylabel() == early_cell
    assert result_ax.get_xlabel() == "X_umap"
    assert early_cell in result_ax.get_title()
    assert saved == [("early_cell.pdf", False), ("01_early_cell.png", True)]

    plt.close(fig)


def test_plot_early_cell_uses_first_two_dimensions_of_3d_embedding(
    monkeypatch, preprocessed_adata
):
    """A 3D embedding is passed to Palantir as its first two dimensions only."""
    adata = preprocessed_adata.copy()
    adata.obsm["X_umap"] = np.random.default_rng(0).standard_normal((adata.n_obs, 3))
    embedding_before = adata.obsm["X_umap"].copy()
    fig, ax = plt.subplots()
    calls = []

    def fake_highlight_cells_on_umap(data, cells, **kwargs):
        # Palantir only knows x and y, so anything else would fail there.
        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["x", "y"]
        np.testing.assert_array_equal(data.to_numpy(), embedding_before[:, :2])
        assert list(data.index) == list(adata.obs_names)
        calls.append(cells)
        return fig, ax

    monkeypatch.setattr(
        pla.palantir.plot,
        "highlight_cells_on_umap",
        fake_highlight_cells_on_umap,
    )

    result = pla.plot_early_cell(adata, early_cell=adata.obs_names[0], show=False)

    assert result == (fig, ax)
    assert calls == [[adata.obs_names[0]]]
    np.testing.assert_array_equal(adata.obsm["X_umap"], embedding_before)
    assert adata.obsm["X_umap"].shape == (adata.n_obs, 3)

    plt.close(fig)


@pytest.mark.parametrize(
    ("early_cell", "embedding_key", "match"),
    [
        ("missing_cell", "X_umap", "Early cell"),
        (None, "missing_embedding", "Embedding"),
    ],
    ids=["missing-early-cell", "missing-embedding"],
)
def test_plot_early_cell_invalid_input_raises(
    preprocessed_adata,
    early_cell,
    embedding_key,
    match,
):
    """Invalid early-cell plotting inputs should raise ValueError."""
    if early_cell is None:
        early_cell = preprocessed_adata.obs_names[0]

    with pytest.raises(ValueError, match=match):
        pla.plot_early_cell(
            preprocessed_adata,
            early_cell=early_cell,
            embedding_key=embedding_key,
            show=False,
        )


# ------------------------- plot_interactive_embedding_cell_ids -------------------------#


@pytest.mark.parametrize(
    ("n_dimensions", "trace_type"),
    [
        (2, "scatter"),
        (3, "scatter3d"),
        (4, "scatter3d"),
    ],
    ids=["2d", "3d", "4d"],
)
def test_plot_interactive_embedding_cell_ids_supports_dimensions(
    preprocessed_adata,
    n_dimensions,
    trace_type,
):
    """Interactive embedding plots should handle 2D, 3D, and higher-dimensional inputs."""
    adata = preprocessed_adata.copy()
    rng = np.random.default_rng(42)
    adata.obsm["X_test"] = rng.standard_normal((adata.n_obs, n_dimensions))

    fig = pla.plot_interactive_embedding_cell_ids(
        adata,
        embedding_key="X_test",
        color_by="celltype",
        renderer=None,
        show=False,
    )

    assert isinstance(fig, PlotlyFigure)
    assert len(fig.data) > 0
    assert all(trace.type == trace_type for trace in fig.data)


@pytest.mark.parametrize("show", [True, False], ids=["show", "no-show"])
def test_plot_interactive_embedding_cell_ids_initializes_notebook_mode(
    monkeypatch,
    preprocessed_adata,
    show,
):
    """Displaying an interactive embedding should initialize Plotly notebook mode."""
    initializations = []

    monkeypatch.setattr(
        pla.po.offline,
        "init_notebook_mode",
        lambda **kwargs: initializations.append(kwargs),
    )
    monkeypatch.setattr(PlotlyFigure, "show", lambda self, **kwargs: None)

    pla.plot_interactive_embedding_cell_ids(
        preprocessed_adata,
        embedding_key="X_umap",
        renderer=None,
        show=show,
    )

    assert len(initializations) == (1 if show else 0)


@pytest.mark.parametrize(
    ("embedding_key", "color_by", "n_dimensions", "match"),
    [
        ("missing_embedding", None, None, "not found"),
        ("X_1d", None, 1, "at least two coordinate dimensions"),
        ("X_umap", "missing_column", None, "not found"),
    ],
    ids=["missing-embedding", "one-dimension", "missing-color-column"],
)
def test_plot_interactive_embedding_cell_ids_invalid_input_raises(
    preprocessed_adata,
    embedding_key,
    color_by,
    n_dimensions,
    match,
):
    """Invalid interactive-embedding inputs should raise a ValueError."""
    # Work on a copy so the added embedding does not leak into other tests.
    adata = preprocessed_adata.copy()

    if n_dimensions is not None:
        adata.obsm[embedding_key] = np.random.randn(adata.n_obs, n_dimensions)

    with pytest.raises(ValueError, match=match):
        pla.plot_interactive_embedding_cell_ids(
            adata,
            embedding_key=embedding_key,
            color_by=color_by,
            renderer=None,
            show=False,
        )


# ------------------------- run_diffusion_and_magic -------------------------#


@pytest.mark.parametrize(
    ("cache", "n_components", "n_steps", "expected_components"),
    [
        (None, 3, 1, 3),
        (None, 2, 3, 2),
        ("complete", 2, 3, 5),
        ("incomplete", 2, 3, 2),
    ],
    ids=["no-cache", "no-cache-3-steps", "complete-cache", "incomplete-cache"],
)
def test_run_diffusion_and_magic_stores_results(
    monkeypatch,
    preprocessed_adata,
    tmp_path,
    cache,
    n_components,
    n_steps,
    expected_components,
):
    """Diffusion and MAGIC results should be stored in the returned AnnData.

    The patched Palantir calls size their output from the arguments they are
    handed - the multiscale array is ``n_components`` wide and the MAGIC layer is
    the expression matrix scaled by ``n_steps`` - so these assertions check the
    function's own effect on the AnnData instead of captured keyword arguments.
    A complete cache keeps its own five-component array because it is returned
    without recomputation.
    """
    _patch_diffusion_magic(monkeypatch, preprocessed_adata.n_obs)
    expected = preprocessed_adata

    if cache is not None:
        expected = preprocessed_adata.copy()
        expected.obsm["DM_EigenVectors_multiscaled"] = np.ones((expected.n_obs, 5))

        if cache == "complete":
            expected.layers["MAGIC_imputed_data"] = expected.X.copy() * n_steps

        monkeypatch.setattr(pla.os, "makedirs", lambda *args, **kwargs: None)
        monkeypatch.setattr(pla.os.path, "exists", lambda _path: True)
        monkeypatch.setattr(pla.sc, "read_h5ad", lambda _path: expected)

    result = pla.run_diffusion_and_magic(
        preprocessed_adata,
        save_path=str(tmp_path / "palantir_magic.h5ad"),
        n_components=n_components,
        n_steps=n_steps,
        recompute_magic=False,
    )

    assert result is expected
    assert result.obsm["DM_EigenVectors_multiscaled"].shape == (
        preprocessed_adata.n_obs,
        expected_components,
    )
    assert result.layers["MAGIC_imputed_data"].max() == pytest.approx(
        result.X.max() * n_steps
    )


def test_run_diffusion_and_magic_forwards_parameters(
    monkeypatch, preprocessed_adata, tmp_path
):
    """`knn`, `kernel_backend`, `n_jobs` and `imputation_key` should reach Palantir."""
    _patch_diffusion_magic(monkeypatch, preprocessed_adata.n_obs)
    captured = {}
    fake_diffusion_maps = pla.palantir.utils.run_diffusion_maps
    fake_magic_imputation = pla.palantir.utils.run_magic_imputation

    def spy_diffusion_maps(adata, **kwargs):
        captured["diffusion"] = kwargs
        return fake_diffusion_maps(adata, **kwargs)

    def spy_magic_imputation(**kwargs):
        captured["magic"] = kwargs
        return fake_magic_imputation(**kwargs)

    monkeypatch.setattr(pla.palantir.utils, "run_diffusion_maps", spy_diffusion_maps)
    monkeypatch.setattr(pla.palantir.utils, "run_magic_imputation", spy_magic_imputation)

    result = pla.run_diffusion_and_magic(
        preprocessed_adata,
        save_path=str(tmp_path / "palantir_magic.h5ad"),
        n_components=3,
        knn=7,
        kernel_backend="sklearn",
        n_jobs=2,
        imputation_key="custom_magic",
    )

    # The fake falls back to its default key unless the custom one is forwarded.
    assert "custom_magic" in result.layers
    assert "MAGIC_imputed_data" not in result.layers
    assert captured["diffusion"]["knn"] == 7
    assert captured["diffusion"]["kernel_backend"] == "sklearn"
    assert captured["magic"]["n_jobs"] == 2


# ------------------------- detect_and_plot_terminal_states -------------------------#


@pytest.mark.parametrize(
    ("cluster_labels", "expected", "embedding_dims"),
    [
        (None, None, 2),
        (["Macrophage", "Macrophage"], ["Macrophage", "Macrophage_2"], 2),
        (["TypeA", "TypeA"], ["TypeA", "TypeA_2"], 2),
        (None, None, 3),
    ],
    ids=[
        "unique-labels",
        "duplicate-macrophage",
        "duplicate-typea",
        "3d-embedding",
    ],
)
def test_detect_and_plot_terminal_states_manual(
    monkeypatch,
    gene_trend_adata,
    cluster_labels,
    expected,
    embedding_dims,
):
    """Manual terminal cells should become a Series with unique cluster labels.

    A three-dimensional embedding is plotted using its first two dimensions
    while the embedding stored in ``adata.obsm`` keeps all three.
    """
    adata = gene_trend_adata.copy()
    terminal_cells = [adata.obs_names[1], adata.obs_names[2]]
    plotted = {}

    if embedding_dims == 3:
        adata.obsm["X_umap"] = np.random.randn(adata.n_obs, 3)

    def fake_highlight_cells_on_umap(data, cells):
        # Palantir only knows x and y, so anything else would fail there.
        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["x", "y"]
        plotted["shape"] = data.shape
        plotted["cells"] = list(cells)

    monkeypatch.setattr(
        pla.palantir.plot,
        "highlight_cells_on_umap",
        fake_highlight_cells_on_umap,
    )
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    adata.obs["celltype"] = "Macrophage"
    if cluster_labels is not None:
        adata.obs.loc[terminal_cells, "celltype"] = cluster_labels

    result = pla.detect_and_plot_terminal_states(
        adata,
        early_cell=adata.obs_names[0],
        terminal_states=terminal_cells,
        clusters_to_analyze="celltype",
        embedding_key="X_umap",
        eigvec_key="DM_EigenVectors_multiscaled",
    )

    assert isinstance(result, pd.Series)
    assert list(result.index) == terminal_cells
    assert plotted["cells"] == terminal_cells
    assert plotted["shape"] == (adata.n_obs, 2)
    assert adata.obsm["X_umap"].shape == (adata.n_obs, embedding_dims)

    if expected is not None:
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


@pytest.mark.parametrize(
    "trailing_shape",
    [(1,), (2, 2)],
    ids=["one-coordinate-dimension", "three-dimensional-array"],
)
def test_detect_and_plot_terminal_states_invalid_embedding_shape_raises(
    gene_trend_adata, trailing_shape
):
    """An embedding that is not a 2D array with at least two coordinates should raise."""
    adata = gene_trend_adata.copy()
    adata.obsm["X_invalid"] = np.zeros((adata.n_obs, *trailing_shape))

    with pytest.raises(ValueError, match="2D array with at least two coordinate dimensions"):
        pla.detect_and_plot_terminal_states(
            adata,
            early_cell=adata.obs_names[0],
            embedding_key="X_invalid",
        )


def test_detect_and_plot_terminal_states_auto(monkeypatch, gene_trend_adata):
    """Automatic detection should keep only the Palantir cells present in adata."""
    terminal_cells = [
        "missing_cell",
        gene_trend_adata.obs_names[1],
        gene_trend_adata.obs_names[2],
    ]

    def fake_identify_terminal_states(diffusion_df, early_cell, knn, num_waypoints, **kwargs):
        # Palantir's result depends on these three arguments; the fake only finds
        # terminal cells when they arrive unchanged.
        if (early_cell, knn, num_waypoints) != (gene_trend_adata.obs_names[0], 10, 20):
            return [], []

        return terminal_cells, [gene_trend_adata.obs_names[3]]

    monkeypatch.setattr(
        pla.palantir.core,
        "identify_terminal_states",
        fake_identify_terminal_states,
    )

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
        plot=False,
    )

    assert isinstance(result, pd.Series)
    assert list(result.index) == [
        gene_trend_adata.obs_names[1],
        gene_trend_adata.obs_names[2],
    ]
    assert list(result.values) == ["terminal_state_1", "terminal_state_2"]


# ------------------------- analyze_top_genes_communities -------------------------#


# Gene2 carries the largest score but fails the pvals_adj cutoff, so it is only
# ever selected if the significance filter is skipped.
SINGLE_GROUP_MARKERS = {
    "GroupA": pd.DataFrame(
        {
            "names": ["Gene0", "Gene1", "Gene2"],
            "logfoldchanges": [2.0, 1.0, 0.5],
            "scores": [10.0, 5.0, 100.0],
            "pvals_adj": [0.01, 0.04, 0.20],
        }
    )
}

TWO_GROUP_MARKERS = SINGLE_GROUP_MARKERS | {
    "GroupB": pd.DataFrame(
        {
            "names": ["Gene3", "Gene4"],
            "logfoldchanges": [1.5, 0.3],
            "scores": [8.0, 1.0],
            "pvals_adj": [0.02, 0.03],
        }
    )
}

# Gene0 has the larger score, Gene1 the larger log fold change.
OPPOSITE_ORDER_MARKERS = {
    "GroupA": pd.DataFrame(
        {
            "names": ["Gene0", "Gene1"],
            "logfoldchanges": [0.1, 5.0],
            "scores": [10.0, 5.0],
            "pvals_adj": [0.01, 0.02],
        }
    )
}

# Same table without the optional logfoldchanges column.
NO_LOGFC_MARKERS = {
    "GroupA": SINGLE_GROUP_MARKERS["GroupA"].drop(columns="logfoldchanges")
}


@pytest.mark.parametrize(
    ("marker_table", "top_n", "expected_selected", "rejected"),
    [
        (SINGLE_GROUP_MARKERS, 1, {"Gene0"}, "Gene2"),
        (NO_LOGFC_MARKERS, 1, {"Gene0"}, "Gene2"),
        (TWO_GROUP_MARKERS, 4, {"Gene0", "Gene1", "Gene3"}, "Gene2"),
        (OPPOSITE_ORDER_MARKERS, 1, {"Gene0"}, "Gene1"),
    ],
    ids=["single-group", "without-logfoldchanges", "two-groups", "score-not-logfoldchange"],
)
def test_analyze_top_genes_communities_ranks_markers_by_score(
    monkeypatch,
    gene_trend_adata,
    marker_table,
    top_n,
    expected_selected,
    rejected,
):
    """Markers should be ranked by score magnitude after pvals_adj filtering."""
    _patch_gene_program_clustering(monkeypatch)

    selected_genes, communities = pla.analyze_top_genes_communities(
        gene_trend_adata,
        trend_branches=["BranchA"],
        gene_source="markers",
        marker_table=marker_table,
        top_n=top_n,
        store_key="marker_store",
    )

    assert expected_selected <= set(selected_genes)
    # The rejected gene fails the pvals_adj cutoff or has the lower score.
    assert rejected not in selected_genes
    assert list(communities["BranchA"].index) == selected_genes
    assert (
        "marker_store" in gene_trend_adata.uns["sctoolbox"]["palantir_gene_communities"]
    )


@pytest.mark.parametrize(
    ("gene_source", "marker_table", "drop_hvg", "match"),
    [
        (
            "markers",
            {"GroupA": pd.DataFrame({"names": ["Gene0"], "logfoldchanges": [1.0]})},
            False,
            "missing required columns",
        ),
        (
            "markers",
            {"GroupA": pd.DataFrame(columns=["names", "scores"])},
            False,
            "all marker tables are empty",
        ),
        ("markers", None, False, "requires marker_table"),
        ("hvg", None, True, "highly_variable"),
    ],
    ids=[
        "missing-score-column",
        "empty-marker-tables",
        "no-marker-table",
        "missing-hvg-annotation",
    ],
)
def test_analyze_top_genes_communities_invalid_input_raises(
    gene_trend_adata, gene_source, marker_table, drop_hvg, match
):
    """Unusable gene-selection input should raise a ValueError."""
    adata = gene_trend_adata.copy()

    if drop_hvg:
        del adata.var["highly_variable"]

    with pytest.raises(ValueError, match=match):
        pla.analyze_top_genes_communities(
            adata,
            trend_branches="BranchA",
            gene_source=gene_source,
            marker_table=marker_table,
            top_n=2,
        )


def test_analyze_top_genes_communities_hvg_selection(monkeypatch, gene_trend_adata):
    """HVG-based community analysis should return selected genes and communities."""
    _patch_gene_program_clustering(monkeypatch)

    selected_genes, communities = pla.analyze_top_genes_communities(
        gene_trend_adata,
        trend_branches="BranchA",
        gene_source="hvg",
        top_n=4,
        n_neighbors=2,
        store_key="test_store",
    )

    assert selected_genes == ["Gene0", "Gene1", "Gene2", "Gene3"]
    assert isinstance(communities["BranchA"], pd.Series)
    assert list(communities["BranchA"].index) == selected_genes
    assert (
        "test_store" in gene_trend_adata.uns["sctoolbox"]["palantir_gene_communities"]
    )


def test_analyze_top_genes_communities_forwards_n_neighbors(
    monkeypatch, gene_trend_adata
):
    """`n_neighbors` should reach Palantir's clustering, which sets the cluster count."""
    _patch_gene_program_clustering(monkeypatch)
    # One cluster per neighbor: with the default of 30, four genes would give four clusters.
    monkeypatch.setattr(
        pla.palantir.presults,
        "cluster_gene_trends",
        lambda adata, branch, genes, n_neighbors: pd.Series(
            [str(i % n_neighbors) for i in range(len(genes))], index=genes
        ),
    )

    _, communities = pla.analyze_top_genes_communities(
        gene_trend_adata,
        trend_branches="BranchA",
        gene_source="hvg",
        top_n=4,
        n_neighbors=3,
    )

    assert communities["BranchA"].nunique() == 3


def test_analyze_top_genes_communities_rejects_unexpected_palantir_output(
    monkeypatch, gene_trend_adata
):
    """A non-tabular Palantir result should raise a TypeError."""
    monkeypatch.setattr(
        pla.palantir.presults,
        "cluster_gene_trends",
        lambda *args, **kwargs: ["not", "a", "table"],
    )

    with pytest.raises(TypeError, match="Unexpected return type"):
        pla.analyze_top_genes_communities(
            gene_trend_adata,
            trend_branches="BranchA",
            gene_source="hvg",
        )


def test_analyze_top_genes_communities_writes_excel_with_trends(
    monkeypatch, gene_trend_adata, tmp_path
):
    """Branch sheets should carry the community table joined with its trends."""
    _patch_gene_program_clustering(monkeypatch)
    written = {}

    monkeypatch.setattr(
        pla.utils.tables,
        "write_excel",
        lambda table_dict, filename, **kwargs: written.update(table_dict),
    )
    saved = _record_saves(monkeypatch)

    pla.analyze_top_genes_communities(
        gene_trend_adata,
        trend_branches="BranchA",
        gene_source="hvg",
        top_n=4,
        save_excel="gene_communities.xlsx",
        save_plots=str(tmp_path / "communities"),
    )

    assert set(written) == {"BranchA"}
    assert "gene" in written["BranchA"].columns
    assert any(column.startswith("pt_") for column in written["BranchA"].columns)
    assert saved == [(f"{tmp_path / 'communities'}_BranchA.pdf", False)]


# ------------------------- plot_branch_gene_programs -------------------------#


@pytest.mark.parametrize(
    "mode",
    ["range", "var", "union"],
)
def test_plot_branch_gene_programs_ranking_modes(monkeypatch, gene_trend_adata, mode):
    """Every supported ranking mode should return a usable branch result."""
    _patch_gene_program_clustering(monkeypatch)

    result = pla.plot_branch_gene_programs(
        gene_trend_adata,
        branch="BranchA",
        n_dynamic_genes=4,
        gene_ranking_mode=mode,
        trend_knn_neighbors=4,
        plot_representative_genes=False,
    )

    assert "BranchA" in result
    assert len(result["BranchA"]["selected_genes"]) == 4
    assert set(result["BranchA"]["cluster_patterns"]) == set(
        result["BranchA"]["centroid_z"].index
    )


def test_plot_branch_gene_programs_forwards_trend_knn_neighbors(
    monkeypatch, gene_trend_adata
):
    """`trend_knn_neighbors` should reach Palantir's clustering, which sets the cluster count."""
    _patch_gene_program_clustering(monkeypatch)
    # One cluster per neighbor: with the default of 30, four genes would give four clusters.
    monkeypatch.setattr(
        pla.palantir.presults,
        "cluster_gene_trends",
        lambda adata, branch, genes, n_neighbors: pd.Series(
            [str(i % n_neighbors) for i in range(len(genes))], index=genes
        ),
    )

    result = pla.plot_branch_gene_programs(
        gene_trend_adata,
        branch="BranchA",
        n_dynamic_genes=4,
        trend_knn_neighbors=3,
        plot_representative_genes=False,
    )

    assert result["BranchA"]["trend_clusters"].nunique() == 3


@pytest.mark.parametrize("failing_step", ["clustering", "plotting"])
@pytest.mark.parametrize("branch", [None, "BranchA"], ids=["all-branches", "one-branch"])
def test_plot_branch_gene_programs_branch_failure(
    monkeypatch,
    gene_trend_adata,
    branch,
    failing_step,
):
    """A failing branch is skipped when looping and re-raised when requested."""
    plotted_branches = _patch_gene_program_clustering(
        monkeypatch,
        failing_branch="BranchA",
        failing_step=failing_step,
    )

    kwargs = {
        "branch": branch,
        "n_dynamic_genes": 4,
        "gene_ranking_mode": "union",
        "trend_knn_neighbors": 2,
        "plot_representative_genes": False,
    }

    if branch is not None:
        with pytest.raises(ValueError, match="simulated"):
            pla.plot_branch_gene_programs(gene_trend_adata, **kwargs)
        return

    result = pla.plot_branch_gene_programs(gene_trend_adata, **kwargs)

    assert "BranchA" not in result
    assert "BranchB" in result
    assert plotted_branches == ["BranchB"]


@pytest.mark.parametrize(
    ("flatten_gene0", "n_dynamic_genes", "expected_selected"),
    [(True, 8, 7), (False, 3, 3)],
    ids=["excludes-flat-genes", "limits-n-dynamic-genes"],
)
def test_plot_branch_gene_programs_gene_selection(
    monkeypatch, gene_trend_adata, flatten_gene0, n_dynamic_genes, expected_selected
):
    """Flat genes are dropped and no more than n_dynamic_genes are selected."""
    adata = gene_trend_adata.copy()

    if flatten_gene0:
        adata.varm["gene_trends_BranchA"].loc["Gene0"] = 0.5

    _patch_gene_program_clustering(monkeypatch)

    result = pla.plot_branch_gene_programs(
        adata,
        branch="BranchA",
        n_dynamic_genes=n_dynamic_genes,
        gene_ranking_mode="union",
        trend_knn_neighbors=2,
        plot_representative_genes=False,
    )

    assert len(result["BranchA"]["selected_genes"]) == expected_selected

    if flatten_gene0:
        assert "Gene0" not in result["BranchA"]["selected_genes"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"branch": "MissingBranch"},
        {"gene_ranking_mode": "invalid_mode"},
    ],
    ids=["unknown-branch", "invalid-ranking-mode"],
)
def test_plot_branch_gene_programs_returns_empty(monkeypatch, gene_trend_adata, kwargs):
    """Unusable input should be skipped gracefully and return an empty result."""
    _patch_gene_program_clustering(monkeypatch)

    result = pla.plot_branch_gene_programs(
        gene_trend_adata,
        plot_representative_genes=False,
        **kwargs,
    )

    assert result == {}


@pytest.mark.parametrize(
    "with_representatives",
    [False, True],
    ids=["without-representatives", "with-representatives"],
)
def test_plot_branch_gene_programs_smoke(
    monkeypatch, gene_trend_adata, tmp_path, with_representatives
):
    """Branch gene programs should be generated from small synthetic trends."""
    _patch_gene_program_clustering(monkeypatch)

    result = pla.plot_branch_gene_programs(
        gene_trend_adata,
        branch=None,
        n_dynamic_genes=6,
        gene_ranking_mode="union",
        trend_knn_neighbors=2,
        plot_representative_genes=with_representatives,
        representatives_per_cluster=2,
        save=str(tmp_path / "gene_programs"),
        plot_format="png",
    )

    assert set(result) == {"BranchA", "BranchB"}

    expected_keys = {
        "selected_genes",
        "trend_clusters",
        "centroid_z",
        "cluster_patterns",
        "cluster_summary",
        "cluster_representatives",
        "figures",
    }

    for branch in ("BranchA", "BranchB"):
        branch_result = result[branch]
        representatives = branch_result["cluster_representatives"]

        assert expected_keys <= branch_result.keys()
        assert len(branch_result["selected_genes"]) == 6
        assert (tmp_path / f"gene_programs_branch-{branch}_clusters.png").exists()
        assert (
            tmp_path / f"gene_programs_branch-{branch}_centroid_heatmap.png"
        ).exists()

        if not with_representatives:
            assert representatives == {}
            assert branch_result["figures"]["representatives"] is None
            continue

        assert set(representatives) == set(branch_result["centroid_z"].index)
        for genes in representatives.values():
            assert 1 <= len(genes) <= 2
            assert set(genes) <= set(branch_result["selected_genes"])

        assert branch_result["figures"]["representatives"] is not None
        assert (
            tmp_path / f"gene_programs_branch-{branch}_representatives.png"
        ).exists()


def test_plot_branch_gene_programs_writes_one_excel_workbook(
    monkeypatch,
    gene_trend_adata,
):
    """The Excel export should be written once per call, covering every branch."""
    written = []

    _patch_gene_program_clustering(monkeypatch)
    monkeypatch.setattr(
        pla.utils.tables,
        "write_excel",
        lambda table_dict, filename, **kwargs: written.append(sorted(table_dict)),
    )

    pla.plot_branch_gene_programs(
        gene_trend_adata,
        branch=None,
        n_dynamic_genes=4,
        gene_ranking_mode="union",
        trend_knn_neighbors=2,
        plot_representative_genes=False,
        save_excel="palantir_gene_programs.xlsx",
    )

    assert written == [["BranchA", "BranchB", "summary"]]


# ------------------------- plot_branch_early_late_genes -------------------------#


@pytest.mark.parametrize(
    "with_umap",
    [True, False],
    ids=["with-umap", "without-umap"],
)
def test_plot_branch_early_late_genes_smoke(monkeypatch, gene_trend_adata, with_umap):
    """Early/late gene plotting should return trend and UMAP result handles."""
    adata = gene_trend_adata.copy()

    if not with_umap:
        adata.obsm.pop("X_umap", None)

    def fake_plot_embedding(*args, **kwargs):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig

    monkeypatch.setattr(pla.pl, "plot_embedding", fake_plot_embedding)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    result = pla.plot_branch_early_late_genes(
        adata,
        branches=["BranchA"],
        top_n=2,
        quantile_cutoff=0.25,
        umap_ncols=2,
    )

    assert result["BranchA"]["trend_fig"] is not None
    assert result["BranchA"]["top_genes"]["early"][0] == "Gene3"
    assert result["BranchA"]["top_genes"]["late"][0] == "Gene0"

    if with_umap:
        assert result["BranchA"]["umap_fig"] is not None
    else:
        assert result["BranchA"]["umap_fig"] is None
        assert result["BranchA"]["umap_axes"].size == 0


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


# ------------------------- plot_branch_gene_trend_panels -------------------------#


@pytest.mark.parametrize(
    ("branches", "genes", "expected_branches", "expected_genes"),
    [
        (None, ["Gene0", "Gene1"], {"BranchA", "BranchB"}, ["Gene0", "Gene1"]),
        ("BranchA", ["Gene0", "Gene1"], {"BranchA"}, ["Gene0", "Gene1"]),
        (["BranchA", "BranchB"], ["Gene0"], {"BranchA", "BranchB"}, ["Gene0"]),
        ("BranchA", "Gene0", {"BranchA"}, ["Gene0"]),
    ],
    ids=["all-branches", "single-branch", "branch-list", "single-gene-string"],
)
def test_plot_branch_gene_trend_panels_branch_selection(
    monkeypatch,
    gene_trend_adata,
    branches,
    genes,
    expected_branches,
    expected_genes,
):
    """Branch and gene selection should accept None, a string, and a list."""
    plotted = _patch_plot_trend(monkeypatch)

    result = pla.plot_branch_gene_trend_panels(
        gene_trend_adata,
        computed=None,
        branches=branches,
        genes=genes,
        masks_key=None,
        show=False,
    )

    assert set(result) == expected_branches
    assert plotted == [
        (branch, gene) for branch in sorted(expected_branches) for gene in expected_genes
    ]

    for branch in expected_branches:
        assert result[branch]["genes"] == expected_genes
        assert result[branch]["gene_selection"] == "manual"
        plt.close(result[branch]["figure"])


@pytest.mark.parametrize(
    "branches",
    ["BranchA", ["BranchA", "BranchB"]],
    ids=["computed-branch", "missing-computed-branch"],
)
def test_plot_branch_gene_trend_panels_automatic_genes(
    monkeypatch,
    gene_trend_adata,
    branches,
):
    """Representative genes come from ``computed``; a missing branch fails early."""
    plotted = _patch_plot_trend(monkeypatch)

    computed = {
        "BranchA": {
            "cluster_representatives": {"0": ["Gene0"], "1": ["Gene1"]},
            "centroid_z": pd.DataFrame(index=["0", "1"]),
        }
    }
    kwargs = {
        "computed": computed,
        "branches": branches,
        "genes": None,
        "genes_per_cluster": 1,
        "masks_key": None,
        "show": False,
    }

    if branches != "BranchA":
        with pytest.raises(ValueError, match="BranchB"):
            pla.plot_branch_gene_trend_panels(gene_trend_adata, **kwargs)

        # Validation happens before any figure is created.
        assert plotted == []
        return

    result = pla.plot_branch_gene_trend_panels(gene_trend_adata, **kwargs)

    assert result["BranchA"]["genes"] == ["Gene0", "Gene1"]
    assert result["BranchA"]["gene_selection"] == "automatic"
    plt.close(result["BranchA"]["figure"])


@pytest.mark.parametrize(
    ("format_kwargs", "expected_extension"),
    [
        ({}, "pdf"),
        ({"plot_format": "svg"}, "svg"),
    ],
    ids=["default-pdf", "explicit-svg"],
)
def test_plot_branch_gene_trend_panels_report_figures_stay_png(
    monkeypatch,
    gene_trend_adata,
    format_kwargs,
    expected_extension,
):
    """Report figures should be png while other figures follow ``plot_format``."""
    _patch_plot_trend(monkeypatch)
    saved = _record_saves(monkeypatch)

    result = pla.plot_branch_gene_trend_panels(
        gene_trend_adata,
        branches="BranchA",
        genes=["Gene0", "Gene1"],
        masks_key=None,
        save="palantir_branch_gene_trends",
        report_save="08_branch_gene_trends",
        show=False,
        **format_kwargs,
    )

    assert saved == [
        (f"palantir_branch_gene_trends_branch-BranchA.{expected_extension}", False),
        ("08_branch_gene_trends_branch-BranchA.png", True),
    ]

    plt.close(result["BranchA"]["figure"])


def test_plot_branch_gene_trend_panels_sanitizes_branch_name_in_filenames(
    monkeypatch,
    gene_trend_adata,
):
    """Spaces and slashes in a branch name should not reach the figure filenames."""
    adata = gene_trend_adata.copy()
    adata.varm["gene_trends_T cell/CD4"] = adata.varm.pop("gene_trends_BranchA")

    plotted = _patch_plot_trend(monkeypatch)
    saved = _record_saves(monkeypatch)

    result = pla.plot_branch_gene_trend_panels(
        adata,
        branches="T cell/CD4",
        genes=["Gene0", "Gene1"],
        masks_key=None,
        save="palantir_branch_gene_trends",
        report_save="08_branch_gene_trends",
        show=False,
    )

    assert saved == [
        ("palantir_branch_gene_trends_branch-T_cell_CD4.pdf", False),
        ("08_branch_gene_trends_branch-T_cell_CD4.png", True),
    ]

    # Data access and return values keep the original branch name.
    assert list(result) == ["T cell/CD4"]
    assert {branch for branch, _ in plotted} == {"T cell/CD4"}

    plt.close(result["T cell/CD4"]["figure"])


# ------------------------- plot_fate_probability_projection -------------------------#


@pytest.mark.parametrize(
    ("n_fates", "row", "as_dataframe", "color_by"),
    [
        (2, None, True, "celltype"),
        (3, None, True, "celltype"),
        (5, None, True, "celltype"),
        (3, [0.2, 0.3, 0.5], False, None),
        (3, [0.20, 0.30, 0.49], True, None),
    ],
    ids=[
        "two-fates",
        "three-fates",
        "five-fates",
        "ndarray-with-uns-labels",
        "non-unit-rows",
    ],
)
def test_plot_fate_probability_projection_probability_storage(
    preprocessed_adata, n_fates, row, as_dataframe, color_by
):
    """Projections accept 2+ fates as a labelled DataFrame or ndarray plus uns labels."""
    adata = preprocessed_adata.copy()
    fate_names = [f"Fate{i}" for i in range(n_fates)]

    if row is None:
        rng = np.random.default_rng(42)
        values = rng.random((adata.n_obs, n_fates))
        values /= values.sum(axis=1, keepdims=True)
    else:
        values = np.tile([row], (adata.n_obs, 1))

    if as_dataframe:
        adata.obsm["palantir_fate_probabilities"] = pd.DataFrame(
            values,
            index=adata.obs_names,
            columns=fate_names,
        )
    else:
        adata.obsm["palantir_fate_probabilities"] = values
        adata.uns["palantir_fate_probabilities_columns"] = fate_names

    fig, ax = pla.plot_fate_probability_projection(
        adata,
        color_by=color_by,
        show=False,
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert set(fate_names) <= {text.get_text() for text in ax.texts}

    plt.close(fig)


@pytest.mark.parametrize("as_dataframe", [True, False], ids=["dataframe", "ndarray"])
def test_plot_fate_probability_projection_normalizes_rows_without_mutating_input(
    preprocessed_adata, as_dataframe
):
    """Rows that do not sum to 1 are normalized for the projection only."""
    adata = preprocessed_adata.copy()
    fate_names = ["FateA", "FateB", "FateC"]
    values = np.tile([0.4, 0.2, 0.2], (adata.n_obs, 1))

    if as_dataframe:
        adata.obsm["palantir_fate_probabilities"] = pd.DataFrame(
            values, index=adata.obs_names, columns=fate_names
        )
    else:
        adata.obsm["palantir_fate_probabilities"] = values
        adata.uns["palantir_fate_probabilities_columns"] = fate_names

    fig, ax = pla.plot_fate_probability_projection(adata, show=False)

    # Three fates sit on a regular triangle, the first at the top, then clockwise.
    angles = np.pi / 2 - 2 * np.pi * np.arange(3) / 3
    vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    expected = np.array([0.5, 0.25, 0.25]) @ vertices

    # The first collection holds the projected cells.
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(), np.tile(expected, (adata.n_obs, 1))
    )
    np.testing.assert_array_equal(
        np.asarray(adata.obsm["palantir_fate_probabilities"]), values
    )

    plt.close(fig)


def test_plot_fate_probability_projection_tolerates_rounding_excursions(preprocessed_adata):
    """Values a few ulp outside [0, 1] are accepted and snapped onto the simplex."""
    adata = preprocessed_adata.copy()
    values = np.tile([1.0 + 1e-12, -1e-12, 0.0], (adata.n_obs, 1))
    adata.obsm["palantir_fate_probabilities"] = pd.DataFrame(
        values, index=adata.obs_names, columns=["FateA", "FateB", "FateC"]
    )

    fig, ax = pla.plot_fate_probability_projection(adata, show=False)

    # All weight sits on the first fate, which is the top vertex of the triangle.
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(), np.tile([0.0, 1.0], (adata.n_obs, 1)), atol=1e-9
    )

    plt.close(fig)


@pytest.mark.parametrize(
    ("provide_ax", "color_by"),
    [(True, None), (True, "celltype"), (False, "celltype")],
    ids=["given-axes", "given-axes-categorical", "own-axes"],
)
def test_plot_fate_probability_projection_axes(
    fate_probability_adata, provide_ax, color_by
):
    """A provided Axes is drawn on and returned; otherwise a new pair is created."""
    provided_fig, provided_ax = plt.subplots() if provide_ax else (None, None)

    fig, ax = pla.plot_fate_probability_projection(
        fate_probability_adata,
        color_by=color_by,
        ax=provided_ax,
        show=False,
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_figure() is fig
    assert len(ax.collections) > 0

    if provide_ax:
        assert (fig, ax) == (provided_fig, provided_ax)

    plt.close(fig)


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        ([-0.1, 0.5, 0.6], None, "must lie between 0 and 1"),
        ([1.1, 0.0, 0.0], None, "must lie between 0 and 1"),
        ([np.nan, 0.5, 0.5], None, "must contain only finite values"),
        ([np.inf, 0.0, 0.0], None, "must contain only finite values"),
        ([0.0, 0.0, 0.0], None, "must have a positive sum"),
        ([1.0], None, "at least two are required"),
        ([0.2, 0.3, 0.5], ["FateA", "FateB"], "contains 2 terminal-state labels"),
    ],
    ids=[
        "below-zero",
        "above-one",
        "nan",
        "infinite",
        "zero-sum",
        "single-fate",
        "mismatched-ndarray-labels",
    ],
)
def test_plot_fate_probability_projection_invalid_probabilities_raise(
    preprocessed_adata,
    probabilities,
    labels,
    message,
):
    """Invalid fate probabilities or labels should raise a ValueError."""
    adata = preprocessed_adata.copy()
    values = np.tile([probabilities], (adata.n_obs, 1))

    if labels is None:
        adata.obsm["palantir_fate_probabilities"] = pd.DataFrame(
            values,
            index=adata.obs_names,
            columns=[f"Fate{i}" for i in range(len(probabilities))],
        )
    else:
        # An ndarray takes its terminal-state names from adata.uns.
        adata.obsm["palantir_fate_probabilities"] = values
        adata.uns["palantir_fate_probabilities_columns"] = labels

    with pytest.raises(ValueError, match=message):
        pla.plot_fate_probability_projection(adata, show=False)


# ------------------------- notebook helper functions -------------------------#


@pytest.mark.parametrize(
    "plotting_fails", [False, True], ids=["plot-succeeds", "plot-raises"]
)
def test_plot_palantir_results_removes_temporary_obs_columns(
    monkeypatch,
    fate_probability_adata,
    plotting_fails,
):
    """Fate-probability columns should exist only while the embedding is drawn."""
    captured = {}

    def fake_plot_embedding(adata, **kwargs):
        captured.update(kwargs)
        captured["columns_during_plot"] = list(adata.obs.columns)

        if plotting_fails:
            raise ValueError("simulated embedding failure")

        return np.array([plt.subplots()[1]])

    monkeypatch.setattr(pla.pl, "plot_embedding", fake_plot_embedding)

    obs_columns_before = list(fate_probability_adata.obs.columns)

    def call():
        return pla.plot_palantir_results(
            fate_probability_adata,
            save="palantir_results.pdf",
            report_save="06_palantir_results.png",
            show=False,
        )

    if plotting_fails:
        with pytest.raises(ValueError, match="simulated embedding failure"):
            call()
    else:
        assert isinstance(call(), np.ndarray)

    assert list(captured["color"])[:2] == ["palantir_pseudotime", "palantir_entropy"]
    assert set(list(captured["color"])[2:]) == {
        "fate_prob: FateA",
        "fate_prob: FateB",
        "fate_prob: FateC",
    }
    assert captured["save"] == "palantir_results.pdf"
    assert captured["report"] == "06_palantir_results.png"
    assert "fate_prob: FateA" in captured["columns_during_plot"]
    # The columns are a plotting detail and must not survive the call.
    assert list(fate_probability_adata.obs.columns) == obs_columns_before


def test_plot_palantir_results_keeps_existing_fate_column(
    monkeypatch, fate_probability_adata
):
    """A pre-existing `fate_prob: ...` column must be refused, not overwritten or dropped."""
    plotted = []
    monkeypatch.setattr(pla.pl, "plot_embedding", lambda *args, **kwargs: plotted.append(1))

    sentinel = np.arange(fate_probability_adata.n_obs, dtype=float)
    fate_probability_adata.obs["fate_prob: FateA"] = sentinel
    obs_before = fate_probability_adata.obs.copy()

    with pytest.raises(ValueError, match="already contains"):
        pla.plot_palantir_results(fate_probability_adata, show=False)

    assert not plotted
    pd.testing.assert_frame_equal(fate_probability_adata.obs, obs_before)


def test_plot_palantir_results_keeps_unrelated_columns_on_error(
    monkeypatch, fate_probability_adata
):
    """Unrelated obs columns keep their values when plotting fails."""

    def failing_plot_embedding(*args, **kwargs):
        raise ValueError("simulated embedding failure")

    monkeypatch.setattr(pla.pl, "plot_embedding", failing_plot_embedding)

    fate_probability_adata.obs["my_column"] = np.arange(fate_probability_adata.n_obs)
    obs_before = fate_probability_adata.obs.copy()

    with pytest.raises(ValueError, match="simulated embedding failure"):
        pla.plot_palantir_results(fate_probability_adata, show=False)

    pd.testing.assert_frame_equal(fate_probability_adata.obs, obs_before)


@pytest.mark.parametrize(
    ("probabilities", "match"),
    [
        (None, "not found in adata.obsm"),
        (np.zeros((1, 3)), "must be a DataFrame"),
    ],
    ids=["missing-probabilities", "unlabelled-probabilities"],
)
def test_plot_palantir_results_requires_probability_table(
    preprocessed_adata, probabilities, match
):
    """Missing or unlabelled fate probabilities should raise a ValueError."""
    if probabilities is not None:
        preprocessed_adata.obsm["palantir_fate_probabilities"] = np.tile(
            [[0.2, 0.3, 0.5]],
            (preprocessed_adata.n_obs, 1),
        )

    with pytest.raises(ValueError, match=match):
        pla.plot_palantir_results(preprocessed_adata, show=False)


def test_plot_branch_cell_selection_saves_both_destinations(monkeypatch, branch_adata):
    """Branch-selection plotting should forward both save destinations."""
    figure = plt.figure()

    monkeypatch.setattr(
        pla.palantir.plot,
        "plot_branch_selection",
        lambda adata, **kwargs: figure,
    )
    saved = _record_saves(monkeypatch)

    result = pla.plot_branch_cell_selection(
        branch_adata,
        save="palantir_branch_selection.pdf",
        report_save="03_branch_selection.png",
        show=False,
    )

    assert result is figure
    assert saved == [
        ("palantir_branch_selection.pdf", False),
        ("03_branch_selection.png", True),
    ]

    plt.close(figure)


def test_plot_branch_trajectories_builds_three_panels(monkeypatch, branch_adata):
    """The trajectory summary should draw pseudotime, entropy and branch panels."""
    plotted_colors = []

    def fake_plot_trajectories(adata, cell_color, **kwargs):
        plotted_colors.append(cell_color)

    monkeypatch.setattr(pla.palantir.plot, "plot_trajectories", fake_plot_trajectories)
    saved = _record_saves(monkeypatch)

    # Converting colors for h5ad is save_h5ad's job; the plot must leave them alone.
    fate_colors = {"BranchA": (1.0, 0.0, 0.0, 1.0)}
    branch_adata.uns["palantir_fates_colors"] = dict(fate_colors)

    fig, axes = pla.plot_branch_trajectories(
        branch_adata,
        save="palantir_branch_summary.pdf",
        report_save="03_branch_summary.png",
        show=False,
    )

    assert plotted_colors == ["palantir_pseudotime", "palantir_entropy", "branch_selection"]
    assert [ax.get_title() for ax in axes] == ["Pseudotime", "Entropy", "Branch Selection"]
    assert branch_adata.uns["palantir_fates_colors"] == fate_colors
    assert saved == [
        ("palantir_branch_summary.pdf", False),
        ("03_branch_summary.png", True),
    ]

    plt.close(fig)


@pytest.mark.parametrize(
    ("kwargs", "expected_key"),
    [
        ({}, "palantir_pseudotime"),
        ({"pseudotime_key": "custom_pseudotime"}, "custom_pseudotime"),
    ],
    ids=["default-key", "custom-key"],
)
def test_plot_branch_trajectories_forwards_pseudotime_key(
    monkeypatch, branch_adata, kwargs, expected_key
):
    """Every trajectory panel should follow the requested pseudotime key."""
    calls = []

    def fake_plot_trajectories(adata, pseudo_time_key, cell_color, **kwargs):
        calls.append((pseudo_time_key, cell_color))

    monkeypatch.setattr(pla.palantir.plot, "plot_trajectories", fake_plot_trajectories)

    fig, _ = pla.plot_branch_trajectories(branch_adata, show=False, **kwargs)

    assert calls == [
        (expected_key, expected_key),
        (expected_key, "palantir_entropy"),
        (expected_key, "branch_selection"),
    ]

    plt.close(fig)


@pytest.mark.parametrize(
    ("with_representatives", "expected"),
    [
        (
            True,
            [
                ("04_gene_programs_BranchA.png", True),
                ("04_gene_programs_BranchB.png", True),
            ],
        ),
        (False, []),
    ],
    ids=["with-representatives", "without-representatives"],
)
def test_plot_branch_gene_programs_report_save(
    monkeypatch, gene_trend_adata, with_representatives, expected
):
    """The representative figure of each branch is saved as a png for the report."""
    _patch_gene_program_clustering(monkeypatch)
    saved = _record_saves(monkeypatch)

    pla.plot_branch_gene_programs(
        gene_trend_adata,
        branch=None,
        n_dynamic_genes=6,
        gene_ranking_mode="union",
        trend_knn_neighbors=2,
        plot_representative_genes=with_representatives,
        representatives_per_cluster=2,
        report_save="04_gene_programs",
    )

    assert saved == expected


def test_plot_branch_gene_programs_sanitizes_branch_names_in_filenames(
    monkeypatch, gene_trend_adata, tmp_path
):
    """Spaces and slashes in branch names should give flat, sanitized filenames."""
    adata = gene_trend_adata.copy()
    adata.varm["gene_trends_T cell/CD4"] = adata.varm.pop("gene_trends_BranchA")
    adata.varm["gene_trends_B cell"] = adata.varm.pop("gene_trends_BranchB")

    plotted_branches = _patch_gene_program_clustering(monkeypatch)
    saved = _record_saves(monkeypatch)
    prefix = tmp_path / "gene_programs"

    result = pla.plot_branch_gene_programs(
        adata,
        branch=None,
        n_dynamic_genes=6,
        gene_ranking_mode="union",
        trend_knn_neighbors=2,
        plot_representative_genes=True,
        representatives_per_cluster=2,
        save=str(prefix),
        plot_format="png",
        report_save="04_gene_programs",
    )

    assert saved == [
        (f"{prefix}_branch-B_cell_clusters.png", False),
        (f"{prefix}_branch-B_cell_centroid_heatmap.png", False),
        (f"{prefix}_branch-B_cell_representatives.png", False),
        ("04_gene_programs_B_cell.png", True),
        (f"{prefix}_branch-T_cell_CD4_clusters.png", False),
        (f"{prefix}_branch-T_cell_CD4_centroid_heatmap.png", False),
        (f"{prefix}_branch-T_cell_CD4_representatives.png", False),
        ("04_gene_programs_T_cell_CD4.png", True),
    ]

    # The slash must not have created a subdirectory next to the figures.
    assert not [path for path in tmp_path.iterdir() if path.is_dir()]

    # Data access and return values keep the original branch names.
    assert set(result) == {"T cell/CD4", "B cell"}
    assert sorted(plotted_branches) == ["B cell", "T cell/CD4"]

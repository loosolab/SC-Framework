"""Helper functions for Palantir."""

import os
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse, spmatrix

# Visualization
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import plotly as po
import plotly.express as px
import seaborn as sns
from plotly.graph_objects import Figure as PlotlyFigure

# Single-cell analysis
import palantir
import scanpy as sc
from anndata import AnnData

# sctoolbox
import sctoolbox.utils as utils
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
from sctoolbox.plotting import embedding as pl
from sctoolbox.plotting.general import _save_figure
from sctoolbox.tools import marker_genes

# Type hints / decorators
from beartype import beartype
from beartype.typing import Any, Iterable, Literal, Optional, Union


logger = settings.logger
Severity = Literal[False, "warn", "error"]


@deco.log_anndata
@beartype
def is_preprocessed(  # noqa: C901
    adata: AnnData,
    normalize: Severity = "warn",
    log1p: Severity = "warn",
    pca: Severity = "error",
    embedding: Severity = "error",
    neighbors: Severity = "error",
    pca_key: str = "X_pca",
    umap_key: str = "X_umap",
    tsne_key: str = "X_tsne",
    neighbors_key: str = "neighbors",
    sample_size: int = 50000,
    random_state: int = 0,
) -> bool:
    """Validate whether an AnnData object fulfills preprocessing requirements.

    This function performs configurable preflight checks before running diffusion maps
    and/or MAGIC. The normalization and log1p checks are heuristic and inspect `adata.X`.

    Parameters
    ----------
    adata : AnnData
        AnnData object to inspect.
    normalize : Literal[False, "warn", "error"], default="warn"
        Heuristic check whether `adata.X` looks like raw integer counts.
    log1p : Literal[False, "warn", "error"], default="warn"
        Heuristic check whether `adata.X` looks log-transformed based on value range.
    pca : Literal[False, "warn", "error"], default="error"
        Require PCA coordinates in `adata.obsm[pca_key]`.
    embedding : Literal[False, "warn", "error"], default="error"
        Require either UMAP or t-SNE embedding in `adata.obsm`.
    neighbors : Literal[False, "warn", "error"], default="error"
        Require a neighbor graph in `adata.uns[neighbors_key]`.
    pca_key : str, default="X_pca"
        Key for PCA coordinates in `adata.obsm`.
    umap_key : str, default="X_umap"
        Key for UMAP coordinates in `adata.obsm`.
    tsne_key : str, default="X_tsne"
        Key for t-SNE coordinates in `adata.obsm`.
    neighbors_key : str, default="neighbors"
        Key for the neighbor graph in `adata.uns`.
    sample_size : int, default=50000
        Number of values sampled from `adata.X` for heuristic checks (performance guard).
    random_state : int, default=0
        Random seed for sampling from `adata.X`.

    Returns
    -------
    bool
        True if no checks configured as "error" failed, otherwise False.
    """
    mandatory_failed = False

    def emit(severity: Severity, message: str) -> bool:
        """Log a message depending on severity.

        Parameters
        ----------
        severity : Literal[False, "warn", "error"]
            Logging severity.
        message : str
            Message to emit.

        Returns
        -------
        bool
            True if `severity == "error"`, otherwise False.
        """
        if severity is False:
            return False
        if severity == "error":
            logger.error(message)
            return True
        logger.warning(message)
        return False

    def _X_values_sample(X: spmatrix | np.ndarray, sample_size: int, random_state: int) -> np.ndarray:
        """Return a 1D sample of values from an expression matrix.

        For sparse matrices, only stored values (`.data`) are sampled.

        Parameters
        ----------
        X : scipy.sparse.spmatrix | numpy.ndarray
            Expression matrix to sample from (typically `adata.X`).
        sample_size : int
            Maximum number of values to sample.
        random_state : int
            Random seed for sampling.

        Returns
        -------
        numpy.ndarray
            1D array of sampled values (may be empty).
        """
        if issparse(X):
            data = X.data
        else:
            data = np.asarray(X).ravel()

        n = int(sample_size)
        if data.size <= n:
            return data

        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(data.size, size=n, replace=False)
        return data[idx]

    def _looks_like_raw_integer_counts() -> bool:
        """Heuristic check whether `adata.X` looks like raw integer counts.

        This is a sample-based, fast check that tests if values are integer-like.
        Returns False if the sampled array is empty.

        Returns
        -------
        bool
            True if sampled values are integer-like, otherwise False.
        """
        x = _X_values_sample(adata.X, sample_size=sample_size, random_state=random_state)
        if x.size == 0:
            return False
        return bool(np.all(np.equal(x, np.round(x))))

    def _max_in_X() -> float:
        """Return the max of sampled values from `adata.X`.

        Returns 0.0 if the sampled array is empty.

        Returns
        -------
        float
            Maximum sampled value (0.0 if empty).
        """
        x = _X_values_sample(adata.X, sample_size=sample_size, random_state=random_state)
        if x.size == 0:
            return 0.0
        return float(np.nanmax(x))

    if normalize and _looks_like_raw_integer_counts():
        mandatory_failed = mandatory_failed or emit(
            normalize,
            "Normalization check: adata.X looks like raw integer counts (not normalized).",
        )

    if log1p:
        max_val = _max_in_X()
        if max_val >= 50:
            mandatory_failed = mandatory_failed or emit(
                log1p,
                f"Log1p check: max sampled value in adata.X is {max_val:.2f}; may not be log-transformed.",
            )

    if pca and (pca_key not in adata.obsm):
        mandatory_failed = mandatory_failed or emit(
            pca,
            f"PCA missing: adata.obsm['{pca_key}'] not found.",
        )

    has_umap = umap_key in adata.obsm
    has_tsne = tsne_key in adata.obsm
    if embedding and (not has_umap) and (not has_tsne):
        mandatory_failed = mandatory_failed or emit(
            embedding,
            f"Embedding missing: neither adata.obsm['{umap_key}'] nor adata.obsm['{tsne_key}'] found.",
        )

    if neighbors and (neighbors_key not in adata.uns):
        mandatory_failed = mandatory_failed or emit(
            neighbors,
            f"Neighbors missing: adata.uns['{neighbors_key}'] not found.",
        )

    return not mandatory_failed


@deco.log_anndata
@beartype
def run_diffusion_and_magic(
    adata: AnnData,
    save_path: str | None = None,
    n_components: int = 10,
    knn: int = 30,
    kernel_backend: str = "scanpy",
    n_jobs: int = 10,
    n_steps: int = 3,
    imputation_key: str = "MAGIC_imputed_data",
    recompute_magic: bool = False,
) -> AnnData:
    """Run diffusion maps, derive the multiscale space, and perform MAGIC imputation.

    This function computes Palantir diffusion maps, derives the multiscale diffusion
    space, and runs MAGIC imputation on an ``AnnData`` object. Diffusion-map results
    are stored in ``adata.obsm``, ``adata.obsp``, and ``adata.uns``; the multiscale
    space is stored in ``adata.obsm["DM_EigenVectors_multiscaled"]``; and MAGIC-imputed
    expression is stored in ``adata.layers[imputation_key]``.

    If ``save_path`` exists and ``recompute_magic=False``, the cached ``AnnData`` is
    loaded and returned. If the cached object is missing ``layers[imputation_key]``,
    diffusion maps and MAGIC are recomputed on the cached object.

    Parameters
    ----------
    adata : AnnData
        AnnData object used for diffusion maps and MAGIC imputation.
    save_path : str | None, default=None
        Path used for caching the processed AnnData object. If ``None``, the path is
        set to
        ``{settings.adata_output_dir}/{adata.obs_names.name or "adata"}_magic.h5ad``.
    n_components : int, default=10
        Number of diffusion-map components.
    knn : int, default=30
        Number of nearest neighbors used for kernel construction.
    kernel_backend : str, default="scanpy"
        Backend used by Palantir for kNN/kernel construction. Use ``"scanpy"`` for
        the Scanpy-based backend or ``"sklearn"`` for exact kNN.
    n_jobs : int, default=10
        Number of parallel jobs used for MAGIC imputation.
    n_steps : int, default=3
        Number of diffusion steps used for MAGIC imputation.
    imputation_key : str, default="MAGIC_imputed_data"
        Key used to store MAGIC-imputed expression in ``adata.layers``.
    recompute_magic : bool, default=False
        If ``True``, ignore any existing cache file and recompute diffusion maps,
        multiscale space, and MAGIC imputation.

    Returns
    -------
    AnnData
        AnnData object containing Palantir diffusion results, the multiscale diffusion
        space in ``adata.obsm["DM_EigenVectors_multiscaled"]``, and MAGIC-imputed
        expression in ``adata.layers[imputation_key]``.
    """
    # ------------------------------------------------------------------
    # (1) Resolve output path (cache file)
    # ------------------------------------------------------------------
    if save_path is None:
        filename = f"{adata.obs_names.name or 'adata'}_magic.h5ad"
        save_path = os.path.join(settings.adata_output_dir, filename)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ------------------------------------------------------------------
    # (2) Load cache if allowed
    # ------------------------------------------------------------------
    if os.path.exists(save_path) and not recompute_magic:
        logger.info("File '%s' exists. Loading cached AnnData.", save_path)
        cached = sc.read_h5ad(save_path)

        # Minimal validity check: if MAGIC layer exists, we treat cache as complete.
        if imputation_key in cached.layers:
            logger.info("Found MAGIC layer '%s' in cache. Returning cached AnnData.", imputation_key)
            return cached

        logger.warning("Cached AnnData is missing MAGIC layer '%s'. Recomputing.", imputation_key)
        adata = cached  # keep any cached preprocessing state for recomputation

    # ------------------------------------------------------------------
    # (3) Compute diffusion maps
    # ------------------------------------------------------------------
    logger.info("Running diffusion maps with n_components=%d, knn=%d, kernel_backend='%s'.",
        n_components,
        knn,
        kernel_backend,
    )
    dm_res = palantir.utils.run_diffusion_maps(
        adata,
        n_components=n_components,
        knn=knn,
        kernel_backend=kernel_backend,
        pca_key='X_pca',
        kernel_key='DM_Kernel',
        sim_key='DM_Similarity',
        eigval_key='DM_EigenValues',
        eigvec_key='DM_EigenVectors',
    )
    # ------------------------------------------------------------------
    # (4) Compute multiscale diffusion space and store it
    # ------------------------------------------------------------------
    logger.info("Computing multiscale diffusion space...")
    ms_data = palantir.utils.determine_multiscale_space(dm_res)
    adata.obsm["DM_EigenVectors_multiscaled"] = ms_data.values

    # ------------------------------------------------------------------
    # (5) MAGIC imputation
    # ------------------------------------------------------------------
    logger.info("Running MAGIC imputation...")
    palantir.utils.run_magic_imputation(
        data=adata,
        dm_res=dm_res,
        n_jobs=n_jobs,
        n_steps=n_steps,
        sim_key="DM_Similarity",
        imputation_key=imputation_key,
    )

    # ------------------------------------------------------------------
    # (6) Save cache and return
    # ------------------------------------------------------------------
    utils.adata.save_h5ad(adata, save_path)
    logger.info("Processed AnnData saved to: %s", save_path)

    return adata


@deco.log_anndata
@beartype
def compare_all_layers(adata: AnnData, exclude_layers: list[str] | None = None) -> tuple[plt.Figure, np.ndarray]:
    """
    Compare total counts per cell for `adata.X` and all layers in `adata.layers`, working with both dense and sparse matrices.

    Parameters
    ----------
    adata : AnnData
        The annotated data object with multiple layers.
    exclude_layers : list[str] | None, optional
        Layers to skip from comparison.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure containing all histograms.
    axs : np.ndarray
        Array of axes objects, one for each histogram.
    """
    if exclude_layers is None:
        exclude_layers = []

    # Helper function to sum values per cell
    def row_sums(matrix: np.ndarray) -> np.ndarray:
        if issparse(matrix):
            return matrix.sum(axis=1).A1
        return np.sum(matrix, axis=1)

    # Dictionary to store sums
    sums_dict = {"adata.X": row_sums(adata.X)}

    for layer_name, layer_matrix in adata.layers.items():
        if layer_name is None:
            continue
        if layer_name not in exclude_layers:
            sums_dict[f"layers['{layer_name}']"] = row_sums(layer_matrix)

    # Create subplots
    n_layers = len(sums_dict)
    fig, axs = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4), sharey=True)
    axs = np.atleast_1d(axs)

    # Plot histograms
    for ax, (name, sums) in zip(axs, sums_dict.items()):
        sns.histplot(sums, bins=50, kde=True, ax=ax)
        ax.set_title(f"{name} – Total per cell")
        ax.set_xlabel("Sum of values")
        ax.set_ylabel("Number of cells")

    plt.tight_layout()
    return fig, axs


@deco.log_anndata
@beartype
def remove_cells_by_label(
    adata: AnnData,
    column_name: str | None = None,
    labels_to_remove: str | list[str] | None = None,
) -> AnnData:
    """
    Remove cells from an AnnData object that match specific labels in a .obs column.

    Returns a new AnnData containing only the remaining cells. If no labels are
    provided, the original AnnData is returned unchanged.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing cells to filter.
    column_name : str, optional
        The column in `adata.obs` where the cell labels are stored.
        Required if `labels_to_remove` is given.
    labels_to_remove : str or list of str, optional
        The label(s) to remove. Can be a single string or a list of strings.
        If None or empty, no cells are removed.

    Returns
    -------
    AnnData
        Filtered AnnData object (or original if nothing removed).

    Raises
    ------
    ValueError
        - If `column_name` is not provided while `labels_to_remove` is given,
        - if `column_name` does not exist in `adata.obs`,
        - or if none of the requested labels are present in the column.
    """
    if not labels_to_remove:
        logger.info("No labels provided for removal — returning original AnnData.")
        return adata

    if column_name is None:
        raise ValueError("You must specify 'column_name' when providing labels_to_remove.")
    if column_name not in adata.obs.columns:
        raise ValueError(f"Column '{column_name}' does not exist in adata.obs.")

    # Normalize to a list once
    labels = [labels_to_remove] if isinstance(labels_to_remove, str) else list(labels_to_remove)

    col = adata.obs[column_name]

    # Use categories if possible (faster and stable)
    if hasattr(col, "cat"):
        present = set(map(str, col.cat.categories))
        col_str = col.astype(str)
    else:
        # Keep behavior consistent with string labels
        col_str = col.astype(str)
        present = set(col_str.unique())

    requested = set(map(str, labels))
    found = requested & present
    missing = sorted(requested - present)

    if not found:
        raise ValueError(
            f"None of the requested labels are present in adata.obs['{column_name}'] "
            f"(requested={sorted(requested)[:20]}{'...' if len(requested) > 20 else ''})."
        )
    if missing:
        logger.warning("Ignoring missing labels in '%s': %s", column_name, missing)

    mask = ~col_str.isin(found)
    filtered_adata = adata[mask].copy()

    removed_count = adata.shape[0] - filtered_adata.shape[0]
    logger.info("Removed %d cells with labels: %s.", removed_count, sorted(found))
    logger.info("%d cells remain after filtering.", filtered_adata.shape[0])

    return filtered_adata


@deco.log_anndata
@beartype
def select_early_cell(
    adata: AnnData,
    celltype: str,
    celltype_column: str,
    early_cell: str | None = None,
    fallback_seed: int | None = 42,
    eigvec_key: str = "DM_EigenVectors_multiscaled",
) -> str:
    """Select or validate the early cell used for Palantir.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    celltype : str
        Cell type used for automatic early-cell selection.
    celltype_column : str
        Column in ``adata.obs`` containing the cell-type labels.
    early_cell : str | None, default=None
        Manually selected early-cell ID. If ``None``, Palantir selects
        the cell automatically.
    fallback_seed : int | None, default=42
        Seed used by Palantir for fallback early-cell selection. ``None`` disables
        the fallback; Palantir then raises ``CellNotFoundException`` if no cell of
        ``celltype`` lies at an extreme of the diffusion components.
    eigvec_key : str, default="DM_EigenVectors_multiscaled"
        Key in ``adata.obsm`` holding the diffusion components used for automatic
        early-cell selection.

    Returns
    -------
    str
        Selected early-cell ID.

    Raises
    ------
    ValueError
        If a manually provided early-cell ID is not found in ``adata.obs_names``.
    """
    if early_cell is None:
        return palantir.utils.early_cell(
            adata,
            celltype=celltype,
            celltype_column=celltype_column,
            eigvec_key=eigvec_key,
            fallback_seed=fallback_seed,
        )

    if early_cell not in adata.obs_names:
        raise ValueError(
            f"Provided early_cell '{early_cell}' not found in adata.obs_names."
        )

    return early_cell


@deco.log_anndata
@beartype
def plot_early_cell(
    adata: AnnData,
    early_cell: str,
    embedding_key: str = "X_umap",
    save: str | None = None,
    report_save: str | None = None,
    show: bool = True,
) -> tuple[Figure, Any]:
    """Highlight the selected Palantir early cell on an embedding.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    early_cell : str
        Cell ID to highlight.
    embedding_key : str, default="X_umap"
        Key in ``adata.obsm`` containing the embedding. Embeddings with more than
        two dimensions use the first two dimensions.
    save : str | None, default=None
        Path and filename appended to ``settings.full_figure_prefix``.
    report_save : str | None, default=None
        Path and filename appended to ``settings.report_dir``.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    tuple[Figure, Any]
        Matplotlib figure and axis containing the highlighted early cell.

    Raises
    ------
    ValueError
        If ``early_cell`` or ``embedding_key`` is not present in ``adata``.
    """
    if early_cell not in adata.obs_names:
        raise ValueError(f"Early cell '{early_cell}' not found in adata.obs_names.")

    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm.")

    # Palantir's highlight plot only knows x and y; pass higher-dimensional
    # embeddings as a two-dimensional table instead of modifying adata.
    plot_data = adata
    embedding = np.asarray(adata.obsm[embedding_key])

    if embedding.shape[1] > 2:
        logger.warning(
            "Embedding '%s' contains %d dimensions. "
            "Using the first two dimensions for the early-cell plot.",
            embedding_key,
            embedding.shape[1],
        )
        plot_data = pd.DataFrame(embedding[:, :2], index=adata.obs_names, columns=["x", "y"])

    fig, ax = palantir.plot.highlight_cells_on_umap(
        plot_data,
        [early_cell],
        embedding_basis=embedding_key,
        figsize=(6, 6),
        annotation_offset=0.03,
        s=1,
        s_highlighted=20,
    )

    ax.set_title(f"Early cell identification\nCell ID: {early_cell}")
    fig.tight_layout()

    if save is not None:
        _save_figure(path=save)

    if report_save is not None:
        _save_figure(path=report_save, report=True)

    if show:
        plt.show()

    return fig, ax


@deco.log_anndata
@beartype
def plot_interactive_embedding_cell_ids(
    adata: AnnData,
    color_by: str | None = None,
    embedding_key: str = "X_umap",
    title: str = "Interactive embedding (hover to see cell ID)",
    renderer: str | None = None,
    show: bool = True,
) -> PlotlyFigure:
    """Plot an interactive 2D or 3D embedding and show cell IDs on hover.

    This optional helper supports manual terminal-state selection by showing
    cell IDs when hovering over points in the embedding.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the embedding in ``adata.obsm`` and optional
        metadata in ``adata.obs``.
    color_by : str | None, default=None
        Column in ``adata.obs`` used to color the points. If ``None``, all
        cells are shown without grouping.
    embedding_key : str, default="X_umap"
        Key in ``adata.obsm`` containing an embedding with at least two dimensions.
        Two-dimensional embeddings are shown as scatter plots. Embeddings with
        three or more dimensions use the first three dimensions.
    title : str, default="Interactive embedding (hover to see cell ID)"
        Title shown above the plot.
    renderer : str | None, default=None
        Plotly renderer used only when ``show=True``. If ``None``, Plotly uses the
        currently configured default renderer.
    show : bool, default=True
        Whether to display the figure immediately.

    Returns
    -------
    PlotlyFigure
        Plotly figure containing the interactive embedding plot.

    Raises
    ------
    ValueError
        If ``embedding_key`` is missing in ``adata.obsm``, if the embedding is
        not a 2D array of cells by coordinates, if it contains fewer than two
        coordinate dimensions, or if ``color_by`` is not found in ``adata.obs``.
    """
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm.")

    embedding = np.asarray(adata.obsm[embedding_key])

    if embedding.ndim != 2 or embedding.shape[1] < 2:
        raise ValueError(
            f"Embedding '{embedding_key}' must be a 2D array with at least two "
            "coordinate dimensions."
        )

    if embedding.shape[1] > 3:
        logger.warning(
            "Embedding '%s' contains %d dimensions. "
            "Using the first three dimensions for the interactive plot.",
            embedding_key,
            embedding.shape[1],
        )

    used_dimensions = min(embedding.shape[1], 3)

    df = pd.DataFrame(
        embedding[:, :used_dimensions],
        columns=[f"Dim{i}" for i in range(1, used_dimensions + 1)],
        index=adata.obs_names,
    )

    if color_by is not None:
        if color_by not in adata.obs.columns:
            raise ValueError(f"Column '{color_by}' not found in adata.obs.")
        df[color_by] = adata.obs[color_by].astype(str)

    if used_dimensions == 2:
        fig = px.scatter(
            df,
            x="Dim1",
            y="Dim2",
            color=color_by,
            hover_name=df.index,
            title=title,
        )
    else:
        fig = px.scatter_3d(
            df,
            x="Dim1",
            y="Dim2",
            z="Dim3",
            color=color_by,
            hover_name=df.index,
            title=title,
        )

    fig.update_layout(width=800, height=700)

    if show:
        po.offline.init_notebook_mode(connected=True)  # prints a dict when not run in notebook
        fig.show(renderer=renderer)

    return fig


@deco.log_anndata
@beartype
def detect_and_plot_terminal_states(  # noqa: C901
    adata: AnnData,
    early_cell: str,
    terminal_states: list[str] | tuple[str, ...] | str | None = None,
    clusters_to_analyze: str | None = None,
    embedding_key: str = "X_umap",
    eigvec_key: str = "DM_EigenVectors_multiscaled",
    knn: int = 30,
    num_waypoints: int = 500,
    n_jobs: int = -1,
    max_iterations: int = 25,
    seed: int = 42,
    title: str | None = None,
    show_excluded_boundaries: bool = True,
    save: str | None = None,
    report_save: str | None = None,
    plot: bool = True,
) -> pd.Series:
    """Determine terminal states, plot them, and return the Series for ``run_palantir``.

    Terminal states can either be provided manually or detected automatically with
    ``palantir.core.identify_terminal_states``.

    The returned object matches the format that the current Palantir code path
    expects in ``palantir.core.run_palantir(...)``:

    - index: cell IDs
    - values: terminal-state labels

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    early_cell : str
        Start cell ID used by Palantir.
    terminal_states : list[str] | tuple[str, ...] | str | None, default=None
        Terminal cell IDs. If ``None`` or an empty string, terminal states are
        detected automatically.
    clusters_to_analyze : str | None, default=None
        Column in ``adata.obs`` used to assign labels to terminal states. If
        ``None``, generic labels are used.
    embedding_key : str, default="X_umap"
        Key in ``adata.obsm`` containing the embedding used for plotting.
        Embeddings with more than two dimensions use the first two dimensions.
    eigvec_key : str, default="DM_EigenVectors_multiscaled"
        Key in ``adata.obsm`` containing the diffusion representation used for
        automatic terminal-state detection.
    knn : int, default=30
        Number of nearest neighbors for Palantir.
    num_waypoints : int, default=500
        Number of waypoints for Palantir.
    n_jobs : int, default=-1
        Number of parallel jobs.
    max_iterations : int, default=25
        Maximum number of pseudotime refinement iterations.
    seed : int, default=42
        Random seed.
    title : str | None, default=None
        Plot title. If ``None``, a default title is used.
    show_excluded_boundaries : bool, default=True
        Whether to show excluded boundary cells from automatic detection as QC markers.
    save : str | None, default=None
        Path and filename appended to ``settings.full_figure_prefix``.
    report_save : str | None, default=None
        Path and filename appended to ``settings.report_dir``.
    plot : bool, default=True
        Whether to create and display the terminal-state plot.

    Returns
    -------
    pd.Series
        Terminal states for direct use in ``palantir.core.run_palantir(...)``,
        with cell IDs as index and labels as values.

    Raises
    ------
    ValueError
        If required inputs are missing or invalid, or if no valid terminal cells
        are found.
    """
    # Normalize string input so that all downstream logic only has to handle:
    # - None -> automatic detection
    # - list of cell IDs -> manual terminal states
    if isinstance(terminal_states, str):
        terminal_states = terminal_states.strip()
        terminal_states = None if terminal_states == "" else [terminal_states]

    # Validate all required inputs early, before running Palantir.
    # This makes errors fail fast and keeps later code simpler.
    if eigvec_key not in adata.obsm:
        raise ValueError(f"Diffusion embedding '{eigvec_key}' not found in adata.obsm.")
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm.")
    if early_cell not in adata.obs_names:
        raise ValueError(f"Early cell '{early_cell}' not found in adata.obs_names.")
    if clusters_to_analyze is not None and clusters_to_analyze not in adata.obs.columns:
        raise ValueError(f"Cluster column '{clusters_to_analyze}' not found in adata.obs.")

    # The embedding is only used for plotting.
    # At least two dimensions are required; higher-dimensional embeddings
    # are reduced to the first two dimensions below.
    embedding = np.asarray(adata.obsm[embedding_key])

    if embedding.ndim != 2 or embedding.shape[1] < 2:
        raise ValueError(
            f"Embedding '{embedding_key}' must be a 2D array with at least two "
            "coordinate dimensions."
        )

    if embedding.shape[1] > 2:
        logger.warning(
            "Embedding '%s' contains %d dimensions. "
            "Using the first two dimensions for the terminal-state plot.",
            embedding_key,
            embedding.shape[1],
        )

    # Store excluded diffusion boundaries from automatic detection.
    # These are not used as final terminal states, but can be shown for QC.
    excluded_boundaries = pd.Index([])

    # ------------------------------------------------------------------
    # Step 1: Determine terminal-state candidates
    # ------------------------------------------------------------------
    if terminal_states is None:
        logger.info("No terminal cells provided — running Palantir automatic detection.")

        # Convert the diffusion representation into a DataFrame with cell IDs as index.
        # Palantir uses these cell IDs internally to identify terminal states.
        diffusion_df = pd.DataFrame(adata.obsm[eigvec_key], index=adata.obs_names)

        # Automatic mode:
        # Palantir returns:
        # - detected terminal cells
        # - excluded boundary cells that were considered but not selected
        detected_terminal_states, excluded_boundaries = palantir.core.identify_terminal_states(
            diffusion_df,
            early_cell=early_cell,
            knn=knn,
            num_waypoints=num_waypoints,
            n_jobs=n_jobs,
            max_iterations=max_iterations,
            seed=seed,
        )

        candidate_cells = detected_terminal_states
        logger.info("Excluded diffusion-map boundaries: %s", list(excluded_boundaries))
    else:
        # Manual mode:
        # Use the provided cell IDs directly as terminal-state candidates.
        logger.info("Using manually provided terminal cells.")
        candidate_cells = terminal_states

    # Keep only valid cell IDs that actually exist in the AnnData object.
    # This protects against typos or outdated cell IDs.
    terminal_cells = [str(cell_id) for cell_id in candidate_cells if str(cell_id) in adata.obs_names]
    if not terminal_cells:
        raise ValueError("No valid terminal cells found in adata.obs_names.")

    logger.info("Valid terminal cells detected: %s", terminal_cells)

    # ------------------------------------------------------------------
    # Step 2: Build readable labels for the terminal states
    # ------------------------------------------------------------------
    if clusters_to_analyze is None:
        # If no annotation column is given, create placeholder labels first.
        # These are replaced below with generic labels such as terminal_state_1.
        raw_labels = pd.Series(
            [None] * len(terminal_cells),
            index=terminal_cells,
            dtype=object,
        )
    else:
        # Extract cell-group labels from adata.obs for the detected terminal cells.
        raw_labels = adata.obs.loc[terminal_cells, clusters_to_analyze]

    labels: list[str] = []
    used_labels: set[str] = set()

    # Create one unique label per terminal cell.
    # If cluster labels repeat, add suffixes such as _2, _3, ...
    for i, cell_id in enumerate(terminal_cells, start=1):
        raw_label = raw_labels.loc[cell_id]

        # Fall back to a generic label if no cluster annotation is available.
        if pd.isna(raw_label) or str(raw_label).strip() == "":
            base_label = f"terminal_state_{i}"
        else:
            base_label = str(raw_label).strip()

        # Make sure labels are unique, because Palantir fate probabilities
        # will later use these names as terminal-state labels.
        label = base_label
        counter = 2
        while label in used_labels:
            label = f"{base_label}_{counter}"
            counter += 1

        used_labels.add(label)
        labels.append(label)

    # Build the exact object that is passed into run_palantir(...).
    # IMPORTANT:
    # - index = cell IDs
    # - values = terminal-state labels
    terminal_state_series = pd.Series(
        data=labels,
        index=terminal_cells,
        dtype=object,
    )

    logger.info("Terminal states passed to run_palantir (index=cell_id, value=label):\n%s", terminal_state_series)
    # ------------------------------------------------------------------
    # Step 3: Plot terminal cells on the chosen 2D embedding
    # ------------------------------------------------------------------
    if plot:
        # Palantir's highlight plot only knows x and y; pass the first two
        # dimensions as a table instead of copying adata. The same table
        # serves the QC overlay below.
        coords = pd.DataFrame(embedding[:, :2], index=adata.obs_names, columns=["x", "y"])
        palantir.plot.highlight_cells_on_umap(coords, terminal_cells)

        # Optional QC overlay:
        # Show excluded boundaries as "x" markers.
        # These cells were considered during automatic detection, but were not chosen
        # as final terminal states by Palantir.
        if show_excluded_boundaries and len(excluded_boundaries) > 0:
            excluded_valid = [cell_id for cell_id in excluded_boundaries if cell_id in coords.index]
            if excluded_valid:
                plt.scatter(
                    coords.loc[excluded_valid, "x"],
                    coords.loc[excluded_valid, "y"],
                    s=20,
                    marker="x",
                    alpha=0.8,
                )

        # Annotate each selected terminal cell with:
        # - the assigned terminal-state label
        # - the corresponding cell ID
        for cell_id, label in terminal_state_series.items():
            x, y = coords.loc[cell_id, ["x", "y"]]
            plt.text(x, y, f"{label}\n{cell_id}", fontsize=9)

        plt.title("Terminal state identification" if title is None else title)
        plt.tight_layout()

        if save is not None:
            _save_figure(path=save)

        if report_save is not None:
            _save_figure(path=report_save, report=True)

        plt.show()

    return terminal_state_series


@beartype
def _rank_highly_variable_genes(adata: AnnData) -> tuple[list[str], str]:
    """Order the highly variable genes of ``adata`` from most to least variable.

    The first usable ranking column in ``adata.var`` is used:
    ``highly_variable_rank`` (smaller is more variable, e.g. ``flavor="seurat_v3"``),
    then ``dispersions_norm`` (larger is more variable, e.g. ``flavor="seurat"``).
    Genes without a finite score come last, and ties are broken by gene name, so the
    order does not depend on the gene order in ``adata``. Without a usable ranking
    column, the genes keep their ``adata.var`` order and a warning is logged.

    Parameters
    ----------
    adata : AnnData
        AnnData with a boolean ``adata.var["highly_variable"]`` column.

    Returns
    -------
    tuple[list[str], str]
        Ordered highly variable genes and the name of the ranking used.
    """
    hvg = adata.var.loc[adata.var["highly_variable"].eq(True)]

    for column, ascending in (("highly_variable_rank", True), ("dispersions_norm", False)):
        if column not in hvg.columns:
            continue
        score = pd.to_numeric(hvg[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if score.isna().all():
            continue

        n_missing = int(score.isna().sum())
        if n_missing:
            logger.warning("%d highly variable genes have no '%s' value and are ranked last.", n_missing, column)

        ranking = pd.DataFrame({"score": score.to_numpy(), "gene": hvg.index.astype(str)})
        ranking = ranking.sort_values(
            ["score", "gene"], ascending=[ascending, True], na_position="last", kind="stable"
        )
        return ranking["gene"].tolist(), column

    if len(hvg):
        logger.warning(
            "adata.var has no usable 'highly_variable_rank' or 'dispersions_norm' column; "
            "highly variable genes are used in their adata.var order."
        )
    return hvg.index.astype(str).tolist(), "adata.var order"


@deco.log_anndata
@beartype
def analyze_top_genes_communities(  # noqa: C901
    adata: AnnData,
    trend_branches: str | list[str],
    gene_source: Literal["markers", "hvg"] = "markers",
    marker_table: str | dict[str, pd.DataFrame] | None = None,
    gene_trend_base_key: str = "gene_trends",
    top_n: int = 300,
    n_neighbors: int = 30,
    save_excel: str | None = None,
    save_plots: str | None = None,
    store_key: str | None = None,
    excel_prefix: str | Path | None = None,
) -> tuple[list[str], dict[str, pd.Series | pd.DataFrame]]:
    """Cluster Palantir gene-trend profiles into communities for selected branches.

    The function selects a gene set (HVGs or marker genes) and clusters their
    branch-specific Palantir gene trends using
    ``palantir.presults.cluster_gene_trends``. It can optionally save Palantir
    cluster plots, export Excel tables, and store a compact result object in
    ``adata.uns["sctoolbox"]``.

    Parameters
    ----------
    adata : AnnData
        AnnData containing Palantir results and branch-wise gene trends in
        ``adata.varm``.
    trend_branches : str | list[str]
        Branch name(s) to cluster.
    gene_source : Literal["markers", "hvg"], default="markers"
        Which gene set to use for clustering. With ``"hvg"``, the ``top_n`` most
        variable genes flagged in ``adata.var["highly_variable"]`` are used, ranked by
        ``adata.var["highly_variable_rank"]`` (smaller first) or, if absent,
        ``adata.var["dispersions_norm"]`` (larger first). Ties are broken by gene name
        and genes without a score come last. Without either column the genes keep
        their ``adata.var`` order and a warning is logged.
    marker_table : str | dict[str, pd.DataFrame] | None, default=None
        Marker definitions used when ``gene_source="markers"``.

        - If ``str``, interpreted as a key in ``adata.uns`` and resolved via ``marker_genes.get_rank_genes_tables``.
        - If ``dict``, expected format is ``group -> DataFrame``.

        Each table needs a ``names`` (or ``gene``) and a ``scores`` column. If
        ``pvals_adj`` is present, markers are first filtered to
        ``pvals_adj <= 0.05``; the remaining markers are ranked by ``abs(scores)``
        only. ``logfoldchanges`` is optional and not part of the ranking, and no
        effect-size filter is applied here. Pass a pre-filtered table (e.g. the
        ``*_filtered`` output of ``marker_genes.run_rank_genes``) if effect-size
        filtering is wanted.

    gene_trend_base_key : str, default="gene_trends"
        Base key for branch trend matrices in ``adata.varm``. Expected format:
        ``{gene_trend_base_key}_{branch}``.
    top_n : int, default=300
        Number of genes selected for clustering (all available genes if fewer).
    n_neighbors : int, default=30
        Neighborhood size used by Palantir for clustering gene-trend profiles.
    save_excel : str | None, default=None
        If set, write one Excel file with one sheet per exported table.
        Branch sheets can optionally include appended Palantir trend values.
    save_plots : str | None, default=None
        If set, save Palantir cluster plots as ``{save_plots}_{branch}.pdf``.
        The value acts as a filename prefix and may include directories.
    store_key : str | None, default=None
        If set, store results in
        ``adata.uns["sctoolbox"]["palantir_gene_communities"][store_key]``.
    excel_prefix : str | pathlib.Path | None, default=None
        Optional prefix or directory prepended to ``save_excel``.

    Returns
    -------
    tuple[list[str], dict[str, pd.Series | pd.DataFrame]]
        A tuple containing:
        - the selected genes used for clustering
        - a mapping ``branch -> communities`` as returned by Palantir

    Raises
    ------
    ValueError
        If gene selection fails or required inputs are missing.
    TypeError
        If Palantir returns an unexpected object type.
    """
    # Normalize branch input so the rest of the function always works with a list.
    if isinstance(trend_branches, str):
        trend_branch_list = [trend_branches]
    else:
        trend_branch_list = list(trend_branches)

    # Keep the original key only for metadata if markers were provided via adata.uns.
    marker_key = marker_table if isinstance(marker_table, str) else None

    # Resolve marker tables from AnnData if the user provided a key.
    if gene_source == "markers" and isinstance(marker_table, str):
        marker_table = marker_genes.get_rank_genes_tables(
            adata,
            key=marker_key,
            n_genes=None,
            out_group_fractions=True,
        )
        logger.info("Loaded marker tables from adata using key='%s'.", marker_key)

    # `group_tables` collects sheets for optional Excel export.
    group_tables: dict[str, pd.DataFrame] = {}
    selected_genes: list[str]

    if gene_source == "hvg":
        if "highly_variable" not in adata.var.columns:
            raise ValueError("gene_source='hvg' requires adata.var['highly_variable'].")

        ranked_hvgs, ranking = _rank_highly_variable_genes(adata)
        selected_genes = ranked_hvgs[:top_n]
        logger.info(
            "Selected %d/%d HVGs for clustering (ranked by %s).", len(selected_genes), len(ranked_hvgs), ranking
        )

    else:
        if not marker_table:
            raise ValueError(
                "gene_source='markers' requires marker_table (dict or adata.uns key)."
            )

        valid_tables = {
            key: value
            for key, value in marker_table.items()
            if value is not None and not value.empty
        }
        if not valid_tables:
            raise ValueError("marker_table provided but all marker tables are empty.")

        alpha = 0.05
        per_group = max(1, int(np.ceil(top_n / len(valid_tables))))

        selected_set: set[str] = set()
        ranked_rows: list[pd.DataFrame] = []

        # Keep cleaned marker tables for optional Excel export.
        for group, df in valid_tables.items():
            df_raw = df.copy()

            if "names" not in df_raw.columns and "gene" in df_raw.columns:
                df_raw = df_raw.rename(columns={"gene": "names"})
            if "names" not in df_raw.columns:
                raise ValueError(f"Marker table '{group}' has no 'names'/'gene' column.")

            df_raw = df_raw[df_raw["names"].notna() & (df_raw["names"] != "")]
            group_tables[group] = df_raw

        # Build a quota-based marker selection so no single group dominates.
        for group, df in valid_tables.items():
            df2 = df.copy()

            if "names" not in df2.columns and "gene" in df2.columns:
                df2 = df2.rename(columns={"gene": "names"})
            if "names" not in df2.columns:
                raise ValueError(f"Marker table '{group}' has no 'names'/'gene' column.")

            df2 = df2[df2["names"].notna() & (df2["names"] != "")]

            if "pvals_adj" in df2.columns:
                df2 = df2[df2["pvals_adj"].notna() & (df2["pvals_adj"] <= alpha)]

            missing = [col for col in ("scores",) if col not in df2.columns]
            if missing:
                raise ValueError(
                    f"Marker table '{group}' is missing required columns: {missing}. "
                    "Expected at least: ['names' (or 'gene'), 'scores']."
                )

            # Rank significant markers by the magnitude of the marker-test score only.
            # logfoldchanges is deliberately not part of the score; effect-size
            # filtering is left to the caller (see docstring).
            df2["rank_score"] = df2["scores"].abs()

            df2 = (
                df2.sort_values("rank_score", ascending=False)
                .drop_duplicates(subset="names", keep="first")
            )

            top_group = df2.head(per_group)
            selected_set.update(top_group["names"].tolist())
            ranked_rows.append(df2[["names", "rank_score"]])

        # Keep a stable ranking-based order instead of converting the set directly.
        combined = pd.concat(ranked_rows, axis=0, ignore_index=True)
        combined = combined[combined["names"].isin(selected_set)]
        combined = (
            combined.sort_values("rank_score", ascending=False)
            .drop_duplicates(subset="names", keep="first")
        )

        if len(selected_set) > top_n:
            selected_genes = combined.head(top_n)["names"].tolist()
        else:
            selected_genes = combined["names"].tolist()

        logger.info(
            "Selected %d marker genes across %d groups (quota=%d/group, alpha=%s).",
            len(selected_genes),
            len(valid_tables),
            per_group,
            alpha,
        )

    if not selected_genes:
        raise ValueError("No genes selected for clustering (selected_genes is empty).")

    communities_dict: dict[str, pd.Series | pd.DataFrame] = {}

    for br in trend_branch_list:
        logger.info("Clustering gene trends for branch: %s", br)

        communities = palantir.presults.cluster_gene_trends(
            adata,
            br,
            selected_genes,
            n_neighbors=n_neighbors,
        )
        communities_dict[br] = communities

        # Build one branch table for optional Excel export.
        if isinstance(communities, pd.Series):
            df_br = (
                communities.rename("community")
                .reset_index()
                .rename(columns={"index": "gene"})
            )
        elif isinstance(communities, pd.DataFrame):
            df_br = communities.copy()
            if "gene" not in df_br.columns:
                df_br = df_br.reset_index().rename(columns={"index": "gene"})
        else:
            raise TypeError(
                "Unexpected return type from "
                f"palantir.presults.cluster_gene_trends for branch '{br}': "
                f"{type(communities)}. Expected pandas.Series or pandas.DataFrame."
            )

        if "community" in df_br.columns:
            df_br = df_br.sort_values("community").reset_index(drop=True)

        group_tables[br] = df_br

    # Plot branch-specific community panels and optionally save them.
    if save_plots is not None:
        Path(save_plots).parent.mkdir(parents=True, exist_ok=True)

    for br in trend_branch_list:
        palantir.plot.plot_gene_trend_clusters(adata, br)

        if save_plots is not None:
            filename = f"{save_plots}_{br}.pdf"
            _save_figure(path=filename)
            logger.info("Saved plot for branch '%s' as '%s'.", br, filename)

        plt.show()
        plt.close()

    # Optionally store a compact result object in adata.uns.
    if store_key is not None:
        uns_path = ["palantir_gene_communities", store_key]

        communities_tables: dict[str, pd.DataFrame] = {}
        for br, comm in communities_dict.items():
            if isinstance(comm, pd.Series):
                communities_tables[br] = (
                    comm.rename("community")
                    .reset_index()
                    .rename(columns={"index": "gene"})
                )
            else:
                df_comm = comm.copy()
                if "gene" not in df_comm.columns:
                    df_comm = df_comm.reset_index().rename(columns={"index": "gene"})
                communities_tables[br] = df_comm

        result_object = {
            "selected_genes": selected_genes,
            "communities": communities_tables,
            "parameters": {
                "top_n": top_n,
                "n_neighbors": n_neighbors,
                "gene_source": gene_source,
                "marker_source": marker_key if marker_key is not None else "dict_or_none",
                "trend_branches": trend_branch_list,
            },
        }

        if utils.adata.in_uns(adata, ["sctoolbox"] + uns_path):
            logger.warning(
                "Overwriting existing entry in adata.uns at path %s",
                ["sctoolbox"] + uns_path,
            )

        utils.adata.add_uns_info(
            adata=adata,
            key=uns_path,
            value=result_object,
            how="overwrite",
        )
        logger.info("Stored results in adata.uns['sctoolbox'] at path %s.", uns_path)

    # Optionally write one Excel file using the framework table writer.
    if save_excel is not None and group_tables:
        out_path = Path(excel_prefix) / save_excel if excel_prefix else Path(save_excel)
        filename = Path(settings.full_table_prefix) / out_path
        filename.parent.mkdir(parents=True, exist_ok=True)

        sheets: dict[str, pd.DataFrame] = {}

        for name, df in group_tables.items():
            df_copy = df.copy()

            # Round common numeric columns for readability.
            for col in ("scores", "logfoldchanges", "rank_score"):
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].round(3)

            # Standardize the gene column name before optional trend merging.
            if "gene" not in df_copy.columns:
                if "names" in df_copy.columns:
                    df_copy = df_copy.rename(columns={"names": "gene"})
                else:
                    df_copy.index.name = None
                    df_copy["gene"] = df_copy.index.astype(str)

            # Only branch sheets receive appended trend matrices.
            if name in trend_branch_list:
                trend_df = adata.varm.get(f"{gene_trend_base_key}_{name}")
                if isinstance(trend_df, pd.DataFrame):
                    common_genes = [gene for gene in df_copy["gene"] if gene in trend_df.index]
                    if common_genes:
                        trend_subset = trend_df.loc[common_genes].copy()
                        trend_subset.index.name = None
                        trend_subset.columns = [
                            f"pt_{float(col):.2f}" if pd.notna(col) else "pt_nan"
                            for col in trend_subset.columns
                        ]
                        trend_subset = trend_subset.round(3)

                        df_copy = (
                            df_copy.set_index("gene")
                            .loc[common_genes]
                            .join(trend_subset, how="left")
                            .reset_index()
                        )

            sheets[str(name)] = df_copy

        utils.tables.write_excel(
            table_dict=sheets,
            filename=str(filename),
            index=False,
        )
        logger.info("Saved gene community tables to '%s'.", filename)

    return selected_genes, communities_dict


@deco.log_anndata
@beartype
def branch_qc(
    adata: AnnData,
    masks_key: str = "branch_masks",
    pseudotime_key: str = "palantir_pseudotime",
    min_cells: int = 150,
    n_bins: int = 20,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Compute minimal QC metrics for Palantir branches.

    The function summarizes each branch (defined by boolean masks) by:
    (1) number of cells, (2) pseudotime range, and (3) pseudotime coverage based
    on occupied histogram bins. A ``keep`` flag indicates whether the branch
    passes the provided thresholds.

    Parameters
    ----------
    adata : AnnData
        AnnData with boolean branch masks in ``adata.obsm[masks_key]`` and numeric
        pseudotime values in ``adata.obs[pseudotime_key]``.
    masks_key : str, default="branch_masks"
        Key in ``adata.obsm`` with boolean branch masks (cells x branches).
    pseudotime_key : str, default="palantir_pseudotime"
        Key in ``adata.obs`` with numeric pseudotime.
    min_cells : int, default=150
        Minimum number of cells required per branch.
    n_bins : int, default=20
        Number of histogram bins used to estimate pseudotime coverage.
    min_coverage : float, default=0.5
        Minimum fraction of occupied bins required.

    Returns
    -------
    pd.DataFrame
        One row per branch with QC metrics and a ``keep`` flag.
    """
    out_cols = ["branch", "n_cells", "pt_min", "pt_max", "coverage", "keep"]

    if masks_key not in adata.obsm:
        logger.warning("Key '%s' not found in adata.obsm. Returning empty QC table.", masks_key)
        return pd.DataFrame(columns=out_cols)

    if pseudotime_key not in adata.obs:
        logger.warning("Key '%s' not found in adata.obs. Returning empty QC table.", pseudotime_key)
        return pd.DataFrame(columns=out_cols)

    masks = adata.obsm[masks_key]
    pt = adata.obs[pseudotime_key]

    rows: list[dict[str, Union[str, int, float, bool]]] = []

    for branch in masks.columns:
        member = masks[branch].to_numpy(dtype=bool, copy=False)
        cell_ids = masks.index[member]
        n_cells = int(len(cell_ids))

        if n_cells == 0:
            rows.append(
                {"branch": branch, "n_cells": 0, "pt_min": np.nan, "pt_max": np.nan, "coverage": 0.0, "keep": False}
            )
            continue

        pt_branch = pt.loc[cell_ids].dropna()
        if pt_branch.empty:
            rows.append(
                {"branch": branch, "n_cells": n_cells, "pt_min": np.nan, "pt_max": np.nan, "coverage": 0.0, "keep": False}
            )
            continue

        pt_min = float(pt_branch.min())
        pt_max = float(pt_branch.max())

        if pt_max > pt_min:
            bins = np.linspace(pt_min, pt_max, int(n_bins) + 1)
            hist, _ = np.histogram(pt_branch.to_numpy(), bins=bins)
            coverage = float((hist > 0).mean())
        else:
            # Degenerate: all cells at same pseudotime -> effectively 1 occupied bin.
            coverage = 1.0 / float(n_bins)

        keep = (n_cells >= int(min_cells)) and (coverage >= float(min_coverage))

        rows.append(
            {"branch": branch, "n_cells": n_cells, "pt_min": pt_min, "pt_max": pt_max, "coverage": coverage, "keep": keep}
        )

    qc = (
        pd.DataFrame(rows, columns=out_cols)
        .sort_values(["keep", "n_cells"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return qc


@deco.log_anndata
@beartype
def plot_branch_qc_distributions(  # noqa: C901
    adata: AnnData,
    qc: pd.DataFrame,
    *,
    masks_key: str = "branch_masks",
    pseudotime_key: str = "palantir_pseudotime",
    branches: Optional[Iterable[str]] = None,
    max_branches: int = 12,
    bins: int = 30,
    ncols: int = 3,
    show_pt_bounds: bool = True,
    figsize: tuple[float, float] = (4.8, 3.2),
    save: Optional[str] = None,
    report: bool = False,
    dpi: Optional[Union[int, float]] = None,
    rasterize: bool = False,
    savefig_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Figure, np.ndarray] | None:
    """Plot pseudotime distributions per branch for QC inspection.

    This function visualizes branch-wise pseudotime histograms to assess whether
    branches have sufficient cell support and cover the pseudotime range. Optionally,
    pseudotime bounds from the QC table (``pt_min``/``pt_max``) are shown as vertical
    guide lines.

    Parameters
    ----------
    adata : AnnData
        AnnData with boolean branch masks in ``adata.obsm[masks_key]`` and numeric
        pseudotime values in ``adata.obs[pseudotime_key]``.
    qc : pandas.DataFrame
        QC table returned by :func:`branch_qc`. Must contain the columns ``branch``,
        ``n_cells`` and ``keep``. Optional columns: ``pt_min``, ``pt_max``.
    masks_key : str, default="branch_masks"
        Key in ``adata.obsm`` that stores boolean branch masks (cells x branches).
    pseudotime_key : str, default="palantir_pseudotime"
        Key in ``adata.obs`` that stores numeric pseudotime values.
    branches : Optional[Iterable[str]], default=None
        Branch names to plot. If None, the largest branches by ``n_cells`` are plotted.
    max_branches : int, default=12
        Maximum number of branches to display if ``branches`` is None.
    bins : int, default=30
        Number of histogram bins per branch.
    ncols : int, default=3
        Number of subplot columns.
    show_pt_bounds : bool, default=True
        If True, draw vertical lines for ``pt_min`` and ``pt_max`` (if present in ``qc``).
    figsize : tuple[float, float], default=(4.8, 3.2)
        Base size of one subplot. The total figure size scales with the grid.
    save : Optional[str], default=None
        If provided, save the figure via :func:`~sctoolbox.plotting.general._save_figure`.
        The file extension controls the format. If no extension is given, PNG is used.
    report : bool, default=False
        Forwarded to :func:`~sctoolbox.plotting.general._save_figure`.
    dpi : Optional[int | float], default=None
        Forwarded to :func:`~sctoolbox.plotting.general._save_figure`.
    rasterize : bool, default=False
        Forwarded to :func:`~sctoolbox.plotting.general._save_figure`.
    savefig_kwargs : Optional[dict[str, Any]], default=None
        Extra keyword arguments forwarded to ``matplotlib.pyplot.savefig`` via ``_save_figure``.

    Returns
    -------
    tuple[Figure, np.ndarray] | None
        Matplotlib figure and a flat array of axes, or ``None`` if no plot can be created.
    """
    # ------------------------------------------------------------------
    # Step 0 — Initialize optional save arguments
    # ------------------------------------------------------------------
    # Avoid mutable default argument issues
    if savefig_kwargs is None:
        savefig_kwargs = {}

    # ------------------------------------------------------------------
    # Step 1 — Validate required AnnData keys (soft fail)
    # ------------------------------------------------------------------
    # We need:
    # - branch masks in adata.obsm
    # - pseudotime values in adata.obs
    missing = []
    if masks_key not in adata.obsm:
        missing.append(f"adata.obsm['{masks_key}']")
    if pseudotime_key not in adata.obs:
        missing.append(f"adata.obs['{pseudotime_key}']")

    # If anything is missing, do not crash and return without creating a figure.
    if missing:
        logger.warning("Missing required key(s): %s. Nothing to plot.", ", ".join(missing))
        return None

    pt = adata.obs[pseudotime_key]      # pseudotime vector
    masks = adata.obsm[masks_key]      # boolean cell × branch matrix

    # ------------------------------------------------------------------
    # Step 2 — Decide which branches to visualize
    # ------------------------------------------------------------------
    # If user specifies branches → restrict to those
    # Otherwise → take largest branches by cell count
    if branches is not None:
        qc_plot = qc[qc["branch"].isin(set(branches))]
    else:
        qc_plot = qc.sort_values("n_cells", ascending=False).head(int(max_branches))

    # Ensure branches actually exist in masks
    mask_branches = set(masks.columns)
    missing_branches = sorted(set(qc_plot["branch"]) - mask_branches)

    # Warn once (not inside loop)
    if missing_branches:
        logger.warning(
            "Some branches are not present in masks '%s' and will be skipped: %s",
            masks_key,
            missing_branches,
        )
        qc_plot = qc_plot[qc_plot["branch"].isin(mask_branches)]

    n = int(len(qc_plot))
    if n == 0:
        logger.warning("No branches available for plotting (after filtering).")
        return None

    # ------------------------------------------------------------------
    # Step 3 — Compute global pseudotime range
    # ------------------------------------------------------------------
    # Using a shared x-axis makes distributions comparable
    pt_all = pt.dropna().to_numpy()
    if pt_all.size == 0:
        logger.warning("Pseudotime column '%s' contains no finite values.", pseudotime_key)
        return None

    x_min = float(np.min(pt_all))
    x_max = float(np.max(pt_all))

    # Small padding so bars do not touch axis border
    x_pad = 0.02 * (x_max - x_min + 1e-12)
    x_lim = (x_min - x_pad, x_max + x_pad)

    # ------------------------------------------------------------------
    # Step 4 — Create subplot grid
    # ------------------------------------------------------------------
    ncols_eff = max(1, int(ncols))
    nrows_eff = int(np.ceil(n / ncols_eff))

    fig, axes_grid = plt.subplots(
        nrows_eff,
        ncols_eff,
        figsize=(figsize[0] * ncols_eff, figsize[1] * nrows_eff),
        sharex=True,
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes_grid).ravel()

    # Check once if QC bounds exist
    has_bounds = bool(
        show_pt_bounds
        and ("pt_min" in qc_plot.columns)
        and ("pt_max" in qc_plot.columns)
    )

    bins_eff = int(bins)

    # ------------------------------------------------------------------
    # Step 5 — Plot histogram for each branch
    # ------------------------------------------------------------------
    for i, row in enumerate(qc_plot.itertuples(index=False)):
        branch = row.branch

        # Boolean mask selecting cells belonging to this branch
        member_mask = masks[branch].to_numpy(dtype=bool, copy=False)

        # Extract pseudotime values of branch members
        cell_ids = masks.index[member_mask]
        vals = pt.loc[cell_ids].dropna().to_numpy()

        ax = axes[i]

        # Histogram of pseudotime distribution
        ax.hist(vals, bins=bins_eff, alpha=0.85)
        ax.set_xlim(x_lim)

        # Title includes QC metrics
        ax.set_title(f"{branch} (n={row.n_cells}, keep={row.keep})", fontsize=10)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Cells")
        ax.grid(alpha=0.2)

        # Optional QC bounds (vertical reference lines)
        if has_bounds:
            if np.isfinite(row.pt_min):
                ax.axvline(row.pt_min, linewidth=1, alpha=0.7)
            if np.isfinite(row.pt_max):
                ax.axvline(row.pt_max, linewidth=1, alpha=0.7)

    # Turn off unused subplot axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # ------------------------------------------------------------------
    # Step 6 — Save figure (optional)
    # ------------------------------------------------------------------
    if save is not None:
        _save_figure(
            path=save,
            dpi=dpi,
            report=report,
            rasterize=rasterize,
            **savefig_kwargs,
        )

    return fig, axes


@beartype
def _padded_limits(values: np.ndarray, pad_fraction: float = 0.05) -> tuple[float, float] | None:
    """Return axis limits that contain every finite value plus a relative margin.

    Parameters
    ----------
    values : np.ndarray
        Values that must lie inside the limits. NaN and infinite values are ignored.
    pad_fraction : float, default=0.05
        Margin added below the minimum and above the maximum, as a fraction of the
        value range.

    Returns
    -------
    tuple[float, float] | None
        Lower and upper limit, or ``None`` if ``values`` contains no finite value.
    """
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None

    low, high = float(finite.min()), float(finite.max())
    # A constant curve has no range; pad by its magnitude (at least 1) so it stays visible.
    span = high - low if high > low else max(abs(high), 1.0)
    return low - pad_fraction * span, high + pad_fraction * span


@beartype
def _share_trend_and_cell_scale(trend_ax: Axes, cell_ax: Axes) -> None:
    """Put a trend axis and its twin cell axis on one common y-scale.

    The limits span the trend curves of ``trend_ax`` and the cell points of
    ``cell_ax``. As both axes then show the same numbers, the tick labels and the
    label of the twin axis are hidden.

    Parameters
    ----------
    trend_ax : Axes
        Axis holding the trend curves.
    cell_ax : Axes
        Twin axis holding the individual cells as scatter points.
    """
    values = [np.asarray(line.get_ydata(), dtype=float) for line in trend_ax.lines]
    values += [np.asarray(points.get_offsets(), dtype=float)[:, 1] for points in cell_ax.collections]
    limits = _padded_limits(np.concatenate(values)) if values else None
    if limits is None:
        return

    trend_ax.set_ylim(*limits)
    cell_ax.set_ylim(*limits)
    cell_ax.tick_params(axis="y", right=False, labelright=False)
    cell_ax.set_ylabel("")


@beartype
def _classify_trend_pattern(curve: pd.Series) -> str:
    """Classify a centroid trend into a simple biological pattern label.

    Parameters
    ----------
    curve : pd.Series
        One centroid trend across pseudotime bins.

    Returns
    -------
    str
        Pattern label describing the overall shape of the trend. Possible values are:

        - ``"stable"``: little overall change across pseudotime.
        - ``"transient peak"``: intermediate increase followed by a decrease.
        - ``"transient dip"``: intermediate decrease followed by a recovery.
        - ``"late increase"``: expression increases mainly at late pseudotime.
        - ``"early high, then decrease"``: high early expression followed by a decline.
        - ``"broad increase"``: expression increases broadly across pseudotime.
        - ``"broad decrease"``: expression decreases broadly across pseudotime.
        - ``"complex"``: no simpler pattern above applies.
    """
    y = np.asarray(curve, dtype=float)

    # Very short or nearly constant curves are treated as stable.
    if y.size < 3 or np.allclose(y, y[0]):
        return "stable"

    # Summarize the beginning, middle, and end of the curve.
    start = float(np.nanmean(y[: max(1, y.size // 5)]))
    end = float(np.nanmean(y[-max(1, y.size // 5):]))
    mid_start = y.size // 3
    mid_end = max(mid_start + 1, (2 * y.size) // 3)
    middle = float(np.nanmean(y[mid_start:mid_end]))

    # Locate the main maximum and minimum positions.
    peak_idx = int(np.nanargmax(y))
    through_idx = int(np.nanargmin(y))

    # Use overall change and total spread as simple shape summaries.
    delta = end - start
    spread = float(np.nanmax(y) - np.nanmin(y))

    # Low-spread curves are treated as stable.
    if spread < 0.5:
        return "stable"

    # Define rough early and late regions of the pseudotime axis.
    early_cut = max(1, int(0.25 * y.size))
    late_cut = min(y.size - 1, int(0.75 * y.size))

    # Middle higher than both ends suggests a transient peak.
    if middle > start + 0.5 and middle > end + 0.5 and early_cut <= peak_idx <= late_cut:
        return "transient peak"

    # Middle lower than both ends suggests a transient dip.
    if middle < start - 0.5 and middle < end - 0.5 and early_cut <= through_idx <= late_cut:
        return "transient dip"

    # Strong increase with a late maximum suggests late activation.
    if delta >= 0.75 and peak_idx >= late_cut:
        return "late increase"

    # Strong decrease with an early maximum suggests early activation followed by decline.
    if delta <= -0.75 and peak_idx <= early_cut:
        return "early high, then decrease"

    # More gradual monotonic patterns.
    if delta >= 0.5:
        return "broad increase"

    if delta <= -0.5:
        return "broad decrease"

    # Remaining shapes are grouped as complex.
    return "complex"


@deco.log_anndata
@beartype
def plot_branch_gene_programs(  # noqa: C901
    adata: AnnData,
    gene_trend_base_key: str = "gene_trends",
    n_dynamic_genes: int = 1500,
    gene_ranking_mode: str = "union",
    trend_knn_neighbors: int = 30,
    max_pseudotime_bins_heatmap: int = 120,
    plot_representative_genes: bool = True,
    representatives_per_cluster: int = 3,
    representatives_ncols: int = 3,
    branch: str | None = None,
    save: str | None = None,
    save_excel: str | None = None,
    plot_format: str = "pdf",
    report_save: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Identify and visualize branch-wise gene programs from Palantir gene trends.

    For each branch, this function selects dynamic genes from Palantir-smoothed
    gene trends, clusters them into branch-specific trend programs, visualizes
    these programs, and returns a structured result dictionary.

    The workflow includes:
    - filtering non-informative genes with flat trends
    - ranking genes by dynamic range, variance, or a union of both
    - clustering selected genes with Palantir
    - summarizing each trend cluster by a centroid curve
    - assigning a simple biological pattern label to each trend cluster
    - optionally plotting representative genes per cluster
    - optionally exporting one Excel sheet per branch and one summary sheet

    Parameters
    ----------
    adata : AnnData
        AnnData containing branch-specific Palantir gene trends in ``adata.varm``.
    gene_trend_base_key : str, default="gene_trends"
        Base key used to locate branch-specific trends in ``adata.varm``.
    n_dynamic_genes : int, default=1500
        Number of dynamic genes selected per branch before clustering.
    gene_ranking_mode : {"range", "var", "union"}, default="union"
        Ranking mode used to select dynamic genes.
    trend_knn_neighbors : int, default=30
        Neighborhood size used by Palantir for gene-trend clustering.
    max_pseudotime_bins_heatmap : int, default=120
        Maximum number of pseudotime bins shown in the centroid heatmap.
        If more bins are present, the heatmap is downsampled.
    plot_representative_genes : bool, default=True
        If True, plot representative gene trajectories per trend cluster.
        Representative genes are selected based on centroid similarity,
        dynamic trend strength, and branch specificity.
    representatives_per_cluster : int, default=3
        Number of representative genes per cluster.
    representatives_ncols : int, default=3
        Number of columns in the representative-gene panel.
    branch : str | None, default=None
        If provided, process only this branch. Otherwise, process all matching
        branches found in ``adata.varm``.
    save : str | None, default=None
        Filename prefix for saving figures. If None, figures are not written.
        Generated figures use the format specified by ``plot_format``.
    save_excel : str | None, default=None
        Output path for an optional Excel file. If provided, the file contains
        one sheet per branch with trend cluster, cluster pattern, representative
        flag, and full gene-level trend values, plus one summary sheet with
        cluster-level information across branches.
    plot_format : str, default="pdf"
        File format used when saving plots.
    report_save : str | None, default=None
        Filename prefix appended to ``settings.report_dir``. The representative
        gene trend figure of each branch is saved as
        ``{report_save}_{branch}.png`` (characters in the branch name other than
        letters, digits, ``_`` and ``-`` are replaced by ``_``). If None, nothing is
        written to the report directory.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping ``branch -> results``.

        Each branch result contains:
        - ``selected_genes``: selected dynamic genes
        - ``trend_clusters``: gene-to-cluster assignment
        - ``centroid_z``: z-scored centroid trends per cluster
        - ``cluster_patterns``: simple biological pattern labels per cluster
        - ``cluster_summary``: per-cluster summary table
        - ``cluster_representatives``: representative genes per cluster
        - ``cluster_gene_ranking``: all genes of each cluster, ordered by the
          representative score (most representative first); the first
          ``representatives_per_cluster`` genes equal ``cluster_representatives``.
          Empty if ``plot_representative_genes=False``.
        - ``figures``: generated matplotlib figure handles

    Raises
    ------
    KeyError
        If gene-trend clustering for an explicitly requested branch fails because
        required trend data are missing.
    ValueError
        If gene-trend clustering for an explicitly requested branch fails because
        of invalid input, or if duplicate export columns are created during
        pseudotime-column renaming.
    """
    # Build the prefix that identifies all branch-specific trend tables.
    # Example: "gene_trends_branchA", "gene_trends_branchB", ...
    varm_prefix = f"{gene_trend_base_key}_"

    # Collect all available branch trend tables once.
    # These tables are later reused to estimate branch specificity when
    # choosing representative genes.
    branch_trend_tables = {
        key[len(varm_prefix):]: value
        for key, value in adata.varm.items()
        if key.startswith(varm_prefix) and isinstance(value, pd.DataFrame)
    }

    # Decide which branches should be processed.
    # If the user provided one branch explicitly, validate that it exists.
    # Otherwise, process all detected branches.
    if branch is not None:
        trend_key = f"{gene_trend_base_key}_{branch}"
        if trend_key not in adata.varm:
            logger.warning(
                "Requested branch '%s' not found in adata.varm (expected '%s').",
                branch,
                trend_key,
            )
            return {}
        branches_to_process = [branch]
    else:
        branches_to_process = sorted(branch_trend_tables.keys())
        if not branches_to_process:
            logger.warning(
                "No branches found in adata.varm with prefix '%s'.",
                varm_prefix
            )
            return {}

    # Helper for saving matplotlib figures with the existing framework utility.
    # The function first makes the correct figure active and then delegates
    # the actual saving to `_save_figure`.
    def _save_fig(fig: Any, suffix: str) -> None:
        if save is None:
            return
        path = f"{save}{suffix}"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(fig.number)
        _save_figure(path=path)

    # Main outputs returned to the caller.
    results: dict[str, dict[str, Any]] = {}

    # Tables prepared for optional Excel export.
    excel_tables: dict[str, pd.DataFrame] = {}

    # One summary table per branch. These are combined into a single summary sheet.
    summary_tables: list[pd.DataFrame] = []

    # Process each branch independently.
    for br in branches_to_process:
        # Filename-safe branch name; `br` itself stays unchanged for data access.
        safe_branch = re.sub(r"[^\w-]", "_", str(br))
        trend_key = f"{gene_trend_base_key}_{br}"
        trends_obj = adata.varm.get(trend_key)

        # Validate the branch-specific trend table.
        if trends_obj is None:
            logger.warning(
                "Missing trends in adata.varm['%s']. Skipping '%s'.",
                trend_key,
                br,
            )
            continue
        if not isinstance(trends_obj, pd.DataFrame):
            logger.warning(
                "Expected DataFrame in adata.varm['%s'], got %s. Skipping '%s'.",
                trend_key,
                type(trends_obj),
                br,
            )
            continue

        # Work on a copy so the original data in `adata.varm` remains untouched.
        trend_df = trends_obj.copy()

        # Convert pseudotime-bin labels to numeric values.
        # Non-numeric columns are removed because they cannot be interpreted
        # as valid pseudotime positions.
        trend_df.columns = pd.to_numeric(trend_df.columns, errors="coerce")
        trend_df = trend_df.loc[:, ~trend_df.columns.isna()]
        if trend_df.shape[1] == 0:
            logger.warning(
                "Branch '%s': no numeric pseudotime bins in '%s'.",
                br,
                trend_key,
            )
            continue

        # Sort bins from early to late pseudotime.
        trend_df = trend_df.reindex(sorted(trend_df.columns), axis=1)

        # Remove invalid values and genes that contain only missing entries.
        trend_df = trend_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all")
        if trend_df.empty:
            logger.warning("Branch '%s': empty trend table after cleaning.", br)
            continue

        # Remove flat genes because they do not show any dynamic behavior
        # across pseudotime and are therefore not informative for clustering.
        dyn_range_all = trend_df.max(axis=1) - trend_df.min(axis=1)
        trend_df = trend_df.loc[dyn_range_all > 0]
        if trend_df.shape[0] < 2:
            logger.warning(
                "Branch '%s': too few non-constant genes after filtering.",
                br,
            )
            continue

        # Compute two complementary dynamic scores:
        # 1. dynamic range: captures overall amplitude
        # 2. variance: captures overall variability of the trend curve
        dynamic_range = trend_df.max(axis=1) - trend_df.min(axis=1)
        dynamic_var = trend_df.var(axis=1)

        # Normalize the ranking mode and cap the requested number of genes
        # at the number of available genes.
        mode = gene_ranking_mode.lower().strip()
        n_avail = int(trend_df.shape[0])
        n_select = min(int(n_dynamic_genes), n_avail)

        if n_select < int(n_dynamic_genes):
            logger.warning(
                "Branch '%s': reducing n_dynamic_genes %d -> %d (only %d genes).",
                br,
                n_dynamic_genes,
                n_select,
                n_avail,
            )

        # Select dynamic genes according to the chosen ranking strategy.
        if mode == "range":
            # Select genes with the largest total amplitude.
            selected_genes = (
                dynamic_range.sort_values(ascending=False)
                .head(n_select)
                .index
                .tolist()
            )
        elif mode == "var":
            # Select genes with the largest variance across pseudotime.
            selected_genes = (
                dynamic_var.sort_values(ascending=False)
                .head(n_select)
                .index
                .tolist()
            )
        elif mode == "union":
            # Combine both rankings so that both strong monotonic trends
            # and more complex patterns can be captured.
            n_by_range = n_select // 2
            n_by_var = n_select - n_by_range

            top_range = dynamic_range.sort_values(ascending=False).head(n_by_range).index
            top_var = dynamic_var.sort_values(ascending=False).head(n_by_var).index
            selected = pd.Index(top_range).union(pd.Index(top_var)).tolist()

            # If overlaps reduced the size of the union, fill the missing slots
            # with genes from a combined ranking.
            if len(selected) < n_select:
                rank_range = dynamic_range.rank(ascending=False, method="average")
                rank_var = dynamic_var.rank(ascending=False, method="average")
                combined = (rank_range + rank_var).sort_values()
                missing = n_select - len(selected)
                fill = (
                    combined.loc[~combined.index.isin(selected)]
                    .head(missing)
                    .index
                    .tolist()
                )
                selected.extend(fill)

            selected_genes = selected[:n_select]
        else:
            logger.warning(
                "Invalid gene_ranking_mode='%s'. Use 'range', 'var', or 'union'. Skipping '%s'.",
                gene_ranking_mode,
                br,
            )
            continue

        if len(selected_genes) < 2:
            logger.warning(
                "Branch '%s': too few selected genes (%d).",
                br,
                len(selected_genes),
            )
            continue

        # Restrict the trend matrix to the selected dynamic genes only.
        trend_df_sel = trend_df.loc[selected_genes].copy()

        # Cluster selected gene trends with Palantir.
        # Output is a mapping from gene to trend-cluster label.
        try:
            trend_clusters = palantir.presults.cluster_gene_trends(
                adata,
                br,
                selected_genes,
                n_neighbors=int(trend_knn_neighbors),
            ).astype(str)
        except (KeyError, ValueError) as exc:
            # When all branches are processed, one failing branch should not stop
            # the remaining branches.
            if branch is None:
                logger.exception(
                    "Branch '%s' failed during gene-trend clustering and will be skipped: %s",
                    br,
                    exc,
                )
                continue

            # If the user explicitly requested one branch, keep the error visible.
            raise

        # Keep only genes that are present in both the cleaned trend table
        # and the Palantir cluster output.
        common = trend_df_sel.index.intersection(trend_clusters.index)
        if common.empty:
            logger.warning(
                "Branch '%s': no overlap between trends and cluster labels.",
                br,
            )
            continue

        trend_df_sel = trend_df_sel.loc[common]
        trend_clusters = trend_clusters.loc[common]

        # Plot Palantir's original trend-cluster panels.
        # The plot titles are updated to also show the number of genes per cluster.
        try:
            palantir.plot.plot_gene_trend_clusters(adata, br)
        except (KeyError, ValueError) as exc:
            if branch is None:
                logger.exception(
                    "Branch '%s' failed during gene-trend plotting and will be skipped: %s",
                    br,
                    exc,
                )
                continue

            raise

        fig_clusters = plt.gcf()
        axes_clusters = [ax for ax in fig_clusters.axes if ax.has_data()]
        cluster_sizes = trend_clusters.value_counts()

        for ax in axes_clusters:
            title = (ax.get_title() or "").strip()
            parts = title.split(maxsplit=1)
            if len(parts) != 2 or parts[0] != "Cluster":
                continue
            cid = parts[1].strip()
            n = int(cluster_sizes.get(cid, 0))
            ax.set_title(f"Cluster {cid} (n={n})", fontsize=10)

        fig_clusters.suptitle(
            f"Branch '{br}' — Palantir trend clusters (n={len(selected_genes)} genes)",
            y=1.02,
        )
        _save_fig(fig_clusters, f"_branch-{safe_branch}_clusters.{plot_format}")
        plt.show()

        # Compute one centroid curve per trend cluster using the median trend.
        # Each centroid is then z-scored across pseudotime so that cluster shapes
        # can be compared independently of absolute magnitude.
        centroid_abs = trend_df_sel.groupby(trend_clusters).median()
        vals = centroid_abs.values
        mu = np.nanmean(vals, axis=1, keepdims=True)
        sd = np.nanstd(vals, axis=1, keepdims=True)
        sd[sd == 0] = 1.0

        centroid_z = (vals - mu) / sd
        centroid_z = np.nan_to_num(centroid_z, nan=0.0, posinf=0.0, neginf=0.0)
        centroid_z = pd.DataFrame(
            centroid_z,
            index=centroid_abs.index.astype(str),
            columns=centroid_abs.columns.astype(str),
        )

        # Assign a simple biological pattern label to each trend cluster.
        # This makes the output easier to interpret than cluster numbers alone.
        cluster_patterns = {
            cluster_id: _classify_trend_pattern(centroid_z.loc[cluster_id])
            for cluster_id in centroid_z.index
        }

        # Build a compact cluster-level summary table.
        # This is useful for interpretation and for the summary Excel sheet.
        cluster_summary = pd.DataFrame(
            {
                "branch": br,
                "trend_cluster": centroid_z.index.astype(str),
                "cluster_pattern": [cluster_patterns[c] for c in centroid_z.index],
                "n_genes": [int((trend_clusters == c).sum()) for c in centroid_z.index],
                "start_value": centroid_z.iloc[:, 0].values,
                "end_value": centroid_z.iloc[:, -1].values,
                "peak_bin": centroid_z.idxmax(axis=1).values,
                "through_bin": centroid_z.idxmin(axis=1).values,
            }
        )

        # Build the centroid heatmap.
        # If there are many pseudotime bins, downsample them to keep the heatmap readable.
        heatmap_mat = centroid_z.copy()
        if heatmap_mat.shape[1] > int(max_pseudotime_bins_heatmap):
            keep_idx = np.linspace(
                0,
                heatmap_mat.shape[1] - 1,
                int(max_pseudotime_bins_heatmap),
            ).astype(int)
            heatmap_mat = heatmap_mat.iloc[:, keep_idx]

        fig_h = max(4.0, 0.35 * float(heatmap_mat.shape[0]))
        cg = sns.clustermap(
            heatmap_mat,
            cmap="vlag",
            center=0,
            row_cluster=True,
            col_cluster=False,
            figsize=(12, fig_h),
            yticklabels=True,
            xticklabels=False,
            cbar_kws={"label": "z-score"},
        )

        # Restore readable pseudotime tick labels on the heatmap x-axis.
        ax_hm = cg.ax_heatmap
        pt = heatmap_mat.columns.astype(float).values
        tick_pos = np.linspace(0, len(pt) - 1, 6).astype(int)
        ax_hm.set_xticks(tick_pos)
        ax_hm.set_xticklabels([f"{pt[i]:.2f}" for i in tick_pos], rotation=0)
        ax_hm.set_xlabel("Pseudotime")
        ax_hm.set_ylabel("Trend cluster (hierarchical similarity)")

        cg.fig.suptitle(f"Branch '{br}' — centroid similarity", y=1.02)
        _save_fig(cg.fig, f"_branch-{safe_branch}_centroid_heatmap.{plot_format}")
        plt.show()

        # Select representative genes per trend cluster.
        # The representative score combines:
        # - similarity to the cluster centroid
        # - dynamic strength in the current branch
        # - branch specificity relative to the other available branches
        cluster_representatives: dict[str, list[str]] = {}
        cluster_gene_ranking: dict[str, list[str]] = {}
        fig_reps = None

        if plot_representative_genes:
            # Z-score each gene trend across pseudotime so shapes can be compared.
            gene_vals = trend_df_sel.values
            gmu = np.nanmean(gene_vals, axis=1, keepdims=True)
            gsd = np.nanstd(gene_vals, axis=1, keepdims=True)
            gsd[gsd == 0] = 1.0

            gene_z = (gene_vals - gmu) / gsd
            gene_z = np.nan_to_num(gene_z, nan=0.0, posinf=0.0, neginf=0.0)
            gene_z = pd.DataFrame(
                gene_z,
                index=trend_df_sel.index,
                columns=trend_df_sel.columns,
            )

            # Estimate branch specificity using trend amplitude.
            # Higher values indicate that the gene changes more strongly in the
            # current branch than in the other available branches.
            current_branch_range = trend_df_sel.max(axis=1) - trend_df_sel.min(axis=1)

            other_branch_ranges: dict[str, float] = {}
            for gene in trend_df_sel.index:
                other_ranges = []

                for other_br, other_df in branch_trend_tables.items():
                    if other_br == br or gene not in other_df.index:
                        continue

                    other_curve = other_df.loc[gene]
                    other_curve = pd.to_numeric(other_curve, errors="coerce")
                    other_curve = other_curve.replace([np.inf, -np.inf], np.nan).dropna()

                    if other_curve.empty:
                        continue

                    other_ranges.append(float(other_curve.max() - other_curve.min()))

                other_branch_ranges[gene] = float(np.mean(other_ranges)) if other_ranges else 0.0

            branch_specificity = pd.Series(
                {
                    gene: float(current_branch_range.loc[gene])
                    / (other_branch_ranges[gene] + 1e-12)
                    for gene in trend_df_sel.index
                }
            )

            # Rank genes inside each cluster with a combined score.
            for cl in centroid_z.index:
                genes_in_cluster = trend_clusters[trend_clusters == cl].index
                if len(genes_in_cluster) == 0:
                    continue

                # Similarity between each gene curve and the cluster centroid.
                x_gene = gene_z.loc[genes_in_cluster].values
                centroid = centroid_z.loc[cl].values
                sim = (x_gene @ centroid) / (
                    np.linalg.norm(x_gene, axis=1)
                    * (np.linalg.norm(centroid) + 1e-12)
                    + 1e-12
                )
                sim_series = pd.Series(sim, index=genes_in_cluster)

                # Additional criteria describing dynamic strength and specificity.
                gene_range = (
                    trend_df_sel.loc[genes_in_cluster].max(axis=1)
                    - trend_df_sel.loc[genes_in_cluster].min(axis=1)
                )
                gene_var = trend_df_sel.loc[genes_in_cluster].var(axis=1)
                gene_specificity = branch_specificity.loc[genes_in_cluster]

                # Convert all criteria to ranks so they can be combined robustly.
                sim_rank = sim_series.rank(ascending=False, method="average")
                range_rank = gene_range.rank(ascending=False, method="average")
                var_rank = gene_var.rank(ascending=False, method="average")
                specificity_rank = gene_specificity.rank(ascending=False, method="average")

                representative_score = (
                    sim_rank + range_rank + var_rank + specificity_rank
                )

                # Keep the full ranking: callers can show more genes of one cluster
                # than the representatives drawn here.
                ranked_genes = representative_score.sort_values(ascending=True).index.tolist()
                cluster_gene_ranking[str(cl)] = ranked_genes
                cluster_representatives[str(cl)] = ranked_genes[: int(representatives_per_cluster)]

            # Reuse the heatmap row order so the representative plots follow the same order.
            row_order = cg.dendrogram_row.reordered_ind
            clusters_in_order = heatmap_mat.index[row_order].tolist()
            clusters_to_plot = [
                cl for cl in clusters_in_order if cl in cluster_representatives
            ]

            if clusters_to_plot:
                ncols_eff = max(1, int(representatives_ncols))
                nrows_eff = int(np.ceil(len(clusters_to_plot) / ncols_eff))

                fig_reps, axes_grid = plt.subplots(
                    nrows_eff,
                    ncols_eff,
                    figsize=(14, 3.4 * nrows_eff),
                    sharex=True,
                    sharey=True,
                )
                axes_arr = np.atleast_1d(axes_grid).ravel()
                pt_gene = trend_df_sel.columns.astype(float).values

                # Use common y-limits so all panels can be compared visually. The limits
                # span every plotted value, so no representative curve is clipped.
                rep_genes_all = [
                    gene
                    for cl in clusters_to_plot
                    for gene in cluster_representatives[cl]
                    if gene in trend_df_sel.index
                ]
                y_limits = _padded_limits(
                    trend_df_sel.loc[rep_genes_all].values if rep_genes_all else trend_df_sel.values
                )

                for i, cl in enumerate(clusters_to_plot):
                    ax_i = axes_arr[i]
                    pattern = cluster_patterns.get(cl, "unknown")

                    # Plot all representative genes for the current trend cluster.
                    for gene in cluster_representatives[cl]:
                        if gene in trend_df_sel.index:
                            ax_i.plot(
                                pt_gene,
                                trend_df_sel.loc[gene].values,
                                linewidth=2,
                                label=gene,
                            )

                    ax_i.set_title(f"Cluster {cl}: {pattern}", fontsize=11)
                    ax_i.set_xlim(pt_gene.min(), pt_gene.max())
                    if y_limits is not None:
                        ax_i.set_ylim(*y_limits)
                    ax_i.grid(alpha=0.2)

                    # Name the curves in a legend inside the panel: labels at the curve
                    # ends overlap when curves end close together, and long names
                    # (e.g. ATAC peaks) run into the neighbouring panel.
                    if ax_i.lines:
                        ax_i.legend(fontsize=8, loc="best", frameon=False)

                    if i % ncols_eff == 0:
                        ax_i.set_ylabel("Trend")
                    if i >= (nrows_eff - 1) * ncols_eff:
                        ax_i.set_xlabel("Pseudotime")

                # Hide unused panels if the grid is larger than the number of clusters.
                for j in range(len(clusters_to_plot), len(axes_arr)):
                    axes_arr[j].axis("off")

                fig_reps.suptitle(
                    f"Branch '{br}' — representative feature trends",
                    y=1.02,
                )
                fig_reps.tight_layout()
                _save_fig(fig_reps, f"_branch-{safe_branch}_representatives.{plot_format}")

                # Save for the report before plt.show(), which may close the figure.
                if report_save is not None:
                    plt.figure(fig_reps.number)
                    _save_figure(path=f"{report_save}_{safe_branch}.png", report=True)

                plt.show()

        # Build a gene-level export table for this branch.
        # Each row contains one gene, cluster metadata, and the full trend curve.
        representative_genes = {
            gene
            for genes in cluster_representatives.values()
            for gene in genes
        }

        export_df = trend_df_sel.copy()

        # Remove the index name so "gene" exists only as a regular column.
        export_df.index.name = None
        export_df.insert(0, "gene", export_df.index.astype(str))

        cluster_ids = trend_clusters.loc[export_df.index].astype(str)
        export_df.insert(0, "trend_cluster", cluster_ids.values)
        export_df.insert(
            1,
            "cluster_pattern",
            [cluster_patterns[c] for c in cluster_ids.values],
        )
        export_df.insert(
            3,
            "representative",
            [gene in representative_genes for gene in export_df.index],
        )

        # Rename pseudotime columns to unique, Excel-friendly names.
        metadata_columns = {
            "trend_cluster",
            "cluster_pattern",
            "gene",
            "representative",
        }

        renamed_columns = []
        pt_index = 0

        for col in export_df.columns:
            if isinstance(col, str) and col in metadata_columns:
                renamed_columns.append(col)
            else:
                renamed_columns.append(f"pt_{pt_index}_{float(col):.8f}")
                pt_index += 1

        export_df.columns = renamed_columns

        # Fail early if column renaming produced duplicate names.
        if export_df.columns.duplicated().any():
            dup_cols = export_df.columns[export_df.columns.duplicated()].tolist()
            raise ValueError(
                f"Duplicate export columns detected after pseudotime renaming: {dup_cols}"
            )

        # Sort by cluster, then representative status, then gene name.
        export_df = export_df.sort_values(
            by=["trend_cluster", "representative", "gene"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

        excel_tables[br] = export_df
        summary_tables.append(cluster_summary)

        # Store branch-level outputs for downstream use in notebooks or pipelines.
        results[br] = {
            "selected_genes": selected_genes,
            "trend_clusters": trend_clusters,
            "centroid_z": centroid_z,
            "cluster_patterns": cluster_patterns,
            "cluster_summary": cluster_summary,
            "cluster_representatives": cluster_representatives,
            "cluster_gene_ranking": cluster_gene_ranking,
            "figures": {
                "clusters": fig_clusters,
                "centroid_heatmap": cg.fig,
                "representatives": fig_reps,
            },
        }

    # Optionally save all branch tables into one Excel file. This runs after the
    # branch loop so the workbook is written once, with every branch in it.
    # The framework helper sanitizes sheet names internally.
    if save_excel is not None and excel_tables:
        sheets: dict[str, pd.DataFrame] = {}

        for branch_name, table in excel_tables.items():
            sheet = table.copy()

            # Round pseudotime columns for readability in Excel.
            # Round each column separately to avoid assignment issues with duplicated labels.
            pt_cols = [
                col for col in sheet.columns
                if isinstance(col, str) and col.startswith("pt_")
            ]
            for col in pt_cols:
                sheet[col] = sheet[col].round(3)

            sheets[str(branch_name)] = sheet

        if summary_tables:
            summary_df = pd.concat(summary_tables, ignore_index=True).copy()

            # Round selected summary columns for readability.
            for col in ("start_value", "end_value"):
                if col in summary_df.columns:
                    summary_df[col] = summary_df[col].round(3)

            sheets["summary"] = summary_df

        filename = Path(settings.full_table_prefix) / save_excel
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Use the framework table writer for consistent Excel export handling.
        utils.tables.write_excel(
            table_dict=sheets,
            filename=str(filename),
            index=False,
        )
        logger.info("Saved branch gene program tables to '%s'.", filename)

    return results


@deco.log_anndata
@beartype
def plot_branch_early_late_genes(  # noqa: C901
    adata: AnnData,
    gene_trend_key: str = "gene_trends",
    pseudotime_key: str = "palantir_pseudotime",
    entropy_key: str = "palantir_entropy",
    top_n: int = 5,
    quantile_cutoff: float = 0.25,
    force_equal_ylim: bool = True,
    figsize: tuple[float, float] = (20.0, 8.0),
    trend_linewidth: float = 2.0,
    umap_ncols: int = 4,
    branches: Iterable[str] | None = None,
    save: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Plot early/late gene-trend panels per branch and optionally a matching UMAP panel.

    For each branch with trends stored in `adata.varm[f"{gene_trend_key}_{branch}"]`,
    this function:
    1) loads the branch trend matrix (genes × pseudotime bins),
    2) defines early/late windows via `quantile_cutoff`,
    3) ranks genes by `delta = mean(early) - mean(late)`,
    4) plots two trend panels (top early genes vs top late genes),
    5) optionally plots a UMAP panel colored by pseudotime, entropy, and the top early/late gene.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing gene trends in `adata.varm`.
    gene_trend_key : str, default="gene_trends"
        Base prefix for branch-specific trend matrices in `adata.varm`.
    pseudotime_key : str, default="palantir_pseudotime"
        Key in `adata.obs` used for UMAP pseudotime coloring.
    entropy_key : str, default="palantir_entropy"
        Key in `adata.obs` used for UMAP entropy coloring.
    top_n : int, default=5
        Number of top early and top late genes to plot per branch.
    quantile_cutoff : float, default=0.25
        Fraction of the pseudotime grid used to define early and late windows.
        Must satisfy `0 < quantile_cutoff < 0.5`.
    force_equal_ylim : bool, default=True
        If True, enforce identical y-limits for both trend panels.
    figsize : tuple[float, float], default=(20.0, 8.0)
        Figure size for the trend line plots.
    trend_linewidth : float, default=2.0
        Line width for gene trend curves.
    umap_ncols : int, default=4
        Number of columns in the UMAP panel grid.
    branches : Iterable[str] | None, default=None
        Subset of branches to process. If None, all detected branches are processed.
    save : str | None, default=None
        Filename prefix for saving figures. If None, no files are written.
        Files are saved as:
        - `{save}_branch-<branch>_trends.pdf`
        - `{save}_branch-<branch>_umap.pdf` (only if the UMAP panel was created)

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping `branch -> result dict` with figure handles and selected genes.

    Raises
    ------
    ValueError
        If `quantile_cutoff` is not in the range (0, 0.5) or if `top_n < 1`.
    """
    # --- validate user inputs (avoid silent weirdness) --------------------------
    if not (0.0 < float(quantile_cutoff) < 0.5):
        raise ValueError("quantile_cutoff must satisfy 0 < quantile_cutoff < 0.5.")
    if int(top_n) < 1:
        raise ValueError("top_n must be >= 1.")

    prefix = f"{gene_trend_key}_"  # varm keys look like "gene_trends_<branch>"

    # --- detect branches from varm ---------------------------------------------
    detected = [k.split(prefix, 1)[1] for k in adata.varm.keys() if k.startswith(prefix)]
    detected = list(dict.fromkeys(detected))  # de-duplicate while preserving order

    if branches is not None:
        wanted = set(branches)
        branch_list = [b for b in detected if b in wanted]
    else:
        branch_list = detected

    if not branch_list:
        return {}

    # --- optional UMAP availability --------------------------------------------
    can_plot_umap = (
        "X_umap" in adata.obsm
        and pseudotime_key in adata.obs
        and entropy_key in adata.obs
    )

    # --- save helper (sc-framework _save_figure is stateful) -------------------
    def _save_current(fig: Any, path: str) -> None:
        if save is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(fig.number)  # set current figure for stateful saver
        _save_figure(path=path)

    results: dict[str, dict[str, Any]] = {}
    logged_umap_skip = False  # avoid spamming the same info message

    for br in branch_list:
        key = f"{gene_trend_key}_{br}"
        trends_obj = adata.varm.get(key)
        if trends_obj is None:
            continue

        # --- load trends into df (genes x bins) and pt_vals (numeric bins) ------
        if isinstance(trends_obj, pd.DataFrame):
            df = trends_obj.copy()
            df.columns = pd.to_numeric(df.columns, errors="coerce")
            df = df.loc[:, ~df.columns.isna()]
            if df.empty or df.shape[1] == 0:
                continue
            pt_vals = df.columns.to_numpy(dtype=float)

        elif isinstance(trends_obj, np.ndarray):
            # fallback: array trends require an explicit pseudotime grid in adata.uns
            pt_uns = adata.uns.get(f"{key}_pseudotime")
            if pt_uns is None:
                continue
            pt_vals = np.asarray(pt_uns, dtype=float)
            df = pd.DataFrame(trends_obj, index=adata.var_names, columns=pt_vals)
            if df.empty or df.shape[1] == 0:
                continue

        else:
            continue

        # --- ensure bins are ordered early -> late ------------------------------
        order = np.argsort(pt_vals)
        pt_vals = pt_vals[order]
        df = df.iloc[:, order]

        # --- early/late windows based on quantiles of the bin grid --------------
        q_low, q_high = np.quantile(pt_vals, [quantile_cutoff, 1.0 - quantile_cutoff])
        early_mask = pt_vals <= q_low
        late_mask = pt_vals >= q_high
        if not np.any(early_mask) or not np.any(late_mask):
            continue

        # --- gene ranking: delta = mean(early) - mean(late) ---------------------
        # delta > 0 => higher early; delta < 0 => higher late
        # Positive delta => early-high genes; negative delta => late-high genes.
        early_mean = df.loc[:, early_mask].mean(axis=1)
        late_mean = df.loc[:, late_mask].mean(axis=1)
        delta = early_mean - late_mean

        top_n_eff = min(int(top_n), int(df.shape[0]))
        early_ranked = delta.sort_values(ascending=False).head(top_n_eff)
        late_ranked = delta.sort_values(ascending=True).head(top_n_eff)
        if early_ranked.empty or late_ranked.empty:
            continue

        early_genes = list(map(str, early_ranked.index))
        late_genes = list(map(str, late_ranked.index))
        early_top = early_genes[0]
        late_top = late_genes[0]

        # ================================================================
        # (1) Trend line plot (two panels)
        # ================================================================
        # Create a new figure for the trend panels (left = early genes, right = late genes)
        trend_fig = plt.figure(figsize=figsize)

        # Use a 1x2 GridSpec so we can control spacing between the two panels
        grid = trend_fig.add_gridspec(nrows=1, ncols=2, wspace=0.25)

        # Add a shared title for the entire figure (the current branch name)
        trend_fig.suptitle(f"Branch: {br}", fontsize=14, fontweight="bold", y=1.02)

        # Create the two subplots: early panel (left) and late panel (right)
        ax_early = trend_fig.add_subplot(grid[0, 0])  # early-high genes
        ax_late = trend_fig.add_subplot(grid[0, 1])   # late-high genes

        # Plot the trend curves for the top "early" genes
        for g in early_genes:
            if g in df.index:  # defensive check: only plot if gene exists in the trend table
                ax_early.plot(
                    pt_vals,                      # x-axis: pseudotime grid (sorted)
                    df.loc[g].to_numpy(),         # y-axis: smoothed trend values for this gene
                    label=g,                      # legend entry = gene name
                    linewidth=trend_linewidth,    # consistent line thickness across genes
                )

        # Plot the trend curves for the top "late" genes
        for g in late_genes:
            if g in df.index:  # defensive check: only plot if gene exists in the trend table
                ax_late.plot(
                    pt_vals,
                    df.loc[g].to_numpy(),
                    label=g,
                    linewidth=trend_linewidth,
                )

        # This makes the two panels directly comparable (same y-scale).
        if force_equal_ylim:
            # Consider only genes that we actually plotted (present in the trend table)
            plotted = [g for g in (early_genes + late_genes) if g in df.index]
            if plotted:
                # Collect all y-values from the plotted genes across both panels
                y = df.loc[plotted].to_numpy().ravel()

                # Drop NaN/inf values to avoid crashing min/max
                y = y[np.isfinite(y)]

                if y.size:
                    # Global y-range across both panels
                    y_min = float(np.min(y))
                    y_max = float(np.max(y))

                    # Apply the same limits to both axes (fair visual comparison)
                    ax_early.set_ylim(y_min, y_max)
                    ax_late.set_ylim(y_min, y_max)

        for ax in (ax_early, ax_late):
            ax.axvline(float(q_low), linestyle="--", alpha=0.3)
            ax.axvline(float(q_high), linestyle="--", alpha=0.3)
            ax.set_xlabel("Pseudotime")
            ax.set_ylabel("Trend / Expression")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)

        ax_early.set_title(f"Top early genes (n={len(early_genes)})")
        ax_late.set_title(f"Top late genes (n={len(late_genes)})")

        if save is not None:
            _save_current(trend_fig, f"{save}_branch-{br}_trends.pdf")

        plt.show()

        # ================================================================
        # (2) UMAP panel (optional)
        # ================================================================
        umap_fig = None
        umap_axes = np.array([], dtype=object)

        if can_plot_umap:
            umap_colors = [pseudotime_key, entropy_key, early_top, late_top]

            pl.plot_embedding(
                adata,
                method="umap",
                color=umap_colors,
                ncols=int(umap_ncols),
                show_title=True,
                frameon=False,
            )

            umap_fig = plt.gcf()
            umap_axes = np.array(umap_fig.axes, dtype=object)

            if save is not None:
                _save_current(umap_fig, f"{save}_branch-{br}_umap.pdf")

            plt.show()
        else:
            if not logged_umap_skip:
                logger.info(
                    "UMAP panel skipped (missing adata.obsm['X_umap'] or required obs keys)."
                )
                logged_umap_skip = True

        results[br] = {
            "trend_fig": trend_fig,
            "trend_axes": (ax_early, ax_late),
            "umap_fig": umap_fig,
            "umap_axes": umap_axes,
            "top_genes": {"early": early_genes, "late": late_genes},
            "quantiles": {"q_low": float(q_low), "q_high": float(q_high)},
        }

    return results


@deco.log_anndata
@beartype
def plot_fate_probability_projection(  # noqa: C901
    adata: AnnData,
    fate_prob_key: str = "palantir_fate_probabilities",
    color_by: str | None = None,
    fate_order: Iterable[str] | None = None,
    point_size: float = 8.0,
    alpha: float = 0.45,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    label_wrap: int | None = 28,
    legend_height: float | None = None,
    legend_marker_scale: float = 2.5,
    title: str = "Palantir fate probability projection",
    show_equal_probability: bool = True,
    save: str | None = None,
    report_save: str | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot Palantir fate probabilities as a joint two-dimensional projection.

    Terminal states are placed at fixed endpoints and each cell is positioned
    at the probability-weighted average of these endpoints. Fate-probability
    rows are normalized to sum to one before projection.

    For two terminal states, the projection preserves the full probability
    vector on a line. For three terminal states, the triangular projection
    represents the complete two-dimensional probability simplex. For more than
    three terminal states, the regular polygon is a lossy two-dimensional
    projection.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing Palantir fate probabilities.
    fate_prob_key : str, default="palantir_fate_probabilities"
        Key in ``adata.obsm`` containing cell-by-terminal-state fate probabilities.
        If stored as a NumPy array, terminal-state labels are read from
        ``adata.uns[f"{fate_prob_key}_columns"]`` when available.
    color_by : str | None, default=None
        Column in ``adata.obs`` used to color cells.
    fate_order : Iterable[str] | None, default=None
        Optional order of terminal states in the projection. If None, the
        column order in the fate-probability table is retained. For more than
        three terminal states, this order also determines which states are
        adjacent in the polygon.
    point_size : float, default=8.0
        Size of individual cell points.
    alpha : float, default=0.45
        Transparency of individual cell points.
    ax : Axes | None, default=None
        Axes to draw the projection on. If None, a new figure and axes are
        created. A categorical legend is placed on a dedicated axes below the
        projection only when the axes are created internally; otherwise it is
        drawn on ``ax``. ``figsize`` is ignored when ``ax`` is given.
    figsize : tuple[float, float] | None, default=None
        Figure size. If None, a suitable size is chosen automatically.
    label_wrap : int | None, default=28
        Maximum width used to wrap fate and categorical legend labels. Set to
        None to disable wrapping.
    legend_height : float | None, default=None
        Positive height reserved for a categorical legend below the projection.
        If None, the required height is estimated automatically.
    legend_marker_scale : float, default=2.5
        Scale factor applied to categorical legend markers.
    title : str, default="Palantir fate probability projection"
        Figure title.
    show_equal_probability : bool, default=True
        Whether to mark the equal-probability reference position.
    save : str | None, default=None
        Filename used to save the figure to the regular figure directory.
    report_save : str | None, default=None
        Relative path used to save the figure to the report directory.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and projection axes.

    Raises
    ------
    ValueError
        If required inputs or plotting parameters are invalid, including when
        fewer than two terminal states are provided, probabilities are
        non-numeric, non-finite, outside the valid range, or contain a zero-sum
        row, or when ndarray terminal-state labels have an inconsistent count.
    """
    if fate_prob_key not in adata.obsm:
        raise ValueError(f"Fate probabilities '{fate_prob_key}' not found in adata.obsm.")
    if color_by is not None and color_by not in adata.obs.columns:
        raise ValueError(f"Column '{color_by}' not found in adata.obs.")
    if point_size <= 0:
        raise ValueError("point_size must be > 0.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must satisfy 0 <= alpha <= 1.")
    if label_wrap is not None and label_wrap < 1:
        raise ValueError("label_wrap must be >= 1 or None.")
    if legend_height is not None and legend_height <= 0:
        raise ValueError("legend_height must be > 0 or None.")
    if legend_marker_scale <= 0:
        raise ValueError("legend_marker_scale must be > 0.")

    # Label-based alignment (.loc / intersection) is ambiguous with duplicated cell IDs.
    if not adata.obs_names.is_unique and (
        color_by is not None or isinstance(adata.obsm[fate_prob_key], pd.DataFrame)
    ):
        raise ValueError("adata.obs_names must be unique to align fate probabilities with adata.obs.")

    fate_probabilities = adata.obsm[fate_prob_key]

    if isinstance(fate_probabilities, pd.DataFrame):
        prob_df = fate_probabilities.copy()
        if not prob_df.index.is_unique:
            raise ValueError(f"Cell IDs in adata.obsm['{fate_prob_key}'] must be unique.")
        common_cells = adata.obs_names.intersection(prob_df.index)

        if len(common_cells) == 0:
            raise ValueError(f"No cells in adata.obs_names overlap with '{fate_prob_key}'.")

        if len(common_cells) < adata.n_obs:
            logger.warning(
                "Fate probabilities contain %d/%d AnnData cells. Only overlapping cells will be plotted.",
                len(common_cells),
                adata.n_obs,
            )

        prob_df = prob_df.loc[common_cells]

    else:
        probability_array = np.asarray(fate_probabilities)

        if probability_array.ndim != 2:
            raise ValueError(
                f"adata.obsm['{fate_prob_key}'] must be a two-dimensional cell-by-fate matrix."
            )
        if probability_array.shape[0] != adata.n_obs:
            raise ValueError(
                f"adata.obsm['{fate_prob_key}'] contains {probability_array.shape[0]} rows "
                f"but adata has {adata.n_obs} cells."
            )

        columns_key = f"{fate_prob_key}_columns"

        if columns_key in adata.uns:
            fate_names = list(adata.uns[columns_key])

            if len(fate_names) != probability_array.shape[1]:
                raise ValueError(
                    f"adata.uns['{columns_key}'] contains {len(fate_names)} terminal-state labels "
                    f"but adata.obsm['{fate_prob_key}'] contains {probability_array.shape[1]} columns."
                )
        else:
            logger.warning(
                "Fate probabilities in adata.obsm['%s'] have no column labels and "
                "adata.uns['%s'] is missing. Using generic terminal-state names.",
                fate_prob_key,
                columns_key,
            )
            fate_names = [
                f"terminal_state_{i}"
                for i in range(1, probability_array.shape[1] + 1)
            ]

        prob_df = pd.DataFrame(
            probability_array,
            index=adata.obs_names,
            columns=fate_names,
        )

    prob_df.columns = [str(column) for column in prob_df.columns]

    if prob_df.columns.duplicated().any():
        duplicates = prob_df.columns[prob_df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate terminal-state labels in '{fate_prob_key}': {duplicates}")

    n_fates = prob_df.shape[1]
    if n_fates == 0:
        raise ValueError(f"adata.obsm['{fate_prob_key}'] contains no terminal states.")
    if n_fates == 1:
        raise ValueError(
            "The fate-probability projection is not informative with only one terminal state; "
            "at least two are required."
        )

    if fate_order is not None:
        requested_order = [str(fate) for fate in fate_order]

        if len(requested_order) != len(set(requested_order)):
            raise ValueError("fate_order contains duplicate terminal-state labels.")
        if len(requested_order) != n_fates or set(requested_order) != set(prob_df.columns):
            raise ValueError(
                "fate_order must contain every terminal-state label exactly once. "
                f"Available states: {list(prob_df.columns)}"
            )

        prob_df = prob_df.loc[:, requested_order]

    fate_names = list(prob_df.columns)

    try:
        probability_values = prob_df.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fate probabilities must contain numeric values only.") from exc

    if not np.isfinite(probability_values).all():
        raise ValueError("Fate probabilities must contain only finite values.")

    # Palantir projects waypoint probabilities onto cells with a weighted sum
    # (``W.T @ branch_probs``) and does not clip above 1 or renormalize rows,
    # so genuine results can sit a few ulp outside [0, 1]. Anything beyond the
    # tolerance is not a probability matrix and is rejected; the remaining
    # within-tolerance excursions are snapped back so the convex weights below
    # are exactly non-negative.
    tolerance = 1e-8
    if np.any(probability_values < -tolerance) or np.any(
        probability_values > 1 + tolerance
    ):
        raise ValueError("Fate probabilities must lie between 0 and 1.")

    probability_values = np.clip(probability_values, 0, 1)

    row_sums = probability_values.sum(axis=1)

    if np.any(row_sums <= 0):
        raise ValueError("Fate-probability rows must have a positive sum.")

    non_unit_rows = ~np.isclose(row_sums, 1.0, atol=1e-6, rtol=0)

    if non_unit_rows.any():
        logger.warning(
            "%d fate-probability rows do not sum to 1 and will be normalized "
            "before projection (maximum deviation: %.4g).",
            non_unit_rows.sum(),
            np.max(np.abs(row_sums[non_unit_rows] - 1.0)),
        )

    # The projection uses convex weights, therefore normalize each row.
    probability_values /= row_sums[:, None]

    if n_fates == 2:
        vertices = np.array([[-1.0, 0.0], [1.0, 0.0]])
    else:
        # Regular polygon: first vertex at the top, then clockwise in fate order, so
        # neighbouring states in fate_order are adjacent on the polygon.
        angles = np.pi / 2 - 2 * np.pi * np.arange(n_fates) / n_fates
        vertices = np.column_stack((np.cos(angles), np.sin(angles)))

        if n_fates > 3:
            logger.warning(
                "Plotting %d terminal states as a 2D circular projection. "
                "For more than three terminal states this representation is lossy.",
                n_fates,
            )

    # Probability-weighted vertex average: p=1 puts a cell on its vertex, an even mix at
    # the centroid.
    projected_coordinates = probability_values @ vertices
    plot_df = pd.DataFrame(
        projected_coordinates,
        columns=["projection_x", "projection_y"],
        index=prob_df.index,
    )

    color_values = None
    categorical_color = False
    hue_order = None
    legend_columns = 0

    if color_by is not None:
        color_values = adata.obs.loc[plot_df.index, color_by]
        # pandas counts bool as numeric; treat it as categorical (True/False).
        categorical_color = (
            not pd.api.types.is_numeric_dtype(color_values)
            or pd.api.types.is_bool_dtype(color_values)
        )

        if categorical_color:
            observed = color_values.astype("string")
            existing = set(observed.dropna())
            is_categorical = isinstance(color_values.dtype, pd.CategoricalDtype)
            if is_categorical:
                existing |= {str(c) for c in color_values.cat.categories}

            # Missing values get their own label; never merge them with a genuine value.
            # Deterministic fallbacks: "NA", "NA (missing)", "NA (missing 2)", ...
            missing_label = "NA"
            suffix = 0
            while missing_label in existing:
                suffix += 1
                missing_label = "NA (missing)" if suffix == 1 else f"NA (missing {suffix})"

            color_labels = observed.fillna(missing_label).astype(str)

            if is_categorical:
                # Respect the declared category order, restricted to categories present.
                present = set(color_labels)
                hue_order = [
                    str(category)
                    for category in color_values.cat.categories
                    if str(category) in present
                ]
                if observed.isna().any():
                    hue_order.append(missing_label)
            else:
                hue_order = color_labels.drop_duplicates().tolist()

            n_categories = len(hue_order)

            legend_columns = min(5, max(1, n_categories // 5))

            max_label_length = max(len(label) for label in hue_order)
            wrapped_length = (
                min(max_label_length, label_wrap)
                if label_wrap is not None
                else max_label_length
            )
            wrapped_lines = max(
                1,
                int(np.ceil(max_label_length / max(wrapped_length, 1))),
            )
            legend_rows = int(np.ceil(n_categories / legend_columns))

            if legend_height is None:
                legend_height = max(0.6, 0.20 * legend_rows * wrapped_lines)

    if figsize is None:
        figsize = (9.0, 7.0 + legend_height) if categorical_color else (7.0, 7.0)

    legend_ax = None

    if ax is not None:
        fig = ax.get_figure()
    elif categorical_color:
        fig = plt.figure(figsize=figsize)

        grid = fig.add_gridspec(
            2,
            1,
            height_ratios=(
                max(1.0, figsize[1] - legend_height),
                legend_height,
            ),
            hspace=0.0,
        )

        ax = fig.add_subplot(grid[0, 0])
        legend_ax = fig.add_subplot(grid[1, 0])
        legend_ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=figsize)

    if color_by is None:
        ax.scatter(
            plot_df["projection_x"],
            plot_df["projection_y"],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    elif not categorical_color:
        numeric_colors = pd.to_numeric(color_values, errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )

        # Only finite values define the color scale; NaN/inf cells are drawn in light gray
        # so they stay visible without corrupting the norm.
        finite_colors = np.isfinite(numeric_colors)
        x_coords = plot_df["projection_x"].to_numpy()
        y_coords = plot_df["projection_y"].to_numpy()

        if not finite_colors.all():
            if finite_colors.any():
                logger.warning(
                    "%d cells have non-finite '%s' values; they are drawn in light gray.",
                    (~finite_colors).sum(),
                    color_by,
                )

            ax.scatter(
                x_coords[~finite_colors],
                y_coords[~finite_colors],
                color="lightgray",
                s=point_size,
                alpha=alpha,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )

        if finite_colors.any():
            scatter = ax.scatter(
                x_coords[finite_colors],
                y_coords[finite_colors],
                c=numeric_colors[finite_colors],
                s=point_size,
                alpha=alpha,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            colorbar.set_label(color_by)
        else:
            logger.warning("'%s' has no finite values; all cells are drawn in gray.", color_by)

    else:
        # Private column name: color_by may equal "projection_x"/"projection_y" and would
        # otherwise overwrite the coordinates.
        plot_df["_sctoolbox_color"] = color_labels

        sns.scatterplot(
            data=plot_df,
            x="projection_x",
            y="projection_y",
            hue="_sctoolbox_color",
            hue_order=hue_order,
            s=point_size,
            alpha=alpha,
            linewidth=0,
            rasterized=True,
            ax=ax,
            zorder=2,
        )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()

        legend_labels = [
            textwrap.fill(label, width=label_wrap) if label_wrap is not None else label
            for label in labels
        ]
        legend_title = (
            textwrap.fill(color_by, width=label_wrap)
            if label_wrap is not None
            else color_by
        )

        # Separate axis for the legend: keeps a wide, multi-column legend below the
        # projection from shrinking or distorting the (equal-aspect) projection axes.
        # Caller-supplied axes have no such companion, so the legend goes on them.
        (legend_ax if legend_ax is not None else ax).legend(
            handles,
            legend_labels,
            title=legend_title,
            loc="center" if legend_ax is not None else "best",
            frameon=False,
            ncol=legend_columns,
            markerscale=legend_marker_scale,
        )

    if n_fates == 2:
        ax.plot(vertices[:, 0], vertices[:, 1], linewidth=1.5, color="0.55", zorder=1)
    else:
        closed_vertices = np.vstack([vertices, vertices[0]])
        ax.plot(
            closed_vertices[:, 0],
            closed_vertices[:, 1],
            linewidth=1.5,
            color="0.55",
            zorder=1,
        )

    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        s=100,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        zorder=4,
    )

    for fate_name, (x_coord, y_coord) in zip(fate_names, vertices):
        text_x, text_y = 1.2 * x_coord, 1.2 * y_coord

        fate_label = (
            textwrap.fill(fate_name, width=label_wrap)
            if label_wrap is not None
            else fate_name
        )

        ha = "left" if text_x > 0.15 else "right" if text_x < -0.15 else "center"
        va = "bottom" if text_y > 0.15 else "top" if text_y < -0.15 else "center"

        ax.text(
            text_x,
            text_y,
            fate_label,
            ha=ha,
            va=va,
            fontsize=10,
            fontweight="bold",
        )

    if show_equal_probability:
        equal_position = np.full(n_fates, 1 / n_fates) @ vertices

        ax.scatter(
            equal_position[0],
            equal_position[1],
            s=35,
            facecolors="white",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )

        reference_label = (
            "equal fate probability"
            if n_fates <= 3
            else "equal-probability reference"
        )

        # Explain the marker in a compact legend in the upper-left corner: cells always lie
        # inside the polygon, so this corner stays free of data for two to five fates.
        # A categorical legend drawn on the same axes is kept as a separate artist.
        reference_handle = Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            label=reference_label,
        )
        category_legend = ax.get_legend()
        ax.legend(
            handles=[reference_handle],
            loc="upper left",
            frameon=False,
            fontsize=8,
            handletextpad=0.4,
            borderaxespad=0.2,
        )
        if category_legend is not None:
            ax.add_artist(category_legend)

    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.30, 1.35)
    ax.axis("off")
    fig.tight_layout()

    plt.figure(fig.number)

    if save is not None:
        _save_figure(path=save)
    if report_save is not None:
        _save_figure(path=report_save, report=True)
    if show:
        plt.show()

    return fig, ax


@deco.log_anndata
@beartype
def plot_branch_gene_trend_panels(  # noqa: C901
    adata: AnnData,
    computed: dict[str, dict[str, Any]] | None = None,
    branches: str | Iterable[str] | None = None,
    genes: str | Iterable[str] | None = None,
    genes_per_cluster: int = 1,
    color: str | None = None,
    position_layer: str | None = None,
    masks_key: str | None = "branch_masks",
    gene_trend_key: str = "gene_trends",
    pseudotime_key: str = "palantir_pseudotime",
    ncols: int = 3,
    point_size: float = 8.0,
    alpha: float = 0.35,
    figsize: tuple[float, float] = (4.5, 3.5),
    trend_linewidth: float = 2.0,
    save: str | None = None,
    report_save: str | None = None,
    plot_format: str = "pdf",
    show: bool = True,
    trend_layer: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Plot branch-specific Palantir gene trends together with individual cells.

    Genes can be supplied manually or selected automatically from representative
    trend-cluster genes returned by :func:`plot_branch_gene_programs`. A single
    selected gene uses Palantir's native ``plot_trend`` representation, while
    multiple genes are combined into a shared panel figure.

    Parameters
    ----------
    adata : AnnData
        AnnData containing Palantir results and expression data.
    computed : dict[str, dict[str, Any]] | None, default=None
        Output of :func:`plot_branch_gene_programs`. Only used when
        ``genes=None``: representative genes are then taken from each branch's
        ``cluster_representatives``, so every plotted branch must be present and
        ``plot_branch_gene_programs`` must have run with
        ``plot_representative_genes=True``. Can be left ``None`` when ``genes``
        is given; it is ignored in that case.
    branches : str | Iterable[str] | None, default=None
        Branches to plot. If None, all branches with stored gene trends in
        ``adata.varm`` are used.
    genes : str | Iterable[str] | None, default=None
        Genes to plot. If None, representative genes are selected automatically.
    genes_per_cluster : int, default=1
        Number of representative genes selected per trend cluster.
    color : str | None, default=None
        ``adata.obs`` column or ``adata.var_names`` entry used to color
        individual cells, like ``color`` in ``scanpy.pl.embedding``.
    position_layer : str | None, default=None
        Expression layer used for individual cell positions.
    masks_key : str | None, default="branch_masks"
        Key in ``adata.obsm`` containing branch masks.
    gene_trend_key : str, default="gene_trends"
        Base key for branch-specific Palantir gene trends.
    pseudotime_key : str, default="palantir_pseudotime"
        Column in ``adata.obs`` containing Palantir pseudotime.
    ncols : int, default=3
        Maximum number of subplot columns.
    point_size : float, default=8.0
        Size of individual cell points.
    alpha : float, default=0.35
        Transparency of individual cell points.
    figsize : tuple[float, float], default=(4.5, 3.5)
        Width and height of one multi-gene panel.
    trend_linewidth : float, default=2.0
        Width of the Palantir trend line in multi-gene panels.
    save : str | None, default=None
        Filename prefix for saving figures.
    report_save : str | None, default=None
        Filename prefix for saving figures to the report directory.
    plot_format : str, default="pdf"
        File format used for figures written to ``save``. Figures written to
        ``report_save`` are always png.
    show : bool, default=True
        Whether to display generated figures.
    trend_layer : str | None, default=None
        Layer in ``adata.layers`` from which the gene trends were computed, i.e. the
        ``expression_key`` passed to ``palantir.presults.compute_gene_trends``.
        Palantir does not record this layer, so it has to be stated here. If it
        equals ``position_layer``, trend curves and cells share one y-scale.
        Otherwise (including ``None``), both keep separate y-scales, because their
        values are not on a comparable scale.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from branch name to generated figure, axes, genes, and selection mode.

    Raises
    ------
    ValueError
        If required data or plotting parameters are invalid.
    """
    if genes_per_cluster < 1:
        raise ValueError("genes_per_cluster must be >= 1.")
    if ncols < 1:
        raise ValueError("ncols must be >= 1.")
    if point_size <= 0:
        raise ValueError("point_size must be > 0.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must satisfy 0 <= alpha <= 1.")
    if trend_linewidth <= 0:
        raise ValueError("trend_linewidth must be > 0.")

    if position_layer is not None and position_layer not in adata.layers:
        raise ValueError(f"Layer '{position_layer}' not found in adata.layers.")
    if trend_layer is not None and trend_layer not in adata.layers:
        raise ValueError(f"Layer '{trend_layer}' not found in adata.layers.")

    # Trend curves and cell values are on one numeric scale only if both come from the
    # same layer; a shared axis for different layers would suggest a false agreement.
    share_y_scale = trend_layer is not None and trend_layer == position_layer
    if trend_layer is not None and not share_y_scale:
        logger.warning(
            "trend_layer '%s' differs from position_layer '%s'; trends and cells keep separate y-scales.",
            trend_layer,
            position_layer,
        )

    if masks_key is not None and masks_key not in adata.obsm:
        raise ValueError(f"Branch masks '{masks_key}' not found in adata.obsm.")
    if pseudotime_key not in adata.obs:
        raise ValueError(f"Pseudotime '{pseudotime_key}' not found in adata.obs.")
    if color is not None and color not in adata.obs.columns and color not in adata.var_names:
        raise ValueError(f"Color '{color}' not found in adata.obs or adata.var_names.")

    trend_prefix = f"{gene_trend_key}_"

    # Stored branch-specific trend tables define which branches can be plotted;
    # `computed` is only needed later for automatic representative-gene selection.
    if branches is None:
        branch_list = [
            key[len(trend_prefix):]
            for key in adata.varm
            if key.startswith(trend_prefix)
        ]
    elif isinstance(branches, str):
        branch_list = [branches]
    else:
        branch_list = list(branches)

    if not branch_list:
        raise ValueError(
            f"No branch-specific gene trends with prefix '{trend_prefix}' found for plotting."
        )

    if isinstance(genes, str):
        requested_genes = [genes]
    elif genes is not None:
        requested_genes = list(genes)
    else:
        requested_genes = []

    # Validate automatic-gene inputs before plotting so a missing branch cannot
    # fail midway after earlier branch figures have already been created or saved.
    if genes is None:
        if computed is None:
            raise ValueError(
                "Automatic gene selection (genes=None) requires computed results "
                "from plot_branch_gene_programs."
            )

        missing_computed_branches = [
            branch for branch in branch_list if branch not in computed
        ]
        if missing_computed_branches:
            raise ValueError(
                "Automatic gene selection is missing computed results for "
                f"branches: {missing_computed_branches}."
            )

    def _save_branch_figure(
        fig: Figure,
        prefix: str | None,
        branch_name: str,
        report: bool,
    ) -> None:
        """Save one branch figure, or do nothing if ``prefix`` is None.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        prefix : str | None
            Filename prefix; the file is written as
            ``{prefix}_branch-{branch_name}.{extension}``. If None, nothing is saved.
        branch_name : str
            Branch name; characters other than letters, digits, ``_`` and ``-``
            are replaced by ``_`` for the filename.
        report : bool
            If True, save as png to ``settings.report_dir``; otherwise use
            ``plot_format`` and ``settings.full_figure_prefix``.
        """
        if prefix is None:
            return

        safe_branch = re.sub(r"[^\w-]", "_", branch_name)
        # Report figures are png: the report deck does not use high-resolution
        # images. Every other figure follows `plot_format`, which defaults to the
        # more easily editable pdf.
        extension = "png" if report else plot_format
        path = f"{prefix}_branch-{safe_branch}.{extension}"
        plt.figure(fig.number)
        _save_figure(path=path, report=report)

    results: dict[str, dict[str, Any]] = {}

    # Each branch gets its own figure.
    for branch in branch_list:
        trend_key = f"{gene_trend_key}_{branch}"
        if trend_key not in adata.varm:
            raise ValueError(f"Branch trends '{trend_key}' not found in adata.varm.")

        trends = adata.varm[trend_key]

        if not isinstance(trends, pd.DataFrame):
            raise ValueError(
                f"Branch trends '{trend_key}' must be a DataFrame in adata.varm."
            )

        # Manual genes are plotted for every branch as given; otherwise take the
        # top `genes_per_cluster` representatives of each trend cluster.
        if genes is not None:
            branch_genes = requested_genes.copy()
            selection_mode = "manual"
        else:
            representatives = computed[branch]["cluster_representatives"]
            if not representatives:
                raise ValueError(
                    f"No representative genes available for branch '{branch}'. "
                    "Provide genes manually or run plot_branch_gene_programs with "
                    "plot_representative_genes=True."
                )

            # Preserve the trend-cluster order from plot_branch_gene_programs().
            cluster_order = computed[branch]["centroid_z"].index.astype(str).tolist()

            branch_genes = []
            for cluster in cluster_order:
                cluster_genes = representatives.get(str(cluster), [])
                branch_genes.extend(str(gene) for gene in cluster_genes[:genes_per_cluster])

            selection_mode = "automatic"

        # Genes without a stored trend for this branch are skipped with a warning,
        # so one missing gene does not stop the remaining panels.
        available_genes = set(map(str, trends.index))
        missing_genes = [gene for gene in branch_genes if gene not in available_genes]

        if missing_genes:
            logger.warning(
                "Branch '%s': skipping genes without stored trends: %s",
                branch,
                missing_genes,
            )

        branch_genes = [gene for gene in branch_genes if gene in available_genes]
        if not branch_genes:
            logger.warning("Branch '%s': no genes available for plotting.", branch)
            continue

        # A single gene needs no panel grid: Palantir creates the figure itself,
        # including its own legend or colorbar.
        if len(branch_genes) == 1:
            gene = branch_genes[0]

            palantir.plot.plot_trend(
                adata,
                branch,
                gene,
                color=color,
                masks_key=masks_key,
                gene_trend_key=gene_trend_key,
                pseudo_time_key=pseudotime_key,
                position_layer=position_layer,
                s=point_size,
                alpha=alpha,
            )

            fig = plt.gcf()

            if share_y_scale:
                # Palantir draws the cells on a twin axis at the trend axis' position.
                trend_ax = fig.axes[0]
                cell_axes = [
                    axis for axis in fig.axes[1:]
                    if axis.collections and axis.get_position().bounds == trend_ax.get_position().bounds
                ]
                if cell_axes:
                    _share_trend_and_cell_scale(trend_ax, cell_axes[0])
                    trend_ax.set_ylabel(f"{gene} ({trend_layer})")

            axes = np.array(fig.axes, dtype=object)

            _save_branch_figure(fig, save, branch, report=False)
            _save_branch_figure(fig, report_save, branch, report=True)

            if show:
                plt.show()

            results[branch] = {
                "figure": fig,
                "axes": axes,
                "genes": branch_genes,
                "gene_selection": selection_mode,
            }
            continue

        ncols_eff = min(ncols, len(branch_genes))
        nrows_eff = int(np.ceil(len(branch_genes) / ncols_eff))

        # Non-numeric obs columns get one shared legend. Booleans count as categorical
        # although pandas treats them as numeric. Numeric columns and genes get one
        # shared colorbar instead.
        has_categorical_legend = (
            color is not None
            and color in adata.obs.columns
            and (
                not pd.api.types.is_numeric_dtype(adata.obs[color])
                or pd.api.types.is_bool_dtype(adata.obs[color])
            )
        )
        has_numeric_color = color is not None and not has_categorical_legend

        # The legend row is added on top of the panel height, so panels keep their size.
        legend_height = 1.4 if has_categorical_legend else 0.0
        figure_height = figsize[1] * nrows_eff + legend_height
        fig = plt.figure(figsize=(figsize[0] * ncols_eff, figure_height))

        if has_categorical_legend:
            grid = fig.add_gridspec(
                nrows_eff + 1,
                ncols_eff,
                height_ratios=[1.0] * nrows_eff + [legend_height / figsize[1]],
                hspace=0.25,
                wspace=0.30,
            )
            axes = np.array(
                [
                    fig.add_subplot(grid[row, col])
                    for row in range(nrows_eff)
                    for col in range(ncols_eff)
                ],
                dtype=object,
            )
            # Use one categorical legend instead of repeating it in every gene panel.
            legend_ax = fig.add_subplot(grid[-1, :])
            legend_ax.axis("off")
        else:
            grid = fig.add_gridspec(
                nrows_eff,
                ncols_eff,
                hspace=0.25,
                wspace=0.30,
            )
            axes = np.array(
                [
                    fig.add_subplot(grid[row, col])
                    for row in range(nrows_eff)
                    for col in range(ncols_eff)
                ],
                dtype=object,
            )
            legend_ax = None

        legend_handles = None
        legend_labels = None
        numeric_mappable = None

        for i, gene in enumerate(branch_genes):
            ax = axes[i]
            # `plot_trend()` creates a secondary expression axis for the cell values.
            # Track newly created axes so per-panel legends can be removed and the scatter
            # mappable can be reused for the shared legend or numeric colorbar.
            axes_before = set(fig.axes)

            palantir.plot.plot_trend(
                adata,
                branch,
                gene,
                color=color,
                masks_key=masks_key,
                gene_trend_key=gene_trend_key,
                ax=ax,
                pseudo_time_key=pseudotime_key,
                position_layer=position_layer,
                color_bar_bounds=None,
                s=point_size,
                alpha=alpha,
            )

            # The first line on the main axis is Palantir's trend curve.
            if ax.lines:
                ax.lines[0].set_linewidth(trend_linewidth)

            # Axis labels only on the outer panels: x on the last row, y on the first column.
            row = i // ncols_eff
            col = i % ncols_eff
            last_row = (len(branch_genes) - 1) // ncols_eff

            ax.set_title(gene, fontsize=11)
            ax.set_xlabel("Pseudotime" if row == last_row else "", labelpad=2)
            ax.set_ylabel("Palantir trend" if col == 0 else "")
            ax.grid(alpha=0.2)

            new_axes = [axis for axis in fig.axes if axis not in axes_before]
            if new_axes:
                cell_ax = new_axes[0]
                cell_ax.set_title("")
                cell_ax.set_ylabel("Feature value" if col == ncols_eff - 1 else "")

                if share_y_scale:
                    _share_trend_and_cell_scale(ax, cell_ax)
                    ax.set_ylabel(f"Feature value ({trend_layer})" if col == 0 else "")

                if has_numeric_color and numeric_mappable is None and cell_ax.collections:
                    numeric_mappable = cell_ax.collections[0]

                legend = cell_ax.get_legend()
                if legend is not None:
                    if legend_handles is None:
                        legend_handles, legend_labels = cell_ax.get_legend_handles_labels()
                    legend.remove()

        # Hide unused cells of the last grid row.
        for i in range(len(branch_genes), len(axes)):
            axes[i].axis("off")

        # Reserve a fixed height (in inches) above the panels for the suptitle and the
        # panel titles; a fixed fraction of the figure is too little for tall grids.
        title_space_inches = 0.8
        layout_top = 1.0 - title_space_inches / figure_height

        fig.suptitle(
            f"Branch '{branch}' — Palantir feature trends",
            fontsize=14,
            y=1.0 - 0.15 / figure_height,
            va="top",
        )

        # List only the categories of the cells that are actually drawn (the branch
        # cells); handles and their colors are kept as Palantir created them.
        if legend_handles and legend_labels and masks_key is not None:
            masks = adata.obsm[masks_key]
            if isinstance(masks, pd.DataFrame) and branch in masks.columns:
                branch_cells = masks[branch].to_numpy(dtype=bool)
                drawn = set(adata.obs.loc[branch_cells, color].astype(str))
                kept = [
                    (handle, label)
                    for handle, label in zip(legend_handles, legend_labels)
                    if label in drawn
                ]
                legend_handles = [handle for handle, _ in kept]
                legend_labels = [label for _, label in kept]

        if legend_ax is not None and legend_handles and legend_labels:
            n_categories = len(legend_labels)

            if n_categories <= 6:
                legend_columns = n_categories
            elif n_categories <= 12:
                legend_columns = 4
            else:
                legend_columns = 5

            legend_ax.legend(
                legend_handles,
                legend_labels,
                title=color,
                loc="center",
                bbox_to_anchor=(0.5, 0.40),
                frameon=False,
                ncol=legend_columns,
                markerscale=2.5,
            )

        fig.subplots_adjust(top=layout_top)

        # One colorbar for all panels, built from the first panel's scatter.
        if has_numeric_color and numeric_mappable is not None:
            colorbar = fig.colorbar(
                numeric_mappable,
                ax=list(axes[: len(branch_genes)]),
                fraction=0.025,
                pad=0.03,
                shrink=0.75,
                aspect=30,
            )
            colorbar.set_label(color)

        _save_branch_figure(fig, save, branch, report=False)
        _save_branch_figure(fig, report_save, branch, report=True)

        if show:
            plt.show()

        results[branch] = {
            "figure": fig,
            "axes": axes[: len(branch_genes)],
            "genes": branch_genes,
            "gene_selection": selection_mode,
        }

    return results


@deco.log_anndata
@beartype
def plot_palantir_results(
    adata: AnnData,
    fate_prob_key: str = "palantir_fate_probabilities",
    pseudotime_key: str = "palantir_pseudotime",
    entropy_key: str = "palantir_entropy",
    method: str = "umap",
    ncols: int = 4,
    point_size: int | float = 30,
    save: str | None = None,
    report_save: str | None = None,
    show: bool = True,
    cmap: str = "viridis",
) -> np.ndarray:
    """Plot Palantir pseudotime, entropy and per-fate probabilities on one embedding.

    Terminal states are ordered by their highest fate probability, so the most
    informative panels come first. The probabilities are written to ``adata.obs``
    only for the duration of the plot and removed again afterwards, leaving
    ``adata`` unchanged.

    Parameters
    ----------
    adata : AnnData
        AnnData containing Palantir results.
    fate_prob_key : str, default="palantir_fate_probabilities"
        Key in ``adata.obsm`` containing cell-by-terminal-state fate probabilities.
    pseudotime_key : str, default="palantir_pseudotime"
        Column in ``adata.obs`` containing Palantir pseudotime.
    entropy_key : str, default="palantir_entropy"
        Column in ``adata.obs`` containing Palantir entropy.
    method : str, default="umap"
        Embedding used for plotting.
    ncols : int, default=4
        Maximum number of panel columns.
    point_size : int | float, default=30
        Size of individual cell points.
    save : str | None, default=None
        Path and filename appended to ``settings.full_figure_prefix``.
    report_save : str | None, default=None
        Path and filename appended to ``settings.report_dir``.
    show : bool, default=True
        Whether to display the figure.
    cmap : str, default="viridis"
        Continuous colormap for all panels. The embedding default would draw the
        lowest values in grey, which reads as "no data" for pseudotime, entropy and
        probabilities.

    Returns
    -------
    np.ndarray
        Array of axes, one per plotted panel.

    Raises
    ------
    ValueError
        If ``fate_prob_key`` is missing from ``adata.obsm`` or does not hold a
        DataFrame with terminal-state columns, or if ``adata.obs`` already
        contains a ``"fate_prob: <terminal state>"`` column.
    """
    if fate_prob_key not in adata.obsm:
        raise ValueError(f"Fate probabilities '{fate_prob_key}' not found in adata.obsm.")

    fate_probabilities = adata.obsm[fate_prob_key]

    if not isinstance(fate_probabilities, pd.DataFrame):
        raise ValueError(
            f"adata.obsm['{fate_prob_key}'] must be a DataFrame with terminal-state columns."
        )

    ordered_fates = fate_probabilities.max().sort_values(ascending=False).index.tolist()
    fate_columns = [f"fate_prob: {fate}" for fate in ordered_fates]

    # The temporary columns are dropped again below; never touch existing ones.
    existing = [column for column in fate_columns if column in adata.obs.columns]
    if existing:
        raise ValueError(f"adata.obs already contains column(s) {existing}; rename or remove them first.")

    adata.obs[fate_columns] = fate_probabilities[ordered_fates].to_numpy()

    try:
        axes = pl.plot_embedding(
            adata,
            method=method,
            color=[pseudotime_key, entropy_key, *fate_columns],
            ncols=ncols,
            show_title=True,
            size=point_size,
            frameon=False,
            color_map=cmap,
            save=save,
            report=report_save,
        )

        if show:
            plt.show()
    finally:
        # The probability columns are a plotting detail; never leave them behind.
        adata.obs.drop(columns=fate_columns, inplace=True)

    return axes


@deco.log_anndata
@beartype
def plot_branch_cell_selection(
    adata: AnnData,
    pseudotime_key: str = "palantir_pseudotime",
    fate_prob_key: str = "palantir_fate_probabilities",
    masks_key: str = "branch_masks",
    embedding_key: str = "X_umap",
    figsize: tuple[float, float] = (15.0, 5.0),
    save: str | None = None,
    report_save: str | None = None,
    show: bool = True,
) -> Figure:
    """Plot the cells selected for each Palantir branch along pseudotime.

    Parameters
    ----------
    adata : AnnData
        AnnData containing Palantir results and branch masks.
    pseudotime_key : str, default="palantir_pseudotime"
        Column in ``adata.obs`` containing Palantir pseudotime.
    fate_prob_key : str, default="palantir_fate_probabilities"
        Key in ``adata.obsm`` containing the fate probabilities.
    masks_key : str, default="branch_masks"
        Key in ``adata.obsm`` containing the branch masks.
    embedding_key : str, default="X_umap"
        Key in ``adata.obsm`` containing the embedding.
    figsize : tuple[float, float], default=(15.0, 5.0)
        Figure size.
    save : str | None, default=None
        Path and filename appended to ``settings.full_figure_prefix``.
    report_save : str | None, default=None
        Path and filename appended to ``settings.report_dir``.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    Figure
        Matplotlib figure containing one branch-selection panel per branch.
    """
    fig = palantir.plot.plot_branch_selection(
        adata,
        pseudo_time_key=pseudotime_key,
        fate_prob_key=fate_prob_key,
        masks_key=masks_key,
        fates=None,
        embedding_basis=embedding_key,
        figsize=figsize,
    )

    plt.figure(fig.number)

    if save is not None:
        _save_figure(path=save)
    if report_save is not None:
        _save_figure(path=report_save, report=True)
    if show:
        plt.show()

    return fig


@deco.log_anndata
@beartype
def plot_branch_trajectories(
    adata: AnnData,
    pseudotime_key: str = "palantir_pseudotime",
    entropy_key: str = "palantir_entropy",
    branch_key: str = "branch_selection",
    pseudotime_interval: tuple[float, float] = (0.0, 0.9),
    cmap: str = "viridis",
    figsize: tuple[float, float] = (15.0, 5.0),
    save: str | None = None,
    report_save: str | None = None,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot Palantir pseudotime, entropy and branch selection side by side.

    Parameters
    ----------
    adata : AnnData
        AnnData containing Palantir results and branch masks.
    pseudotime_key : str, default="palantir_pseudotime"
        Column in ``adata.obs`` used to color the pseudotime panel.
    entropy_key : str, default="palantir_entropy"
        Column in ``adata.obs`` used to color the entropy panel.
    branch_key : str, default="branch_selection"
        Cell coloring used for the branch-selection panel.
    pseudotime_interval : tuple[float, float], default=(0.0, 0.9)
        Pseudotime interval shown in each panel.
    cmap : str, default="viridis"
        Colormap used for the continuous pseudotime and entropy panels.
    figsize : tuple[float, float], default=(15.0, 5.0)
        Figure size.
    save : str | None, default=None
        Path and filename appended to ``settings.full_figure_prefix``.
    report_save : str | None, default=None
        Path and filename appended to ``settings.report_dir``.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    tuple[Figure, np.ndarray]
        Matplotlib figure and its three panel axes.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    palantir.plot.plot_trajectories(
        adata,
        pseudo_time_key=pseudotime_key,
        cell_color=pseudotime_key,
        pseudotime_interval=pseudotime_interval,
        ax=axes[0],
        scanpy_kwargs={"cmap": cmap},
    )

    axes[0].set_title("Pseudotime", fontsize=16)

    palantir.plot.plot_trajectories(
        adata,
        pseudo_time_key=pseudotime_key,
        cell_color=entropy_key,
        pseudotime_interval=pseudotime_interval,
        ax=axes[1],
        scanpy_kwargs={"cmap": cmap},
    )
    axes[1].set_title("Entropy", fontsize=16)

    palantir.plot.plot_trajectories(
        adata,
        pseudo_time_key=pseudotime_key,
        cell_color=branch_key,
        pseudotime_interval=pseudotime_interval,
        ax=axes[2],
    )
    axes[2].set_title("Branch Selection", fontsize=16)

    fig.tight_layout()
    plt.figure(fig.number)

    if save is not None:
        _save_figure(path=save)
    if report_save is not None:
        _save_figure(path=report_save, report=True)
    if show:
        plt.show()

    return fig, axes

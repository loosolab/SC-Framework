"""Tools for Hotspot."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt
import hotspot
import sctoolbox.plotting as pl
import gseapy as gp
import sctoolbox.utils.decorator as deco
import scanpy.external as sce

from anndata import AnnData
from IPython.display import display
from beartype import beartype
from beartype.typing import Optional, Any
from pathlib import Path
from sctoolbox import settings

logger = settings.logger


def was_subsetted(
    original_adata: AnnData,
    current_adata: AnnData,
) -> bool:
    """
    Check whether current_adata contains fewer cells than original_adata.

    This is useful to decide whether PCA/neighbors/UMAP should be recomputed.

    Returns
    -------
    bool
        True if current_adata contains fewer cells than original_adata,
        otherwise False.
    """
    return current_adata.n_obs < original_adata.n_obs


def ensure_pca(
    adata: AnnData,
    pca_key: str = "X_pca",
    n_comps: int = 50,
    use_highly_variable: bool = True,
    layer: str | None = None,
    svd_solver: str | None = None,
    random_state: int = 0,
    force_recompute: bool = False,
) -> str:
    """
    Ensure that a PCA representation exists in adata.obsm.

    If pca_key already exists and force_recompute=False, it is reused.
    Otherwise, PCA is computed and stored in adata.obsm[pca_key].

    Returns
    -------
    str
        Key under which the PCA representation is stored in adata.obsm.
    """

    if pca_key in adata.obsm and not force_recompute:
        existing_dims = adata.obsm[pca_key].shape[1]
        print(f"Existing PCA found: adata.obsm['{pca_key}'] with {existing_dims} dimensions.")
        return pca_key

    print(f"Computing PCA and storing it in adata.obsm['{pca_key}']...")

    key_added = None if pca_key == "X_pca" else pca_key

    sc.pp.pca(
        adata,
        n_comps=n_comps,
        use_highly_variable=use_highly_variable,
        layer=layer,
        svd_solver=svd_solver,
        random_state=random_state,
        key_added=key_added,
    )

    return pca_key


def ensure_harmony(
    adata: AnnData,
    batch_key: str | None,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    force_recompute: bool = False,
) -> str | None:
    """
    Ensure that a Harmony-corrected latent representation exists.

    Requires:
    - batch_key in adata.obs
    - PCA basis in adata.obsm[basis]
    - scanpy.external and harmonypy installed

    Returns
    -------
    str or None
        Key of the existing or newly computed Harmony representation.
        Returns None if no valid batch key is available or Harmony
        cannot be computed.

    Raises
    ------
    ValueError
        If basis is not present in adata.obsm.
    """

    if adjusted_basis in adata.obsm and not force_recompute:
        print(f"Existing Harmony representation found: adata.obsm['{adjusted_basis}'].")
        return adjusted_basis

    if batch_key is None:
        print("No batch_key provided. Harmony will not be computed.")
        return None

    if batch_key not in adata.obs.columns:
        print(f"batch_key '{batch_key}' not found in adata.obs. Harmony will not be computed.")
        return None

    if basis not in adata.obsm:
        raise ValueError(
            f"Basis '{basis}' not found in adata.obsm. "
            "Please run ensure_pca first."
        )

    print(
        f"Computing Harmony using batch_key='{batch_key}', "
        f"basis='{basis}', adjusted_basis='{adjusted_basis}'..."
    )

    try:
        sce.pp.harmony_integrate(
            adata,
            key=batch_key,
            basis=basis,
            adjusted_basis=adjusted_basis,
        )

        return adjusted_basis

    except Exception as e:
        print("Harmony could not be computed.")
        print(f"Reason: {e}")
        return None


def ensure_umap(
    adata: AnnData,
    use_rep: str,
    neighbors_key: str = "neighbors",
    n_neighbors: int = 30,
    n_pcs: int = 30,
    metric: str = "euclidean",
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    random_state: int = 0,
    force_recompute: bool = False,
) -> str:
    """
    Ensure that adata.obsm['X_umap'] exists.

    UMAP is used for visualization, not as the main Hotspot input.

    Returns
    -------
    str
        Key of the UMAP representation in adata.obsm, namely
        "X_umap".

    Raises
    ------
    ValueError
        If use_rep is not present in adata.obsm.
    """

    if "X_umap" in adata.obsm and not force_recompute:
        print("Existing UMAP found: adata.obsm['X_umap'].")
        return "X_umap"

    if use_rep not in adata.obsm:
        raise ValueError(
            f"use_rep='{use_rep}' not found in adata.obsm. "
            "Cannot compute UMAP."
        )

    print(f"Computing neighbors using representation: '{use_rep}'...")

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs if use_rep == "X_pca" else None,
        use_rep=use_rep,
        metric=metric,
        key_added=neighbors_key,
    )

    print("Computing UMAP...")

    sc.tl.umap(
        adata,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        random_state=random_state,
        neighbors_key=neighbors_key,
    )

    return "X_umap"


def ensure_representation_for_umap(
    adata: AnnData,
    use_rep: str = "X_pca",
    batch_key: str | None = None,
    pca_key: str = "X_pca",
    harmony_key: str = "X_pca_harmony",
    n_comps: int = 50,
    use_highly_variable: bool = True,
    layer: str | None = None,
    svd_solver: str | None = None,
    random_state: int = 0,
    force_recompute: bool = False,
) -> str:
    """
    Ensure that the representation used for UMAP exists.

    Automatically computable:
    - X_pca
    - X_pca_harmony

    Other representations, e.g. X_scVI or X_scanorama, must already exist.

    Returns
    -------
    str
        Key of the selected representation in adata.obsm.

    Raises
    ------
    KeyError
        If the requested representation is not present in adata.obsm.
    ValueError
        If the Harmony representation cannot be computed.
    """

    if use_rep == pca_key:
        return ensure_pca(
            adata,
            pca_key=pca_key,
            n_comps=n_comps,
            use_highly_variable=use_highly_variable,
            layer=layer,
            svd_solver=svd_solver,
            random_state=random_state,
            force_recompute=force_recompute,
        )

    if use_rep == harmony_key:
        ensure_pca(
            adata,
            pca_key=pca_key,
            n_comps=n_comps,
            use_highly_variable=use_highly_variable,
            layer=layer,
            svd_solver=svd_solver,
            random_state=random_state,
            force_recompute=force_recompute,
        )

        harmony_rep = ensure_harmony(
            adata,
            batch_key=batch_key,
            basis=pca_key,
            adjusted_basis=harmony_key,
            force_recompute=force_recompute,
        )

        if harmony_rep is None:
            raise ValueError(
                "Harmony representation could not be computed. "
                "Please check batch_key or use use_rep='X_pca'."
            )

        return harmony_rep

    if use_rep not in adata.obsm:
        raise KeyError(
            f"use_rep '{use_rep}' was not found in adata.obsm. "
            "Use an existing representation or choose 'X_pca' / 'X_pca_harmony'."
        )

    print(f"Using existing representation: adata.obsm['{use_rep}']")
    return use_rep


def ensure_umap_for_visualization(
    adata: AnnData,
    use_rep: str = "X_pca",
    batch_key: str | None = None,
    is_subset: bool = False,
    recompute_if_subset: bool = True,
    force_recompute_umap: bool = False,
    force_recompute_rep: bool = False,
    pca_key: str = "X_pca",
    harmony_key: str = "X_pca_harmony",
    n_comps: int = 50,
    use_highly_variable: bool = True,
    layer: str | None = None,
    svd_solver: str | None = None,
    neighbors_key: str = "neighbors_umap",
    n_neighbors: int = 30,
    n_pcs: int = 30,
    metric: str = "euclidean",
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    random_state: int = 0,
) -> str:
    """
    Ensure that UMAP exists for visualization.

    This function only computes PCA/Harmony/neighbors/UMAP if needed.
    It does not plot.

    Returns
    -------
    str
        Key of the UMAP representation stored in adata.obsm,
        namely "X_umap".
    """

    force_rep_effective = force_recompute_rep or (is_subset and recompute_if_subset)
    force_umap_effective = force_recompute_umap or (is_subset and recompute_if_subset)

    print(f"UMAP use_rep: {use_rep}")
    print(f"is_subset: {is_subset}")
    print(f"force_recompute_rep_effective: {force_rep_effective}")
    print(f"force_recompute_umap_effective: {force_umap_effective}")

    use_rep = ensure_representation_for_umap(
        adata,
        use_rep=use_rep,
        batch_key=batch_key,
        pca_key=pca_key,
        harmony_key=harmony_key,
        n_comps=n_comps,
        use_highly_variable=use_highly_variable,
        layer=layer,
        svd_solver=svd_solver,
        random_state=random_state,
        force_recompute=force_rep_effective,
    )

    umap_key = ensure_umap(
        adata,
        use_rep=use_rep,
        neighbors_key=neighbors_key,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        metric=metric,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        random_state=random_state,
        force_recompute=force_umap_effective,
    )

    return umap_key


def plot_umap(
    adata: AnnData,
    condition_column: str,
    ncols: int = 3,
    title: str | None = None,
    save: str | None = None,
    report: str | None = None,
    rasterize: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Plot UMAP using sctoolbox plotting style.

    Returns
    -------
    Any
        Plot object returned by pl.embedding.plot_embedding.

    Raises
    ------
    KeyError
        If condition_column is not present in adata.obs or
        adata.var_names.
    """

    if condition_column not in adata.obs.columns and condition_column not in adata.var_names:
        raise KeyError(
            f"'{condition_column}' was not found in adata.obs or adata.var_names."
        )

    if title is None:
        title = "UMAP\ncolored by " + condition_column

    return pl.embedding.plot_embedding(
        adata,
        method="umap",
        color=condition_column,
        ncols=ncols,
        suptitle=title,
        save=save,
        report=report,
        rasterize=rasterize,
        **kwargs,
    )


def ensure_hotspot_latent_representation(  # noqa: C901
    adata: AnnData,
    representation_settings: dict[str, Any],
    pca_settings: dict[str, Any],
    harmony_settings: dict[str, Any],
    is_subset: bool = False,
) -> str:
    """
    Ensure and select a latent representation for Hotspot.

    This function is independent from UMAP.

    Supported automatic computation:
    - X_pca
    - X_pca_harmony

    External representations such as X_scVI or X_scanorama are only used
    if they already exist in adata.obsm. They are not recomputed automatically.

    Returns
    -------
    str
        Key of the selected latent representation in ``adata.obsm``.

    Raises
    ------
    KeyError
        If the requested representation or a required configuration key
        is missing.
    ValueError
        If the requested representation cannot be computed or no suitable
        representation can be selected.
    """

    preferred_rep = representation_settings.get("preferred_rep", "auto")
    priority = representation_settings.get(
        "priority",
        ["X_pca_harmony", "X_scVI", "X_scanorama", "X_pca"],
    )

    compute_pca_if_missing = representation_settings.get("compute_pca_if_missing", True)
    compute_harmony_if_missing = representation_settings.get("compute_harmony_if_missing", True)

    recompute_if_subset = representation_settings.get("recompute_if_subset", True)

    force_recompute_pca = representation_settings.get("force_recompute_pca", False)
    force_recompute_harmony = representation_settings.get("force_recompute_harmony", False)

    allow_existing_external_after_subset = representation_settings.get(
        "allow_existing_external_after_subset",
        False,
    )

    pca_key = pca_settings["pca_key"]
    harmony_key = harmony_settings["adjusted_basis"]

    external_reps = {"X_scVI", "X_scanorama"}

    force_pca_effective = force_recompute_pca or (
        is_subset and recompute_if_subset
    )

    force_harmony_effective = force_recompute_harmony or (
        is_subset and recompute_if_subset
    )

    print(f"Preferred Hotspot latent representation: {preferred_rep}")
    print(f"is_subset: {is_subset}")
    print(f"force_pca_effective: {force_pca_effective}")
    print(f"force_harmony_effective: {force_harmony_effective}")

    # --------------------------------------------------------
    # Helper: compute PCA
    # --------------------------------------------------------
    def _compute_pca() -> str:
        return ensure_pca(
            adata,
            force_recompute=force_pca_effective,
            **pca_settings,
        )

    # --------------------------------------------------------
    # Helper: compute Harmony
    # --------------------------------------------------------
    def _compute_harmony() -> str:
        ensure_pca(
            adata,
            force_recompute=force_pca_effective,
            **pca_settings,
        )

        harmony_rep = ensure_harmony(
            adata,
            force_recompute=force_harmony_effective,
            **harmony_settings,
        )

        if harmony_rep is None:
            raise ValueError(
                "Harmony was requested but could not be computed. "
                "Please check batch_key or use preferred_rep='X_pca'."
            )

        return harmony_rep

    # --------------------------------------------------------
    # Case 1: User explicitly requests one representation
    # --------------------------------------------------------
    if preferred_rep != "auto":

        if preferred_rep == pca_key:
            return _compute_pca()

        if preferred_rep == harmony_key:
            return _compute_harmony()

        if preferred_rep in external_reps:
            if preferred_rep not in adata.obsm:
                raise KeyError(
                    f"Requested representation '{preferred_rep}' was not found "
                    "in adata.obsm. This tool cannot compute it automatically."
                )

            if is_subset and recompute_if_subset and not allow_existing_external_after_subset:
                raise ValueError(
                    f"'{preferred_rep}' exists, but the data was subsetted and "
                    "recompute_if_subset=True. This tool cannot recompute external "
                    "representations such as scVI or Scanorama automatically. "
                    "Either recompute it before this step, use preferred_rep='X_pca' "
                    "or 'X_pca_harmony', or set "
                    "allow_existing_external_after_subset=True."
                )

            print(f"Using existing external representation: adata.obsm['{preferred_rep}']")
            return preferred_rep

        if preferred_rep in adata.obsm:
            if is_subset and recompute_if_subset:
                print(
                    f"Warning: '{preferred_rep}' exists, but it is not automatically "
                    "recomputed by this tool. Using existing representation."
                )

            return preferred_rep

        raise KeyError(
            f"Requested representation '{preferred_rep}' was not found in adata.obsm."
        )

    # --------------------------------------------------------
    # Case 2: Auto mode with priority
    # --------------------------------------------------------
    for rep in priority:

        # PCA is recomputable
        if rep == pca_key:
            if rep in adata.obsm and not force_pca_effective:
                print(f"Using existing PCA representation: adata.obsm['{rep}']")
                return rep

            if compute_pca_if_missing:
                print("Using PCA as Hotspot latent representation.")
                return _compute_pca()

        # Harmony is recomputable
        if rep == harmony_key:
            if rep in adata.obsm and not force_harmony_effective:
                print(f"Using existing Harmony representation: adata.obsm['{rep}']")
                return rep

            if compute_harmony_if_missing:
                print("Using Harmony as Hotspot latent representation.")
                try:
                    return _compute_harmony()
                except ValueError as e:
                    print(f"Skipping Harmony: {e}")

        # External representations are only reusable
        if rep in external_reps:
            if rep in adata.obsm:
                if is_subset and recompute_if_subset and not allow_existing_external_after_subset:
                    print(
                        f"Skipping existing '{rep}' because data was subsetted and "
                        "external representations are not recomputed automatically."
                    )
                    continue

                print(f"Using existing external representation: adata.obsm['{rep}']")
                return rep

    raise ValueError(
        "No suitable Hotspot latent representation could be selected or computed. "
        "Please check preferred_rep, priority, or enable PCA/Harmony computation."
    )


def subset_adata(  # noqa: C901
    adata: AnnData,
    subset_filters: dict[str, Any] | None = None,
    copy: bool = True,
    drop_unused_categories: bool = True,
) -> AnnData:
    """
    Subset AnnData based on one or multiple columns in `adata.obs`.

    Parameters
    ----------
    adata
        AnnData object to subset.
    subset_filters
        Dictionary mapping obs column names to one or multiple values.

        Example
        -------
        {
            "sample": [3, 6],
            "subcelltype": ["Epcam+", "EPCAM-"]
        }

        Within one column, values are combined with OR.
        Across columns, filters are combined with AND.

    copy
        Whether to return a copy of the subsetted AnnData.
    drop_unused_categories
        Whether to remove unused categories from categorical obs columns.

    Returns
    -------
    AnnData
        Subsetted AnnData object.

    Raises
    ------
    KeyError
        If a column specified in subset_filters is not present in
        adata.obs.
    TypeError
        If subset_filters is not a dictionary.
    ValueError
        If requested values are missing, no values are provided, or the
        subsetting removes all cells.
    """
    if subset_filters is None or len(subset_filters) == 0:
        return adata.copy() if copy else adata

    if not isinstance(subset_filters, dict):
        raise TypeError("`subset_filters` must be a dictionary mapping obs columns to values.")

    missing_columns = [col for col in subset_filters if col not in adata.obs.columns]
    if missing_columns:
        raise KeyError(
            f"Columns not found in `adata.obs`: {missing_columns}. "
            f"Available columns are: {list(adata.obs.columns)}"
        )

    mask = pd.Series(True, index=adata.obs_names)

    for col, values in subset_filters.items():
        if values is None:
            continue

        if not isinstance(values, (list, tuple, set, pd.Index, np.ndarray)):
            values = [values]

        if len(values) == 0:
            raise ValueError(f"No values were provided for subset column '{col}'.")

        obs_values_str = adata.obs[col].astype(str)
        requested_values_str = [str(v) for v in values]

        available_values = sorted(obs_values_str.unique().tolist())
        missing_values = sorted(set(requested_values_str) - set(available_values))

        if missing_values:
            preview = available_values[:20]
            raise ValueError(
                f"Values {missing_values} were not found in `adata.obs['{col}']`. "
                f"Example available values: {preview}"
            )

        col_mask = obs_values_str.isin(requested_values_str)
        mask &= col_mask

    if mask.sum() == 0:
        raise ValueError("Subsetting removed all cells. Please check `subset_filters`.")

    adata_subset = adata[mask].copy() if copy else adata[mask]

    if drop_unused_categories:
        for col in adata_subset.obs.columns:
            if pd.api.types.is_categorical_dtype(adata_subset.obs[col]):
                adata_subset.obs[col] = adata_subset.obs[col].cat.remove_unused_categories()

    return adata_subset


def inspect_and_filter_genes_by_cell_detection(
    adata: AnnData,
    layer_key: str | None = "raw",
    min_cells: int = 50,
    thresholds_to_show: tuple[int, ...] = (10, 20, 30, 50, 100),
    filter_genes: bool = False,
    plot: bool = True,
    bins: int = 50,
    add_to_var: bool = True,
    var_column: str = "n_cells_detected",
    copy: bool = True,
) -> tuple[AnnData, pd.DataFrame, dict[str, Any]]:
    """
    Inspect and optionally filter genes based on the number of cells in which each gene is detected.

    A gene is considered detected in a cell if its expression/count value is > 0.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    layer_key : str or None
        Layer to use for gene detection counts.
        If None, adata.X is used.
    min_cells : int
        Minimum number of cells in which a gene must be detected.
    thresholds_to_show : list or tuple
        Thresholds for reporting how many genes would be kept.
    filter_genes : bool
        If False, only statistics and plots are shown.
        If True, genes detected in fewer than min_cells cells are removed.
    plot : bool
        Whether to plot a histogram of gene detection frequencies.
    bins : int
        Number of histogram bins.
    add_to_var : bool
        Whether to save detection counts in adata.var[var_column].
    var_column : str
        Column name for detection counts in adata.var.
    copy : bool
        If True, return a copied filtered AnnData object.
        If False and filter_genes=True, return a view.

    Returns
    -------
    adata_out : AnnData
        Original or filtered AnnData object, depending on filter_genes.
    gene_detection : pandas.DataFrame
        DataFrame with gene names and number of cells detected.
    summary : dict
        Summary information about filtering.

    Raises
    ------
    KeyError
        If layer_key is not present in adata.layers.
    """

    # --------------------------------------------------------
    # Select expression matrix
    # --------------------------------------------------------
    if layer_key is None:
        X = adata.X
        used_layer = "adata.X"
    else:
        if layer_key not in adata.layers:
            raise KeyError(
                f"Layer '{layer_key}' not found in adata.layers. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        X = adata.layers[layer_key]
        used_layer = f"adata.layers['{layer_key}']"

    print(f"Using {used_layer} for gene detection.")

    # --------------------------------------------------------
    # Count number of cells where each gene is detected
    # --------------------------------------------------------
    if sp.issparse(X):
        n_cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
    else:
        n_cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()

    gene_detection = pd.DataFrame(
        {
            "gene": adata.var_names,
            "n_cells_detected": n_cells_per_gene,
        },
        index=adata.var_names,
    )

    # --------------------------------------------------------
    # Save to adata.var if wanted
    # --------------------------------------------------------
    if add_to_var:
        adata.var[var_column] = n_cells_per_gene
        print(f"Added gene detection counts to adata.var['{var_column}'].")

    # --------------------------------------------------------
    # Show basic statistics
    # --------------------------------------------------------
    print("\nGene detection frequency summary:")
    display(gene_detection.head())
    print(gene_detection["n_cells_detected"].describe())

    # --------------------------------------------------------
    # Plot histogram
    # --------------------------------------------------------
    if plot:
        plt.figure(figsize=(7, 4))
        plt.hist(gene_detection["n_cells_detected"], bins=bins)
        plt.xlabel("Number of cells in which gene is detected")
        plt.ylabel("Number of genes")
        plt.title("Gene detection frequency")
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # Show threshold comparison
    # --------------------------------------------------------
    print("\nGenes kept at different min_cells thresholds:")

    threshold_summary = {}

    for threshold in thresholds_to_show:
        n_keep = int((gene_detection["n_cells_detected"] >= threshold).sum())
        threshold_summary[threshold] = n_keep
        print(f"min_cells={threshold}: keep {n_keep} / {adata.n_vars} genes")

    # --------------------------------------------------------
    # Apply filtering only if requested
    # --------------------------------------------------------
    keep_genes = n_cells_per_gene >= min_cells

    summary = {
        "used_layer": used_layer,
        "min_cells": min_cells,
        "n_genes_before": int(adata.n_vars),
        "n_genes_after": int(keep_genes.sum()),
        "n_genes_removed": int(adata.n_vars - keep_genes.sum()),
        "filter_genes": filter_genes,
        "threshold_summary": threshold_summary,
    }

    print("\nSelected filtering threshold:")
    print(
        f"min_cells={min_cells}: keep {summary['n_genes_after']} / "
        f"{summary['n_genes_before']} genes "
        f"and remove {summary['n_genes_removed']} genes."
    )

    if filter_genes:
        print("\nFiltering genes now...")

        adata_out = adata[:, keep_genes]

        if copy:
            adata_out = adata_out.copy()

        adata_out.uns["gene_detection_filtering"] = summary

        print(
            f"Filtered AnnData created: {adata_out.n_obs} cells x "
            f"{adata_out.n_vars} genes."
        )

    else:
        print("\nfilter_genes=False, so no genes were removed.")
        adata_out = adata

    return adata_out, gene_detection, summary


def run_hotspot(  # noqa: C901
    adata: AnnData,
    layer_key: str = "raw",
    latent_obsm_key: str | None = None,
    umi_counts_obs_key: str = "total_counts",
    model: str = "danb",
    n_neighbors: int = 30,
    weighted_graph: bool = False,
    jobs: int = 8,
    top_n: int | None = 2000,
    gene_selection_method: str = "fdr",
    autocorr_fdr_threshold: float | None = 0.05,
    min_gene_threshold: int = 30,
    core_only: bool = True,
    module_fdr_threshold: float = 0.05,
    compute_local_correlations: bool = True,
    compute_modules: bool = True,
    compute_module_scores: bool = True,
    attach_results_to_adata: bool = False,
    module_prefix: str = "Hotspot module",
    overwrite_obs_key: bool = False,
    overwrite_attached_results: bool = True,
) -> dict[str, Any]:
    """
    Run Hotspot analysis on an AnnData object.

    Parameters
    ----------
    adata
        AnnData object containing expression data and embeddings.
    layer_key
        Layer used as count input for Hotspot.
    latent_obsm_key
        Embedding key in `adata.obsm`. If None, common defaults are tried.
    umi_counts_obs_key
        Name of obs column containing per-cell UMI / total count values.
        If missing, it will be computed from `layer_key`.
    model
        Hotspot model, e.g. 'danb'.
    n_neighbors
        Number of neighbors for KNN graph construction.
    weighted_graph
        Whether to build a weighted KNN graph.
    jobs
        Number of parallel jobs.
    top_n
        Number of top genes to use for local correlations if
        `gene_selection_method='top_n_z'`.
    gene_selection_method
        Strategy for selecting genes for local correlations.
        Supported: 'top_n_z', 'fdr'
    autocorr_fdr_threshold
        FDR threshold for gene selection if `gene_selection_method='fdr'`.
    min_gene_threshold
        Minimum genes per module.
    core_only
        Whether to keep only core module genes.
    module_fdr_threshold
        FDR threshold used in module creation.
    compute_local_correlations
        Whether to compute local correlations.
    compute_modules
        Whether to create modules.
    compute_module_scores
        Whether to calculate module scores.
    attach_results_to_adata
        Whether to attach Hotspot results to `adata.obs`, `adata.var` and `adata.uns`.
    module_prefix
        Prefix for module score columns added to `adata.obs`.
    overwrite_obs_key
        Whether to overwrite existing `umi_counts_obs_key` in `adata.obs`.
    overwrite_attached_results
        Whether to overwrite already attached hotspot result columns in `adata`.

    Returns
    -------
    dict
        Dictionary with Hotspot object and result tables.

    Raises
    ------
    TypeError
        If ``adata`` is not an AnnData object.
    KeyError
        If the requested layer, latent representation, or a required result
        column is missing.
    ValueError
        If the selected configuration is invalid, all genes have zero
        variance, or the requested analysis steps are incompatible.
    """
    if not isinstance(adata, AnnData):
        raise TypeError("`adata` must be an AnnData object.")

    if layer_key not in adata.layers:
        raise KeyError(f"Layer '{layer_key}' not found in adata.layers.")

    if latent_obsm_key is None:
        for candidate in ("X_pca_harmony", "X_pca"):
            if candidate in adata.obsm:
                latent_obsm_key = candidate
                break

    if latent_obsm_key is None or latent_obsm_key not in adata.obsm:
        raise KeyError(
            "No valid latent representation found. Provide `latent_obsm_key` "
            "or ensure one of ['X_pca_harmony', 'X_pca'] exists in adata.obsm."
        )

    if umi_counts_obs_key not in adata.obs or overwrite_obs_key:
        adata.obs[umi_counts_obs_key] = np.asarray(adata.layers[layer_key].sum(axis=1)).ravel()

        # Remove genes with zero variance in the selected layer
    X = adata.layers[layer_key]

    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        gene_var = mean_sq - mean**2
    else:
        gene_var = np.asarray(X.var(axis=0)).ravel()

    keep_genes = gene_var > 0

    if keep_genes.sum() == 0:
        raise ValueError(
            "All genes have zero variance in the selected subset/layer. "
            "Please check your subsetting or input layer."
        )

    n_removed = int((~keep_genes).sum())
    if n_removed > 0:
        adata = adata[:, keep_genes].copy()

    hs = hotspot.Hotspot(
        adata,
        layer_key=layer_key,
        model=model,
        latent_obsm_key=latent_obsm_key,
        umi_counts_obs_key=umi_counts_obs_key,
    )

    hs.create_knn_graph(
        weighted_graph=weighted_graph,
        n_neighbors=n_neighbors,
    )

    autocorrelations = hs.compute_autocorrelations(jobs=jobs)

    selected_genes = None
    local_correlations = None
    modules = None
    module_scores = None

    if compute_local_correlations:
        if gene_selection_method == "top_n_z":
            if top_n is None:
                raise ValueError("`top_n` must be provided for gene_selection_method='top_n_z'.")
            if "Z" not in autocorrelations.columns:
                raise KeyError("Hotspot autocorrelation results do not contain column 'Z'.")
            selected_genes = autocorrelations.sort_values("Z", ascending=False).head(top_n).index

        elif gene_selection_method == "fdr":
            if autocorr_fdr_threshold is None:
                raise ValueError(
                    "`autocorr_fdr_threshold` must be provided for gene_selection_method='fdr'."
                )
            if "FDR" not in autocorrelations.columns:
                raise KeyError("Hotspot autocorrelation results do not contain column 'FDR'.")
            selected_genes = autocorrelations.loc[
                autocorrelations["FDR"] < autocorr_fdr_threshold
            ].index

        else:
            raise ValueError(
                "Unsupported gene_selection_method. Use one of: ['top_n_z', 'fdr']."
            )

        selected_genes = pd.Index(selected_genes)
        local_correlations = hs.compute_local_correlations(selected_genes, jobs=jobs)

    if compute_modules:
        modules = hs.create_modules(
            min_gene_threshold=min_gene_threshold,
            core_only=core_only,
            fdr_threshold=module_fdr_threshold,
        )

    if compute_module_scores:
        if not compute_modules:
            raise ValueError("`compute_module_scores=True` requires `compute_modules=True`.")
        module_scores = hs.calculate_module_scores()

    results = {
        "hotspot": hs,
        "autocorrelations": autocorrelations,
        "selected_genes": selected_genes,
        "local_correlations": local_correlations,
        "modules": modules,
        "module_scores": module_scores,
        "latent_obsm_key_used": latent_obsm_key,
        "umi_counts_obs_key_used": umi_counts_obs_key,
    }

    if attach_results_to_adata:
        # add module scores to obs
        if module_scores is not None:
            module_scores_obs = module_scores.copy()
            module_scores_obs.columns = [f"{module_prefix} {c}" for c in module_scores_obs.columns]

            if overwrite_attached_results:
                adata.obs = adata.obs.drop(columns=module_scores_obs.columns, errors="ignore")

            adata.obs = adata.obs.join(module_scores_obs)

        # add autocorrelation stats to var
        autocorr_cols = {
            "C": "hotspot_C",
            "Z": "hotspot_Z",
            "Pval": "hotspot_Pval",
            "FDR": "hotspot_FDR",
        }

        for src, dst in autocorr_cols.items():
            if src in autocorrelations.columns:
                if overwrite_attached_results or dst not in adata.var.columns:
                    adata.var[dst] = autocorrelations[src].reindex(adata.var_names)

        # add module assignments to var
        if modules is not None:
            module_series = pd.Series(modules, name="hotspot_module")
            if overwrite_attached_results or "hotspot_module" not in adata.var.columns:
                adata.var["hotspot_module"] = module_series.reindex(adata.var_names)

        # add metadata to uns
        adata.uns["hotspot"] = {
            "latent_obsm_key_used": latent_obsm_key,
            "umi_counts_obs_key_used": umi_counts_obs_key,
            "gene_selection_method": gene_selection_method,
            "autocorr_fdr_threshold": autocorr_fdr_threshold,
            "top_n": top_n,
            "n_neighbors": n_neighbors,
            "weighted_graph": weighted_graph,
            "model": model,
            "min_gene_threshold": min_gene_threshold,
            "core_only": core_only,
            "module_fdr_threshold": module_fdr_threshold,
            "n_selected_genes": None if selected_genes is None else len(selected_genes),
            "n_modules": None if modules is None else int(pd.Series(modules).loc[pd.Series(modules) != -1].nunique()),
            "module_prefix": module_prefix,
            "n_genes_input": int(gene_var.shape[0]),
            "n_zero_variance_genes_removed": n_removed,
            "n_genes_used": int(adata.n_vars),
        }

    return results


def summarize_modules(
    modules: pd.Series | None,
    autocorrelations: pd.DataFrame | None = None,
    top_n_genes: int = 5,
) -> pd.DataFrame:
    """
    Summarize Hotspot modules for easier biological interpretation.

    Parameters
    ----------
    modules
        Gene-to-module assignment.
    autocorrelations
        Hotspot autocorrelation table.
    top_n_genes
        Number of top genes per module to report.

    Returns
    -------
    pd.DataFrame
        Summary table with module size and representative genes.
    """
    if modules is None:
        return pd.DataFrame(columns=["module", "n_genes", "top_genes"])

    modules = pd.Series(modules).copy()
    modules = modules[modules != -1]

    if modules.empty:
        return pd.DataFrame(columns=["module", "n_genes", "top_genes"])

    summary_rows = []
    for module_id, genes in modules.groupby(modules):
        gene_list = list(genes.index)

        if autocorrelations is not None and "Z" in autocorrelations.columns:
            ranked = (
                autocorrelations.loc[autocorrelations.index.intersection(gene_list)]
                .sort_values("Z", ascending=False)
                .index.tolist()
            )
        else:
            ranked = gene_list

        top_genes = ", ".join(ranked[:top_n_genes])

        summary_rows.append(
            {
                "module": module_id,
                "n_genes": len(gene_list),
                "top_genes": top_genes,
            }
        )

    return pd.DataFrame(summary_rows).sort_values("module").reset_index(drop=True)


def save_hotspot_results(
    results: dict[str, Any],
    output_dir: str | Path,
    output_prefix: str = "hotspot",
    module_summary: pd.DataFrame | None = None,
    adata: AnnData | None = None,
    adata_path: str | Path | None = None,
) -> None:
    """
    Save Hotspot results to disk.

    Parameters
    ----------
    results
        Result dictionary returned by `run_hotspot`.
    output_dir
        Output directory.
    output_prefix
        Prefix for output files.
    module_summary
        Optional module summary table.
    adata
        Optional AnnData object to save.
    adata_path
        Optional path for saving AnnData.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    autocorrelations = results.get("autocorrelations")
    selected_genes = results.get("selected_genes")
    local_correlations = results.get("local_correlations")
    modules = results.get("modules")
    module_scores = results.get("module_scores")

    if autocorrelations is not None:
        autocorrelations.to_csv(output_dir / f"{output_prefix}_autocorrelations.csv.gz")

    if selected_genes is not None:
        pd.Series(selected_genes, name="gene").to_csv(
            output_dir / f"{output_prefix}_selected_genes.csv",
            index=False,
        )

    if local_correlations is not None:
        local_correlations.to_csv(output_dir / f"{output_prefix}_local_correlations.csv.gz")

    if modules is not None:
        pd.Series(modules, name="module").to_csv(
            output_dir / f"{output_prefix}_module_assignments.csv"
        )

    if module_scores is not None:
        module_scores.to_csv(output_dir / f"{output_prefix}_module_scores.csv.gz")

    if module_summary is not None:
        module_summary.to_csv(output_dir / f"{output_prefix}_module_summary.csv", index=False)

    if adata is not None and adata_path is not None:
        adata.write(adata_path)


def show_hotspot_results(  # noqa: C901
    adata: AnnData,
    results: dict[str, Any],
    fdr_threshold: float = 0.05,
    module_score_prefix: str = "Hotspot module",
    ncols: int = 3,
    output_prefix: str = "hotspot",
    plot_umap_modules: bool = True,
    plot_local_correlation_heatmap: bool = True,
    plot_autocorrelation_distribution: bool = True,
    plot_module_sizes: bool = True,
    save: bool = True,
    report: bool = True,
) -> dict[str, Any]:
    """
    Show and save the main Hotspot results.

    This function summarizes:
    - autocorrelation results
    - number of significant / selected genes
    - autocorrelation distributions
    - local correlation heatmap
    - module score UMAPs
    - module size table and barplot

    Parameters
    ----------
    adata : AnnData
        AnnData object used for Hotspot plotting. Must contain X_umap if UMAP plots are requested.
    results : dict
        Dictionary returned by the Hotspot workflow.
    fdr_threshold : float
        FDR threshold used to count significant genes.
    module_score_prefix : str
        Prefix for module score columns added to adata.obs.
    ncols : int
        Number of columns for UMAP module score plots.
    output_prefix : str
        Prefix for saved figure names.
    plot_umap_modules : bool
        Whether to plot module scores on UMAP.
    plot_local_correlation_heatmap : bool
        Whether to plot Hotspot local correlation heatmap.
    plot_autocorrelation_distribution : bool
        Whether to plot distributions of autocorrelation results.
    plot_module_sizes : bool
        Whether to show and plot module sizes.
    save : bool
        Whether to save figures.
    report : bool
        Whether to save report figures if supported by sctoolbox.

    Returns
    -------
    dict[str, Any]
        Summary containing information such as the number of tested genes,
        selected genes, modules, module sizes, and module-score columns.
    """
    summary = {}

    # ========================================================
    # 1. Autocorrelation results
    # ========================================================

    hs_results = results.get("autocorrelations", None)

    if hs_results is not None:
        print("Autocorrelation results")

        with pd.option_context("display.max.rows", 10, "display.max.columns", None):
            display(hs_results)

        n_total = hs_results.shape[0]

        if "FDR" in hs_results.columns:
            n_sig = int((hs_results["FDR"] < fdr_threshold).sum())
            print(f"{n_sig} / {n_total} genes have FDR < {fdr_threshold}")
        else:
            n_sig = None
            print("Column 'FDR' not found in autocorrelation results.")

        summary["n_total_genes_tested"] = int(n_total)
        summary["n_significant_genes"] = n_sig

        # ----------------------------------------------------
        # Autocorrelation distribution plots
        # ----------------------------------------------------

        if plot_autocorrelation_distribution:
            numeric_columns = [
                col for col in ["C", "Z", "Pval", "FDR"]
                if col in hs_results.columns
            ]

            for col in numeric_columns:
                values = hs_results[col].replace([np.inf, -np.inf], np.nan).dropna()

                plt.figure(figsize=(7, 4))
                plt.hist(values, bins=50)
                plt.xlabel(col)
                plt.ylabel("Number of genes")
                plt.title(f"Distribution of Hotspot {col}")
                plt.tight_layout()

                if save:
                    plt.savefig(f"{output_prefix}_autocorrelation_{col}_distribution.pdf", bbox_inches="tight")

                if report:
                    plt.savefig(f"{output_prefix}_autocorrelation_{col}_distribution.png", bbox_inches="tight", dpi=300)

                plt.show()

    else:
        print("No autocorrelation results found in results['autocorrelations'].")

    # ========================================================
    # 2. Selected genes
    # ========================================================

    selected_genes = results.get("selected_genes", None)

    if selected_genes is not None:
        print("\nSelected genes")
        print(f"Selected genes: {len(selected_genes)}")

        display(pd.Series(selected_genes, name="gene").head(20))

        summary["n_selected_genes"] = int(len(selected_genes))

    else:
        print("\nNo selected genes found in results['selected_genes'].")
        summary["n_selected_genes"] = None

    # ========================================================
    # 3. Local correlation results and heatmap
    # ========================================================

    local_corr = results.get("local_correlations", None)
    hs = results.get("hotspot", None)

    if local_corr is not None:
        print("\nLocal correlation matrix preview")

        with pd.option_context("display.max.rows", 10, "display.max.columns", 10):
            display(local_corr.iloc[:10, :10])

        summary["local_correlation_shape"] = local_corr.shape

        if plot_local_correlation_heatmap and hs is not None:
            print("\nPlotting local correlation heatmap...")

            try:
                hs.plot_local_correlations()

                if save:
                    plt.savefig(f"{output_prefix}_local_correlations_heatmap.pdf", bbox_inches="tight")

                if report:
                    plt.savefig(f"{output_prefix}_local_correlations_heatmap.png", bbox_inches="tight", dpi=300)

                plt.show()

            except Exception as e:
                print("Could not plot local correlation heatmap.")
                print(f"Reason: {e}")

    else:
        print("\nNo local correlations found in results['local_correlations'].")

    # ========================================================
    # 4. Modules and module sizes
    # ========================================================

    modules = results.get("modules", None)

    if modules is None and hs is not None and getattr(hs, "modules", None) is not None:
        modules = hs.modules

    if modules is not None:
        modules = pd.Series(modules).copy()
        modules.name = "Module"

        print("\nModules preview")

        with pd.option_context("display.max.rows", 10, "display.max.columns", None):
            display(modules)

        module_sizes = modules[modules != -1].value_counts().sort_index()
        module_sizes.name = "n_genes"

        print("\nModule sizes")
        display(module_sizes)

        summary["n_modules"] = int(module_sizes.shape[0])
        summary["module_sizes"] = module_sizes

        if plot_module_sizes:
            plt.figure(figsize=(10, 4))
            module_sizes.plot(kind="bar")
            plt.xlabel("Hotspot module")
            plt.ylabel("Number of genes")
            plt.title("Hotspot module sizes")
            plt.tight_layout()

            if save:
                plt.savefig(f"{output_prefix}_module_sizes.pdf", bbox_inches="tight")

            if report:
                plt.savefig(f"{output_prefix}_module_sizes.png", bbox_inches="tight", dpi=300)

            plt.show()

    else:
        print("\nNo modules found in results['modules'] or results['hotspot'].modules.")
        module_sizes = None
        summary["n_modules"] = None

    # ========================================================
    # 5. Module score UMAPs
    # ========================================================

    module_scores = results.get("module_scores", None)

    if module_scores is not None:
        print("\nModule scores")

        module_scores_plot = module_scores.copy()

        module_scores_plot.columns = [
            f"{module_score_prefix} {c}" for c in module_scores_plot.columns
        ]

        # align scores to adata cells
        module_scores_plot = module_scores_plot.reindex(adata.obs_names)

        if module_scores_plot.isna().any().any():
            print(
                "Warning: Some module scores are NaN after reindexing. "
                "Please check whether module_scores.index matches adata.obs_names."
            )

        # add module scores to adata.obs
        adata.obs[module_scores_plot.columns] = module_scores_plot

        with pd.option_context("display.max.rows", 5, "display.max.columns", None):
            display(module_scores_plot)

        summary["n_module_score_columns"] = int(module_scores_plot.shape[1])
        summary["module_score_columns"] = module_scores_plot.columns.tolist()

        if plot_umap_modules:
            if "X_umap" not in adata.obsm:
                print("No UMAP found in adata.obsm['X_umap']. Module UMAPs cannot be plotted.")
            else:
                print("\nPlotting module score UMAPs...")

                try:
                    _ = pl.embedding.plot_embedding(
                        adata,
                        method="umap",
                        color=module_scores_plot.columns.tolist(),
                        ncols=ncols,
                        suptitle="",
                        save=f"{output_prefix}_module_scores_umap.pdf" if save else None,
                        report=f"{output_prefix}_module_scores_umap.png" if report else None,
                        color_map="viridis",
                    )

                except ValueError as e:
                    print("Could not plot with color_map='viridis'. Trying without color_map.")
                    print(f"Reason: {e}")

                    _ = pl.embedding.plot_embedding(
                        adata,
                        method="umap",
                        color=module_scores_plot.columns.tolist(),
                        ncols=ncols,
                        suptitle="",
                        save=f"{output_prefix}_module_scores_umap.pdf" if save else None,
                        report=f"{output_prefix}_module_scores_umap.png" if report else None,
                    )

    else:
        print("\nNo module scores found in results['module_scores'].")

    # ========================================================
    # 6. Final summary
    # ========================================================

    print("\nHotspot result summary")
    summary_display = {
        key: value
        for key, value in summary.items()
        if key not in ["module_sizes", "module_score_columns"]
    }

    display(pd.Series(summary_display, name="value"))

    return summary


def add_hotspot_results_to_adata(  # noqa: C901
    adata: AnnData,
    results: dict[str, Any],
    prefix: str = "hotspot",
    fdr_threshold: float = 0.05,
    store_module_scores: bool = True,
    store_autocorrelations: bool = True,
    store_modules: bool = True,
    store_selected_genes: bool = True,
    store_local_correlations: bool = False,
    copy: bool = False,
) -> AnnData:
    """
    Add Hotspot results to AnnData before saving.

    Stores:
    - module scores in adata.obs
    - autocorrelation statistics in adata.var
    - module assignments in adata.var
    - selected gene flags in adata.var
    - summary information in adata.uns

    Does not store the Hotspot object itself.

    Returns
    -------
    AnnData
        AnnData object containing the selected Hotspot results.
    """

    if copy:
        adata = adata.copy()

    summary = {}

    # ========================================================
    # 1. Autocorrelation results -> adata.var
    # ========================================================

    autocorr = results.get("autocorrelations", None)

    if store_autocorrelations and autocorr is not None:
        autocorr_aligned = autocorr.reindex(adata.var_names)

        for col in autocorr_aligned.columns:
            adata.var[f"{prefix}_{col}"] = autocorr_aligned[col]

        if "FDR" in autocorr_aligned.columns:
            adata.var[f"{prefix}_significant"] = (
                autocorr_aligned["FDR"] < fdr_threshold
            ).fillna(False)

            summary["n_significant_genes"] = int(
                adata.var[f"{prefix}_significant"].sum()
            )

        summary["n_genes_tested"] = int(autocorr.shape[0])

    # ========================================================
    # 2. Selected genes -> adata.var
    # ========================================================

    selected_genes = results.get("selected_genes", None)

    if store_selected_genes and selected_genes is not None:
        selected_genes = set(selected_genes)

        adata.var[f"{prefix}_selected"] = adata.var_names.isin(selected_genes)

        summary["n_selected_genes"] = int(
            adata.var[f"{prefix}_selected"].sum()
        )

    # ========================================================
    # 3. Module assignments -> adata.var
    # ========================================================

    modules = results.get("modules", None)

    if modules is None:
        hs = results.get("hotspot", None)
        if hs is not None and getattr(hs, "modules", None) is not None:
            modules = hs.modules

    if store_modules and modules is not None:
        modules = pd.Series(modules).copy()
        modules_aligned = modules.reindex(adata.var_names)

        # Missing genes get -1 = unassigned
        module_values = (
            pd.to_numeric(modules_aligned, errors="coerce")
            .fillna(-1)
            .astype(int)
        )

        adata.var[f"{prefix}_module"] = module_values

        module_sizes = (
            module_values[module_values != -1]
            .value_counts()
            .sort_index()
        )

        module_sizes_df = module_sizes.rename("n_genes").to_frame()
        module_sizes_df.index.name = "module"

        adata.uns[f"{prefix}_module_sizes"] = module_sizes_df

        summary["n_modules"] = int(module_sizes.shape[0])

    # ========================================================
    # 4. Module scores -> adata.obs
    # ========================================================

    module_scores = results.get("module_scores", None)

    if store_module_scores and module_scores is not None:
        module_scores = module_scores.copy()

        module_scores.columns = [
            f"{prefix}_module_score_{c}" for c in module_scores.columns
        ]

        module_scores = module_scores.reindex(adata.obs_names)

        for col in module_scores.columns:
            adata.obs[col] = module_scores[col]

        summary["n_module_scores"] = int(module_scores.shape[1])

    # ========================================================
    # 5. Local correlations -> optional
    # ========================================================

    local_corr = results.get("local_correlations", None)

    if store_local_correlations and local_corr is not None:
        adata.uns[f"{prefix}_local_correlations"] = local_corr
        summary["local_correlations_stored"] = True
        summary["local_correlations_shape"] = tuple(local_corr.shape)
    else:
        summary["local_correlations_stored"] = False

    # ========================================================
    # 6. Store summary
    # ========================================================

    summary["fdr_threshold"] = float(fdr_threshold)
    summary["prefix"] = prefix

    adata.uns[f"{prefix}_summary"] = summary

    print("Hotspot results added to AnnData.")
    print(summary)

    return adata


@beartype
def run_hotspot_module_enrichment(  # noqa: C901
    adata: sc.AnnData,
    results: dict,
    gene_sets: Optional[list[str]] = None,
    organism: str = "Fish",
    min_module_size: int = 5,
    exclude_unassigned: bool = True,
    uppercase_genes: bool = False,
    overwrite: bool = False,
    inplace: bool = False,
    save_table: Optional[str] = "hotspot_module_enrichment.tsv",
    **kwargs: Any,
) -> Optional[sc.AnnData]:
    """
    Run Enrichr-based gene set enrichment for Hotspot modules.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object.
    results : dict
        Result dictionary from the Hotspot workflow.
        Expected keys include "modules".
    gene_sets : Optional[list[str]], default None
        Enrichr libraries to use. If None, GO_Biological_Process_2018 is used.
        For zebrafish/Fish, 2018 libraries are often more reliably available
        than newer GO_*_2023 libraries.
    organism : str, default "Fish"
        Organism passed to gseapy.enrichr.
    min_module_size : int, default 5
        Minimum number of genes required to test a module.
    exclude_unassigned : bool, default True
        Whether to exclude module label -1.
    uppercase_genes : bool, default False
        Whether to convert module gene names to uppercase before enrichment.
        Keep False for zebrafish symbols unless your library requires uppercase.
    overwrite : bool, default False
        If False and enrichment already exists in adata.uns, skip.
    inplace : bool, default False
        If True, modify adata in place. Otherwise return a modified copy.
    save_table : Optional[str], default "hotspot_module_enrichment.tsv"
        File name for saving the combined enrichment table in settings.table_dir.
        If None, no table is saved.
    **kwargs : Any
        Additional parameters forwarded to gseapy.enrichr.

    Returns
    -------
    Optional[sc.AnnData]
        Modified AnnData object if inplace=False, otherwise None.

    Raises
    ------
    ValueError
        If the module assignments or enrichment configuration are invalid.
    """

    if gene_sets is None:
        gene_sets = ["GO_Biological_Process_2018"]

    modified_adata = adata if inplace else adata.copy()

    hotspot_uns = modified_adata.uns.setdefault("sctoolbox", {}).setdefault("hotspot", {})

    if "module_enrichment" in hotspot_uns and not overwrite:
        logger.warning(
            "Hotspot module enrichment already exists. "
            "Skipping. Set overwrite=True to replace."
        )
        if inplace:
            return None
        return modified_adata

    # --------------------------------------------------------
    # Get modules
    # --------------------------------------------------------

    modules = results.get("modules", None)

    if modules is None:
        hs = results.get("hotspot", None)
        if hs is not None and getattr(hs, "modules", None) is not None:
            modules = hs.modules

    if modules is None:
        raise ValueError(
            "No module assignments found. Expected results['modules'] "
            "or results['hotspot'].modules."
        )

    modules = pd.Series(modules).copy()
    modules.name = "Module"

    if exclude_unassigned:
        modules = modules[modules != -1]

    if modules.empty:
        raise ValueError("No assigned Hotspot modules found after filtering.")

    # --------------------------------------------------------
    # Run enrichment per module
    # --------------------------------------------------------

    enrichment_results = {}
    all_rows = []

    default_kwargs = {
        "outdir": None,
        "cutoff": 1.0,
        "no_plot": True,
        "verbose": False,
    }
    kwargs = {**default_kwargs, **kwargs}

    module_ids = sorted(modules.dropna().unique())

    for i, module_id in enumerate(module_ids, start=1):
        genes = modules[modules == module_id].index.astype(str).tolist()

        if uppercase_genes:
            genes = [g.upper() for g in genes]

        logger.info(
            f"Running Enrichr for Hotspot module {module_id} "
            f"({i}/{len(module_ids)}), n_genes={len(genes)}."
        )

        if len(genes) < min_module_size:
            logger.warning(
                f"Skipping module {module_id}: only {len(genes)} genes "
                f"(< min_module_size={min_module_size})."
            )
            enrichment_results[int(module_id)] = pd.DataFrame()
            continue

        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=gene_sets,
                organism=organism,
                **kwargs,
            )

            df = enr.results.copy()

            if df is None or df.empty:
                enrichment_results[int(module_id)] = pd.DataFrame()
                continue

            df["module"] = int(module_id)
            df["n_module_genes"] = len(genes)

            enrichment_results[int(module_id)] = df
            all_rows.append(df)

        except Exception as e:
            logger.warning(f"Module {module_id} enrichment failed: {e}")
            enrichment_results[int(module_id)] = pd.DataFrame()

    if all_rows:
        enrichment_table = pd.concat(all_rows, ignore_index=True)

        sort_cols = [
            col for col in ["module", "Adjusted P-value", "P-value"]
            if col in enrichment_table.columns
        ]

        if sort_cols:
            enrichment_table = enrichment_table.sort_values(sort_cols)
    else:
        enrichment_table = pd.DataFrame()

    # --------------------------------------------------------
    # Store in adata.uns
    # --------------------------------------------------------

    module_sizes = modules.value_counts().sort_index()
    module_sizes.name = "n_genes"

    hotspot_uns["module_enrichment"] = {
        "method": "enrichr",
        "organism": organism,
        "gene_sets": gene_sets,
        "min_module_size": min_module_size,
        "exclude_unassigned": exclude_unassigned,
        "uppercase_genes": uppercase_genes,
        "module_sizes": module_sizes.to_frame(),
        "enrichment_table": enrichment_table,
    }

    logger.info(
        "Saving Hotspot module enrichment in "
        "adata.uns['sctoolbox']['hotspot']['module_enrichment']."
    )

    # --------------------------------------------------------
    # Save table
    # --------------------------------------------------------

    if save_table is not None and not enrichment_table.empty:
        outfile = Path(settings.table_dir) / save_table
        logger.info(f"Saving Hotspot module enrichment table to {outfile}")
        enrichment_table.to_csv(outfile, sep="\t", header=True, index=False)

    if inplace:
        return None

    return modified_adata


@beartype
def summarize_hotspot_module_enrichment(
    adata: sc.AnnData,
    top_n: int = 5,
    pvalue_column: str = "Adjusted P-value",
) -> pd.DataFrame:
    """
    Return the top enriched terms per Hotspot module.

    Returns
    -------
    pd.DataFrame
        Table containing the top enriched terms for each Hotspot module.
    """

    enrichment_info = (
        adata.uns
        .get("sctoolbox", {})
        .get("hotspot", {})
        .get("module_enrichment", {})
    )

    enrichment_table = enrichment_info.get("enrichment_table", None)

    if enrichment_table is None or enrichment_table.empty:
        logger.warning("No Hotspot module enrichment table found.")
        return pd.DataFrame()

    rows = []

    for module_id, df in enrichment_table.groupby("module"):
        if pvalue_column in df.columns:
            top_df = df.sort_values(pvalue_column).head(top_n).copy()
        else:
            top_df = df.head(top_n).copy()

        rows.append(top_df)

    if not rows:
        return pd.DataFrame()

    summary = pd.concat(rows, ignore_index=True)

    sort_cols = [
        col for col in ["module", pvalue_column]
        if col in summary.columns
    ]

    if sort_cols:
        summary = summary.sort_values(sort_cols)

    return summary


@deco.log_anndata
@beartype
def plot_hotspot_module_enrichment(  # noqa: C901
    adata: sc.AnnData,
    top_n: int = 5,
    pvalue_column: str = "Adjusted P-value",
    term_column: str = "Term",
    save: Optional[str] = "hotspot_module_enrichment.pdf",
    report: Optional[str] = "hotspot_module_enrichment.png",
    figsize: tuple[float, float] = (10, 8),
) -> pd.DataFrame:
    """
    Plot Hotspot module enrichment results.

    Creates:
    - top enriched terms table
    - module size table
    - dotplot of top enriched terms per module
    - module size barplot

    Returns
    -------
    pd.DataFrame
        Top enrichment terms per module.

    Raises
    ------
    ValueError
        If no valid enrichment results are available for plotting.
    """

    enrichment_info = (
        adata.uns
        .get("sctoolbox", {})
        .get("hotspot", {})
        .get("module_enrichment", {})
    )

    enrichment_table = enrichment_info.get("enrichment_table", None)
    module_sizes = enrichment_info.get("module_sizes", None)

    if enrichment_table is None or enrichment_table.empty:
        raise ValueError(
            "No Hotspot module enrichment results found. "
            "Run run_hotspot_module_enrichment first."
        )

    # --------------------------------------------------------
    # Top terms table
    # --------------------------------------------------------

    top_terms = summarize_hotspot_module_enrichment(
        adata,
        top_n=top_n,
        pvalue_column=pvalue_column,
    )

    print("Top enriched terms per Hotspot module")

    columns_to_show = [
        col for col in [
            "module",
            "Gene_set",
            term_column,
            pvalue_column,
            "P-value",
            "Overlap",
            "Genes",
        ]
        if col in top_terms.columns
    ]

    with pd.option_context("display.max.rows", 100, "display.max.columns", None):
        display(top_terms[columns_to_show])

    # --------------------------------------------------------
    # Module sizes
    # --------------------------------------------------------

    if module_sizes is not None:
        print("\nHotspot module sizes")
        display(module_sizes)

    # --------------------------------------------------------
    # Dotplot
    # --------------------------------------------------------

    plot_df = top_terms.copy()

    if pvalue_column not in plot_df.columns:
        raise ValueError(f"Column '{pvalue_column}' not found in enrichment table.")

    if term_column not in plot_df.columns:
        raise ValueError(f"Column '{term_column}' not found in enrichment table.")

    plot_df[pvalue_column] = pd.to_numeric(plot_df[pvalue_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[pvalue_column])

    if plot_df.empty:
        logger.warning("No valid adjusted p-values found for plotting.")
        return top_terms

    # avoid -log10(0)
    plot_df["minus_log10_padj"] = -np.log10(
        plot_df[pvalue_column].clip(lower=1e-300)
    )

    plot_df["module"] = plot_df["module"].astype(str)

    # Make labels readable but not too long
    plot_df["term_label"] = plot_df[term_column].astype(str).str.slice(0, 80)

    plt.figure(figsize=figsize)

    scatter = plt.scatter(
        x=plot_df["module"],
        y=plot_df["term_label"],
        s=plot_df["minus_log10_padj"] * 25,
        c=plot_df["minus_log10_padj"],
    )

    plt.xlabel("Hotspot module")
    plt.ylabel("Enriched term")
    plt.title("Hotspot module enrichment")
    plt.colorbar(scatter, label="-log10 adjusted p-value")
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches="tight")

    if report:
        plt.savefig(report, bbox_inches="tight", dpi=300)

    plt.show()

    # --------------------------------------------------------
    # Module size barplot
    # --------------------------------------------------------

    if module_sizes is not None:
        plt.figure(figsize=(10, 4))

        if isinstance(module_sizes, pd.DataFrame):
            size_series = module_sizes.iloc[:, 0]
        else:
            size_series = pd.Series(module_sizes)

        size_series.plot(kind="bar")
        plt.xlabel("Hotspot module")
        plt.ylabel("Number of genes")
        plt.title("Hotspot module sizes")
        plt.tight_layout()

        if save:
            size_save = save.replace(".pdf", "_module_sizes.pdf")
            plt.savefig(size_save, bbox_inches="tight")

        if report:
            size_report = report.replace(".png", "_module_sizes.png")
            plt.savefig(size_report, bbox_inches="tight", dpi=300)

        plt.show()

    return top_terms


def stringify_adata_keys_for_saving(adata: AnnData) -> AnnData:
    """
    Convert non-string keys/columns in obs, var and uns to strings so that sctoolbox.utils.adata.save_h5ad can handle them.

    Returns
    -------
    AnnData
        The modified AnnData object with string-compatible keys.
    """

    def stringify_nested_keys(obj: Any) -> Any:
        if isinstance(obj, dict):
            new_items = {}
            for k, v in list(obj.items()):
                new_k = str(k)
                new_items[new_k] = stringify_nested_keys(v)

            obj.clear()
            obj.update(new_items)
            return obj

        elif isinstance(obj, pd.DataFrame):
            obj.columns = obj.columns.map(str)
            obj.index = obj.index.map(str)
            return obj

        elif isinstance(obj, pd.Series):
            obj.index = obj.index.map(str)
            return obj

        else:
            return obj

    adata.obs.columns = adata.obs.columns.map(str)
    adata.var.columns = adata.var.columns.map(str)

    stringify_nested_keys(adata.uns)

    return adata

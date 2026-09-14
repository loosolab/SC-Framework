"""Tools for Hotspot."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt
import hotspot
import sctoolbox.plotting as pl
import sctoolbox.tools.qc_filter as qc_filter
import sctoolbox.utils as utils
import gseapy as gp
import sctoolbox.utils.decorator as deco
import scanpy.external as sce

from anndata import AnnData
from IPython.display import display
from beartype import beartype
from beartype.typing import Optional, Any, Sequence
from sctoolbox import settings
from sctoolbox.plotting.general import _save_figure

logger = settings.logger


@beartype
def was_subsetted(
    original_adata: AnnData,
    current_adata: AnnData,
) -> bool:
    """
    Check whether current_adata contains fewer cells than original_adata.

    This is useful to decide whether PCA/neighbors/UMAP should be recomputed.

    Parameters
    ----------
    original_adata : AnnData
        AnnData object before subsetting.
    current_adata : AnnData
        AnnData object after potential subsetting.

    Returns
    -------
    bool
        True if current_adata contains fewer cells than original_adata,
        otherwise False.
    """
    return current_adata.n_obs < original_adata.n_obs


@beartype
def ensure_pca(
    adata: AnnData,
    pca_key: str = "X_pca",
    n_comps: int = 50,
    use_highly_variable: bool = True,
    layer: Optional[str] = None,
    svd_solver: Optional[str] = None,
    random_state: int = 0,
    force_recompute: bool = False,
) -> str:
    """
    Ensure that a PCA representation exists in adata.obsm.

    If pca_key already exists and force_recompute=False, it is reused.
    Otherwise, PCA is computed and stored in adata.obsm[pca_key].

    Parameters
    ----------
    adata : AnnData
        AnnData object to add the PCA representation to.
    pca_key : str, default "X_pca"
        Key in adata.obsm the PCA representation is stored under.
    n_comps : int, default 50
        Number of principal components to compute.
    use_highly_variable : bool, default True
        Whether to restrict the PCA to genes flagged in adata.var["highly_variable"].
    layer : Optional[str], default None
        Layer used for the PCA. If None, adata.X is used.
    svd_solver : Optional[str], default None
        SVD solver passed to scanpy. If None, scanpy chooses.
    random_state : int, default 0
        Random seed for reproducible results.
    force_recompute : bool, default False
        Whether to recompute even if pca_key already exists.

    Returns
    -------
    str
        Key under which the PCA representation is stored in adata.obsm.
    """

    if pca_key in adata.obsm and not force_recompute:
        existing_dims = adata.obsm[pca_key].shape[1]
        logger.info(
            f"Existing PCA found: adata.obsm['{pca_key}'] with {existing_dims} dimensions."
        )
        return pca_key

    logger.info(f"Computing PCA and storing it in adata.obsm['{pca_key}'].")

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


@beartype
def ensure_harmony(
    adata: AnnData,
    batch_key: str,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    force_recompute: bool = False,
) -> str:
    """
    Ensure that a Harmony-corrected latent representation exists in adata.obsm.

    If adjusted_basis already exists and force_recompute is False, it is reused
    and no existing representation is overwritten.

    Parameters
    ----------
    adata : AnnData
        AnnData object to add the Harmony representation to.
    batch_key : str
        Column in adata.obs used as batch variable.
    basis : str, default "X_pca"
        Key in adata.obsm used as input for Harmony.
    adjusted_basis : str, default "X_pca_harmony"
        Key in adata.obsm the Harmony representation is stored under.
    force_recompute : bool, default False
        Whether to recompute even if adjusted_basis already exists.

    Returns
    -------
    str
        Key under which the Harmony representation is stored in adata.obsm.

    Raises
    ------
    KeyError
        If batch_key is not in adata.obs or basis is not in adata.obsm.
    ImportError
        If harmonypy is not installed.
    """
    if adjusted_basis in adata.obsm and not force_recompute:
        logger.info(f"Existing Harmony representation found: adata.obsm['{adjusted_basis}'].")
        return adjusted_basis

    if batch_key not in adata.obs.columns:
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs.")

    if basis not in adata.obsm:
        raise KeyError(f"Basis '{basis}' not found in adata.obsm. Run ensure_pca first.")

    logger.info(f"Computing Harmony (batch_key='{batch_key}', basis='{basis}').")

    try:
        sce.pp.harmony_integrate(adata, key=batch_key, basis=basis, adjusted_basis=adjusted_basis)
    except ImportError as e:
        raise ImportError(
            "Harmony requires harmonypy, which is not installed. "
            "Install it with: pip install 'SC-Framework[hotspot]'"
        ) from e

    return adjusted_basis


@beartype
def ensure_umap(
    adata: AnnData,
    use_rep: str,
    pca_key: str = "X_pca",
    neighbors_key: str = "neighbors",
    n_neighbors: int = 30,
    n_pcs: int = 30,
    metric: str = "euclidean",
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    random_state: int = 0,
    umap_key: str = "X_umap",
    force_recompute: bool = False,
) -> str:
    """
    Ensure that a UMAP representation exists in adata.obsm.

    UMAP is used for visualization, not as the main Hotspot input.

    Parameters
    ----------
    adata : AnnData
        AnnData object to add the UMAP to.
    use_rep : str
        Key in adata.obsm used to build the neighbor graph.
    pca_key : str, default "X_pca"
        Key of the PCA representation. Only if use_rep equals this key is
        n_pcs applied; other representations are used in full.
    neighbors_key : str, default "neighbors"
        Key in adata.uns the neighbor graph is stored under.
    n_neighbors : int, default 30
        Number of neighbors for the graph.
    n_pcs : int, default 30
        Number of components used when use_rep is the PCA key.
    metric : str, default "euclidean"
        Distance metric for the neighbor graph.
    min_dist : float, default 0.5
        UMAP min_dist, controls how tightly cells are packed.
    spread : float, default 1.0
        UMAP spread, controls the overall scale of the layout.
    n_components : int, default 2
        Number of UMAP dimensions.
    random_state : int, default 0
        Random seed for reproducible results.
    umap_key : str, default "X_umap"
        Key in adata.obsm the UMAP is stored under. An existing
        representation under this key is reused unless force_recompute
        is True.
    force_recompute : bool, default False
        Whether to recompute even if umap_key already exists in adata.obsm.

    Returns
    -------
    str
        Key of the UMAP representation in adata.obsm.

    Raises
    ------
    ValueError
        If use_rep is not present in adata.obsm.
    """

    if umap_key in adata.obsm and not force_recompute:
        logger.info(f"Existing UMAP found: adata.obsm['{umap_key}'].")
        return umap_key

    if use_rep not in adata.obsm:
        raise ValueError(
            f"use_rep='{use_rep}' not found in adata.obsm. "
            "Cannot compute UMAP."
        )

    logger.info(f"Computing neighbors using representation: '{use_rep}'.")

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs if use_rep == pca_key else None,
        use_rep=use_rep,
        metric=metric,
        key_added=neighbors_key,
    )

    logger.info("Computing UMAP.")

    sc.tl.umap(
        adata,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        random_state=random_state,
        neighbors_key=neighbors_key,
        key_added=None if umap_key == "X_umap" else umap_key,
    )

    return umap_key


@beartype
def ensure_representation_for_umap(
    adata: AnnData,
    use_rep: str = "X_pca",
    batch_key: Optional[str] = None,
    pca_key: str = "X_pca",
    harmony_key: str = "X_pca_harmony",
    n_comps: int = 50,
    use_highly_variable: bool = True,
    layer: Optional[str] = None,
    svd_solver: Optional[str] = None,
    random_state: int = 0,
    force_recompute: bool = False,
) -> str:
    """
    Ensure that the representation used for UMAP exists.

    Automatically computable:
    - X_pca
    - X_pca_harmony

    Other representations, e.g. X_scVI or X_scanorama, must already exist.

    Parameters
    ----------
    adata : AnnData
        AnnData object to select or compute the representation for.
    use_rep : str, default "X_pca"
        Key of the representation to use.
    batch_key : Optional[str], default None
        Batch column used for Harmony. Required if use_rep is harmony_key.
    pca_key : str, default "X_pca"
        Key of the PCA representation.
    harmony_key : str, default "X_pca_harmony"
        Key of the Harmony representation.
    n_comps : int, default 50
        Number of principal components, forwarded to ensure_pca.
    use_highly_variable : bool, default True
        Whether to restrict the PCA to highly variable genes.
    layer : Optional[str], default None
        Layer used for the PCA. If None, adata.X is used.
    svd_solver : Optional[str], default None
        SVD solver passed to scanpy.
    random_state : int, default 0
        Random seed for reproducible results.
    force_recompute : bool, default False
        Whether to recompute the representation even if it already exists.

    Returns
    -------
    str
        Key of the selected representation in adata.obsm.

    Raises
    ------
    KeyError
        If the requested representation is not present in adata.obsm.
    ValueError
        If use_rep is the Harmony key but no batch_key was provided.
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
        if batch_key is None:
            raise ValueError(
                f"use_rep='{harmony_key}' requires a batch_key. "
                f"Provide one or use use_rep='{pca_key}'."
            )

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

        return ensure_harmony(
            adata,
            batch_key=batch_key,
            basis=pca_key,
            adjusted_basis=harmony_key,
            force_recompute=force_recompute,
        )

    if use_rep not in adata.obsm:
        raise KeyError(
            f"use_rep '{use_rep}' was not found in adata.obsm. "
            f"Use an existing representation or choose '{pca_key}' / '{harmony_key}'."
        )

    logger.info(f"Using existing representation: adata.obsm['{use_rep}']")
    return use_rep


@beartype
def ensure_umap_for_visualization(
    adata: AnnData,
    use_rep: str = "X_pca",
    batch_key: Optional[str] = None,
    is_subset: bool = False,
    recompute_if_subset: bool = True,
    force_recompute_umap: bool = False,
    force_recompute_rep: bool = False,
    pca_key: str = "X_pca",
    harmony_key: str = "X_pca_harmony",
    n_comps: int = 50,
    use_highly_variable: bool = True,
    layer: Optional[str] = None,
    svd_solver: Optional[str] = None,
    neighbors_key: str = "neighbors_umap",
    n_neighbors: int = 30,
    n_pcs: int = 30,
    metric: str = "euclidean",
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    random_state: int = 0,
    umap_key: str = "X_umap",
) -> str:
    """
    Ensure that UMAP exists for visualization.

    This function only computes PCA/Harmony/neighbors/UMAP if needed.
    It does not plot.

    Parameters
    ----------
    adata : AnnData
        AnnData object to compute the UMAP for.
    use_rep : str, default "X_pca"
        Representation used to build the neighbor graph.
    batch_key : Optional[str], default None
        Batch column used for Harmony. Required if use_rep is harmony_key.
    is_subset : bool, default False
        Whether adata is a subset of a larger object.
    recompute_if_subset : bool, default True
        Whether representation and UMAP are recomputed when is_subset is True.
    force_recompute_umap : bool, default False
        Whether to recompute the UMAP regardless of is_subset.
    force_recompute_rep : bool, default False
        Whether to recompute the representation regardless of is_subset.
    pca_key : str, default "X_pca"
        Key of the PCA representation.
    harmony_key : str, default "X_pca_harmony"
        Key of the Harmony representation.
    n_comps : int, default 50
        Number of principal components, forwarded to ensure_pca.
    use_highly_variable : bool, default True
        Whether to restrict the PCA to highly variable genes.
    layer : Optional[str], default None
        Layer used for the PCA. If None, adata.X is used.
    svd_solver : Optional[str], default None
        SVD solver passed to scanpy.
    neighbors_key : str, default "neighbors_umap"
        Key in adata.uns the neighbor graph is stored under.
    n_neighbors : int, default 30
        Number of neighbors for the graph.
    n_pcs : int, default 30
        Number of components used when use_rep is the PCA key.
    metric : str, default "euclidean"
        Distance metric for the neighbor graph.
    min_dist : float, default 0.5
        UMAP min_dist.
    spread : float, default 1.0
        UMAP spread.
    n_components : int, default 2
        Number of UMAP dimensions.
    random_state : int, default 0
        Random seed for reproducible results.
    umap_key : str, default "X_umap"
        Key in adata.obsm the UMAP is stored under.

    Returns
    -------
    str
        Key of the UMAP representation stored in adata.obsm.
    """

    force_rep_effective = force_recompute_rep or (is_subset and recompute_if_subset)
    force_umap_effective = force_recompute_umap or (is_subset and recompute_if_subset)

    logger.debug(
        f"UMAP: use_rep={use_rep}, is_subset={is_subset}, "
        f"force_rep={force_rep_effective}, force_umap={force_umap_effective}"
    )

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

    resolved_umap_key = ensure_umap(
        adata,
        use_rep=use_rep,
        pca_key=pca_key,
        neighbors_key=neighbors_key,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        metric=metric,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        random_state=random_state,
        umap_key=umap_key,
        force_recompute=force_umap_effective,
    )

    return resolved_umap_key


@beartype
def ensure_hotspot_latent_representation(  # noqa: C901
    adata: AnnData,
    representation_settings: dict[str, Any],
    pca_settings: dict[str, Any],
    harmony_settings: dict[str, Any],
    is_subset: bool = False,
) -> str:
    """
    Ensure and select a latent representation for Hotspot.

    This function is independent from UMAP. PCA and Harmony can be computed
    automatically. External representations such as X_scVI or X_scanorama are
    only reused if they already exist in adata.obsm.

    Two modes are supported, and they differ in how strict they are:

    - Explicit (``preferred_rep`` is a key): the requested representation is
      computed or reused, and any problem raises.
    - Automatic (``preferred_rep="auto"``): the entries in ``priority`` are
      tried in order, and an unusable entry is skipped rather than raised.

    Parameters
    ----------
    adata : AnnData
        AnnData object to select or compute the latent representation for.
    representation_settings : dict[str, Any]
        Selection behaviour. Recognised keys: "preferred_rep", "priority",
        "compute_pca_if_missing", "compute_harmony_if_missing",
        "recompute_if_subset", "force_recompute_pca", "force_recompute_harmony",
        "allow_existing_external_after_subset".
    pca_settings : dict[str, Any]
        Keyword arguments forwarded to ensure_pca. Must contain "pca_key".
    harmony_settings : dict[str, Any]
        Keyword arguments forwarded to ensure_harmony. Must contain
        "adjusted_basis" and "batch_key".
    is_subset : bool, default False
        Whether adata is a subset of a larger object. Combined with
        "recompute_if_subset" this forces recomputation of PCA and Harmony.

    Returns
    -------
    str
        Key of the selected latent representation in adata.obsm.

    Raises
    ------
    KeyError
        If a requested representation is missing from adata.obsm and cannot be
        computed, or if a required configuration key is missing.
    ValueError
        If the Harmony key is requested explicitly but no batch_key was given,
        if an external representation exists but the data was subsetted, or if
        no suitable representation can be selected.
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
        "allow_existing_external_after_subset", False
    )

    pca_key = pca_settings["pca_key"]
    harmony_key = harmony_settings["adjusted_basis"]
    batch_key = harmony_settings.get("batch_key")

    external_reps = {"X_scVI", "X_scanorama"}

    force_pca_effective = force_recompute_pca or (is_subset and recompute_if_subset)
    force_harmony_effective = force_recompute_harmony or (is_subset and recompute_if_subset)

    logger.debug(
        f"Hotspot latent representation: preferred_rep={preferred_rep}, "
        f"is_subset={is_subset}, force_pca={force_pca_effective}, "
        f"force_harmony={force_harmony_effective}"
    )

    harmony_possible = batch_key is not None and batch_key in adata.obs.columns

    # --------------------------------------------------------
    # Case 1: User explicitly requests one representation
    # --------------------------------------------------------
    if preferred_rep != "auto":

        if preferred_rep == pca_key:
            return ensure_pca(adata, force_recompute=force_pca_effective, **pca_settings)

        if preferred_rep == harmony_key:
            if batch_key is None:
                raise ValueError(
                    f"preferred_rep='{harmony_key}' requires a batch_key in "
                    f"harmony_settings. Provide one or use preferred_rep='{pca_key}'."
                )
            ensure_pca(adata, force_recompute=force_pca_effective, **pca_settings)
            return ensure_harmony(
                adata, force_recompute=force_harmony_effective, **harmony_settings
            )

        if preferred_rep not in adata.obsm:
            raise KeyError(
                f"Requested representation '{preferred_rep}' was not found in "
                "adata.obsm. This tool cannot compute it automatically."
            )

        if (
            preferred_rep in external_reps
            and is_subset
            and recompute_if_subset
            and not allow_existing_external_after_subset
        ):
            raise ValueError(
                f"'{preferred_rep}' exists, but the data was subsetted and "
                "recompute_if_subset=True. This tool cannot recompute external "
                "representations such as scVI or Scanorama automatically. Either "
                "recompute it before this step, use preferred_rep='X_pca' or "
                f"'{harmony_key}', or set allow_existing_external_after_subset=True."
            )

        if is_subset and recompute_if_subset:
            logger.warning(
                f"'{preferred_rep}' exists but is not recomputed by this tool. "
                "Using the existing representation."
            )

        logger.info(f"Using existing representation: adata.obsm['{preferred_rep}']")
        return preferred_rep

    # --------------------------------------------------------
    # Case 2: Auto mode with priority
    # --------------------------------------------------------
    for rep in priority:

        # PCA is recomputable
        if rep == pca_key:
            if rep in adata.obsm and not force_pca_effective:
                logger.info(f"Using existing PCA representation: adata.obsm['{rep}']")
                return rep
            if not compute_pca_if_missing:
                continue
            logger.info("Using PCA as Hotspot latent representation.")
            return ensure_pca(adata, force_recompute=force_pca_effective, **pca_settings)

        # Harmony is recomputable
        if rep == harmony_key:
            if rep in adata.obsm and not force_harmony_effective:
                logger.info(f"Using existing Harmony representation: adata.obsm['{rep}']")
                return rep
            if not compute_harmony_if_missing:
                continue
            if not harmony_possible:
                logger.info("Skipping Harmony: no usable batch_key.")
                continue
            logger.info("Using Harmony as Hotspot latent representation.")
            ensure_pca(adata, force_recompute=force_pca_effective, **pca_settings)
            return ensure_harmony(
                adata, force_recompute=force_harmony_effective, **harmony_settings
            )

        # External representations are only reusable
        if rep in external_reps and rep in adata.obsm:
            if is_subset and recompute_if_subset and not allow_existing_external_after_subset:
                logger.info(
                    f"Skipping existing '{rep}': data was subsetted and external "
                    "representations are not recomputed automatically."
                )
                continue
            logger.info(f"Using existing external representation: adata.obsm['{rep}']")
            return rep

    raise ValueError(
        "No suitable Hotspot latent representation could be selected or computed. "
        "Please check preferred_rep, priority, or enable PCA/Harmony computation."
    )


@beartype
def subset_adata(  # noqa: C901
    adata: AnnData,
    subset_filters: Optional[dict[str, Any]] = None,
    copy: bool = True,
    drop_unused_categories: bool = True,
) -> AnnData:
    """
    Subset AnnData based on one or multiple columns in `adata.obs`.

    Parameters
    ----------
    adata : AnnData
        AnnData object to subset.
    subset_filters : Optional[dict[str, Any]], default None
        Dictionary mapping obs column names to one or multiple values.

        Example
        -------
        {
            "sample": [3, 6],
            "subcelltype": ["Epcam+", "EPCAM-"]
        }

        Within one column, values are combined with OR.
        Across columns, filters are combined with AND.

    copy : bool, default True
        Whether to return a copy of the subsetted AnnData.
    drop_unused_categories : bool, default True
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
    applied_filters = {}

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

        applied_filters[col] = requested_values_str

        col_mask = obs_values_str.isin(requested_values_str)
        mask &= col_mask

    if mask.sum() == 0:
        raise ValueError("Subsetting removed all cells. Please check `subset_filters`.")

    adata_subset = qc_filter.filter_cells(
        adata,
        cells=adata.obs_names[mask].tolist(),
        invert=True,
        inplace=not copy,
        name="hotspot_subset",
        value={"subset_filters": applied_filters},
        overwrite=True,
    )

    if adata_subset is None:  # inplace=True returns None
        adata_subset = adata

    if drop_unused_categories:
        for col in adata_subset.obs.columns:
            if pd.api.types.is_categorical_dtype(adata_subset.obs[col]):
                adata_subset.obs[col] = adata_subset.obs[col].cat.remove_unused_categories()

    return adata_subset


@beartype
def inspect_and_filter_genes_by_cell_detection(
    adata: AnnData,
    layer_key: Optional[str] = "raw",
    min_cells: int = 50,
    thresholds_to_show: Sequence[int] = (10, 20, 30, 50, 100),
    filter_genes: bool = False,
    plot: bool = True,
    bins: int = 50,
    save: bool = True,
    report: bool = True,
    output_prefix: str = "hotspot",
    var_column: str = "n_cells_by_counts",
    recompute: bool = False,
    copy: bool = True,
) -> tuple[AnnData, pd.DataFrame, dict[str, Any]]:
    """
    Inspect and optionally filter genes based on the number of cells in which each gene is detected.

    A gene is considered detected in a cell if its expression/count value is > 0.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    layer_key : Optional[str], default "raw"
        Layer to use for gene detection counts.
        If None, adata.X is used.
    min_cells : int, default 50
        Minimum number of cells in which a gene must be detected.
        Genes seen in very few cells carry little information for the
        neighbourhood statistics and mostly add noise, so removing them can
        sharpen the Hotspot result. As a rule of thumb, start at roughly
        0.1-1% of the cells in the current (sub)set and check the histogram
        and the threshold table printed by this function before deciding.
        The default of 50 fits datasets of a few thousand to a few tens of
        thousands of cells; for smaller subsets a lower value is usually
        more appropriate. With filter_genes=False nothing is removed and
        this value only affects the reported summary.
    thresholds_to_show : Sequence[int], default (10, 20, 30, 50, 100)
        Thresholds for reporting how many genes would be kept.
    filter_genes : bool, default False
        If False, only statistics and plots are shown.
        If True, genes detected in fewer than min_cells cells are removed.
    plot : bool, default True
        Whether to plot a histogram of gene detection frequencies.
    bins : int, default 50
        Number of histogram bins.
    save : bool, default True
        Whether to save the histogram.
    report : bool, default True
        Whether to additionally write the histogram to sctoolbox.settings.report_dir.
    output_prefix : str, default "hotspot"
        Prefix for saved figure names.
    var_column : str, default "n_cells_by_counts"
        Column in adata.var holding the number of cells each gene is detected in.
        Reused if present. If missing, it is computed with
        sc.pp.calculate_qc_metrics, which also (re)writes the other QC columns in
        adata.var, e.g. "total_counts", "mean_counts" and "pct_dropout_by_counts".
        With a standard workflow the column comes from notebook 2 and nothing is
        recomputed.
    recompute : bool, default False
        Whether to recompute the detection counts even if var_column exists.
        Set this to True after subsetting: counts carried over from an earlier
        notebook refer to the full object and are systematically too high.
    copy : bool, default True
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
        used_layer = "adata.X"
    else:
        if layer_key not in adata.layers:
            raise KeyError(
                f"Layer '{layer_key}' not found in adata.layers. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        used_layer = f"adata.layers['{layer_key}']"

    logger.info(f"Using {used_layer} for gene detection.")

    # --------------------------------------------------------
    # Count number of cells where each gene is detected
    # --------------------------------------------------------
    if var_column in adata.var.columns and not recompute:
        logger.info(f"Using existing adata.var['{var_column}'] for gene detection.")
    else:
        logger.info(f"Computing gene detection counts from {used_layer}.")
        sc.pp.calculate_qc_metrics(
            adata, layer=layer_key, percent_top=None, log1p=False, inplace=True
        )

    n_cells_per_gene = adata.var[var_column].to_numpy()

    gene_detection = pd.DataFrame(
        {
            "gene": adata.var_names,
            var_column: n_cells_per_gene,
        },
        index=adata.var_names,
    )

    # --------------------------------------------------------
    # Show basic statistics
    # --------------------------------------------------------
    logger.info("Gene detection frequency summary:")
    display(gene_detection.head())
    display(gene_detection[var_column].describe())

    # --------------------------------------------------------
    # Plot histogram
    # --------------------------------------------------------
    if plot:
        plt.figure(figsize=(7, 4))
        plt.hist(gene_detection[var_column], bins=bins)
        plt.xlabel("Number of cells in which gene is detected")
        plt.ylabel("Number of genes")
        plt.title("Gene detection frequency")
        plt.tight_layout()

        if save:
            _save_figure(f"{output_prefix}_gene_detection_frequency.pdf", bbox_inches="tight")

        if report:
            _save_figure(f"{output_prefix}_gene_detection_frequency.png", report=True, bbox_inches="tight")

        plt.show()

    # --------------------------------------------------------
    # Show threshold comparison
    # --------------------------------------------------------
    logger.info("Genes kept at different min_cells thresholds:")

    threshold_summary = {}

    for threshold in thresholds_to_show:
        n_keep = int((gene_detection[var_column] >= threshold).sum())
        threshold_summary[threshold] = n_keep
        logger.info(f"min_cells={threshold}: keep {n_keep} / {adata.n_vars} genes")

    # --------------------------------------------------------
    # Apply filtering only if requested
    # --------------------------------------------------------
    keep_genes = n_cells_per_gene >= min_cells

    summary = {
        "used_layer": used_layer,
        "min_cells": min_cells,
        "n_genes_before": adata.n_vars,
        "n_genes_after": int(keep_genes.sum()),
        "n_genes_removed": int(adata.n_vars - keep_genes.sum()),
        "filter_genes": filter_genes,
        "threshold_summary": threshold_summary,
    }

    logger.info(
        f"Selected filtering threshold min_cells={min_cells}: "
        f"keep {summary['n_genes_after']} / {summary['n_genes_before']} genes, "
        f"remove {summary['n_genes_removed']} genes."
    )

    if filter_genes:
        logger.info("Filtering genes.")

        adata_out = qc_filter.filter_genes(
            adata,
            genes=adata.var_names[keep_genes].tolist(),
            invert=True,
            inplace=not copy,
            name="hotspot_gene_detection",
            value={"min_cells": min_cells, "used_layer": used_layer},
            overwrite=True,
        )

        if adata_out is None:  # inplace=True returns None
            adata_out = adata

        logger.info(
            f"Filtered AnnData: {adata_out.n_obs} cells x {adata_out.n_vars} genes."
        )

    else:
        logger.info("filter_genes=False, no genes were removed.")
        adata_out = adata

    return adata_out, gene_detection, summary


@beartype
def run_hotspot(  # noqa: C901
    adata: AnnData,
    layer_key: str = "raw",
    latent_obsm_key: Optional[str] = None,
    umi_counts_obs_key: str = "total_counts",
    model: str = "danb",
    n_neighbors: int = 30,
    weighted_graph: bool = False,
    jobs: Optional[int] = None,
    top_n: Optional[int] = 2000,
    gene_selection_method: str = "fdr",
    autocorr_fdr_threshold: Optional[float] = 0.05,
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
    adata : AnnData
        AnnData object containing expression data and embeddings.
    layer_key : str, default "raw"
        Layer used as count input for Hotspot.
    latent_obsm_key : Optional[str], default None
        Embedding key in adata.obsm. If None, "X_pca_harmony" and "X_pca" are
        tried in that order.
    umi_counts_obs_key : str, default "total_counts"
        Name of obs column containing per-cell UMI / total count values.
        If missing, it is computed from layer_key.
    model : str, default "danb"
        Hotspot model used for the null distribution of gene expression.
        Supported: "danb", "bernoulli", "normal", "none". See
        https://hotspot.readthedocs.io/ for details.
    n_neighbors : int, default 30
        Number of neighbors for KNN graph construction.
    weighted_graph : bool, default False
        Whether to build a weighted KNN graph.
    jobs : Optional[int], default None
        Number of parallel jobs. If None, sctoolbox.settings.get_threads() is used.
    top_n : Optional[int], default 2000
        Number of top genes to use for local correlations if
        gene_selection_method="top_n_z".
    gene_selection_method : str, default "fdr"
        Strategy for selecting genes for local correlations.
        Supported: "top_n_z", "fdr".
    autocorr_fdr_threshold : Optional[float], default 0.05
        FDR threshold for gene selection if gene_selection_method="fdr".
    min_gene_threshold : int, default 30
        Minimum genes per module.
    core_only : bool, default True
        Whether to keep only core module genes.
    module_fdr_threshold : float, default 0.05
        FDR threshold used in module creation.
    compute_local_correlations : bool, default True
        Whether to compute local correlations.
    compute_modules : bool, default True
        Whether to create modules.
    compute_module_scores : bool, default True
        Whether to calculate module scores. Requires compute_modules=True.
    attach_results_to_adata : bool, default False
        Whether to attach Hotspot results to adata.obs, adata.var and adata.uns.
    module_prefix : str, default "Hotspot module"
        Prefix for module score columns added to adata.obs.
    overwrite_obs_key : bool, default False
        Whether to overwrite an existing umi_counts_obs_key in adata.obs.
    overwrite_attached_results : bool, default True
        Whether to overwrite already attached hotspot result columns in adata.

    Returns
    -------
    dict[str, Any]
        Dictionary with Hotspot object and result tables.

    Raises
    ------
    KeyError
        If the requested layer, latent representation, or a required result
        column is missing.
    ValueError
        If the selected configuration is invalid, all genes have zero
        variance, or the requested analysis steps are incompatible.
    """
    if jobs is None:
        jobs = settings.get_threads()

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
        logger.info(f"Removing {n_removed} genes with zero variance in '{layer_key}'.")
        adata_hs = adata[:, keep_genes].copy()
    else:
        adata_hs = adata

    hs = hotspot.Hotspot(
        adata_hs,
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

        # add metadata to uns['sctoolbox']['hotspot']['run']
        utils.adata.add_uns_info(adata, key=["hotspot", "run"], value={
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
            # module label -1 means unassigned and is excluded from the count
            "n_modules": None if modules is None else int(pd.Series(modules).loc[pd.Series(modules) != -1].nunique()),
            "module_prefix": module_prefix,
            "n_genes_input": gene_var.shape[0],
            "n_zero_variance_genes_removed": n_removed,
            "n_genes_used": adata_hs.n_vars,
        })

    return results


@beartype
def show_hotspot_results(  # noqa: C901
    adata: AnnData,
    results: dict[str, Any],
    fdr_threshold: float = 0.05,
    module_score_prefix: str = "Hotspot module",
    color_map: Optional[str] = None,
    ncols: int = 3,
    n_selected_genes_preview: int = 20,
    hist_figsize: tuple[float, float] = (7.0, 4.0),
    hist_bins: int = 50,
    umap_key: str = "X_umap",
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
        AnnData object used for Hotspot plotting. Must contain the UMAP
        given by umap_key if UMAP plots are requested.
    results : dict[str, Any]
        Dictionary returned by run_hotspot.
    fdr_threshold : float, default 0.05
        FDR threshold used to count significant genes.
    module_score_prefix : str, default "Hotspot module"
        Prefix for module score columns added to adata.obs.
    color_map : Optional[str], default None
        Matplotlib colormap for the module score UMAPs. If None, the
        sctoolbox default is used.
    ncols : int, default 3
        Number of columns for UMAP module score plots.
    n_selected_genes_preview : int, default 20
        Number of selected genes shown in the preview table. The full list
        is available in results["selected_genes"].
    hist_figsize : tuple[float, float], default (7.0, 4.0)
        Size of the autocorrelation distribution histograms in inches.
    hist_bins : int, default 50
        Number of bins in the autocorrelation distribution histograms.
    umap_key : str, default "X_umap"
        Key in adata.obsm holding the UMAP used for the module score plots.
    output_prefix : str, default "hotspot"
        Prefix for saved figure names.
    plot_umap_modules : bool, default True
        Whether to plot module scores on UMAP.
    plot_local_correlation_heatmap : bool, default True
        Whether to plot Hotspot local correlation heatmap.
    plot_autocorrelation_distribution : bool, default True
        Whether to plot distributions of autocorrelation results.
    plot_module_sizes : bool, default True
        Whether to show and plot module sizes.
    save : bool, default True
        Whether to save figures.
    report : bool, default True
        Whether to additionally write the figures to sctoolbox.settings.report_dir.

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
        logger.info("Autocorrelation results")

        with pd.option_context("display.max.rows", 10, "display.max.columns", None):
            display(hs_results)

        n_total = hs_results.shape[0]

        if "FDR" in hs_results.columns:
            n_sig = int((hs_results["FDR"] < fdr_threshold).sum())
            logger.info(f"{n_sig} / {n_total} genes have FDR < {fdr_threshold}")
        else:
            n_sig = None
            logger.warning("Column 'FDR' not found in autocorrelation results.")

        summary["n_total_genes_tested"] = n_total
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

                plt.figure(figsize=hist_figsize)
                plt.hist(values, bins=hist_bins)
                plt.xlabel(col)
                plt.ylabel("Number of genes")
                plt.title(f"Distribution of Hotspot {col}")
                plt.tight_layout()

                if save:
                    _save_figure(f"{output_prefix}_autocorrelation_{col}_distribution.pdf", bbox_inches="tight")

                if report:
                    _save_figure(f"{output_prefix}_autocorrelation_{col}_distribution.png", report=True, bbox_inches="tight")

                plt.show()

    else:
        logger.warning("No autocorrelation results found in results['autocorrelations'].")

    # ========================================================
    # 2. Selected genes
    # ========================================================

    selected_genes = results.get("selected_genes", None)

    if selected_genes is not None:
        logger.info(f"Selected genes: {len(selected_genes)}")

        display(pd.Series(selected_genes, name="gene").head(n_selected_genes_preview))

        summary["n_selected_genes"] = len(selected_genes)

    else:
        logger.warning("No selected genes found in results['selected_genes'].")
        summary["n_selected_genes"] = None

    # ========================================================
    # 3. Local correlation results and heatmap
    # ========================================================

    local_corr = results.get("local_correlations", None)
    hs = results.get("hotspot", None)

    if local_corr is not None:
        logger.info("Local correlation matrix preview")

        display(local_corr.iloc[:10, :10])

        summary["local_correlation_shape"] = local_corr.shape

        if plot_local_correlation_heatmap and hs is not None:
            logger.info("Plotting local correlation heatmap...")

            try:
                hs.plot_local_correlations()
            except (ValueError, KeyError) as e:
                logger.warning(f"Could not plot local correlation heatmap: {e}")
            else:
                if save:
                    _save_figure(f"{output_prefix}_local_correlations_heatmap.pdf", bbox_inches="tight")

                if report:
                    _save_figure(f"{output_prefix}_local_correlations_heatmap.png", report=True, bbox_inches="tight")

                plt.show()

    else:
        logger.warning("No local correlations found in results['local_correlations'].")

    # ========================================================
    # 4. Modules and module sizes
    # ========================================================

    modules = results.get("modules", None)

    if modules is None and hs is not None and getattr(hs, "modules", None) is not None:
        modules = hs.modules

    if modules is not None:
        modules = pd.Series(modules).copy()
        modules.name = "Module"

        logger.info("Modules preview")

        with pd.option_context("display.max.rows", 10, "display.max.columns", None):
            display(modules)

        # Hotspot labels genes that could not be assigned to any module with -1
        module_sizes = modules[modules != -1].value_counts().sort_index()
        module_sizes.name = "n_genes"

        logger.info("Module sizes")
        display(module_sizes)

        summary["n_modules"] = module_sizes.shape[0]
        summary["module_sizes"] = module_sizes

        if plot_module_sizes:
            plt.figure(figsize=(10, 4))
            module_sizes.plot(kind="bar")
            plt.xlabel("Hotspot module")
            plt.ylabel("Number of genes")
            plt.title("Hotspot module sizes")
            plt.tight_layout()

            if save:
                _save_figure(f"{output_prefix}_module_sizes.pdf", bbox_inches="tight")

            if report:
                _save_figure(f"{output_prefix}_module_sizes.png", report=True, bbox_inches="tight")

            plt.show()

    else:
        logger.warning("No modules found in results['modules'] or results['hotspot'].modules.")
        module_sizes = None
        summary["n_modules"] = None

    # ========================================================
    # 5. Module score UMAPs
    # ========================================================

    module_scores = results.get("module_scores", None)

    if module_scores is not None:
        logger.info("Module scores")

        module_scores_plot = module_scores.copy()

        module_scores_plot.columns = [
            f"{module_score_prefix} {c}" for c in module_scores_plot.columns
        ]

        # align scores to adata cells
        module_scores_plot = module_scores_plot.reindex(adata.obs_names)

        if module_scores_plot.isna().any().any():
            logger.warning(
                "Some module scores are NaN after reindexing. "
                "Please check whether module_scores.index matches adata.obs_names."
            )

        # add module scores to adata.obs
        adata.obs[module_scores_plot.columns] = module_scores_plot

        with pd.option_context("display.max.rows", 5, "display.max.columns", None):
            display(module_scores_plot)

        summary["n_module_score_columns"] = module_scores_plot.shape[1]
        summary["module_score_columns"] = module_scores_plot.columns.tolist()

        if plot_umap_modules:
            if umap_key not in adata.obsm:
                logger.warning(f"No UMAP found in adata.obsm['{umap_key}']. Module UMAPs cannot be plotted.")
            else:
                logger.info("Plotting module score UMAPs...")

                extra = {} if color_map is None else {"color_map": color_map}

                _ = pl.embedding.plot_embedding(
                    adata,
                    method=umap_key,
                    color=module_scores_plot.columns.tolist(),
                    ncols=ncols,
                    suptitle="",
                    save=f"{output_prefix}_module_scores_umap.pdf" if save else None,
                    report=f"{output_prefix}_module_scores_umap.png" if report else None,
                    **extra,
                )

    else:
        logger.warning("No module scores found in results['module_scores'].")

    # ========================================================
    # 6. Final summary
    # ========================================================

    logger.info("Hotspot result summary")
    summary_display = {
        key: value
        for key, value in summary.items()
        if key not in ["module_sizes", "module_score_columns"]
    }

    display(pd.Series(summary_display, name="value"))

    return summary


@beartype
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

    Parameters
    ----------
    adata : AnnData
        AnnData object to add the results to.
    results : dict[str, Any]
        Result dictionary returned by run_hotspot.
    prefix : str, default "hotspot"
        Prefix for the keys added to obs, var and uns.
    fdr_threshold : float, default 0.05
        FDR threshold used to flag significant genes.
    store_module_scores : bool, default True
        Whether to store module scores in adata.obs.
    store_autocorrelations : bool, default True
        Whether to store autocorrelation statistics in adata.var.
    store_modules : bool, default True
        Whether to store module assignments in adata.var.
    store_selected_genes : bool, default True
        Whether to flag selected genes in adata.var.
    store_local_correlations : bool, default False
        Whether to store the local correlation matrix in adata.uns. This can be
        large and is skipped by default.
    copy : bool, default False
        Whether to modify a copy instead of the given object.

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

        summary["n_genes_tested"] = autocorr.shape[0]

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

        utils.adata.add_uns_info(adata, key=[prefix, "module_sizes"], value=module_sizes_df)

        summary["n_modules"] = module_sizes.shape[0]

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

        summary["n_module_scores"] = module_scores.shape[1]

    # ========================================================
    # 5. Local correlations -> optional
    # ========================================================

    local_corr = results.get("local_correlations", None)

    if store_local_correlations and local_corr is not None:
        utils.adata.add_uns_info(adata, key=[prefix, "local_correlations"], value=local_corr)
        summary["local_correlations_stored"] = True
        summary["local_correlations_shape"] = tuple(local_corr.shape)
    else:
        summary["local_correlations_stored"] = False

    # ========================================================
    # 6. Store summary
    # ========================================================

    summary["fdr_threshold"] = float(fdr_threshold)
    summary["prefix"] = prefix

    utils.adata.add_uns_info(adata, key=[prefix, "summary"], value=summary)

    logger.info(f"Hotspot results added to AnnData: {summary}")

    return adata


@beartype
def run_hotspot_module_enrichment(  # noqa: C901
    adata: AnnData,
    results: dict[str, Any],
    organism: str,
    gene_sets: list[str] = ["GO_Biological_Process_2018"],
    min_module_size: int = 5,
    exclude_unassigned: bool = True,
    uppercase_genes: bool = False,
    overwrite: bool = False,
    inplace: bool = False,
    save_table: Optional[str] = "hotspot_module_enrichment.tsv",
    **kwargs: Any,
) -> Optional[AnnData]:
    """
    Run Enrichr-based gene set enrichment for Hotspot modules.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    results : dict[str, Any]
        Result dictionary from the Hotspot workflow.
        Expected keys include "modules".
    organism : str
        Organism passed to gseapy.enrichr.
    gene_sets : list[str], default ["GO_Biological_Process_2018"]
        Enrichr libraries to use. Availability differs per organism; the
        current list is at https://maayanlab.cloud/Enrichr/#libraries
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
    Optional[AnnData]
        Modified AnnData object if inplace=False, otherwise None.

    Raises
    ------
    ValueError
        If the module assignments or enrichment configuration are invalid.
    """

    modified_adata = adata if inplace else adata.copy()

    if (
        utils.adata.in_uns(modified_adata, ["sctoolbox", "hotspot", "module_enrichment"])
        and not overwrite
    ):
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
        # Hotspot labels unassigned genes with -1; they form no coherent
        # gene program and are not tested for enrichment
        modules = modules[modules != -1]

    if modules.empty:
        raise ValueError("No assigned Hotspot modules found after filtering.")

    # --------------------------------------------------------
    # Run enrichment per module
    # --------------------------------------------------------

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
            continue

        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=gene_sets,
                organism=organism,
                **kwargs,
            )

            df = enr.results.copy()

            if df.empty:
                continue

            df["module"] = int(module_id)
            df["n_module_genes"] = len(genes)

            all_rows.append(df)

        except Exception as e:
            logger.warning(f"Module {module_id} enrichment failed: {e}")

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

    utils.adata.add_uns_info(modified_adata, key=["hotspot", "module_enrichment"], value={
        "method": "enrichr",
        "organism": organism,
        "gene_sets": gene_sets,
        "min_module_size": min_module_size,
        "exclude_unassigned": exclude_unassigned,
        "uppercase_genes": uppercase_genes,
        "module_sizes": module_sizes.to_frame(),
        "enrichment_table": enrichment_table,
    })

    logger.info(
        "Saving Hotspot module enrichment in "
        "adata.uns['sctoolbox']['hotspot']['module_enrichment']."
    )

    # --------------------------------------------------------
    # Save table
    # --------------------------------------------------------

    if save_table is not None and not enrichment_table.empty:
        outfile = settings.full_table_prefix + save_table
        logger.info(f"Saving Hotspot module enrichment table to {outfile}")
        enrichment_table.to_csv(outfile, sep="\t", header=True, index=False)

    if inplace:
        return None

    return modified_adata


@beartype
def summarize_hotspot_module_enrichment(
    adata: AnnData,
    top_n: int = 5,
    pvalue_column: str = "Adjusted P-value",
) -> pd.DataFrame:
    """
    Return the top enriched terms per Hotspot module.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the enrichment results in adata.uns.
    top_n : int, default 5
        Number of top terms to return per module.
    pvalue_column : str, default "Adjusted P-value"
        Column used to rank the enriched terms.

    Returns
    -------
    pd.DataFrame
        Table containing the top enriched terms for each Hotspot module.
    """

    key = ["sctoolbox", "hotspot", "module_enrichment"]
    enrichment_info = utils.adata.get_uns(adata, key) if utils.adata.in_uns(adata, key) else {}

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
    adata: AnnData,
    top_n: int = 5,
    pvalue_column: str = "Adjusted P-value",
    term_column: str = "Term",
    pseudocount: float = 1e-300,
    dot_size_factor: float = 25.0,
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

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the enrichment results in adata.uns.
    top_n : int, default 5
        Number of top terms shown per module.
    pvalue_column : str, default "Adjusted P-value"
        Column used to rank the enriched terms.
    term_column : str, default "Term"
        Column containing the term names.
    pseudocount : float, default 1e-300
        Added to all p-values before the log transform to avoid -log10(0).
    dot_size_factor : float, default 25.0
        Scales the dot area in the dotplot relative to -log10(p-value).
    save : Optional[str], default "hotspot_module_enrichment.pdf"
        File name for the saved figure. If None, the figure is not saved.
    report : Optional[str], default "hotspot_module_enrichment.png"
        File name for the report figure. If None, no report figure is written.
    figsize : tuple[float, float], default (10, 8)
        Size of the dotplot in inches.

    Returns
    -------
    pd.DataFrame
        Top enrichment terms per module.

    Raises
    ------
    ValueError
        If no valid enrichment results are available for plotting.
    """

    key = ["sctoolbox", "hotspot", "module_enrichment"]
    enrichment_info = utils.adata.get_uns(adata, key) if utils.adata.in_uns(adata, key) else {}

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

    logger.info("Top enriched terms per Hotspot module")

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
        logger.info("Hotspot module sizes")
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

    # add pseudocount to avoid -log10(0) without distorting spacing
    plot_df["minus_log10_padj"] = -np.log10(plot_df[pvalue_column] + pseudocount)

    plot_df["module"] = plot_df["module"].astype(str)

    # Make labels readable but not too long
    plot_df["term_label"] = plot_df[term_column].astype(str).str.slice(0, 80)

    plt.figure(figsize=figsize)

    scatter = plt.scatter(
        x=plot_df["module"],
        y=plot_df["term_label"],
        s=plot_df["minus_log10_padj"] * dot_size_factor,
        c=plot_df["minus_log10_padj"],
    )

    plt.xlabel("Hotspot module")
    plt.ylabel("Enriched term")
    plt.title("Hotspot module enrichment")
    plt.colorbar(scatter, label="-log10 adjusted p-value")
    plt.tight_layout()

    if save:
        _save_figure(save, bbox_inches="tight")

    if report:
        _save_figure(report, report=True, bbox_inches="tight")

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
            _save_figure(save.replace(".pdf", "_module_sizes.pdf"), bbox_inches="tight")

        if report:
            _save_figure(report.replace(".png", "_module_sizes.png"), report=True, bbox_inches="tight")

        plt.show()

    return top_terms


@beartype
def stringify_adata_keys_for_saving(adata: AnnData) -> AnnData:
    """
    Convert non-string keys/columns in obs, var and uns to strings so that sctoolbox.utils.adata.save_h5ad can handle them.

    Parameters
    ----------
    adata : AnnData
        AnnData object whose keys should be converted to strings.

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

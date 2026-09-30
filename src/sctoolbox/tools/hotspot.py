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
from beartype import beartype
from beartype.typing import Optional, Any, Sequence, Literal
from sctoolbox import settings
from sctoolbox.plotting.general import _save_figure

logger = settings.logger

try:
    # gseapy >= 1.3 raises its own exception classes
    from gseapy.enrichr import EnrichrError
except ImportError:  # pragma: no cover
    EnrichrError = ValueError

#: Label for genes without a clear module. Genes that Hotspot could not assign
#: to any module, or that were not part of the Hotspot run, are designated as
#: -1 in adata.var["hotspot_module"], which can be interpreted as n/a.
UNASSIGNED_MODULE = -1


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
        Key in adata.obsm the PCA is stored under. An existing PCA under this
        key is reused unless force_recompute is True.
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
            f"Existing PCA found: adata.obsm['{pca_key}'] with {existing_dims} "
            f"dimensions. n_comps={n_comps} is not applied to an existing "
            "representation; set force_recompute=True to recompute it."
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
    **kwargs: Any,
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
    **kwargs : Any
        Additional parameters forwarded to scanpy.external.pp.harmony_integrate
        and from there to harmonypy.run_harmony, e.g. theta, max_iter_harmony
        or random_state. See
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.external.pp.harmony_integrate.html

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
        sce.pp.harmony_integrate(
            adata, key=batch_key, basis=basis, adjusted_basis=adjusted_basis, **kwargs
        )
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
    neighbors_key: str = "neighbors_umap",
    n_neighbors: int = 30,
    n_pcs: Optional[int] = None,
    metric: Literal[
        "cityblock", "cosine", "euclidean", "l1", "l2", "manhattan",
        "braycurtis", "canberra", "chebyshev", "correlation", "dice",
        "hamming", "jaccard", "kulsinski", "mahalanobis", "minkowski",
        "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener",
        "sokalsneath", "sqeuclidean", "yule",
    ] = "euclidean",
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    random_state: int = 0,
    umap_key: str = "X_umap",
    force_recompute: bool = False,
) -> None:
    """
    Ensure that a UMAP representation exists in adata.obsm.

    UMAP is used for visualization, not as the main Hotspot input.

    Parameters
    ----------
    adata : AnnData
        AnnData object to add the UMAP to.
    use_rep : str
        Key in adata.obsm the neighbor graph is built on, e.g. "X_pca",
        "X_pca_harmony" or "X_scVI".
    neighbors_key : str, default "neighbors_umap"
        Key in adata.uns (and prefix in adata.obsp) the neighbor graph is
        stored under. An existing graph under this key is only overwritten
        if force_recompute is True. The default differs from scanpy's
        "neighbors" so that the graph of the clustering is not replaced.
    n_neighbors : int, default 30
        Number of neighbors for the graph.
    n_pcs : Optional[int], default None
        Number of dimensions of use_rep used for the neighbor graph, counted
        from the first. None uses all dimensions. If use_rep has fewer
        dimensions, e.g. because principal components were removed in an
        earlier notebook, all of them are used.
    metric : str, default "euclidean"
        Distance metric for the neighbor graph. Possible values:
        "cityblock", "cosine", "euclidean", "l1", "l2", "manhattan",
        "braycurtis", "canberra", "chebyshev", "correlation", "dice",
        "hamming", "jaccard", "kulsinski", "mahalanobis", "minkowski",
        "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener",
        "sokalsneath", "sqeuclidean", "yule".
        See https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html
    min_dist : float, default 0.5
        UMAP min_dist, controls how tightly cells are packed.
    spread : float, default 1.0
        UMAP spread, controls the overall scale of the layout.
    n_components : int, default 2
        Number of UMAP dimensions.
    random_state : int, default 0
        Random seed for reproducible results.
    umap_key : str, default "X_umap"
        Key in adata.obsm the UMAP is stored under.
    force_recompute : bool, default False
        Whether to recompute the neighbor graph and the UMAP if a UMAP
        already exists under umap_key. If False, the existing one is reused.

    Raises
    ------
    ValueError
        If use_rep is not present in adata.obsm, or if a neighbor graph
        already exists under neighbors_key and force_recompute is False.
    """

    if umap_key in adata.obsm and not force_recompute:
        logger.info(f"Existing UMAP found: adata.obsm['{umap_key}']. Set force_recompute=True to overwrite.")
        return

    if use_rep not in adata.obsm:
        raise ValueError(
            f"use_rep='{use_rep}' not found in adata.obsm. "
            "Cannot compute UMAP."
        )

    # sc.pp.neighbors fails if n_pcs is larger than the number of columns in
    # use_rep. This happens e.g. when PCs were removed in an earlier notebook,
    # so X_pca has 20 columns but n_pcs asks for 30. In that case all
    # available columns are used instead.
    available_dims = adata.obsm[use_rep].shape[1]

    if n_pcs is not None and n_pcs > available_dims:
        logger.info(
            f"adata.obsm['{use_rep}'] holds {available_dims} dimensions, "
            f"using all of them instead of n_pcs={n_pcs}."
        )
        n_pcs = available_dims

    # sc.pp.neighbors overwrites an existing graph under the same key without
    # warning, so this is checked here
    if neighbors_key in adata.uns:
        if not force_recompute:
            raise ValueError(
                f"A neighbor graph already exists under adata.uns['{neighbors_key}']. "
                "Choose another neighbors_key or set force_recompute=True to overwrite it."
            )
        logger.warning(f"Overwriting neighbor graph in adata.uns['{neighbors_key}'].")

    logger.info(f"Computing neighbors using representation: '{use_rep}'.")

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
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
        Key in adata.obsm of the representation to use. Its value decides
        what happens if the key is missing:

        - equal to pca_key: a PCA is computed and stored under pca_key.
        - equal to harmony_key: a PCA is computed (if missing) and Harmony is
          computed on top of it and stored under harmony_key.
        - any other key, e.g. "X_scVI": nothing is computed, the key must
          already exist in adata.obsm.
    batch_key : Optional[str], default None
        Column in adata.obs Harmony corrects for. Required if use_rep is
        harmony_key, ignored otherwise.
    pca_key : str, default "X_pca"
        Key in adata.obsm the PCA is read from or stored under. Only change
        it if the PCA should live under a different name.
    harmony_key : str, default "X_pca_harmony"
        Key in adata.obsm the Harmony representation is read from or stored
        under. Only change it if Harmony should live under a different name.
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
            f"use_rep '{use_rep}' was not found in adata.obsm and cannot be "
            f"computed here. Only use_rep='{pca_key}' (PCA) and "
            f"use_rep='{harmony_key}' (Harmony) are computed automatically; "
            "any other representation must be created beforehand."
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
    n_pcs: Optional[int] = None,
    metric: Literal[
        "cityblock", "cosine", "euclidean", "l1", "l2", "manhattan",
        "braycurtis", "canberra", "chebyshev", "correlation", "dice",
        "hamming", "jaccard", "kulsinski", "mahalanobis", "minkowski",
        "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener",
        "sokalsneath", "sqeuclidean", "yule",
    ] = "euclidean",
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
    n_pcs : Optional[int], default None
        Number of dimensions of use_rep used for the neighbor graph. None
        uses all dimensions.
    metric : str, default "euclidean"
        Distance metric for the neighbor graph, forwarded to ensure_umap.
        See ensure_umap for the possible values.
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

    ensure_umap(
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
        umap_key=umap_key,
        force_recompute=force_umap_effective,
    )

    return umap_key


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
        Key of the selected latent representation in ``adata.obsm``.

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

    # a subset invalidates the existing representations, since they describe
    # the cells and genes they were computed on
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

        # PCA is recomputable; reusing an existing one is handled by ensure_pca
        if rep == pca_key:
            needs_compute = rep not in adata.obsm or force_pca_effective
            if needs_compute and not compute_pca_if_missing:
                continue
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


@deco.log_anndata
@beartype
def inspect_gene_detection(
    adata: AnnData,
    layer_key: Optional[str] = "raw",
    thresholds_to_show: Optional[Sequence[int]] = None,
    var_column: str = "hotspot_n_cells_detected",
    plot: bool = True,
    bins: int = 50,
    save: Optional[str] = None,
    report: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Show in how many cells each gene is detected.

    A gene counts as detected in a cell if its value in the selected layer is
    greater than zero. Nothing is removed here; the result is the basis for
    choosing min_cells in filter_genes_by_cell_detection.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    layer_key : Optional[str], default "raw"
        Layer used for the detection counts. If None, adata.X is used; note
        that anndata >= 0.13 stores .X in adata.layers under the None key, so
        both spellings refer to the same matrix.
    thresholds_to_show : Optional[Sequence[int]], default None
        Thresholds reported in the overview table. If None, they are derived
        from the current cell number as 0.1%, 0.25%, 0.5%, 1% and 2% of
        adata.n_obs, so that the candidates fit the size of the current
        (sub)set instead of a fixed dataset size.
    var_column : str, default "hotspot_n_cells_detected"
        Column in adata.var the detection counts are written to. The counts
        are always computed from the current object rather than reused from
        adata.var["n_cells_by_counts"], which describes the state of the data
        in notebook 2 and is too high after subsetting.
    plot : bool, default True
        Whether to plot a histogram of the gene detection frequencies.
    bins : int, default 50
        Number of histogram bins.
    save : Optional[str], default None
        File name for the saved figure. If None, the figure is not saved.
    report : Optional[str], default None
        Name of the output file used for report creation. Will be silently
        skipped if sctoolbox.settings.report_dir is None.

    Returns
    -------
    gene_detection : pandas.DataFrame
        Per gene: the number of cells it is detected in.
    threshold_table : pandas.DataFrame
        Per candidate threshold: how many genes would be kept.

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
            available = [k for k in adata.layers if k is not None]
            raise KeyError(
                f"Layer '{layer_key}' not found in adata.layers. "
                f"Available layers: {available}"
            )
        used_layer = f"adata.layers['{layer_key}']"

    logger.info(f"Using {used_layer} for gene detection.")

    # --------------------------------------------------------
    # Count number of cells where each gene is detected
    # --------------------------------------------------------
    # counted here rather than reused from the QC columns of notebook 2:
    # those describe the full object and are too high after subsetting
    logger.info(f"Counting gene detection from {used_layer}.")

    X = adata.X if layer_key is None else adata.layers[layer_key]

    # (X > 0) works for sparse and dense alike; asarray/ravel flattens the
    # np.matrix that sparse .sum() returns
    n_cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()

    adata.var[var_column] = n_cells_per_gene

    gene_detection = pd.DataFrame(
        {"gene": adata.var_names, var_column: n_cells_per_gene},
        index=adata.var_names,
    )

    # --------------------------------------------------------
    # Log a compact summary; the full table is returned to the caller
    # --------------------------------------------------------
    stats = gene_detection[var_column].describe()
    logger.info(
        f"Gene detection frequency across {adata.n_obs} cells: "
        f"median={stats['50%']:.0f}, min={stats['min']:.0f}, max={stats['max']:.0f}."
    )

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

        _save_figure(save, bbox_inches="tight")

        if settings.report_dir and report:
            _save_figure(report, report=True, bbox_inches="tight")

        plt.show()

    # --------------------------------------------------------
    # Threshold overview
    # --------------------------------------------------------
    if thresholds_to_show is None:
        # scale the candidates with the current (sub)set instead of using
        # fixed numbers that only fit medium-sized datasets
        thresholds_to_show = sorted(
            {max(1, int(round(adata.n_obs * f))) for f in (0.001, 0.0025, 0.005, 0.01, 0.02)}
        )

    rows = []

    for threshold in thresholds_to_show:
        n_keep = int((gene_detection[var_column] >= threshold).sum())
        rows.append({
            "min_cells": int(threshold),
            "pct_of_cells": round(100 * threshold / adata.n_obs, 3),
            "n_genes_kept": n_keep,
            "pct_genes_kept": round(100 * n_keep / adata.n_vars, 1),
        })

    threshold_table = pd.DataFrame(rows).set_index("min_cells")

    logger.info("Genes kept at different min_cells thresholds:")

    for threshold, row in threshold_table.iterrows():
        logger.info(
            f"min_cells={threshold} ({row['pct_of_cells']}% of cells): "
            f"keep {int(row['n_genes_kept'])} / {adata.n_vars} genes"
        )

    # rule of thumb expressed in the numbers of this dataset rather than in
    # the abstract, so that the user can act on it directly
    lo = max(1, int(adata.n_obs * 0.001))
    hi = max(1, int(adata.n_obs * 0.01))

    logger.info(
        f"Rule of thumb: min_cells between {lo} and {hi} (0.1-1% of the "
        f"{adata.n_obs} cells in this set). Pick a value where the histogram "
        "flattens out and most genes are still kept."
    )

    utils.adata.add_uns_info(adata, key=["hotspot", "gene_detection"], value={
        "used_layer": used_layer,
        "var_column": var_column,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        # stored as two scalars because h5ad has no writer for tuples
        "suggested_min_cells_low": lo,
        "suggested_min_cells_high": hi,
    })

    return gene_detection, threshold_table


@deco.log_anndata
@beartype
def filter_genes_by_cell_detection(
    adata: AnnData,
    min_cells: Optional[int],
    copy: bool = True,
) -> tuple[AnnData, dict[str, Any]]:
    """
    Remove genes detected in fewer than min_cells cells.

    Requires inspect_gene_detection to have been run on the same object, so
    that the decision is made after seeing the distribution.

    Parameters
    ----------
    adata : AnnData
        AnnData object inspected with inspect_gene_detection.
    min_cells : Optional[int]
        Minimum number of cells in which a gene must be detected. None keeps
        all genes, which is the state the notebook ships in. There is no
        numeric default on purpose: choose the value from the histogram and
        the threshold table of inspect_gene_detection. As a starting point,
        0.1-1% of the cells in the current (sub)set.
    copy : bool, default True
        Whether to return a copy instead of filtering in place.

    Returns
    -------
    adata_out : AnnData
        Filtered AnnData object.
    summary : dict
        Gene numbers before and after filtering.

    Raises
    ------
    ValueError
        If inspect_gene_detection was not run, if the counts refer to a
        different cell number, or if the threshold removes all genes.
    """
    if min_cells is None:
        logger.info("min_cells is None, no genes were removed.")
        summary = {
            "min_cells": None,
            "n_genes_before": adata.n_vars,
            "n_genes_after": adata.n_vars,
            "n_genes_removed": 0,
        }
        return (adata.copy() if copy else adata), summary

    key = ["sctoolbox", "hotspot", "gene_detection"]

    if not utils.adata.in_uns(adata, key):
        raise ValueError(
            "No gene detection statistics found. Run inspect_gene_detection "
            "first and choose min_cells based on its output."
        )

    info = utils.adata.get_uns(adata, key)
    var_column = info["var_column"]

    if info["n_cells"] != adata.n_obs:
        raise ValueError(
            f"Gene detection counts were computed on {info['n_cells']} cells, "
            f"but the object now has {adata.n_obs}. Rerun inspect_gene_detection."
        )

    keep_genes = adata.var[var_column].to_numpy() >= min_cells

    if keep_genes.sum() == 0:
        raise ValueError(
            f"min_cells={min_cells} removes all {adata.n_vars} genes. "
            "Choose a lower threshold."
        )

    summary = {
        "used_layer": info["used_layer"],
        "min_cells": min_cells,
        "pct_of_cells": round(100 * min_cells / adata.n_obs, 3),
        "n_genes_before": adata.n_vars,
        "n_genes_after": int(keep_genes.sum()),
        "n_genes_removed": int(adata.n_vars - keep_genes.sum()),
    }

    logger.info(
        f"min_cells={min_cells}: keep {summary['n_genes_after']} / "
        f"{summary['n_genes_before']} genes, remove {summary['n_genes_removed']}."
    )

    adata_out = qc_filter.filter_genes(
        adata,
        genes=adata.var_names[keep_genes].tolist(),
        invert=True,
        inplace=not copy,
        name="hotspot_gene_detection",
        value={"min_cells": min_cells, "used_layer": info["used_layer"]},
        overwrite=True,
    )

    if adata_out is None:  # inplace=True returns None
        adata_out = adata

    logger.info(f"Filtered AnnData: {adata_out.n_obs} cells x {adata_out.n_vars} genes.")

    return adata_out, summary


@deco.log_anndata
@beartype
def run_hotspot(  # noqa: C901
    adata: AnnData,
    layer_key: str = "raw",
    latent_obsm_key: Optional[str] = None,
    umi_counts_obs_key: str = "total_counts",
    model: Literal["danb", "bernoulli", "normal", "none"] = "danb",
    n_neighbors: int = 30,
    weighted_graph: bool = False,
    jobs: Optional[int] = None,
    top_n: Optional[int] = 2000,
    gene_selection_method: Literal["top_n_z", "fdr"] = "fdr",
    autocorr_fdr_threshold: Optional[float] = 0.05,
    min_gene_threshold: int = 30,
    core_only: bool = True,
    module_fdr_threshold: float = 0.05,
    compute_local_correlations: bool = True,
    compute_modules: bool = True,
    compute_module_scores: bool = True,
    store_local_correlations: bool = False,
    module_prefix: str = "Hotspot module",
    overwrite_obs_key: bool = False,
) -> dict[str, Any]:
    """
    Run Hotspot analysis on an AnnData object.

    All results that can be stored are written to adata: autocorrelation
    statistics, module assignments and the selected gene flag to adata.var,
    module scores to adata.obs, run metadata to adata.uns. Genes without a
    clear module, including genes that were not part of the run, are
    designated as -1 (UNASSIGNED_MODULE) in adata.var["hotspot_module"],
    which can be interpreted as n/a. The returned dictionary additionally holds
    the live Hotspot object, which is only needed for the local correlation
    heatmap, see plot_local_correlations.

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
    model : Literal["danb", "bernoulli", "normal", "none"], default "danb"
        Null model for the per-gene expression distribution. Pick the one
        that matches the input in layer_key:

        - "danb": depth-adjusted negative binomial. For UMI counts, i.e. the
          usual choice for scRNA-seq with layer_key="raw".
        - "bernoulli": models detection (zero vs non-zero) only. For binary or
          very sparse data such as scATAC peak matrices, or when counts are
          unreliable.
        - "normal": for continuous data that is already normalised and
          log-transformed, or for non-count modalities.
        - "none": no null model; values are only centered and scaled. Use when
          the data is already variance-stabilised.

        See https://hotspot.readthedocs.io/en/latest/ for the underlying
        statistics.
    n_neighbors : int, default 30
        Number of neighbors for KNN graph construction.
    weighted_graph : bool, default False
        Whether to build a weighted KNN graph.
    jobs : Optional[int], default None
        Number of parallel jobs. If None, sctoolbox.settings.get_threads() is used.
    top_n : Optional[int], default 2000
        Number of top genes to use for local correlations if
        gene_selection_method="top_n_z".
    gene_selection_method : Literal["top_n_z", "fdr"], default "fdr"
        Strategy for selecting genes for local correlations.
    autocorr_fdr_threshold : Optional[float], default 0.05
        FDR threshold for gene selection if gene_selection_method="fdr".
    min_gene_threshold : int, default 30
        Minimum genes per module.
    core_only : bool, default True
        Whether to keep only core module genes, i.e. the genes that cluster
        stably with the module as a whole rather than with a few of its
        members. Genes that are not assigned carry the label -1
        (UNASSIGNED_MODULE). Setting this to False also assigns genes at the
        cluster border, which yields larger but less specific modules.
    module_fdr_threshold : float, default 0.05
        FDR threshold used in module creation.
    compute_local_correlations : bool, default True
        Whether to compute local correlations.
    compute_modules : bool, default True
        Whether to create modules.
    compute_module_scores : bool, default True
        Whether to calculate module scores. Requires compute_modules=True.
    store_local_correlations : bool, default False
        Whether to store the local correlation matrix in adata.uns. The matrix
        grows with the square of the number of selected genes and is therefore
        skipped by default.
    module_prefix : str, default "Hotspot module"
        Prefix for module score columns added to adata.obs.
    overwrite_obs_key : bool, default False
        Whether to overwrite an existing umi_counts_obs_key in adata.obs.

    Returns
    -------
    dict[str, Any]
        Dictionary holding the live Hotspot object under "hotspot" and the raw
        result tables. Downstream functions read from adata instead.

    Raises
    ------
    KeyError
        If the requested layer, latent representation, or a required result
        column is missing.
    ValueError
        If top_n or autocorr_fdr_threshold is missing for the selected
        gene_selection_method, if all genes have zero variance, or if
        compute_module_scores is requested without compute_modules.
    """
    if jobs is None:
        jobs = settings.get_threads()

    if layer_key not in adata.layers:
        # anndata >= 0.13 stores .X in layers under the None key, which is not
        # a layer name the user can pass here
        available = [k for k in adata.layers if k is not None]
        raise KeyError(
            f"Layer '{layer_key}' not found in adata.layers. "
            f"Available layers: {available}"
        )

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

    # scipy sparse has no .var(), so Var(X) = E[X²] - E[X]² is computed by hand
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        gene_var = mean_sq - mean**2
    else:
        gene_var = np.asarray(X.var(axis=0)).ravel()

    # Hotspot's null models divide by the gene variance, so constant genes
    # have to go before the object is handed over
    keep_genes = gene_var > 0

    if keep_genes.sum() == 0:
        raise ValueError(
            "All genes have zero variance in the selected subset/layer. "
            "Please check your subsetting or input layer."
        )

    n_removed = int((~keep_genes).sum())

    if n_removed > 0:
        logger.info(f"Removing {n_removed} genes with zero variance in '{layer_key}'.")
        # Hotspot gets a reduced copy; results are written back to the full
        # adata further down, where the dropped genes end up as NaN / -1
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

    # --------------------------------------------------------
    # Store results in adata so that downstream functions do
    # not depend on the returned dictionary
    # --------------------------------------------------------

    # autocorrelation statistics -> var
    for col in ("C", "Z", "Pval", "FDR"):
        if col in autocorrelations.columns:
            adata.var[f"hotspot_{col}"] = autocorrelations[col].reindex(adata.var_names)

    # module assignments -> var
    n_modules = None

    if modules is not None:
        module_series = pd.Series(modules)

        # reindex introduces NaN for genes outside the run, which count as unassigned
        adata.var["hotspot_module"] = (
            pd.to_numeric(module_series.reindex(adata.var_names), errors="coerce")
            .fillna(UNASSIGNED_MODULE)
            .astype(int)
        )

        n_modules = module_series[module_series != UNASSIGNED_MODULE].nunique()

    # genes that entered the local correlation step -> var
    if selected_genes is not None:
        adata.var["hotspot_selected"] = adata.var_names.isin(selected_genes)

    # module scores -> obs
    module_score_columns = None

    if module_scores is not None:
        module_scores_obs = module_scores.rename(
            columns=lambda c: f"{module_prefix} {c}"
        ).reindex(adata.obs_names)

        module_score_columns = module_scores_obs.columns.tolist()

        # drop first so that a rerun with fewer modules leaves no stale
        # columns from the previous run behind
        adata.obs = adata.obs.drop(columns=module_score_columns, errors="ignore")
        adata.obs = adata.obs.join(module_scores_obs)

    # local correlation matrix -> uns, on request only
    if store_local_correlations and local_correlations is not None:
        utils.adata.add_uns_info(
            adata, key=["hotspot", "local_correlations"], value=local_correlations
        )

    # run results -> uns; the parameters of this call are already recorded
    # by the log_anndata decorator. latent_obsm_key is kept because it is
    # resolved automatically when None is passed.
    utils.adata.add_uns_info(adata, key=["hotspot", "run"], value={
        "latent_obsm_key_used": latent_obsm_key,
        "n_selected_genes": None if selected_genes is None else len(selected_genes),
        "n_modules": n_modules,
        "module_score_columns": module_score_columns,
        # stored as a scalar because h5ad has no writer for tuples
        "n_local_correlation_genes": (
            None if local_correlations is None else local_correlations.shape[0]
        ),
        "n_zero_variance_genes_removed": n_removed,
        "n_genes_used": adata_hs.n_vars,
    })

    return results


@beartype
def plot_local_correlations(
    hs: hotspot.Hotspot,
    save: Optional[str] = None,
    report: Optional[str] = None,
) -> None:
    """
    Plot the Hotspot local correlation heatmap.

    Genes are ordered by the linkage Hotspot computed during module creation,
    so that modules appear as blocks along the diagonal. This is the only
    Hotspot plot that needs the live Hotspot object, because the linkage is
    not stored in AnnData.

    Parameters
    ----------
    hs : hotspot.Hotspot
        Hotspot object as returned by run_hotspot under results["hotspot"].
    save : Optional[str], default None
        File name for the saved figure. If None, the figure is not saved.
    report : Optional[str], default None
        Name of the output file used for report creation. Will be silently
        skipped if sctoolbox.settings.report_dir is None.
    """
    try:
        hs.plot_local_correlations()
    except (ValueError, KeyError) as e:
        logger.warning(f"Could not plot local correlation heatmap: {e}")
        return

    _save_figure(save, bbox_inches="tight")

    if settings.report_dir and report:
        _save_figure(report, report=True, bbox_inches="tight")

    plt.show()


@deco.log_anndata
@beartype
def show_hotspot_results(  # noqa: C901
    adata: AnnData,
    fdr_threshold: float = 0.05,
    color_map: Optional[str] = None,
    ncols: int = 3,
    hist_figsize: tuple[float, float] = (7.0, 4.0),
    hist_bins: int = 50,
    umap_key: str = "X_umap",
    plot_umap_modules: bool = True,
    plot_autocorrelation_distribution: bool = True,
    plot_module_sizes: bool = True,
    save_autocorrelation: Optional[str] = None,
    report_autocorrelation: Optional[str] = None,
    save_module_sizes: Optional[str] = None,
    report_module_sizes: Optional[str] = None,
    save_module_scores: Optional[str] = None,
    report_module_scores: Optional[str] = None,
) -> dict[str, Any]:
    """
    Show and save the main Hotspot results.

    All results are read from adata, which run_hotspot fills. The local
    correlation heatmap is not part of this function because it needs the
    live Hotspot object, see plot_local_correlations.

    This function summarizes:
    - autocorrelation results
    - number of significant / selected genes
    - autocorrelation distributions
    - module size table and barplot
    - module score UMAPs

    Parameters
    ----------
    adata : AnnData
        AnnData object holding the Hotspot results written by run_hotspot.
    fdr_threshold : float, default 0.05
        FDR threshold used to count significant genes.
    color_map : Optional[str], default None
        Matplotlib colormap for the module score UMAPs. If None, the
        sctoolbox default is used.
    ncols : int, default 3
        Number of columns for UMAP module score plots.
    hist_figsize : tuple[float, float], default (7.0, 4.0)
        Size of the autocorrelation distribution histograms in inches.
    hist_bins : int, default 50
        Number of bins in the autocorrelation distribution histograms.
    umap_key : str, default "X_umap"
        Key in adata.obsm holding the UMAP used for the module score plots.
        Must exist if plot_umap_modules is True.
    plot_umap_modules : bool, default True
        Whether to plot module scores on UMAP.
    plot_autocorrelation_distribution : bool, default True
        Whether to plot distributions of autocorrelation results.
    plot_module_sizes : bool, default True
        Whether to plot module sizes.
    save_autocorrelation : Optional[str], default None
        File name for the autocorrelation distributions. One file per
        statistic is written, with the statistic appended to the given name,
        e.g. "..._C.pdf". If None, the figures are not saved.
    report_autocorrelation : Optional[str], default None
        Name of the output file used for report creation, handled like
        save_autocorrelation. Will be silently skipped if
        sctoolbox.settings.report_dir is None.
    save_module_sizes : Optional[str], default None
        File name for the module size barplot. If None, it is not saved.
    report_module_sizes : Optional[str], default None
        Name of the output file used for report creation. Will be silently
        skipped if sctoolbox.settings.report_dir is None.
    save_module_scores : Optional[str], default None
        File name for the module score UMAPs. If None, they are not saved.
    report_module_scores : Optional[str], default None
        Name of the output file used for report creation. Will be silently
        skipped if sctoolbox.settings.report_dir is None.

    Returns
    -------
    dict[str, Any]
        Summary holding the result tables and counts. Tables are returned
        rather than displayed so that the caller decides what to show. The
        "modules" entry is the raw column from adata.var, in which unassigned
        genes carry the label -1 (UNASSIGNED_MODULE).
    """
    summary = {}

    run_key = ["sctoolbox", "hotspot", "run"]
    run_info = utils.adata.get_uns(adata, run_key) if utils.adata.in_uns(adata, run_key) else {}

    # ========================================================
    # 1. Autocorrelation results
    # ========================================================

    autocorr_cols = [c for c in ("C", "Z", "Pval", "FDR") if f"hotspot_{c}" in adata.var.columns]

    if autocorr_cols:
        # genes outside the Hotspot run carry NaN in every column
        hs_results = adata.var[[f"hotspot_{c}" for c in autocorr_cols]].dropna(how="all")
        hs_results.columns = autocorr_cols

        summary["autocorrelations"] = hs_results
        summary["n_total_genes_tested"] = hs_results.shape[0]

        if "FDR" in hs_results.columns:
            n_sig = int((hs_results["FDR"] < fdr_threshold).sum())
            logger.info(
                f"{n_sig} / {hs_results.shape[0]} genes have FDR < {fdr_threshold}"
            )
        else:
            n_sig = None
            logger.warning("Column 'hotspot_FDR' not found in adata.var.")

        summary["n_significant_genes"] = n_sig

        if plot_autocorrelation_distribution:
            for col in autocorr_cols:
                values = hs_results[col].replace([np.inf, -np.inf], np.nan).dropna()

                plt.figure(figsize=hist_figsize)
                plt.hist(values, bins=hist_bins)
                plt.xlabel(col)
                plt.ylabel("Number of genes")
                plt.title(f"Distribution of Hotspot {col}")
                plt.tight_layout()

                # one figure per statistic, so the column is appended to the name
                _save_figure(
                    None if save_autocorrelation is None
                    else save_autocorrelation.replace(".pdf", f"_{col}.pdf"),
                    bbox_inches="tight",
                )

                if settings.report_dir and report_autocorrelation:
                    _save_figure(
                        report_autocorrelation.replace(".png", f"_{col}.png"),
                        report=True,
                        bbox_inches="tight",
                    )

                plt.show()

    else:
        logger.warning("No 'hotspot_*' autocorrelation columns found in adata.var.")

    # ========================================================
    # 2. Selected genes
    # ========================================================

    if "hotspot_selected" in adata.var.columns:
        selected_genes = adata.var_names[adata.var["hotspot_selected"]]

        logger.info(f"Selected genes: {len(selected_genes)}")

        summary["selected_genes"] = pd.Series(selected_genes, name="gene")
        summary["n_selected_genes"] = len(selected_genes)

    else:
        logger.warning("No selected genes found in adata.var['hotspot_selected'].")
        summary["n_selected_genes"] = None

    # ========================================================
    # 3. Local correlations
    # ========================================================
    # The heatmap needs the live Hotspot object, see plot_local_correlations

    n_corr_genes = run_info.get("n_local_correlation_genes")

    if n_corr_genes is not None:
        logger.info(f"Local correlations computed for {n_corr_genes} genes.")

    summary["n_local_correlation_genes"] = n_corr_genes

    # ========================================================
    # 4. Modules and module sizes
    # ========================================================

    if "hotspot_module" in adata.var.columns:
        modules = adata.var["hotspot_module"].rename("Module")

        module_sizes = modules[modules != UNASSIGNED_MODULE].value_counts().sort_index()
        module_sizes.name = "n_genes"

        logger.info(
            f"{module_sizes.shape[0]} modules, "
            f"{int((modules == UNASSIGNED_MODULE).sum())} genes without a module (-1)."
        )

        summary["modules"] = modules
        summary["n_modules"] = module_sizes.shape[0]
        summary["module_sizes"] = module_sizes

        if plot_module_sizes:
            plt.figure(figsize=(10, 4))
            module_sizes.plot(kind="bar")
            plt.xlabel("Hotspot module")
            plt.ylabel("Number of genes")
            plt.title("Hotspot module sizes")
            plt.tight_layout()

            _save_figure(save_module_sizes, bbox_inches="tight")

            if settings.report_dir and report_module_sizes:
                _save_figure(report_module_sizes, report=True, bbox_inches="tight")

            plt.show()

    else:
        logger.warning("No module assignments found in adata.var['hotspot_module'].")
        summary["n_modules"] = None

    # ========================================================
    # 5. Module score UMAPs
    # ========================================================

    # h5ad turns the stored list into an array, so normalise before testing
    stored_columns = run_info.get("module_score_columns")
    score_columns = [] if stored_columns is None else list(stored_columns)

    if score_columns:
        summary["module_scores"] = adata.obs[score_columns]
        summary["n_module_score_columns"] = len(score_columns)
        summary["module_score_columns"] = list(score_columns)

        if plot_umap_modules:
            if umap_key not in adata.obsm:
                logger.warning(
                    f"No UMAP found in adata.obsm['{umap_key}']. "
                    "Module UMAPs cannot be plotted."
                )
            else:
                logger.info("Plotting module score UMAPs...")

                pl.embedding.plot_embedding(
                    adata,
                    method=umap_key,
                    color=score_columns,
                    ncols=ncols,
                    suptitle="",
                    color_map=color_map,
                    save=save_module_scores,
                    report=report_module_scores if settings.report_dir else None,
                )

    else:
        logger.warning("No module score columns recorded in adata.uns.")

    return summary


@deco.log_anndata
@beartype
def run_hotspot_module_enrichment(  # noqa: C901
    adata: AnnData,
    organism: str,
    gene_sets: list[str] = ["GO_Biological_Process_2018"],
    min_module_size: int = 5,
    exclude_unassigned: bool = True,
    overwrite: bool = False,
    save_table: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Run Enrichr-based gene set enrichment for Hotspot modules.

    Gene symbols of both the modules and the libraries are converted to
    uppercase, as in tools.gsea.gene_set_enrichment, because the Enrichr
    databases differ in casing. Gene names in the result table are therefore
    uppercase.

    Parameters
    ----------
    adata : AnnData
        AnnData object holding the module assignments written by run_hotspot
        in adata.var["hotspot_module"]. Results are stored in adata.uns.
    organism : str
        Organism passed to gseapy.enrichr.
    gene_sets : list[str], default ["GO_Biological_Process_2018"]
        Enrichr libraries to use. Availability differs per organism; the
        current list is at https://maayanlab.cloud/Enrichr/#libraries
    min_module_size : int, default 5
        Minimum number of genes required to test a module.
    exclude_unassigned : bool, default True
        Whether to exclude genes that Hotspot could not assign to any module.
        Those genes carry the label -1 (UNASSIGNED_MODULE) in
        adata.var["hotspot_module"].
    overwrite : bool, default False
        If False and enrichment already exists in adata.uns, skip.
    save_table : Optional[str], default None
        File name for the combined enrichment table, saved to
        sctoolbox.settings.table_dir. If None, no table is saved.
    **kwargs : Any
        Additional parameters forwarded to gseapy.enrichr.

    Raises
    ------
    ValueError
        If the module assignments or enrichment configuration are invalid.
    """

    if (
        utils.adata.in_uns(adata, ["sctoolbox", "hotspot", "module_enrichment"])
        and not overwrite
    ):
        logger.warning(
            "Hotspot module enrichment already exists. "
            "Skipping. Set overwrite=True to replace."
        )
        return

    # --------------------------------------------------------
    # Get modules
    # --------------------------------------------------------

    if "hotspot_module" not in adata.var.columns:
        raise ValueError(
            "No module assignments found in adata.var['hotspot_module']. "
            "Run run_hotspot first."
        )

    modules = adata.var["hotspot_module"].rename("Module")

    if exclude_unassigned:
        # unassigned genes form no coherent gene program and are not tested
        modules = modules[modules != UNASSIGNED_MODULE]

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

    module_ids = sorted(modules.unique())

    # Enrichr libraries differ in gene symbol casing, e.g. FishEnrichr stores
    # zebrafish symbols in uppercase while datasets usually use lowercase.
    # As in tools.gsea.gene_set_enrichment, genes and gene sets are both
    # converted to uppercase, so casing never prevents a match.
    libraries = {
        name: {
            term: [g.upper() for g in genes]
            for term, genes in gp.get_library(name=name, organism=organism).items()
        }
        for name in gene_sets
    }

    for i, module_id in enumerate(module_ids, start=1):
        genes = modules[modules == module_id].index.astype(str).str.upper().tolist()

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

        for name, library in libraries.items():
            try:
                enr = gp.enrichr(gene_list=genes, gene_sets=library, **kwargs)
            except (EnrichrError, ValueError) as e:
                logger.warning(f"Module {module_id}, library {name}: enrichment failed: {e}")
                continue

            # Enrichr returns every term that shares at least one gene with
            # the module, significant or not. Without any overlap gseapy
            # leaves an empty list instead of a DataFrame.
            df = enr.results

            if not isinstance(df, pd.DataFrame) or df.empty:
                logger.warning(
                    f"Module {module_id}: none of its genes occurs in {name}. "
                    "Either the genes are not annotated in this library, or "
                    "organism and gene_sets do not match the gene symbols in "
                    "adata.var_names."
                )
                continue

            df = df.copy()
            df["Gene_set"] = name
            df["module"] = int(module_id)
            df["n_module_genes"] = len(genes)

            all_rows.append(df)

    if all_rows:
        enrichment_table = pd.concat(all_rows, ignore_index=True)

        sort_cols = [
            col for col in ["module", "Adjusted P-value", "P-value"]
            if col in enrichment_table.columns
        ]

        if sort_cols:
            enrichment_table = enrichment_table.sort_values(sort_cols)
    else:
        raise ValueError(
            "Enrichment returned no results for any module. The most common "
            "cause is a mismatch between the gene symbols in adata.var_names "
            f"and the libraries {gene_sets} for organism '{organism}'."
        )

    # --------------------------------------------------------
    # Store in adata.uns
    # --------------------------------------------------------

    module_sizes = modules.value_counts().sort_index()
    module_sizes.name = "n_genes"

    utils.adata.add_uns_info(adata, key=["hotspot", "module_enrichment"], value={
        "method": "enrichr",
        "organism": organism,
        "gene_sets": gene_sets,
        "min_module_size": min_module_size,
        "exclude_unassigned": exclude_unassigned,
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
        Column used to rank the enriched terms. If the column is missing, a
        warning is logged and the first top_n terms per module are returned
        in their stored order, i.e. unranked.

    Returns
    -------
    pd.DataFrame
        Top enriched terms per module, sorted by module and pvalue_column.
        If no enrichment results are stored in adata, a warning is logged and
        an empty DataFrame is returned.
    """

    key = ["sctoolbox", "hotspot", "module_enrichment"]
    enrichment_info = utils.adata.get_uns(adata, key) if utils.adata.in_uns(adata, key) else {}

    enrichment_table = enrichment_info.get("enrichment_table", None)

    if enrichment_table is None or enrichment_table.empty:
        logger.warning("No Hotspot module enrichment table found.")
        return pd.DataFrame()

    if pvalue_column not in enrichment_table.columns:
        logger.warning(
            f"Column '{pvalue_column}' not found in the enrichment table. "
            "Terms are returned in stored order without ranking."
        )

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
def plot_hotspot_module_enrichment(
    adata: AnnData,
    top_n: int = 5,
    pvalue_column: str = "Adjusted P-value",
    term_column: str = "Term",
    pseudocount: float = 1e-300,
    dot_size_factor: float = 25.0,
    save: Optional[str] = None,
    report: Optional[str] = None,
    figsize: tuple[float, float] = (10, 8),
) -> pd.DataFrame:
    """
    Plot Hotspot module enrichment results.

    Creates a dotplot of the top enriched terms per module and returns the
    underlying table. Module sizes are shown in show_hotspot_results.

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
    save : Optional[str], default None
        File name for the saved dotplot.
    report : Optional[str], default None
        Name of the output file used for report creation, handled like save.
        Will be silently skipped if sctoolbox.settings.report_dir is None.
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

    logger.info(
        f"Top {top_n} enriched terms per module for "
        f"{top_terms['module'].nunique()} modules."
    )

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

    _save_figure(save, bbox_inches="tight")

    if settings.report_dir and report:
        _save_figure(report, report=True, bbox_inches="tight")

    plt.show()

    return top_terms

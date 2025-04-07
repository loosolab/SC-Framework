"""Module for GSEA analysis."""
import pandas as pd
import gseapy as gp
import scanpy as sc
import tqdm
import warnings
import deprecation

import sctoolbox
import sctoolbox.utils.decorator as deco
from sctoolbox.tools.marker_genes import get_rank_genes_tables

from beartype import beartype
from beartype.typing import Optional, Literal, Any


@deprecation.deprecated(deprecated_in="0.10", removed_in="1.0",
                        current_version=sctoolbox.__version__,
                        details="enrichr_marker_genes() is replaced by gene_set_enrichment().")
def enrichr_marker_genes(adata, **kwargs):
    """Run enrichr method to support older gsea notebook versions."""
    return gene_set_enrichment(adata, method="enrichr", **kwargs)


@deco.log_anndata
@beartype
def gene_set_enrichment(adata: sc.AnnData,
                        marker_key: str,
                        organism: str,
                        method: Literal["prerank", "enrichr"] = "prerank",
                        pvals_adj_tresh: float = 0.05,
                        library_name: str = "GO_Biological_Process_2023",
                        overwrite: bool = False,
                        inplace: bool = False,
                        gene_sets: Optional[dict[str, list[str]]] = None,
                        background: Optional[set[str]] = None,
                        **kwargs: Any
                        ) -> Optional[sc.AnnData]:
    """
    Run gene set enrichment on marker genes.

    Parameters
    ----------
    adata : sc.anndata
        Anndata object.
    marker_key: str
        Key in adata.uns containing rank gene group tables.
    organism : str
        Source organism.
    method : Literal["prerank", "enrichr"], default "prerank"
        Choose between enrichr and prerank(gsea) method.
    pvals_adj_tresh : float, default 0.05
        Threshold for adjusted p-value.
    library_name : str, default GO_Biological_Process_2023
        Name of public GO library.
        Get gene sets from public GO library.
    overwrite : bool, default False
        If True will overwrite existing gsea results.
    inplace : bool, default False
        Whether to copy `adata` or modify it inplace.
    gene_sets : Optional[dict[str, list[str]]], default None
        Dictionary with pathway names as key and gene set as value.
        If given, library name is ignored.
    background : Optional[set[str]], default None
        Set of background genes. Will be automatically determined when library_name or gene_sets is given.
        Only needed for enrichr.
    **kwargs : Any
        Additional parameters forwarded to gseapy.prerank().

    Notes
    -----
    This function only works in combination with the tools.marker_genes.run_rank_genes function.
    Depending on the organism the genes are saved uppercase or lowercase. Since it does not follow
    the typically guidelines all genes are converted to uppercase to prevent failed overlaps.

    Returns
    -------
    Optional[sc.AnnData]
        AnnData object with combined enrichr results stored in .uns['gsea'].

    Raises
    ------
    KeyError
        If marker_key not in adata.uns
    ValueError
        If result dictinary is empty
    """

    if not overwrite and "gsea" in adata.uns:
        warnings.warn("GSEA seems to have been run before! Skipping. Set `overwrite=True` to replace.")

        if inplace:
            return
        else:
            return adata

    modified_adata = adata if inplace else adata.copy()

    # setup dict to store information old data will be overwriten!
    modified_adata.uns['gsea'] = dict()
    modified_adata.uns['gsea']['method'] = method
    modified_adata.uns['gsea']['stat_col'] = "FDR q-val" if method == "prerank" else "Adjusted P-value"
    modified_adata.uns['gsea']['score_col'] = "NES" if method == "prerank" else "Combined Score"

    if marker_key in adata.uns:
        marker_tables = get_rank_genes_tables(modified_adata,
                                              out_group_fractions=True,
                                              key=marker_key,
                                              n_genes = None)
    else:
        raise KeyError("Marker key not found! Please check parameter!")

    if not gene_sets:
        # A public library is used if gene_set is not given
        gene_sets = gp.get_library(name=library_name, organism=organism)
        modified_adata.uns['gsea']['library'] = library_name
    else:
        modified_adata.uns['gsea']['gene_sets'] = gene_sets
    if not background:
        # Generating background if no custom background is given
        background = set([item for sublist in gene_sets.values() for item in sublist])
    if method == "enrichr":
        modified_adata.uns['gsea']['background'] = background

    # Convert gene sets to upper case
    gene_sets = {key: list(map(str.upper, value)) for key, value in gene_sets.items()}

    path_enr = {}
    for ct, deg in tqdm.tqdm(marker_tables.items()):
        enr_list = list()
        # enrichr API
        if len(deg) > 0:
            if method == "enrichr":
                enr = gp.enrichr(list(deg["names"].str.upper()),
                                 gene_sets=gene_sets,
                                 organism=organism,
                                 background=background,
                                 outdir=None,
                                 no_plot=True,
                                 verbose=False)
                enr_list.append(enr.res2d)
            elif method == "prerank":
                # Set default kwargs
                defaultKwargs = {"threads": 4,
                                 "min_size": 5,
                                 "max_size": 1000,
                                 "permutation_num": 1000,
                                 "outdir": None,
                                 "seed": 6,
                                 "verbose": True}
                kwargs = {**defaultKwargs, **kwargs}

                deg["names"] = deg["names"].str.upper()
                deg.index = deg["names"]
                enr = gp.prerank(rnk=deg["scores"],
                                 gene_sets=gene_sets,
                                 **kwargs)
                enr_list.append(enr.res2d)

        # concat results
        if not all(x is None for x in enr_list):
            path_enr[ct] = pd.concat(enr_list)
            path_enr[ct]["Cluster"] = ct
        else:
            warnings.warn(f"No valid pathways found for cluster {ct}")

    # Return combined table
    if not path_enr:
        raise ValueError("No valid pathways found for dataset.")

    modified_adata.uns['gsea']['enrichment_table'] = pd.concat(path_enr.values())

    if not inplace:
        return modified_adata

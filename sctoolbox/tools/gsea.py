"""Module for GSEA analysis."""
import pandas as pd
import gseapy as gp
import scanpy as sc
import tqdm
import warnings

import sctoolbox.utils.decorator as deco
from sctoolbox.tools.marker_genes import get_rank_genes_tables

from beartype import beartype
from beartype.typing import Optional


@deco.log_anndata
@beartype
def enrichr_marker_genes(adata: sc.AnnData,
                         marker_key: str,
                         organism: str,
                         pvals_adj_tresh: float = 0.05,
                         library_name: str = "GO_Biological_Process_2023",
                         gene_sets: Optional[dict[str, list[str]]] = None,
                         background: Optional[set[str]] = None
                         ) -> pd.DataFrame:
    """
    Wrapps enrichr module to use on marker genes per cluster.

    Parameters
    ----------
    adata : sc.anndata
        Anndata object.
    marker_key: str
        Key in adata.uns containing rank gene group tables.
    organism : str
        Source organism.
    pvals_adj_tresh : float, default 0.05
        Threshold for adjusted p-value.
    library_name : str, default GO_Biological_Process_2023
        Name of public GO library.
        Get gene sets from public GO library.
    gene_sets : Optional[dict[str, list[str]]], default None
        Dictionary with pathway names as key and gene set as value.
        If given library name is ignored.
    background : Optional[set[str]], default None
        Set of background genes. Will be automatically determined when library_name or gene_sets is given.

    Notes
    -----
    This function only works in combination with the tools.marker_genes.run_rank_genes function.
    Depending on the organism the genes are saved uppercase or lowercase. Since it does not follow
    the typically guidelines all genes are converted to uppercase to prevent failed overlaps.

    Returns
    -------
    pd.DataFrame
        Combined enrichr results.

    Raises
    ------
    KeyError:
        If marker_key not in adata.uns
    ValueError:
        If result dictinary is empty
    """
    if marker_key in adata.uns:
        marker_tables = get_rank_genes_tables(adata,
                                              out_group_fractions=True,
                                              key=marker_key)
    else:
        raise KeyError("Marker key not found! Please check parameter!")

    if not gene_sets:
        # A public library is used if gene_set is not given
        gene_sets = gp.get_library(name=library_name, organism=organism)
    if not background:
        # Generating background if no custom background is given
        background = set([item for sublist in gene_sets.values() for item in sublist])

    # Convert gene sets to upper case
    gene_sets = {key: list(map(str.upper,value)) for key, value in gene_sets.items()}

    path_enr = {}
    for ct, table in tqdm.tqdm(marker_tables.items()):
        # subset up or down regulated genes
        degs_sig = table[table["pvals_adj"] < pvals_adj_tresh]
        degs_up = degs_sig[degs_sig["logfoldchanges"] > 0]
        degs_dw = degs_sig[degs_sig["logfoldchanges"] < 0]

        degs = {"UP": degs_up, "Down": degs_dw}
        enr_list = list()

        # enrichr API
        for key, deg in degs.items():
            if len(deg) > 0:
                enr = gp.enrichr(list(deg["names"].str.upper()),
                                 gene_sets=gene_sets,
                                 organism=organism,
                                 background=background,
                                 outdir=None,
                                 no_plot=True,
                                 verbose=False)
                enr.res2d['UP_DW'] = key
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
    return (pd.concat(path_enr.values()))

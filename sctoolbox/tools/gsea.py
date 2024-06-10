"""Module for GSEA analysis"""
import pandas as pd
import gseapy as gp
import scanpy as sc
import tqdm
import sctoolbox.utils.decorator as deco
from sctoolbox.tools.marker_genes import get_rank_genes_tables

from beartype import beartype


@deco.log_anndata
@beartype
def enrichr_marker_genes(adata: sc.anndata,
                         marker_key: str,
                         gene_sets: dict[str, list[str]],
                         organism: str,
                         background: set[str],
                         pvals_adj_tresh: float = 0.05
                        ) -> pd.DataFrame:
    """
    Wrapper for enrichr module to use on marker genes per cluster.
    
    Parameter
    ---------
    adata : sc.anndata
        Anndata object.
    marker_key: str
        Key in adata.uns containing rank gene group tables.
    gene_sets : dict[str, list[str]]
        Dictionary with pathway names as key and gene set as value.
    organsim : str
        Source organsim.
    background : set[str]
        Set of background genes.
    pvals_adj_tresh : float
        Threshold for adjusted p-value.

    Returns
    -------
    pd.DataFrame
        Combined enrichr results.
        
    Raises
    ------
    KeyError:
        If marker_key not in adata.uns
    """
    if marker_key in adata.uns:
        marker_tables = get_rank_genes_tables(adata,
                                              out_group_fractions=True,
                                              key=marker_key)
    else:
        raise KeyError("Marker key not found! Please check parameter!")
    
    
    path_enr = {}

    for ct, table in tqdm.tqdm(marker_tables.items()):
        # subset up or down regulated genes
        degs_sig = table[table["pvals_adj"] < pvals_adj_tresh]
        degs_up = degs_sig[degs_sig["logfoldchanges"] > 0]
        degs_dw = degs_sig[degs_sig["logfoldchanges"] < 0]

        degs = {"UP": degs_up, "Down": degs_dw}
        enr_list = list()

        # enrichr API
        for key, deg in degs.items:
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
        path_enr[ct] = pd.concat(enr_list)
        path_enr[ct]["Cluster"] = ct

    # Return combined table
    return(pd.concat(path_enr.values()))

"""Module for GSEA analysis."""
import pandas as pd
import gseapy as gp
import scanpy as sc

import sctoolbox.utils.decorator as deco
from sctoolbox.tools.marker_genes import get_rank_genes_tables

from beartype import beartype
from beartype.typing import Optional, Literal, Any

import sctoolbox.utils as utils
from sctoolbox import settings
logger = settings.logger

_core_uns_path = ['sctoolbox', 'gsea']


@deco.log_anndata
@beartype
def gene_set_enrichment(adata: sc.AnnData,
                        marker_key: str,
                        organism: str,
                        method: Literal["prerank", "enrichr"] = "prerank",
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
        Set of background genes. Will be automatically determined when
        library_name or gene_sets is given. Only needed for enrichr.
    **kwargs : Any
        Additional parameters forwarded to gseapy.prerank().

    Notes
    -----
    This function only works in combination with the
    tools.marker_genes.run_rank_genes function. Depending on the organism the
    genes are saved uppercase or lowercase. Since it does not follow the
    typically guidelines all genes are converted to uppercase to prevent
    failed overlaps.

    Returns
    -------
    Optional[sc.AnnData]
        AnnData object with combined enrichr results stored in .uns['sctoolbox']['gsea'].

    Raises
    ------
    KeyError
        If marker_key not in adata.uns
    ValueError
        If result dictinary is empty
    """

    if not overwrite and utils.adata.in_uns(adata, _core_uns_path):
        logger.warning("GSEA seems to have been run before! Skipping. Set `overwrite=True` to replace.")

        if inplace:
            return
        else:
            return adata

    modified_adata = adata if inplace else adata.copy()

    # setup dict to store information old data will be overwriten!
    utils.adata.add_uns_info(modified_adata,
                             key=['gsea', 'method'],
                             value=method)
    stat_col = "FDR q-val" if method == "prerank" else "Adjusted P-value"
    utils.adata.add_uns_info(modified_adata,
                             key=['gsea', 'stat_col'],
                             value=stat_col)
    score_col = "NES" if method == "prerank" else "Combined Score"
    utils.adata.add_uns_info(modified_adata,
                             key=['gsea', 'score_col'],
                             value=score_col)
    overlap_col = "Tag %" if method == "prerank" else "Overlap"
    utils.adata.add_uns_info(modified_adata,
                             key=['gsea', 'overlap_col'],
                             value=overlap_col)
    utils.adata.add_uns_info(modified_adata,
                             key=['gsea', 'marker_key'],
                             value=marker_key)

    logger.info("Getting gene rank tables.")
    if marker_key in adata.uns:
        marker_tables = get_rank_genes_tables(modified_adata,
                                              out_group_fractions=True,
                                              key=marker_key,
                                              n_genes=None)

        # remove potential "_1", "_2" suffixes
        # TODO I'm assuming that this pattern is uniquely created by .var_names_make_unique and no genes are actually named that way.
        for table in marker_tables.values():
            table.iloc[:, 0] = table.iloc[:, 0].str.replace(r'_\d+$', '', regex=True)
    else:
        msg = "Marker key not found! Please check parameter!"
        logger.error(msg)
        raise KeyError(msg)

    if not gene_sets:
        # A public library is used if gene_set is not given
        logger.info("Downloading gene set library.")
        gene_sets = gp.get_library(name=library_name, organism=organism)
        utils.adata.add_uns_info(modified_adata,
                                 key=['gsea', 'library'],
                                 value=library_name)
    else:
        modified_adata.uns['gsea']['gene_sets'] = gene_sets
    if not background:
        # Generating background if no custom background is given
        background = set([item for sublist in gene_sets.values() for item in sublist])
    if method == "enrichr":
        logger.info("Setting background")
        utils.adata.add_uns_info(modified_adata,
                                 key=['gsea', 'background'],
                                 value=list(background))

    # Convert gene sets to upper case
    gene_sets = {key: list(map(str.upper, value)) for key, value in gene_sets.items()}

    path_enr = {}
    for i, (ct, deg) in enumerate(marker_tables.items(), start=1):
        logger.info(f"Running {method} for cluster {ct} ({i}/{len(marker_tables)}).")
        enr_list = list()
        # enrichr API
        if len(deg) > 0:
            # briefly silence all loggers to disable gseapy logging
            with utils.general.suppress_logging():
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
            logger.warning(f"No valid pathways found for cluster {ct}")

    # Return combined table
    if not path_enr:
        msg = "No valid pathways found for dataset."
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Saving results in 'adata.uns['sctoolbox']['gsea']['enrichment_table']'")
    merged_results = pd.concat(path_enr.values())
    if method == "prerank":
        merged_results[['ES', 'NES', 'NOM p-val', 'FDR q-val', 'FWER p-val']] = merged_results[
            ['ES', 'NES', 'NOM p-val', 'FDR q-val', 'FWER p-val']
        ].astype(float)
    utils.adata.add_uns_info(modified_adata,
                             key=['gsea', 'enrichment_table'],
                             value=merged_results)

    if not inplace:
        return modified_adata

"""Tools for marker gene analyis."""

import re
import glob
import pkg_resources
import pandas as pd
import numpy as np
import scanpy as sc
import itertools
import warnings
import anndata
from pathlib import Path
import matplotlib.pyplot as plt

from beartype import beartype
from beartype.typing import Optional, Tuple, Any, Literal

import sctoolbox.utils as utils
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
from sctoolbox.plotting.general import _save_figure
logger = settings.logger

# path to the internal gene lists (gender, cellcycle, mito, ...)
_GENELIST_LOC = Path(pkg_resources.resource_filename("sctoolbox", "data/gene_lists/"))


@beartype
def get_chromosome_genes(gtf: str,
                         chromosomes: str | list[str]) -> list[str]:
    """
    Get a list of all genes in the gtf for certain chromosome(s).

    Parameters
    ----------
    gtf : str
        Path to the gtf file.
    chromosomes : str | list[str]
        A chromosome or a list of chromosome names to search for genes in.

    Returns
    -------
    list[str]
        A list of all genes in the gtf for the given chromosome(s).

    Notes
    -----
    This function is not directly used by the framework, but is used to create the marker gene lists for 'label_genes'.

    Raises
    ------
    ValueError
        If not all given chromosomes are found in the GTF-file.
    """

    if isinstance(chromosomes, str):
        chromosomes = [chromosomes]

    all_chromosomes = {}

    gene_names = {}
    with open(gtf) as f:
        for line in f:

            columns = line.rstrip().split("\t")
            chrom = columns[0]
            all_chromosomes[chrom] = ""  # save for overview on the chromosomes in the gtf

            # Save gene name if gene in chromosomes
            if chrom in chromosomes:
                m = re.search("gene_name \"(.+?)\";", line)
                if m is not None:
                    name = m.group(1)
                    gene_names[name] = ""

    all_chromosomes = list(all_chromosomes.keys())

    # Check that chromosomes were valid
    for chrom in chromosomes:
        if chrom not in all_chromosomes:
            raise ValueError(f"Chromosome '{chrom}' not found in gtf file. Available chromosomes are: {all_chromosomes}")

    # Collect final gene list
    gene_names = list(gene_names.keys())

    return gene_names


@deco.log_anndata
@beartype
def label_genes(adata: sc.AnnData,
                species: Optional[str] = None,
                gene_column: Optional[str] = None,
                plot: bool = True,
                # mitochondiral args
                m_genes: Optional[list[str] | str | Literal["internal"]] = "internal",
                m_regex: Optional[str] = "^mt",
                # ribosomal args
                r_genes: Optional[list[str] | str | Literal["internal"]] = "internal",
                r_regex: Optional[str] = "^rps|rpl",
                # gender args
                g_genes: Optional[list[str] | str | Literal["internal"]] = "internal",
                g_regex: Optional[str] = None,
                # report args
                report: Optional[str] = None
                ) -> list[str]:
    """
    Label genes as ribosomal, mitochrondrial and gender genes.

    Gene labels are added inplace.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object.
    species : Optional[str]
        Name of the species. Mandatory if any of 'm_genes', 'r_genes' or 'g_genes' is set to 'internal' otherwise unused.
    gene_column : Optional[str], default None
        Name of the column in adata.var that contains the gene names. Uses adata.var.index as default.
    plot : bool, default True
        Enables barplot.
    m_genes : Optional[list[str], str, Literal["internal"]], default "internal"
        Either a list of mitochondrial genes, a file containing one mitochondrial gene name per line or 'internal' to use an sctoolbox provided list.
    m_regex : Optional[str], default "^mt"
        A regex to identify mitochondrial genes if 'm_genes' is not available or failing.
    r_genes : Optional[list[str], str, Literal["internal"]], default "internal"
        Either a list of ribosomal genes, a file containing one ribosomal gene name per line or 'internal' to use an sctoolbox provided list.
    r_regex : Optional[str], default "^rps|rpl"
        A regex to identify ribosomal genes if 'r_genes' is not available or failing.
    g_genes : Optional[list[str], str, Literal["internal"]], default "internal"
        Either a list of gender genes, a file containing one gender gene name per line or 'internal' to use an sctoolbox provided list.
    g_regex : Optional[str]
        A regex to identify gender genes if 'g_genes' is not available or failing.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.

    Raises
    ------
    ValueError
        If 'species' parameter is missing despite having any of 'm_genes', 'r_genes', 'g_genes' set to 'internal'.

    Returns
    -------
    list[str]
        List containing the column names added to adata.var.

    See Also
    --------
    sctoolbox.tools.qc_filter.predict_cell_cycle : for cell cycle prediction.
    """
    if species:
        species = species.lower()
    elif m_genes == "internal" or r_genes == "internal" or g_genes == "internal":
        raise ValueError("Species is mandatory for usage of internal genelists. Either set the parameter 'species' or set 'm_genes', 'r_genes' and 'g_genes' to not be 'internal'.")

    # Get the full list of genes from adata
    if gene_column is None:
        adata_genes = adata.var.index
    else:
        adata_genes = adata.var[gene_column]

    # ------- Annotate genes in adata ------ #
    var_cols = []  # store names of new var columns

    for kind, labeler, regex in [("mito", m_genes, m_regex),
                                 ("ribo", r_genes, r_regex),
                                 ("gender", g_genes, g_regex)]:
        # prepare genelist if needed
        if labeler == "internal":
            available_species = utils.general.clean_flanking_strings(glob.glob(str(_GENELIST_LOC / f"*_{kind}_genes.txt")))
            if species not in available_species:
                avail_str = f" Available species are: {available_species}"
                logger.warning(f"No {kind} genes available for species '{species}'." + (avail_str if available_species else ""))
                logger.warning(f"Falling back to regex '{regex}'...")

                genelist = None
            else:
                genelist = utils.general.read_list_file(str(_GENELIST_LOC / f"{species}_{kind}_genes.txt"))
        elif isinstance(labeler, str):
            try:
                genelist = utils.general.read_list_file(labeler)
            except FileNotFoundError:
                logger.warning(f"File {labeler} not found.")
                logger.warning(f"Falling back to regex '{regex}'...")

                genelist = None
        elif isinstance(labeler, list):
            genelist = labeler
        else:
            genelist = None  # to trigger regex

        # create list of boolean indicators
        bool_label = _annotate(genes=adata_genes, labeler=genelist, regex=regex, kind=kind)

        if bool_label is not None:
            adata.var[f"is_{kind}"] = bool_label
            var_cols.append(f"is_{kind}")

    if plot and var_cols:
        _, ax = plt.subplots(figsize=(5, 3))

        # get the number of assigned genes per category (absolute and percent)
        abs_height = [adata.var[col].sum() for col in var_cols]
        per_height = [v / len(adata.var) * 100 for v in abs_height]

        # absolute
        rects = ax.bar(x=var_cols, height=abs_height)
        ax.bar_label(rects, labels=[f"{a} ({p:.2f} %)" for a, p in zip(abs_height, per_height)], padding=3)
        ax.set_title("Number of genes")
        ax.tick_params(axis="x", rotation=45)
        ax.set(ylim=(0, len(adata.var)), ylabel="count")
        # create secondary y-axis
        secax = ax.secondary_yaxis("right",
                                   functions=(lambda x: x / len(adata.var) * 100,  # translates count -> percentage
                                              lambda x: x * len(adata.var) / 100))  # translates percentage -> count
        secax.set_ylabel("percentage [%]")
        # add text box
        ax.text(x=.975,
                y=.95,
                s=f"Total genes: {len(adata.var)}",
                transform=ax.transAxes,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
                )

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True)

    return var_cols


def _annotate(genes: pd.Series, labeler: Optional[list[str]], regex: Optional[str], kind: str) -> pd.Series:
    """
    Create a boolean list that shows whether a gene is contained in 'labeler' or matches the regex.

    Parameters
    ----------
    genes : pd.Series
        A list of genes that are matched to 'labeler' or if not available 'regex'.
    labeler : Optional[list[str]]
        A second list the first is checked against. If not given uses regex matching instead.
    regex : Optional[str]
        A regex pattern used to match genes. Only used if labeler = None.
    kind : str
        Name of the genes that are annotated. E.g. mito

    Returns
    -------
    pd.Series
        A boolean list of len(genes). True denotes 'genes' that matched either the regex or where contained in the labeler list.
    """
    logger.info(f"Annotating {kind} genes...")
    if labeler:
        # handle elements which are in the labeler list but not in the genes list
        not_found = [lab for lab in labeler if lab not in genes]
        if not_found:
            logger.warning(f"Following genes are not present in the dataset. Please check for typos. {not_found}")
        return genes.isin(labeler)
    elif regex:
        return genes.str.match(regex, case=False)
    logger.warn("Neither genelist nor regex available. Skipping...")


@deco.log_anndata
@beartype
def add_gene_expression(adata: sc.AnnData,
                        gene: str) -> None:
    """
    Add values of gene/feature per cell to the adata.obs dataframe.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object containing gene expression/counts.
    gene : str
        Name of the gene/feature from the adata.var index to be added to adata.obs.

    Raises
    ------
    ValueError
        If gene is not in adata.var.index.
    """

    # Get expression
    if gene in adata.var.index:
        gene_idx = np.argwhere(adata.var.index == gene)[0][0]
        vals = adata.X[:, gene_idx].todense().A1
        adata.obs[gene + "_values"] = vals

    else:
        raise ValueError(f"Gene '{gene}' was not found in adata.var.index")


############################################################
#                 Scanpy rank genes groups                 #
############################################################

@deco.log_anndata
@beartype
def run_rank_genes(adata: sc.AnnData,
                   groupby: str,
                   min_in_group_fraction: float = 0.25,
                   min_fold_change: float = 0.5,
                   max_out_group_fraction: float = 0.8,
                   **kwargs: Any) -> None:
    """
    Run scanpy rank_genes_groups and filter_rank_genes_groups.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object containing gene expression/counts.
    groupby : str
        Column by which the cells in adata should be grouped.
    min_in_group_fraction : float, default 0.25
        Minimum fraction of cells in a group that must express a gene to be considered as a marker gene.
        Parameter forwarded to scanpy.tl.filter_rank_genes_groups.
    min_fold_change : float, default 0.5
        Minimum foldchange (+/-) to be considered as a marker gene.
        Parameter forwarded to scanpy.tl.filter_rank_genes_groups.
    max_out_group_fraction : float, default 0.8
        Maximum fraction of cells in other groups that must express a gene to be considered as a marker gene.
        Parameter forwarded to scanpy.tl.filter_rank_genes_groups.
    **kwargs : Any
        Additional arguments forwarded to scanpy.tl.rank_genes_groups.

    Raises
    ------
    ValueError
        If number of groups defined by the groupby parameter is < 2.
    """

    if adata.obs[groupby].dtype.name != "category":
        adata.obs[groupby] = adata.obs[groupby].astype("category")

    if "log1p" in adata.uns:
        adata.uns['log1p']['base'] = None  # hack for scanpy error; see https://github.com/scverse/scanpy/issues/2239#issuecomment-1104178881

    # Check number of groups in groupby
    if adata.obs[groupby].nunique() < 2:
        raise ValueError("groupby must contain at least two groups.")

    # Catch ImplicitModificationWarning from scanpy
    params = {'method': 't-test',  # prevents warning message "Default of the method has been changed to 't-test' from 't-test_overestim_var'"
              'key_added': f'rank_genes_{groupby}'}  # set default key_added
    params.update(kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=anndata.ImplicitModificationWarning, message="Trying to modify attribute.*")
        sc.tl.rank_genes_groups(adata, groupby=groupby, **params)

    sc.tl.filter_rank_genes_groups(adata,
                                   min_in_group_fraction=min_in_group_fraction,
                                   min_fold_change=min_fold_change,
                                   max_out_group_fraction=max_out_group_fraction,
                                   key=params["key_added"],
                                   key_added=f"{params['key_added']}_filtered"
                                   )


@deco.log_anndata
@beartype
def pairwise_rank_genes(adata: sc.AnnData,
                        groupby: str,
                        foldchange_threshold: int | float = 1,
                        min_in_group_fraction: float = 0.25,
                        max_out_group_fraction: float = 0.5,
                        **kwargs: Any
                        ) -> pd.DataFrame:
    """
    Rank genes pairwise between groups in 'groupby'.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object containing expression data.
    groupby : str
        Key in adata.obs containing groups to be compared.
    foldchange_threshold : int | float, default 1
        Minimum foldchange (+/-) to be considered as a marker gene.
    min_in_group_fraction : float, default 0.25
        Minimum fraction of cells in a group that must express a gene to be considered as a marker gene.
    max_out_group_fraction : float, default 0.5
        Maximum fraction of cells in other groups that must express a gene to be considered as a marker gene.
    **kwargs : Any
        Additional arguments to be passed to scanpy.tl.rank_genes_groups.

    Raises
    ------
    ValueError
        Raised when there are less than two groups in the adata.obs column selected by groupby.

    Returns
    -------
    pd.DataFrame
        Dataframe containinge the pairwise ranked gened between the groups.
    """

    groups = adata.obs[groupby].astype("category").cat.categories
    contrasts = list(itertools.combinations(groups, 2))
    logger.debug(f"Contrasts: {contrasts}")

    if len(contrasts) == 0:
        raise ValueError(f"Number of groups in '{groupby}' must be at least 2. Found: {list(groups)}")

    # Check that fractions are available
    use_fractions = True
    if adata.X.min() < 0:
        logger.warning("adata.X contains negative values (potentially transformed counts), "
                       "meaning that 'min_in_group_fraction' and 'max_out_group_fraction' "
                       "cannot be used for filtering. These parameters will be ignored. "
                       "Consider using raw/normalized data instead.")
        use_fractions = False

    # Calculate marker genes for each contrast
    tables = []
    for contrast in contrasts:
        logger.info(f"Calculating rank genes for contrast: {contrast}")

        # Get adata for contrast
        adata_sub = adata[adata.obs[groupby].isin(contrast)]   # subset to contrast

        # Run rank_genes_groups
        run_rank_genes(adata_sub, groupby=groupby, **kwargs)

        # Get table
        c1, c2 = contrast
        table_dict = get_rank_genes_tables(adata_sub, key=f"rank_genes_{groupby}", n_genes=None, out_group_fractions=True)  # returns dict with each group
        table = table_dict[c1]

        # Reorder columns
        table.set_index("names", inplace=True)
        columns = ["scores", "logfoldchanges", "pvals", "pvals_adj"]
        if use_fractions:
            columns += [c1 + "_fraction", c2 + "_fraction"]
        table = table[columns]  # reorder columns
        table = table.copy(deep=True)  # prevent SettingWithCopyWarning

        # Calculate up/down genes
        c1, c2 = contrast
        groups = ["C1", "C2"]
        if use_fractions:
            conditions = [(table["logfoldchanges"] >= foldchange_threshold) & (table[c1 + "_fraction"] >= min_in_group_fraction) & (table[c2 + "_fraction"] <= max_out_group_fraction),  # up
                          (table["logfoldchanges"] <= -foldchange_threshold) & (table[c1 + "_fraction"] <= max_out_group_fraction) & (table[c2 + "_fraction"] >= min_in_group_fraction)]  # down
        else:
            conditions = [table["logfoldchanges"] >= foldchange_threshold,  # up
                          table["logfoldchanges"] <= -foldchange_threshold]  # down
        table["group"] = np.select(conditions, groups, "NS")

        # Rename columns
        prefix = "/".join(contrast) + "_"
        table.columns = [prefix + col if "fraction" not in col else col for col in table.columns]

        # Add table to list
        tables.append(table)

    # Join individual tables
    merged = pd.concat(tables, join="inner", axis=1)

    # Move fraction columns to the back
    merged = merged.loc[:, ~merged.columns.duplicated()]
    fraction_columns = [col for col in merged.columns if col.endswith("_fraction")]  # might be empty if use_fractions = False
    first_columns = [col for col in merged.columns if col not in fraction_columns]
    merged = merged[first_columns + fraction_columns]

    return merged


@deco.log_anndata
@beartype
def get_rank_genes_tables(adata: sc.AnnData,
                          key: str = "rank_genes_groups",
                          n_genes: Optional[int] = 200,
                          out_group_fractions: bool = False,
                          var_columns: list[str] = [],
                          save_excel: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """
    Get gene tables containing "rank_genes_groups" genes and information per group (from previously chosen `groupby`).

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object containing ranked genes.
    key : str, default "rank_genes_groups"
        The key in adata.uns to be used for fetching ranked genes.
    n_genes : Optional[int], default 200
        Number of genes to be included in the tables. If None, all genes are included.
    out_group_fractions : bool, default False
        If True, the output tables will contain additional columns giving the fraction of genes per group.
    var_columns : list[str], default []
        List of adata.var columns, which will be added to pandas.DataFrame.
    save_excel : Optional[str], default None
        The path to a file for writing the marker gene tables as an excel file (with one sheet per group).

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary with group names as keys, and marker gene tables (pandas DataFrames) per group as values.

    Raises
    ------
    ValueError
        1. If not all columns given in var_columns are in adata.var.
        2. If key cannot be found in adata.uns.
    """

    # Check that all given columns are valid
    if len(var_columns) > 0:
        for col in var_columns:
            if col not in adata.var.columns:
                raise ValueError(f"Column '{col}' not found in adata.var.columns.")

    # Check that key is in adata.uns
    if key not in adata.uns:
        raise ValueError(f"Key '{key}' not found in adata.uns. Please use 'run_rank_genes' first.")

    # Read structure in .uns to pandas dataframes
    tables = {}
    for col in ["names", "scores", "pvals", "pvals_adj", "logfoldchanges"]:
        tables[col] = pd.DataFrame(adata.uns[key][col])

    # Transform to one table per group
    groups = tables["names"].columns
    group_tables = {}
    for group in groups:
        data = {}
        for col in tables:
            data[col] = tables[col][group].values
        table = pd.DataFrame(data)

        # Remove any NaN genes (genes are set to NaN if 'filter_rank_genes_groups' was used)
        table.dropna(inplace=True)

        # Subset to n_genes if chosen
        if n_genes is not None:
            table = table.iloc[:n_genes, :]

        group_tables[group] = table

    # Get in/out group fraction of expressed genes (only works for .X-values above 0)
    if (adata.X.min() < 0) == 0:  # only calculate fractions for raw expression data
        groupby = adata.uns[key]["params"]["groupby"]
        n_cells_dict = adata.obs[groupby].value_counts().to_dict()  # number of cells in groups

        for group in groups:

            # Fraction of cells inside group expressing each gene
            s = np.ravel((adata[adata.obs[groupby].isin([group]), :].X > 0).sum(axis=0))  # sum of cells with expression > 0 for this cluster
            expressed = pd.DataFrame([adata.var.index, s]).T
            expressed.columns = ["names", "n_expr"]

            group_tables[group] = group_tables[group].merge(expressed, left_on="names", right_on="names", how="left")
            group_tables[group][group + "_fraction"] = group_tables[group]["n_expr"] / n_cells_dict[group] if group in n_cells_dict else 0

            # Fraction of cells for individual groups
            if out_group_fractions is True:
                for compare_group in groups:
                    if compare_group != group:
                        s = np.ravel((adata[adata.obs[groupby].isin([compare_group]), :].X > 0).sum(axis=0))
                        expressed = pd.DataFrame(s, index=adata.var.index, dtype="float64")  # expression per gene for this group
                        expressed.columns = [compare_group + "_fraction"]
                        expressed.iloc[:, 0] = expressed.iloc[:, 0] / n_cells_dict[compare_group] if compare_group in n_cells_dict else 0.0

                        group_tables[group] = group_tables[group].merge(expressed, left_on="names", right_index=True, how="left")

            # Fraction of cells outside group expressing each gene
            other_groups = [g for g in groups if g != group]
            s = np.ravel((adata[adata.obs[groupby].isin(other_groups), :].X > 0).sum(axis=0))
            expressed = pd.DataFrame([adata.var.index, s]).T
            expressed.columns = ["names", "n_out_expr"]
            group_tables[group] = group_tables[group].merge(expressed, left_on="names", right_on="names", how="left")

            # If there are only two groups, out_group_fraction -> name of the other group
            if len(groups) == 2:
                out_group_name = [g for g in groups if g != group][0] + "_fraction"
            else:
                out_group_name = "out_group_fraction"

            n_out_group = sum([n_cells_dict[other_group] if other_group in n_cells_dict else 0 for other_group in other_groups])  # sum of cells in other groups
            group_tables[group][out_group_name] = group_tables[group]["n_out_expr"] / n_out_group
            group_tables[group].drop(columns=["n_expr", "n_out_expr"], inplace=True)

    # Add additional columns to table
    if len(var_columns) > 0:
        for group in group_tables:
            group_tables[group] = group_tables[group].merge(adata.var[var_columns], left_on="names", right_index=True, how="left")

    # If chosen: Save tables to joined excel
    if save_excel is not None:

        if not isinstance(save_excel, str):
            raise ValueError("'save_excel' must be a string.")

        filename = settings.full_table_prefix + save_excel

        with pd.ExcelWriter(filename) as writer:
            for group in group_tables:
                table = group_tables[group].copy()

                # Round values of scores/foldchanges
                table["scores"] = table["scores"].round(3)
                table["logfoldchanges"] = table["logfoldchanges"].round(3)

                table.to_excel(writer, sheet_name=utils.tables._sanitize_sheetname(f'{group}'), index=False)

        logger.info(f"Saved marker gene tables to '{filename}'")

    return group_tables


@deco.log_anndata
@beartype
def mask_rank_genes(adata: sc.AnnData,
                    genes: list[str],
                    key: str = "rank_genes_groups",
                    inplace: bool = True) -> Optional[sc.AnnData]:
    """
    Mask names with "nan" in .uns[key]["names"] if they are found in given 'genes'.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object containing ranked genes.
    genes : list[str]
        List of genes to be masked.
    key : str, default "rank_genes_groups"
        The key in adata.uns to be used for fetching ranked genes.
    inplace : bool, default True
        If True, modifies adata.uns[key]["names"] in place. Otherwise, returns a copy of adata.

    Returns
    -------
    Optional[sc.AnnData]
        If inplace = True, modifies adata.uns[key]["names"] in place and returns None.
        Otherwise, returns a copy of adata.

    Raises
    ------
    ValueError
        If genes is not of type list.
    """

    if not inplace:
        adata = adata.copy()

    # Check input
    if not isinstance(genes, list):
        raise ValueError("genes must be a list of strings.")

    # Mask genes
    for group in adata.uns["rank_genes_groups"]["names"].dtype.names:
        adata.uns[key]["names"][group] = np.where(np.isin(adata.uns[key]["names"][group], genes), float('nan'), adata.uns[key]["names"][group])

    if not inplace:
        return adata


#####################################################################
#                       DEseq2 on pseudobulks                       #
#####################################################################

@deco.log_anndata
@beartype
def run_deseq2(adata: sc.AnnData,
               sample_col: str,
               condition_col: str,
               confounders: Optional[str | list[str]] = None,
               layer: Optional[str] = None,
               contrasts: Optional[list[Tuple]] = None,
               min_counts: int = 5,
               percentile_range: Tuple[int, int] = (0, 100),
               threads: Optional[int] = None) -> pd.DataFrame:
    """
    Run pyDESeq2 on counts within adata. Must be run on the raw counts per sample. If the adata contains normalized counts in .X, 'layer' can be used to specify raw counts.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object containing raw counts.
    sample_col : str
        Column name in adata.obs containing sample names.
    condition_col : str
        Column name in adata.obs containing conditions to be compared in contrasts, e.g. condition A vs. condition B.
    confounders : list, default None
        List of additional column names in adata.obs containing confounders to be included in the model.
    layer : Optional[str], default None
        Name of layer containing raw counts to be used for pyDESeq2. Default is None (use .X for counts)
    contrasts : list of tuples, default None
        List of tuples containing the conditions to be compared in contrasts, e.g. [(A, B), (A, C)].
    min_counts : int, default 5
        Minimum number of counts per gene to be included in the analysis.
    percentile_range : tuple, default (0, 100)
        Percentile range of cells to be used for calculating pseudobulks. Setting (0,95) will restrict calculation
        to the cells in the 0-95% percentile ranges. Default is (0, 100), which means all cells are used.
    threads : Optional[int]
        The number of threads to use for parallelizable calculations. If None is given, sctoolbox.settings.threads is used

    Returns
    -------
    pd.DataFrame
        A dataframe containing the results of the DESeq2 analysis.
        Also adds the dataframe to adata.uns["deseq_result"]

    Raises
    ------
    ValueError
        1. If any given column name is not found in adata.obs.
        2. Invalid contrasts are supplied.
        3. Negative counts are encountered.

    Notes
    -----
    Needs the package 'pydeseq2' to be installed.
    This can be obtained by installing the sctoolbox [deseq2] extra with pip using: `pip install .[deseq2]`.

    See Also
    --------
    sctoolbox.utils.bioutils.pseudobulk_table
    """
    utils.checker.check_module("pydeseq2")
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from pydeseq2.default_inference import DefaultInference

    if threads is None:
        threads = settings.threads

    # Setup the design factors
    if confounders is None:
        confounders = []
    elif not isinstance(confounders, list):
        confounders = [confounders]

    # Check that sample_col and condition_col are in adata.obs
    cols = [sample_col, condition_col] + confounders
    utils.checker.check_columns(adata.obs, cols, name="adata.obs", error=True)

    # Build sample_df
    sample_df = adata.obs[cols].reset_index(drop=True).drop_duplicates()
    sample_df.set_index(sample_col, inplace=True)
    sample_df.sort_index(inplace=True)

    logger.debug("sample_df:")
    logger.debug(sample_df)

    conditions = sample_df[condition_col].unique()

    # Establish contrasts
    all_contrasts = list(itertools.combinations(conditions, 2))
    if contrasts is None:
        contrasts = all_contrasts
    else:
        # Check that all contrasts are valid
        all_contrasts_inv = [(C2, C1) for C1, C2 in all_contrasts]  # Check both directions
        for contrast in contrasts:
            if contrast not in all_contrasts + all_contrasts_inv:
                raise ValueError(f"Contrast {contrast} is not valid. Valid contrasts are: {all_contrasts}")

        # Remove cells not in contrasts
        contrast_conditions = set(itertools.chain.from_iterable(contrasts))
        logger.debug(contrast_conditions)
        adata = adata[adata.obs[condition_col].isin(contrast_conditions)]

        # Remove samples not in contrasts
        sample_df = sample_df[sample_df["condition"].isin(contrast_conditions)]

    # Build count matrix
    logger.debug("Building count matrix")
    counts_df = utils.bioutils.pseudobulk_table(adata, sample_col, how="sum", layer=layer,
                                                percentile_range=percentile_range)
    counts_df = counts_df.astype(int)  # pyDESeq2 requires integer counts
    counts_df = counts_df.transpose()  # pyDESeq2 requires genes as columns
    if counts_df.min().min() < 0:
        raise ValueError("Negative counts found. Please make sure that the counts are not normalized, for example by setting 'layer' to choose different counts.")

    # Filter genes using min_counts
    genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= min_counts]
    counts_df = counts_df[genes_to_keep]

    # Run DEseq2
    logger.debug("Running pyDESeq2")
    design_factors = confounders + [condition_col]
    dds = DeseqDataSet(counts=counts_df,
                       metadata=sample_df,
                       design_factors=design_factors,
                       refit_cooks=True,
                       n_cpus=threads)
    dds.deseq2()

    # Create result table with mean values per condition
    deseq_table = pd.DataFrame(index=dds.var.index)

    # Get normalized counts per sample and condition
    for i, condition in enumerate(conditions):
        mean_values = dds.layers["normed_counts"][dds.obs[condition_col] == condition, :].mean(axis=0)
        deseq_table.insert(i, condition + "_mean", mean_values)

    # Get results per contrast
    for C1, C2 in contrasts:
        logger.debug(f"Calculating stats for contrast: {C1} vs. {C2}")
        ds = DeseqStats(dds, contrast=[condition_col, C2, C1], inference=DefaultInference(n_cpus=threads))
        ds.summary()

        # Rename and add to deseq_table
        df = ds.results_df
        df.drop(columns=["lfcSE", "stat"], inplace=True)
        df.columns = [C2 + "/" + C1 + "_" + col for col in df.columns]
        deseq_table = deseq_table.merge(df, left_index=True, right_index=True)

    # Add normalized individual sample counts to the back of table
    # convert .obs.index to list to avoid KeyError
    normalized_counts = pd.DataFrame(dds.layers["normed_counts"], index=list(dds.obs.index), columns=dds.var.index).transpose()
    deseq_table = deseq_table.merge(normalized_counts, left_index=True, right_index=True)

    # Sort by p-value of first contrast
    C1, C2 = contrasts[0]
    deseq_table.sort_values(by=C2 + "/" + C1 + "_pvalue", inplace=True)

    # Add to adata uns
    utils.adata.add_uns_info(adata, "deseq_result", deseq_table)

    return deseq_table


@deco.log_anndata
@beartype
def score_genes(adata: sc.AnnData,
                gene_set: str | list[str],
                score_name: str = 'score',
                inplace: bool = True,
                **kwargs: Any) -> Optional[sc.AnnData]:
    """
    Assign a score to each cell depending on the expression of a set of genes. This is a wrapper for scanpy.tl.score_genes.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object to score.
    gene_set : str | list[str]
        A list of genes or path to a file containing a list of genes.
        The txt file should have one gene per row.
    score_name : str, default "score"
        Name of the column in obs table where the score will be added.
    inplace : bool, default True
        Adds the new column to the original anndata object.
    **kwargs : Any
        Additional arguments to be passed to scanpy.tl.score_genes.

    Returns
    -------
    Optional[sc.AnnData]
        If inplace is False, return a copy of anndata object with the new column in the obs table.

    Raises
    ------
    FileNotFoundError
        If path given in gene_set does not lead to a file.
    """

    if not inplace:
        adata = adata.copy()

    # check if list is in a file
    if isinstance(gene_set, str):
        # check if file exists
        if Path(gene_set).is_file():
            gene_set = [x.strip() for x in open(gene_set)]
        else:
            raise FileNotFoundError('The list was not found!')

    # scale data
    sdata = sc.pp.scale(adata, copy=True)

    # Score the cells
    sc.tl.score_genes(sdata, gene_list=gene_set, score_name=score_name, **kwargs)
    # add score to adata.obs
    adata.obs[score_name] = sdata.obs[score_name]

    return adata if not inplace else None

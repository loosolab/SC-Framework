"""Tools for marker gene analyis."""
import os
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
from typing import Optional

import sctoolbox.utils as utils
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


def get_chromosome_genes(gtf, chromosomes) -> list[str]:
    """
    Get a list of all genes in the gtf for certain chromosome(s).

    Parameters
    ----------
    gtf : str
        Path to the gtf file.
    chromosomes : str or list
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
    ValueError:
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
def label_genes(adata, species, gene_column=None) -> list[str]:
    """
    Label genes as ribosomal, mitochrondrial, cell cycle phase and gender genes.

    Gene labels are added inplace.

    Parameters
    ----------
    adata : anndata.Anndata
        The anndata object.
    species : str
        Name of the species.
    gene_column : str, default None
        Name of the column in adata.var that contains the gene names. Uses adata.var.index as default.

    Returns
    -------
    list[str]
        List containing the column names added to adata.var.
    """

    # Location of gene lists
    genelist_dir = pkg_resources.resource_filename("sctoolbox", "data/gene_lists/")

    species = species.lower()

    # Get the full list of genes from adata
    if gene_column is None:
        adata_genes = adata.var.index
    else:
        adata_genes = adata.var[gene_column]

    # ------- Annotate genes in adata ------ #
    var_cols = []  # store names of new var columns

    # Annotate ribosomal genes
    adata.var["is_ribo"] = adata_genes.str.lower().str.startswith(('rps', 'rpl'))
    var_cols.append("is_ribo")

    # Annotate mitochrondrial genes
    path_mito_genes = genelist_dir + species + "_mito_genes.txt"
    if os.path.exists(path_mito_genes):
        gene_list = utils.read_list_file(path_mito_genes)
        adata.var["is_mito"] = adata_genes.isin(gene_list)  # boolean indicator
    else:
        adata.var["is_mito"] = adata_genes.str.lower().str.startswith("mt")  # fall back to mt search
    var_cols.append("is_mito")

    # Annotate gender genes
    path_gender_genes = genelist_dir + species + "_gender_genes.txt"
    if os.path.exists(path_gender_genes):
        gene_list = utils.read_list_file(path_gender_genes)
        adata.var["is_gender"] = adata_genes.isin(gene_list)  # boolean indicator
        var_cols.append("is_gender")
    else:
        available_files = glob.glob(genelist_dir + "*_gender_genes.txt")
        available_species = utils.clean_flanking_strings(available_files)
        logger.warning(f"No gender genes available for species '{species}'. Available species are: {available_species}")

    return var_cols


@deco.log_anndata
def add_gene_expression(adata, gene) -> None:
    """
    Add values of gene/feature per cell to the adata.obs dataframe.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object containing gene expression/counts.
    gene : str
        Name of the gene/feature from the adata.var index to be added to adata.obs.

    Raises
    ------
    ValueError:
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
def run_rank_genes(adata, groupby,
                   min_in_group_fraction=0.25,
                   min_fold_change=0.5,
                   max_out_group_fraction=0.8,
                   **kwargs) -> None:
    """
    Run scanpy rank_genes_groups and filter_rank_genes_groups.

    Parameters
    ----------
    adata : anndata.AnnData
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
    **kwargs : dict
        Additional arguments forwarded to scanpy.tl.rank_genes_groups.

    Raises
    ------
    ValueError:
        If number of groups defined by the groupby parameter is < 2.
    """

    # Check that adata is an AnnData object
    # if not isinstance(adata, AnnData):
    #     raise ValueError("adata must be an AnnData object.")

    if adata.obs[groupby].dtype.name != "category":
        adata.obs[groupby] = adata.obs[groupby].astype("category")

    if "log1p" in adata.uns:
        adata.uns['log1p']['base'] = None  # hack for scanpy error; see https://github.com/scverse/scanpy/issues/2239#issuecomment-1104178881

    # Check number of groups in groupby
    if adata.obs[groupby].nunique() < 2:
        raise ValueError("groupby must contain at least two groups.")

    # Catch ImplicitModificationWarning from scanpy
    params = {'method': 't-test'}  # prevents warning message "Default of the method has been changed to 't-test' from 't-test_overestim_var'"
    params.update(kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=anndata.ImplicitModificationWarning, message="Trying to modify attribute.*")
        sc.tl.rank_genes_groups(adata, groupby=groupby, **params)

    sc.tl.filter_rank_genes_groups(adata,
                                   min_in_group_fraction=min_in_group_fraction,
                                   min_fold_change=min_fold_change,
                                   max_out_group_fraction=max_out_group_fraction)

    # Copy rank_genes_groups to rank_genes_<groupby>
    adata.uns["rank_genes_" + groupby] = adata.uns["rank_genes_groups"]
    adata.uns["rank_genes_" + groupby + "_filtered"] = adata.uns["rank_genes_groups_filtered"]


@deco.log_anndata
def pairwise_rank_genes(adata, groupby,
                        foldchange_threshold=1,
                        min_in_group_fraction=0.25,
                        max_out_group_fraction=0.5,
                        **kwargs
                        ) -> pd.DataFrame:
    """
    Rank genes pairwise between groups in 'groupby'.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object containing expression data.
    groupby : str
        Key in adata.obs containing groups to be compared.
    foldchange_threshold : float, default 1
        Minimum foldchange (+/-) to be considered as a marker gene.
    min_in_group_fraction : float, default 0.25
        Minimum fraction of cells in a group that must express a gene to be considered as a marker gene.
    max_out_group_fraction : float, default 0.5
        Maximum fraction of cells in other groups that must express a gene to be considered as a marker gene.
    **kwargs : dict
        Additional arguments to be passed to scanpy.tl.rank_genes_groups.

    Returns
    -------
    pd.DataFrame
        Dataframe containinge the pairwise ranked gened between the groups.
    """

    groups = adata.obs[groupby].astype("category").cat.categories
    contrasts = list(itertools.combinations(groups, 2))

    # Calculate marker genes for each contrast
    tables = []
    for contrast in contrasts:
        print(f"Calculating rank genes for contrast: {contrast}")

        # Get adata for contrast
        adata_sub = adata[adata.obs[groupby].isin(contrast)]   # subset to contrast

        # Run rank_genes_groups
        run_rank_genes(adata_sub, groupby=groupby, **kwargs)

        # Get table
        c1, c2 = contrast
        table_dict = get_rank_genes_tables(adata_sub, out_group_fractions=True)  # returns dict with each group
        table = table_dict[c1]

        # Reorder columns
        table.set_index("names", inplace=True)
        table = table[["scores", "logfoldchanges", "pvals", "pvals_adj", c1 + "_fraction", c2 + "_fraction"]]  # reorder columns
        table = table.copy(deep=True)  # prevent SettingWithCopyWarning

        # Calculate up/down genes
        c1, c2 = contrast
        groups = ["C1", "C2"]
        conditions = [(table["logfoldchanges"] >= foldchange_threshold) & (table[c1 + "_fraction"] >= min_in_group_fraction) & (table[c2 + "_fraction"] <= max_out_group_fraction),  # up
                      (table["logfoldchanges"] <= -foldchange_threshold) & (table[c1 + "_fraction"] <= max_out_group_fraction) & (table[c2 + "_fraction"] >= min_in_group_fraction)]  # down
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
    fraction_columns = [col for col in merged.columns if col.endswith("_fraction")]
    first_columns = [col for col in merged.columns if col not in fraction_columns]
    merged = merged[first_columns + fraction_columns]

    return merged


@deco.log_anndata
def get_rank_genes_tables(adata, key="rank_genes_groups", out_group_fractions=False,
                          var_columns=[], save_excel=None) -> dict[str, pd.DataFrame]:
    """
    Get gene tables containing "rank_genes_groups" genes and information per group (from previously chosen `groupby`).

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object containing ranked genes.
    key : str, default "rank_genes_groups"
        The key in adata.uns to be used for fetching ranked genes.
    out_group_fractions : bool, default False
        If True, the output tables will contain additional columns giving the fraction of genes per group.
    var_columns : list. default []
        List of adata.var columns, which will be added to pandas.DataFrame.
    save_excel : str, default None
        The path to a file for writing the marker gene tables as an excel file (with one sheet per group).

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary with group names as keys, and marker gene tables (pandas DataFrames) per group as values.

    Raises
    ------
    ValueError:
        1. If var_columns is no list.
        2. If not all columns given in var_columns are in adata.var.
    """

    # Check input type
    if not isinstance(var_columns, list):
        raise ValueError("var_columns must be a list of strings.")

    # Check that all given columns are valid
    if len(var_columns) > 0:
        for col in var_columns:
            if col not in adata.var.columns:
                raise ValueError(f"Column '{col}' not found in adata.var.columns.")

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

        group_tables[group] = pd.DataFrame(data)

    # Remove any NaN genes (genes are set to NaN if 'filter_rank_genes_groups' was used)
    for group in group_tables:
        group_tables[group].dropna(inplace=True)

    # Get in/out group fraction of expressed genes (only works for .X-values above 0)
    if (adata.X.min() < 0) == 0:  # only calculate fractions for raw expression data
        groupby = adata.uns[key]["params"]["groupby"]
        n_cells_dict = adata.obs[groupby].value_counts().to_dict()  # number of cells in groups

        for group in groups:

            # Fraction of cells inside group expressing each gene
            s = (adata[adata.obs[groupby].isin([group]), :].X > 0).sum(axis=0).A1  # sum of cells with expression > 0 for this cluster
            expressed = pd.DataFrame([adata.var.index, s]).T
            expressed.columns = ["names", "n_expr"]

            group_tables[group] = group_tables[group].merge(expressed, left_on="names", right_on="names", how="left")
            group_tables[group][group + "_fraction"] = group_tables[group]["n_expr"] / n_cells_dict[group]

            # Fraction of cells for individual groups
            if out_group_fractions is True:
                for compare_group in groups:
                    if compare_group != group:
                        s = (adata[adata.obs[groupby].isin([compare_group]), :].X > 0).sum(axis=0).A1
                        expressed = pd.DataFrame(s, index=adata.var.index)  # expression per gene for this group
                        expressed.columns = [compare_group + "_fraction"]
                        expressed.iloc[:, 0] = expressed.iloc[:, 0] / n_cells_dict[compare_group]

                        group_tables[group] = group_tables[group].merge(expressed, left_on="names", right_index=True, how="left")

            # Fraction of cells outside group expressing each gene
            other_groups = [g for g in groups if g != group]
            s = (adata[adata.obs[groupby].isin(other_groups), :].X > 0).sum(axis=0).A1
            expressed = pd.DataFrame([adata.var.index, s]).T
            expressed.columns = ["names", "n_out_expr"]
            group_tables[group] = group_tables[group].merge(expressed, left_on="names", right_on="names", how="left")

            # If there are only two groups, out_group_fraction -> name of the other group
            if len(groups) == 2:
                out_group_name = [g for g in groups if g != group][0] + "_fraction"
            else:
                out_group_name = "out_group_fraction"

            n_out_group = sum([n_cells_dict[other_group] for other_group in other_groups])  # sum of cells in other groups
            group_tables[group][out_group_name] = group_tables[group]["n_out_expr"] / n_out_group
            group_tables[group].drop(columns=["n_expr", "n_out_expr"], inplace=True)

    # Add additional columns to table
    if len(var_columns) > 0:
        for group in group_tables:
            group_tables[group] = group_tables[group].merge(adata.var[var_columns], left_on="names", right_index=True, how="left")

    # If chosen: Save tables to joined excel
    if save_excel is not None:
        with pd.ExcelWriter(save_excel) as writer:
            for group in group_tables:
                table = group_tables[group].copy()

                # Round values of scores/foldchanges
                table["scores"] = table["scores"].round(3)
                table["logfoldchanges"] = table["logfoldchanges"].round(3)

                table.to_excel(writer, sheet_name=utils._sanitize_sheetname(f'{group}'), index=False)

    return group_tables


@deco.log_anndata
def mask_rank_genes(adata, genes, key="rank_genes_groups", inplace=True) -> Optional[anndata.AnnData]:
    """
    Mask names with "nan" in .uns[key]["names"] if they are found in given 'genes'.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object containing ranked genes.
    genes : list
        List of genes to be masked.
    key : str, default "rank_genes_groups"
        The key in adata.uns to be used for fetching ranked genes.
    inplace : bool, default True
        If True, modifies adata.uns[key]["names"] in place. Otherwise, returns a copy of adata.

    Returns
    -------
    Optional[anndata.AnnData]
        If inplace = True, modifies adata.uns[key]["names"] in place and returns None.
        Otherwise, returns a copy of adata.

    Raises
    ------
    ValueError:
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
def run_deseq2(adata, sample_col, condition_col, confounders=None, layer=None, percentile_range=(0, 100)) -> pd.DataFrame:
    """
    Run DESeq2 on counts within adata. Must be run on the raw counts per sample. If the adata contains normalized counts in .X, 'layer' can be used to specify raw counts.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object containing raw counts.
    sample_col : str
        Column name in adata.obs containing sample names.
    condition_col : str
        Column name in adata.obs containing condition names to be compared.
    confounders : list, default None
        List of additional column names in adata.obs containing confounders to be included in the model.
    layer : str, default None
        Name of layer containing raw counts to be used for DESeq2. Default is None (use .X for counts)
    percentile_range : tuple, default (0, 100)
        Percentile range of cells to be used for calculating pseudobulks. Setting (0,95) will restrict calculation
        to the cells in the 0-95% percentile ranges. Default is (0, 100), which means all cells are used.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the results of the DESeq2 analysis.
        Also adds the dataframe to adata.uns["deseq_result"]

    Raises
    ------
    ValueError:
        1. If any given column name is not found in adata.obs.
        2. If any given column name contains characters not compatible with R.

    Notes
    -----
    Needs the package 'diffexpr' to be installed along with 'bioconductor-deseq2' and 'rpy2'.
    These can be obtained by installing the sctoolbox [deseq2] extra with pip using: `pip install . .[deseq2]`.

    See Also
    --------
    sctoolbox.utils.pseudobulk_table
    """

    utils.setup_R()
    from diffexpr.py_deseq import py_DESeq2

    # Setup the design formula
    if confounders is None:
        confounders = []
    elif not isinstance(confounders, list):
        confounders = [confounders]

    design_formula = "~ " + " + ".join(confounders + [condition_col])

    # Check that sample_col and condition_col are in adata.obs
    cols = [sample_col, condition_col] + confounders
    for col in cols:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' was not found in adata.obs.columns.")

        # Check that column is valid for R
        pattern = r'^[a-zA-Z](?:[a-zA-Z0-9_]*\.(?!$))?[\w.]*$'
        if not re.match(pattern, col):
            s = f"Column '{col}' is not a valid column name within R (which is needed for DEseq2). Please adjust the column name. A valid name is defined as: "
            s += "'A syntactically valid name consists of letters, numbers and the dot or underline characters and starts with a letter or the dot not followed by a number.'"
            raise ValueError(s)

    # Build sample_df
    sample_df = adata.obs[cols].reset_index(drop=True).drop_duplicates()
    sample_df.set_index(sample_col, inplace=True)
    sample_df.sort_index(inplace=True)

    conditions = sample_df[condition_col].unique()
    samples_per_cond = {cond: sample_df[sample_df[condition_col] == cond].index.tolist() for cond in conditions}

    # Build count matrix
    print("Building count matrix")
    count_table = utils.pseudobulk_table(adata, sample_col, how="sum", layer=layer,
                                         percentile_range=percentile_range)
    count_table = count_table.astype(int)  # DESeq2 requires integer counts
    count_table.index.name = "gene"
    count_table.reset_index(inplace=True)

    # Run DEseq2
    print("Running DESeq2")
    dds = py_DESeq2(count_matrix=count_table,
                    design_matrix=sample_df,
                    design_formula=design_formula,
                    gene_column='gene')
    dds.run_deseq()

    # Normalizing counts
    dds.normalized_count()
    dds.normalized_count_df.drop(columns="gene", inplace=True)
    dds.normalized_count_df.columns = dds.samplenames

    # Create result table with mean values per condition
    deseq_table = pd.DataFrame(index=dds.normalized_count_df.index)

    for i, condition in enumerate(conditions):
        samples = samples_per_cond[condition]
        mean_values = dds.normalized_count_df[samples].mean(axis=1)
        deseq_table.insert(i, condition + "_mean", mean_values)

    # Get results per contrast
    contrasts = list(itertools.combinations(conditions, 2))
    for C1, C2 in contrasts:

        dds.get_deseq_result(contrast=[condition_col, C2, C1])
        dds.deseq_result.drop(columns="gene", inplace=True)

        # Rename and add to deseq_table
        dds.deseq_result.drop(columns=["lfcSE", "stat"], inplace=True)
        dds.deseq_result.columns = [C2 + "/" + C1 + "_" + col for col in dds.deseq_result.columns]
        deseq_table = deseq_table.merge(dds.deseq_result, left_index=True, right_index=True)

    # Add normalized individual sample counts to the back of table
    deseq_table = deseq_table.merge(dds.normalized_count_df, left_index=True, right_index=True)

    # Sort by p-value of first contrast
    C1, C2 = contrasts[0]
    deseq_table.sort_values(by=C2 + "/" + C1 + "_pvalue", inplace=True)

    # Add to adata uns
    utils.add_uns_info(adata, "deseq_result", deseq_table)

    return deseq_table


@deco.log_anndata
def score_genes(adata, gene_set, score_name='score', inplace=True) -> Optional[anndata.AnnData]:
    """
    Assign a score to each cell depending on the expression of a set of genes.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object to score.
    gene_set : str or list
        A list of genes or path to a file containing a list of genes.
        The txt file should have one gene per row.
    score_name : str, default "score"
        Name of the column in obs table where the score will be added.
    inplace : bool, default True
        Adds the new column to the original anndata object.

    Returns
    -------
    Optional[anndata.AnnData]
        If inplace is False, return a copy of anndata object with the new column in the obs table.

    Raises
    ------
    FileNotFoundError:
        If path given in gene_set does not lead to a file.
    ValueError:
        If gene_set is not a string and not a list.
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

    # check if gene set is a list
    elif not isinstance(gene_set, list):
        raise ValueError('Please provide genes either as a list or txt file!')

    # scale data
    sdata = sc.pp.scale(adata, copy=True)

    # Score the cells
    sc.tl.score_genes(sdata, gene_list=gene_set, score_name=score_name)
    # add score to adata.obs
    adata.obs[score_name] = sdata.obs[score_name]

    return adata if not inplace else None

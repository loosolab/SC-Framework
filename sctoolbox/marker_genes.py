import os
import re
import glob
import pkg_resources
import pandas as pd
import numpy as np
import scanpy as sc
import itertools
from pathlib import Path
from importlib.resources import files

import sctoolbox.utilities as utils
import sctoolbox.creators as creators


def get_chromosome_genes(gtf, chromosomes):
    """
    Get a list of all genes in the gtf for certain chromosome(s)

    Parameters
    ------------
    gtf : str
        Path to the gtf file.
    chromosomes : str or list
        A chromosome or a list of chromosome names to search for genes in.

    Returns
    ---------
    list :
        A list of all genes in the gtf for the given chromosome(s).

    Note
    ------
    This function is not directly used by the framework, but is used to create the marker gene lists for 'label_genes'.
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


def label_genes(adata,
                gene_column=None,
                species=None):
    """
    Label genes as ribosomal, mitochrondrial, cell cycle phase and gender genes.

    Parameters
    ------------
    adata : anndata.Anndata
        adata object
    gene_column : str, optional
        Name of the column in adata.var that contains the gene names. If not provided, adata.var.index is used.
    species : str, optional
        Name of the species. If not provided, the species is inferred adata.uns["infoprocess"]["species"]

    Notes
    --------
    Author: Guilherme Valente & Mette Bentsen

    """

    # Location of gene lists
    genelist_dir = pkg_resources.resource_filename("sctoolbox", "data/gene_lists/")

    # Get organism from the adata object
    if species is None:
        try:
            species = adata.uns['infoprocess']['species']
        except KeyError:
            raise ValueError("Species not provided and could not be found in adata.uns['infoprocess']['species']")
    species = species.lower()

    # Get the full list of genes from adata
    if gene_column is None:
        adata_genes = adata.var.index
    else:
        adata_genes = adata.var[gene_column]

    # ------- Annotate genes in adata ------ #
    added = []

    # Annotate ribosomal genes
    adata.var["is_ribo"] = adata_genes.str.lower().str.startswith(('rps', 'rpl'))
    added.append("is_ribo")

    # Annotate mitochrondrial genes
    path_mito_genes = genelist_dir + species + "_mito_genes.txt"
    if os.path.exists(path_mito_genes):
        gene_list = utils.read_list_file(path_mito_genes)
        adata.var["is_mito"] = adata_genes.isin(gene_list)  # boolean indicator
        added.append("is_mito")

    else:
        adata.var["is_mito"] = adata_genes.str.lower().str.startswith("mt")  # fall back to mt search

    # Annotate cell cycle genes
    path_cellcycle_genes = genelist_dir + species + "_cellcycle_genes.txt"
    if os.path.exists(path_cellcycle_genes):
        table = pd.read_csv(path_cellcycle_genes, header=None, sep="\t")
        cc_dict = dict(zip(table[0], table[1]))

        adata.var["cellcycle"] = [cc_dict.get(gene, "NA") for gene in adata_genes]  # assigns cell cycle phase or "NA"

    else:
        available_files = glob.glob(genelist_dir + "*_cellcycle_genes.txt")
        available_species = utils.clean_flanking_strings(available_files)
        print(f"No cellcycle genes available for species '{species}'. Available species are: {available_species}")

    # Annotate gender genes
    path_gender_genes = genelist_dir + species + "_gender_genes.txt"
    if os.path.exists(path_gender_genes):
        gene_list = utils.read_list_file(path_gender_genes)
        adata.var["is_gender"] = adata_genes.isin(gene_list)  # boolean indicator
        added.append("is_gender")

    else:
        available_files = glob.glob(genelist_dir + "*_gender_genes.txt")
        available_species = utils.clean_flanking_strings(available_files)
        print(f"No gender genes available for species '{species}'. Available species are: {available_species}")

    # --------- Save information -------- #
    creators.build_infor(adata, "genes_labeled", added)


def add_gene_expression(adata, gene):
    """
    Add values of gene/feature per cell to the adata.obs dataframe.

    Parameters
    ------------
    adata : anndata.AnnData
        Anndata object containing gene expression/counts.
    gene : str
        Name of the gene/feature from the adata.var index to be added to adata.obs.

    Returns
    -----------
    None
        A column named "<gene>_values" is added to adata.obs with the expression/count values from .X
    """

    # Get expression
    if gene in adata.var.index:
        gene_idx = np.argwhere(adata.var.index == gene)[0][0]
        vals = adata.X[:, gene_idx].todense().A1
        adata.obs[gene + "_values"] = vals

    else:
        raise ValueError(f"Gene '{gene}' was not found in adata.var.index")


def get_rank_genes_tables(adata, key="rank_genes_groups", out_group_fractions=False, var_columns=[], save_excel=None):
    """ Get gene tables containing "rank_genes_groups" genes and information per group (from previously chosen `groupby`).

    Parameters
    -----------
    adata : anndata.AnnData
        Anndata object containing ranked genes.
    key : str, optional
        The key in adata.uns to be used for fetching ranked genes. Default: "rank_genes_groups".
    out_group_fractions : bool, optional
        If True, the output tables will contain additional columns giving the fraction of genes per group. Default: False.
    save_excel : str, optional
        The path to a file for writing the marker gene tables as an excel file (with one sheet per group). Default: None (no file is written).

    Returns
    --------
    dict :
        A dictionary with group names as keys, and marker gene tables (pandas DataFrames) per group as values.
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
            group_tables[group]["in_group_fraction"] = group_tables[group]["n_expr"] / n_cells_dict[group]

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
            s = (adata[~adata.obs[groupby].isin([group]), :].X > 0).sum(axis=0).A1
            expressed = pd.DataFrame([adata.var.index, s]).T
            expressed.columns = ["names", "n_out_expr"]

            group_tables[group] = group_tables[group].merge(expressed, left_on="names", right_on="names", how="left")
            group_tables[group]["out_group_fraction"] = group_tables[group]["n_out_expr"] / (sum(n_cells_dict.values()) - n_cells_dict[group])
            group_tables[group].drop(columns=["n_expr", "n_out_expr"], inplace=True)

            # Add additional columns to table
            if len(var_columns) > 0:
                group_tables[group] = group_tables[group].merge(adata.var[var_columns], left_on="names", right_index=True, how="left")

    # If chosen: Save tables to joined excel
    if save_excel is not None:
        with pd.ExcelWriter(save_excel) as writer:
            for group in group_tables:
                table = group_tables[group].copy()

                # Round values of scores/foldchanges
                table["scores"] = table["scores"].round(3)
                table["logfoldchanges"] = table["logfoldchanges"].round(3)

                table.to_excel(writer, sheet_name=utils.sanitize_sheetname(f'{group}'), index=False)

    return group_tables


def mask_rank_genes(adata, genes, key="rank_genes_groups", inplace=True):
    """
    Mask names with "nan" in .uns[key]["names"] if they are found in given 'genes'.

    Parameters
    -----------
    adata : anndata.AnnData
        Anndata object containing ranked genes.
    genes : list
        List of genes to be masked.
    key : str, default: "rank_genes_groups"
        The key in adata.uns to be used for fetching ranked genes.
    inplace : bool, default True
        If True, modifies adata.uns[key]["names"] in place. Otherwise, returns a copy of adata.

    Returns
    -------
    anndata.AnnData or None
        If inplace = True, modifies adata.uns[key]["names"] in place and returns None. Otherwise, returns a copy of adata.
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


def run_rank_genes(adata, groupby,
                   method=None,
                   min_in_group_fraction=0.25,
                   min_fold_change=0.5,
                   max_out_group_fraction=0.8):
    """ Run scanpy rank_genes_groups and filter_rank_genes_groups """

    sc.tl.rank_genes_groups(adata, method=method, groupby=groupby)
    sc.tl.filter_rank_genes_groups(adata,
                                   min_in_group_fraction=min_in_group_fraction,
                                   min_fold_change=min_fold_change,
                                   max_out_group_fraction=max_out_group_fraction)

    # adata.uns["rank_genes_" + groupby] = adata.uns["rank_genes_groups"]
    # adata.uns["rank_genes_" + groupby + "_filtered"] = adata.uns["rank_genes_groups_filtered"]


def run_deseq2(adata, sample_col, condition_col, confounders=None, layer=None, percentile_range=(0, 100)):
    """
    Run DESeq2 on counts within adata. Must be run on the raw counts per sample. If the adata contains normalized counts in .X, 'layer' can be used to specify raw counts.

    Note: Needs the package 'diffexpr' to be installed along with 'bioconductor-deseq2' and 'rpy2'.
    These can be obtained by installing the sctoolbox [deseq2] extra with pip using: `pip install . .[deseq2]`.

    Parameters
    -----------
    adata : anndata.AnnData
        Anndata object containing raw counts.
    sample_col : str
        Column name in adata.obs containing sample names.
    condition_col : str
        Column name in adata.obs containing condition names to be compared.
    confounders : list, default: None
        List of additional column names in adata.obs containing confounders to be included in the model.
    layer : str, default: None
        Name of layer containing raw counts to be used for DESeq2. Default is None (use .X for counts)
    percentile_range : tuple, default: (0, 100)
        Percentile range of cells to be used for calculating pseudobulks. Setting (0,95) will restrict calculation
        to the cells in the 0-95% percentile ranges. Default is (0, 100), which means all cells are used.

    Returns
    -----------
    A py_DESeq2 object containing the results of the DESeq2 analysis.
    Also adds the dataframes to adata.uns["deseq_result"] and adata.uns["deseq_normalized"].

    See also
    -----------
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

    # Build sample_df
    cols = [sample_col, condition_col] + confounders
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

    return deseq_table


def get_celltype_assignment(adata, clustering, marker_genes_dict, column_name="celltype"):
    """
    Get cell type assignment based on marker genes.

    Parameters
    -----------
    adata : anndata.AnnData
        Anndata object containing raw counts.
    clustering : str
        Name of clustering column to use for cell type assignment.
    marker_genes_dict : dict
        Dictionary containing cell type names as keys and lists of marker genes as values.
    column_name : str, default: "celltype"
        Name of column to add to adata.obs containing cell type assignment.

    Returns
    -----------
    Returns a dictionary with cluster-to-celltype mapping (key: cluster name, value: cell type)
    Also adds the cell type assignment to adata.obs[<column_name>] in place.
    """

    if column_name in adata.obs.columns:
        raise ValueError("Column name already exists in adata.obs. Please set a different name using 'column_name'.")

    # todo: make this more robust

    marker_genes_list = [[gene] if isinstance(gene, str) else gene for gene in marker_genes_dict.values()]
    marker_genes_list = sum(marker_genes_list, [])
    sub = adata[:, marker_genes_list]

    # Get long marker genes table
    markers = []
    for celltype, gene_list in marker_genes_dict.items():
        for gene in gene_list:
            markers.append({"celltype": celltype, "gene": gene})

    marker_genes_table = pd.DataFrame(markers)

    # Get pseudobulk table
    table = utils.pseudobulk_table(sub, clustering)
    table.index.name = "genes"
    table.reset_index(inplace=True)

    table_long = table.melt(id_vars=["genes"], var_name=clustering)
    table_long.sort_values("value", ascending=False)

    table_long = table_long.merge(marker_genes_table, left_on="genes", right_on="gene")
    table_long = table_long.drop(columns="gene")

    cluster2celltype = {}
    celltype_count = {}
    for idx, sub in table_long.groupby(clustering):
        mu = sub.groupby("celltype").mean(numeric_only=True).sort_values("value", ascending=False)
        cluster2celltype[idx] = mu.index[0]

    # Make unique
    celltype_count = {}
    celltypes = list(cluster2celltype.values())
    for cluster in cluster2celltype:
        celltype = cluster2celltype[cluster]
        celltype_count[celltype] = celltype_count.get(celltype, 0) + 1

        if celltypes.count(celltype) > 1:
            cluster2celltype[cluster] = f"{celltype} {celltype_count[celltype]}"

    # Add assigned celltype to adata.obs
    table = pd.DataFrame().from_dict(cluster2celltype, orient="index")
    table.columns = [column_name]
    adata.obs = adata.obs.merge(table, left_on=clustering, right_index=True, how="left")

    return cluster2celltype


def predict_cell_cycle(adata, species, s_genes=None, g2m_genes=None, inplace=True):
    """
    Assign a score and a phase to each cell depending on the expression of cell cycle genes.

    Parameters
    -----------
    adata : anndata.AnnData
        Anndata object to predict cell cycle on.
    species : str
        The species of data. Available species are: human, mouse, rat and zebrafish.
        If both s_genes and g2m_genes are given, set species=None,
        otherwise species is ignored.
    s_genes : str or list, default None
        If no species is given or desired species is not supported, you can provide
        a list of genes for the S-phase or a txt file containing one gene in each row.
        If only s_genes is provided and species is a supported input, the default
        g2m_genes list will be used, otherwise the function will not run.
    g2m_genes : str or list, default None
        If no species is given or desired species is not supported, you can provide
        a list of genes for the G2M-phase or a txt file containing one gene per row.
        If only g2m_genes is provided and species is a supported input, the default
        s_genes list will be used, otherwise the function will not run.
    inplace : bool, default True
        if True, add new columns to the original anndata object.

    Returns
    -----------
    scanpy.AnnData or None :
        If inplace is False, return a copy of anndata object with the new column in the obs table.
    """

    if not inplace:
        adata = adata.copy()

    # if two lists are given, check if they are lists or paths
    if s_genes is not None:
        if isinstance(s_genes, np.ndarray):
            s_genes = list(s_genes)
        # check if s_genes is neither a list nor a path
        if not isinstance(s_genes, str) and not isinstance(s_genes, list):
            raise ValueError("Please provide a list of genes or a path to a list of genes!")
        # check if s_genes is a file
        if isinstance(s_genes, str):
            # check if file exists
            if Path(s_genes).is_file():
                s_genes = utils.read_list_file(s_genes)
            else:
                raise FileNotFoundError(f'The list {s_genes} was not found!')

    if g2m_genes is not None:
        if isinstance(g2m_genes, np.ndarray):
            g2m_genes = list(g2m_genes)
        # check if g2m_genes is neither a list nor a path
        if not isinstance(g2m_genes, str) and not isinstance(g2m_genes, list):
            raise ValueError("Please provide a list of genes or a path to a list of genes!")
        # check if g2m_genes is a file
        if isinstance(g2m_genes, str):
            # check if file exists
            if Path(g2m_genes).is_file():
                g2m_genes = utils.read_list_file(g2m_genes)
            else:
                raise FileNotFoundError(f'The list {g2m_genes} was not found!')

    # if two lists are given, use both and ignore species
    if s_genes is not None and g2m_genes is not None:
        species = None

    # get gene list for species
    elif species is not None:
        species = species.lower()

        # get path of directory where cell cycles gene lists are saved
        genelist_dir = files(__name__.split('.')[0]).joinpath("data/gene_lists/")

        # check if given species is available
        available_files = [str(path) for path in list(genelist_dir.glob("*_cellcycle_genes.txt"))]
        available_species = utils.clean_flanking_strings(available_files)
        if species not in available_species:
            raise ValueError(f"No cellcycle genes available for species '{species}'. Available species are: {available_species}")

        # get cellcylce genes lists
        path_cellcycle_genes = genelist_dir / f"{species}_cellcycle_genes.txt"
        cell_cycle_genes = pd.read_csv(path_cellcycle_genes, header=None,
                                       sep="\t", names=['gene', 'phase']).set_index('gene')

        # if one list is given as input, get the other list from gene lists dir
        if s_genes is not None:
            print("g2m_genes list is missing! Using default list instead")
            g2m_genes = cell_cycle_genes[cell_cycle_genes['phase'].isin(['g2m_genes'])].index.tolist()
        elif g2m_genes is not None:
            print("s_genes list is missing! Using default list instead")
            s_genes = cell_cycle_genes[cell_cycle_genes['phase'].isin(['s_genes'])].index.tolist()
        else:
            s_genes = cell_cycle_genes[cell_cycle_genes['phase'].isin(['s_genes'])].index.tolist()
            g2m_genes = cell_cycle_genes[cell_cycle_genes['phase'].isin(['g2m_genes'])].index.tolist()

    else:
        raise ValueError("Please provide either a supported species or lists of genes!")

    # Scale the data before scoring
    sdata = sc.pp.scale(adata, copy=True)

    # Score the cells by s phase or g2m phase
    sc.tl.score_genes_cell_cycle(sdata, s_genes=s_genes, g2m_genes=g2m_genes)

    # add results to adata
    adata.obs['S_score'] = sdata.obs['S_score']
    adata.obs['G2M_score'] = sdata.obs['G2M_score']
    adata.obs['phase'] = sdata.obs['phase']

    if not inplace:
        return adata

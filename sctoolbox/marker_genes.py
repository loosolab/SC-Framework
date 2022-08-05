import os
import re
import glob
import pkg_resources
import pandas as pd

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

    return(gene_names)


def label_genes(adata,
                gene_column=None,
                species=None):
    """
    Label genes as ribosomal, mitochrondrial, cell cycle phase and gender genes.

    Parameters
    ------------
    adata : anndata object
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
        species = adata.uns['infoprocess']['species']
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


def get_rank_genes_tables(adata, key="rank_genes_groups", out_group_fractions=False, save_excel=None):
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
    A dictionary with group names as keys, and marker gene tables (pandas DataFrames) per group as values.
    """

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

    # If chosen: Save tables to joined excel
    if save_excel is not None:
        with pd.ExcelWriter(save_excel) as writer:
            for group in group_tables:
                table = group_tables[group].copy()

                # Round values of scores/foldchanges
                table["scores"] = table["scores"].round(3)
                table["logfoldchanges"] = table["logfoldchanges"].round(3)

                table.to_excel(writer, sheet_name=f'{group}', index=False)

    return(group_tables)

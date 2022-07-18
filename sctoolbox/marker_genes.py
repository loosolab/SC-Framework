import pandas as pd
from sctoolbox.checker import *
from sctoolbox.creators import *


def label_genes(ANNDATA, label=False):
    '''
    OPTIONAL: Label gene types in the anndata object as a variable in the adata.var
    If you wanna execute this part, add label=True

    Parameters
    ------------
    ANNDATA : anndata object
        adata object
    label : Boolean. Default: False
        Set True to perform the gene labelling.
    '''
    # Author: Guilherme Valente
    # Message and others

    lst_parameters = ["mitochondrial", "cell_cycle", "gender_genes", "custom"]
    list_species_cellcycle_annotations = ["human", "mouse", "rat", "zebrafish"]
    file_XY_genes = "xy_genes.txt"
    path_cellcycle_genes = "/mnt/agnerds/loosolab_SC_RNA_framework/marker_genes"
    path_XY_genes = path_cellcycle_genes

    dict_opts = {}
    infor = []  # This list will be part of the adata.uns["infoprocess"]
    m1 = "Annotate "
    m2 = "? Choose y or n"
    m3 = "Choose one species: "
    m4 = "Paste the pathway and filename in which your custom list of genes is deposited.\nNOTE: the file must have one gene per row"
    m5 = "Correct the pathway or filename or type q to quit."
    m6 = "Type the string (case sensitivy) used to identify mit genes, e.g., mt, Mt-, so on."
    opt_quit = ["q", "quit"]
    # opt2=["y", "yes", "n", "no"]
    opt_annotation = list_species_cellcycle_annotations

    def fil_dict(DICT, GENETYPE, LIST):
        DICT[GENETYPE] = ["is_" + GENETYPE]
        DICT[GENETYPE].append(LIST)
        return(DICT)

    # Setting which parameters will be annotated
    if label is True:
        for a in lst_parameters:
            answer = input(m1 + a + m2)
            while check_options(answer) is False:  # Annotate?
                answer = input(m1 + a + m2)
            if a == "mitochondrial" and answer.lower() == "y":  # Annotate mitochondrial
                answer = input(m6)  # Which word use to identify mitochondrial genes?
                tmp_list = [answer]
                fil_dict(dict_opts, a, tmp_list)
            if a == "cell_cycle" and answer.lower() == "y":  # Annotate cell cycle
                tmp_list = []
                answer = input(m3 + ', '.join(list_species_cellcycle_annotations))
                while check_options(answer, OPTS1=opt_quit + opt_annotation) is False:  # Give the species name for cell cycle
                    answer = input(m3 + ', '.join(list_species_cellcycle_annotations))
                for b in open(path_cellcycle_genes + "/" + answer + "_cellcycle_genes.txt"):
                    if b.strip():
                        tmp_list.append(b.split("\t")[0].strip())
                fil_dict(dict_opts, a, tmp_list)

            if a == "gender_genes" and answer.lower() == "y": # Annotate gender genes
                tmp_list = []
                for b in open(path_XY_genes + "/" + file_XY_genes):
                    if b.strip():
                        tmp_list.append(b.split("\t")[0].strip())
                fil_dict(dict_opts, a, tmp_list)

            if a == "custom" and answer.lower() == "y": #Annotate customized genes
                tmp_list = []
                answer = input(m4)
                while path.isfile(answer) is False:
                     if answer.lower() in opt1:
                         sys.exit("You quit and lost all modifications :(")
                     print(m5)
                     answer = input(m5)
                for b in open(answer, "r"):
                    if b.strip():
                        tmp_list.append(b.split("\t")[0].strip())
                fil_dict(dict_opts, a, tmp_list)
    # Annotating
    for k, v in dict_opts.items():
        is_what = v[0]
        genes_tag = v[1]
        if k == "mitochondrial":
            ANNDATA.var[is_what] = ANNDATA.var_names.str.startswith(''.join(genes_tag))
            infor.append(is_what)
        else:
            ANNDATA.var[is_what] = ANNDATA.var_names.isin(genes_tag)
            infor.append(is_what)

    # Annotating in adata.uns["infoprocess"]
    if len(infor) > 0:
        build_infor(ANNDATA, "genes_labeled", infor)
    elif label is False:
        build_infor(ANNDATA, "genes_labeled", "None")
    return(ANNDATA)


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

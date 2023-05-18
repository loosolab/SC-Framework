import sys
import pandas as pd
import pkg_resources
import copy
import subprocess

import sctoolbox.utils as utils


#####################################################################
#              Predict cell types from marker genes dict            #
#####################################################################

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

    # if column_name in adata.obs.columns:
    # raise ValueError("Column name already exists in adata.obs. Please set a different name using 'column_name'.")

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


"""

# a subheader


"""


#####################################################################
#                  Predict cell types using SCSA                    #
#####################################################################

def _match_database(marker_db, input_genes):
    """ Find best matching column in the marker database for the input genes"""

    user_database = pd.read_csv(marker_db, sep="\t")

    highest_perc = -1
    for column in user_database.columns:
        database_genes = set(user_database[column].tolist())
        genes_overlap = set(input_genes).intersection(database_genes)

        percent_overlap = len(genes_overlap) / len(input_genes) * 100

        if percent_overlap > highest_perc:
            highest_perc = percent_overlap
            n_overlap = len(genes_overlap)
            highest_perc = percent_overlap
            best_column = column

    if highest_perc == 0:
        print("No match found in the marker database")
        sys.exit()

    print(f"Best match between input genes and database were found in column '{best_column}' with {n_overlap} genes ({highest_perc:.1f}%)")

    return best_column


def _get_rank_genes(d):
    """ Get a list of unique rank genes from the nested adata.uns["rank_genes_groups"] dictionary

    Parameters
    -----------
    d : dictionary
        The dictionary in adata.uns["rank_genes_groups"]

    Returns
    --------
    A list of unique gene names from  adata.uns["rank_genes_groups"]['names']
    """

    names_dict = {}  # collect names in a dict to remove duplicates
    for lst in d["names"]:
        for name in lst:
            names_dict[name] = ""

    genes = list(names_dict.keys())
    return genes


def run_scsa(adata,
             gene_column=None,
             gene_symbol='auto',
             key='rank_genes_groups',
             column_added='SCSA_pred_celltype',
             inplace=True,
             python_path=None,
             species='human',
             fc=1.5,
             pvalue=0.05,
             tissue='All',
             user_db=None,
             celltype_column="cell_name"
             ):
    """
    A function to run SCSA cell type annotation and assign cell types to cluster in an adata object.

    This is a wrapper function that extracts ranked genes generated by scanpy.tl.rank_genes_groups
    function and generates input matrix for SCSA, then runs SCSA and assigns cell types to clusters
    in adata.obs.

    Also adds adata.uns['SCSA'] as a dictionary with the following keys:
    - 'results': SCSA result table
    - 'stderr': SCSA stderr
    - 'stdout': SCSA stdout
    - 'cmd': SCSA command

    Note:
    ------
    SCSA sometimes gives ValueError: MultiIndex (as covered in https://github.com/bioinfo-ibms-pumc/SCSA/issues/19).
    This can be solved by downgrading pandas to 1.2.4.

    Function parameters
    --------------------
    adata : anndata.AnnData
        Adata object to be annotated, must contain ranked genes in adata.uns
    gene_column : str, optional
        Name of the column in adata.var that contains the gene names. Default: None (takes the index of adata.var)
    gene_symbol : str, optional
        The type of gene symbol. One of "auto", "symbol" (gene name) or "id" (ensembl id). Defaults to 'auto'.
    key : str, optional
        The key in adata.uns where ranked genes are stored. Defaults to 'rank_genes_groups'.
    column_added : str, optional
        The column name in adata.obs where the cell types will be added. Defaults to 'SCSA_pred_celltype'.
    inplace : bool, optional
        If True, cell types will be added to adata.obs. Defaults to True.

    SCSA parameters
    ----------------
    python_path : str, optional
        Path to python. If not given, will be inferred from sys.executable.
    species : str, optional
        Supports only human or mouse. Defaults to 'human'. Set to None to use the user defined database given in user_db.
    fc : float, optional
        Fold change threshold to filter genes. Defaults to 1.5.
    pvalue : float, optional
        P-value threshold to filter genes. Defaults to 0.05.
    tissue : float, optional
        A specific tissue can be defined. Defaults to 'All'.
    user_db : str, optional
        Path to the user defined marker database.
        Must contain at least two columns, one named "cell_name" (or set via celltype_column) for the cell type annotation,
        and at least one more column with gene names or ids (selected automatically from best gene overlap).
    celltype_column : str, optional
        The column name in the user_db that contains the cell type annotation. Defaults to 'cell_name'.

    Returns
    --------
        AnnData: If inplace==False, returns adata with cell types in adata.obs
    """

    if species is not None:
        species = species.capitalize()

    # ---- checking if columns exist in adata ---- #
    if key not in adata.uns.keys():
        raise KeyError(f'{key} was not found in adata.uns! Run rank_genes_groups first')

    # Get groupby from adata.uns
    try:
        groupby = adata.uns[key]['params']['groupby']
    except Exception:
        raise KeyError(f"Could not find 'params' within adata.uns[{key}]. Please ensure that this key contains results of rank_genes_groups.")

    # Check species and user.db
    if species not in ['Human', 'Mouse', None]:
        raise ValueError('Supported species are only: human or mouse! To annotate other species, set species=None and provide a user_db')
    if not user_db and not species:
        raise ValueError('If no species is provided, user_db must be given! Supported species are: human or mouse! If you want to annotate other species, please provide a marker genes list using the parameter: user_db')

    # Get paths to scripts and files
    if not python_path:
        python_path = sys.executable

    scsa_path = pkg_resources.resource_filename("sctoolbox", "data/SCSA_custom.py")

    if species is not None:
        marker_db = pkg_resources.resource_filename("sctoolbox", f"data/celltype_markers/cellmarker_{species.lower()}.tsv")
    else:
        marker_db = user_db

    # ---- fetching ranked genes from adata.uns ---- #
    result = copy.deepcopy(adata.uns[key])
    if gene_column is not None:

        # gene_column must be in adata.var
        if gene_column not in adata.var.columns:
            raise KeyError(f'{gene_column} was not found in adata.var')

        # Translate index names to names from gene_column
        idx2name = dict(zip(adata.var.index, adata.var[gene_column]))
        for i in range(len(result["names"])):
            for j in range(len(result["names"][i])):
                result["names"][i][j] = idx2name[result["names"][i][j]]

    # ---- Find out which gene symbol to use ---- #
    all_genes = _get_rank_genes(result)
    print("Found {} genes from input ranked genes".format(len(all_genes)))
    print("Checking if genes are in the database...")

    # Read database and find best matching gene column
    gene_column = _match_database(marker_db, all_genes)

    # ---- Setup table for SCSA input ---- #
    groups = result['names'].dtype.names
    dat = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'logfoldchanges', 'scores', 'pvals']})

    # Fill duplicate genes with _NA
    name_columns = [col for col in dat if col.endswith("_n")]
    for col in name_columns:
        dups = dat[col].duplicated(keep='first')
        dat[col].mask(dups, other="_NA", inplace=True)  # replace all duplicates with _NA

    # Save to file
    csv = './scsa_input.csv'
    dat.to_csv(csv)

    # ---- building the SCSA command ---- #
    results_path = "./scsa_results.txt"
    utils.create_dir(results_path)  # make sure the full path to results exists

    scsa_cmd = f"{python_path} {scsa_path} -i {csv} -f {fc} -p {pvalue} -o {results_path} -m txt "
    scsa_cmd += f"--db {marker_db} "
    scsa_cmd += f"--cellcol {celltype_column} --genecol {gene_column}"

    # ---- run SCSA command ---- #
    print('Running SCSA...')
    p = subprocess.run(scsa_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = p.stderr
    stdout = p.stdout
    if p.returncode != 0:
        raise ValueError(f"SCSA failed with error: {stderr.decode('utf-8')}")

    # ---- read results_path and assign to adata.obs ---- #
    df = pd.read_csv(results_path, sep='\t', engine='python')
    adata.uns["SCSA"] = df

    # Save the celltype with the best z-score to adata.obs
    df_max1 = df.groupby('Cluster').first()
    df_max = df_max1.drop(columns=['Z-score'])
    df_max = df_max.reset_index()
    df_max = df_max.rename(columns={'Cell Type': 'Cell_Type'})
    df_max = df_max.astype(str)
    dictMax = dict(zip(df_max.Cluster, df_max.Cell_Type))

    print(f"Done. Best scoring celltype was added to '{column_added}' and the full results were added to adata.uns['SCSA']")
    for _, row in df.drop_duplicates(subset='Cluster', keep='first').iterrows():
        print(f"Cluster {row['Cluster']} was annotated with celltype: {row['Cell Type']}")

    # Save results to uns dictionary
    scsa_uns_dict = {"SCSA": {"results": df,
                              "stderr": stderr.decode('utf-8'),
                              "stdout": stdout.decode('utf-8'),
                              "cmd": scsa_cmd}}

    # Remove the temporary files
    files = [csv, results_path]
    utils.remove_files(files)

    # Add the annotated celltypes to the anndata-object
    if inplace:
        adata.obs[column_added] = adata.obs[groupby].map(dictMax)
        adata.uns.update(scsa_uns_dict)
    else:
        assigned_adata = adata.copy()
        assigned_adata.obs[column_added] = assigned_adata.obs[groupby].map(dictMax)
        assigned_adata.uns.update(scsa_uns_dict)
        return assigned_adata


#####################################################################
#                  Cell types using custom script                   #
#####################################################################

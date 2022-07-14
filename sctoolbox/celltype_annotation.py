from logging import raiseExceptions
import os
import sys
import pandas as pd
import pkg_resources
import sctoolbox.utilities


def run_scsa(adata,
            results_path='scsa_results.csv', 
            species='human', 
            fc=1.5, 
            pvalue=0.01,
            tissue='All',
            celltype='normal',
            python_path=None, 
            scsa_path=None,
            wholedb_path=None,
            gene_symbol=True, 
            user_db=None, 
            z_score='best', inplace=True, 
            key='rank_genes_groups', 
            key_added='SCSA_pred_celltypes', 
            ):
    function and generates input matrix for SCSA, then runs SCSA and assigns cell types to clusters
        Path where the results csv file will be saved. Defaults to 'scsa_results.csv'.
    python_path : str, optional
        Path to python. If not given, will be inferred from sys.executable.
    scsa_path : str, optional
        Path to SCSA.py. Default is to use the SCSA.py file in the sctoolbox/data folder.
    wholedb_path : str, optional
        Path to whole.db. Default is to use the whole.db file in the sctoolbox/data folder.
    ----------
    adata : anndata.AnnData
        Adata object to be annotated, must contain ranked genes in adata.uns
    scsa_path : str 
        Path to SCSA.py
    wholedb_path : str 
        Path to whole.db
    results_path : str, optional
        Path where the results csv file will be saved. Defaults to ''.
    python : str, optional
        Path to python. If not given, will be infered form sys.executable.
    species : str, optional
        Supports only human or mouse. Defaults to Human.
    source : str, optional
    user_db : str, optional
        Path to the user defined marker database. Defaults to None.
        Fold change threshold to filter genes. Defaults to 1.5.
    pvalue : float, optional
        P value threshold to filter. Defaults to 0.01.
    tissue : float, optional
        A specific tissue can be defined. Defaults to 'All'.
    celltype : str, optional
        Either normal or cancer. Defaults to 'normal'.
    gene_symbol : str, optional
        Whether the genes in adata are gene names. Defaults to True.
    user_db : bool, optional
        Path to the user defined marker database . Defaults to False.
    z_score : str, optional
        Whether to choose the best scoring cell type. Defaults to 'best'.
    inplace : bool, optional
        If True, cell types will be added to adata.obs. Defaults to True.
    key : str, optional
        The key in adata.uns where ranked genes are stored. Defaults to 'rank_genes_groups'.
    key_added : str, optional
        The column name in adata.obs where the cell types will be added. Defaults to 'SCSA_pred_celltypes'.
    clusters_col : str, optional
            Column in adata.obs where the cluster annotation is found. Defaults to 'leiden'.

    Returns
    --------
        AnnData: If inplace==False, returns adata with cell types in adata.obs
    """

    ### checking if columns exist in adata ###
    if key not in adata.uns.keys():
        raise KeyError(f'{key} was not found in adata.uns! Run rank_genes_groups first')
    #Get paths to scripts and files
    if not python_path: 
        python_path = sys.executable
    if not scsa_path:
        pkg_resources.resource_filename("sctoolbox", "data/SCSA.py")
    if not wholedb_path:
        pkg_resources.resource_filename("sctoolbox", "data/whole.db")

    sctoolbox.utilities.create_dir(results_path) #make sure the full path to results exists

    ### fetching ranked genes from adata.uns ###
    result = adata.uns[key]
    groups = result['names'].dtype.names
    dat = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'logfoldchanges','scores','pvals']})
    csv = './ranked_genes.csv'
    dat.to_csv(csv)

    ### building the SCSA command ###
    cmd_output = './output.txt'  # the output of the command will be saved to this file
    
    # setting the path to python
    if not python:
        python=sys.executable
    
    if not results_path:
        results_path = './scsa_results.csv'
    else:
        # getting folder path
        results_path = os.path.dirname(results_path)
        # checking if foldr exists
        if not os.path.isdir(results_path):
            raise FileNotFoundError(f'The folder: ({results_path}) does not exist!')
        # appending /scsa_results.csv to scsa path
        results_path += '/scsa_results.csv'

    if not species:
        scsa_cmd = f"{python} {scsa_path} -d {wholedb_path} -i {csv} -s {source} -k {tissue} -b -f {fc} -p {pvalue} -o {results_path} -m txt"
        if gene_symbol:
            scsa_cmd += ' -E'
        if user_db:
            scsa_cmd += f' -M {user_db} -N'

    elif species.lower() == 'human' or species == 'mouse':
        species = species.capitalize()
        scsa_cmd = f"{python} {scsa_path} -d {wholedb_path} -i {csv} -s {source} -k {tissue} -b -g {species} -f {fc} -p {pvalue} -o {results_path} -m txt"
        if gene_symbol:
            scsa_cmd += ' -E'
        if user_db:
            scsa_cmd += f' -M {user_db} -N'

    else:
        raise ValueError('Supported species are: human or mouse')

    # writing the output of the run to the output file
    scsa_cmd += f" > {cmd_output}"

    ### run SCSA command ###
    print('running SCSA...\n')
    os.system(scsa_cmd)

    ### read results_path and assign to adata.obs ###
    if z_score == 'best':
        df = pd.read_csv(results_path, sep='\t', engine='python')
        df_max1 = df.groupby('Cluster').first()
        df_max = df_max1.drop(columns=['Z-score'])
        df_max = df_max.reset_index()
        df_max = df_max.rename(columns={'Cell Type':'Cell_Type'})
        df_max = df_max.astype(str)
        dictMax = dict(zip(df_max.Cluster, df_max.Cell_Type))

    print('Done')

    ###  Add the annotated celltypes to the anndata-object ###
    if inplace:
        adata.obs['SCSA_pred_celltype'] = adata.obs[clusters_col].map(dictMax)
    else:
        assigned_adata = adata.copy()
        assigned_adata.obs['SCSA_pred_celltype'] = assigned_adata.obs[clusters_col].map(dictMax)
        return assigned_adata

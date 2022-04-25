import pandas as pd

def get_rank_genes_tables(adata, key="rank_genes_groups", save_excel=None):
	""" Get gene tables containing "rank_genes_groups" genes and information per group (from previously chosen `groupby`).
	
	Parameters
	-----------
	adata : AnnData
		Anndata object containing ranked genes.
	key : str, optional
		The key in adata.uns to be used for fetching ranked genes. Default: "rank_genes_groups".
	save_excel : str, optional
		The path to a file for writing the marker gene tables as an excel file (with one sheet per group). Default: None (no file is written).
	
	Returns
	--------
	A dictionary with group names as keys, and marker gene tables (pandas DataFrames) per group as values.
	"""
	
	#Read structure in .uns to pandas dataframes
	tables = {}
	for col in ["names", "scores", "pvals", "pvals_adj", "logfoldchanges"]:
		tables[col] = pd.DataFrame(adata.uns[key][col])

	#Transform to one table per group
	groups = tables["names"].columns
	group_tables = {}
	for group in groups:
		data = {}
		for col in tables:
			data[col] = tables[col][group].values
		
		group_tables[group] = pd.DataFrame(data)
	
	#Get in/out group fraction of expressed genes (only works for .X-values above 0)
	n_negative = sum(sum(adata.X < 0))
	if n_negative == 0: #only calculate fractions for raw expression data
		groupby = adata.uns[key]["params"]["groupby"]
		n_cells_dict = adata.obs[groupby].value_counts().to_dict() #number of cells in groups

		for group in groups: 

			#Fraction of cells inside group expressing each gene
			s = (adata[adata.obs[groupby].isin([group]),:].X > 0).sum(axis=0).A1 #sum of cells with expression > 0 for this cluster
			expressed = pd.DataFrame([adata.var.index, s]).T
			expressed.columns = ["names", "n_expr"]
			
			group_tables[group] = group_tables[group].merge(expressed, left_on="names", right_on="names", how="left")
			group_tables[group]["in_group_fraction"] = group_tables[group]["n_expr"] / n_cells_dict[group]
			
			#Fraction of cells outside group expressing each gene
			s = (adata[~adata.obs[groupby].isin([group]),:].X > 0).sum(axis=0).A1
			expressed = pd.DataFrame([adata.var.index, s]).T
			expressed.columns = ["names", "n_out_expr"]
			
			group_tables[group] = group_tables[group].merge(expressed, left_on="names", right_on="names", how="left")
			group_tables[group]["out_group_fraction"] = group_tables[group]["n_out_expr"] / (sum(n_cells_dict.values()) - n_cells_dict[group])
			
			group_tables[group].drop(columns=["n_expr", "n_out_expr"], inplace=True)
	
	#If chosen: Save tables to joined excel
	if save_excel is not None:
		with pd.ExcelWriter(save_excel) as writer:
			for group in group_tables:
				table = group_tables[group].copy()
				
				#Round values of scores/foldchanges
				table["scores"] = table["scores"].round(3)
				table["logfoldchanges"] = table["logfoldchanges"].round(3)
				
				table.to_excel(writer, sheet_name=f'{group}', index=False)
				
	return(group_tables)
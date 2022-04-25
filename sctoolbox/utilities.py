import pandas as pd

def pseudobulk_table(adata, groupby, how="mean"):
	""" Get a pseudobulk table of values per cluster. 
	
	Parameters
	-----------
	adata : anndata object
		An annotated data matrix containing counts in .X.
	groupby : str
		Name of a column in adata.obs to cluster the pseudobulks by.
	how : str, optional
		How to calculate the value per cluster. Can be one of "mean" or "sum". Default: "mean"
	"""
	
	adata = adata.copy()
	adata.obs[groupby] = adata.obs[groupby].astype('category')
	
	#Fetch the mean/sum counts across each category in cluster_by
	res = pd.DataFrame(columns=adata.var_names, index=adata.obs[groupby].cat.categories)                                                      
	for clust in adata.obs[groupby].cat.categories: 
		
		if how == "mean":
			res.loc[clust] = adata[adata.obs[groupby].isin([clust]),:].X.mean(0)
		elif how == "sum":
			res.loc[clust] = adata[adata.obs[groupby].isin([clust]),:].X.sum(0)
	
	res = res.T #transform to genes x clusters
	return(res)

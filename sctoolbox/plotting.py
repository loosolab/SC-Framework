"""
Modules for plotting single cell data
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scanpy as sc
import qnorm
import sctoolbox.utilities
from sklearn.preprocessing import MinMaxScaler


def search_umap_parameters(adata, dist_min=0.1, dist_max=0.4, dist_step=0.1,
								  spread_min=2.0, spread_max=3.0, spread_step=0.5,
								  metacol="Sample", verbose=True):
	""" Investigate different combinations of dist_min and spread variables for UMAP plots"""
	
	adata = adata.copy()
	
	dists = np.arange(dist_min, dist_max, dist_step)
	dists = np.around(dists, 2)
	spreads = np.arange(spread_min, spread_max, spread_step)
	spreads = np.around(spreads, 2)
	
	#Figure with rows=spread, cols=dist
	fig, axes = plt.subplots(len(spreads), len(dists), figsize=(4*len(dists), 4*len(spreads))) 
	
	#Create umap for each combination of spread/dist
	for i, spread in enumerate(spreads): #rows
		for j, dist in enumerate(dists):  #columns
			
			if verbose == True:
				print(f"Plotting umap for spread={spread} and dist={dist} ({i*len(dists)+j+1}/{len(dists)*len(spreads)})")
		
			#Set legend loc
			#Add legend to last column
			if i == 0 and j == (len(dists)-1):
				legend_loc = "left"
			else:
				legend_loc = "none"
				
			sc.tl.umap(adata, min_dist=dist, spread=spread, n_components=3)
			sc.pl.umap(adata, color=metacol, title='', legend_loc=legend_loc, show=False, ax=axes[i,j])
			
			if j == 0:
				axes[i,j].set_ylabel(f"spread: {spread}")
			else:
				axes[i,j].set_ylabel("")
			
			if i == 0:
				axes[i,j].set_title(f"min_dist: {dist}")
			
			axes[i,j].set_xlabel("")
	
	plt.tight_layout()
	plt.show()


def group_expression_boxplot(adata, gene_list, groupby, figsize=None):
	""" 
	Plot a boxplot showing gene expression of genes in `gene_list` across the groups in `groupby`. The total gene expression is quantile normalized 
	per group, and are subsequently normalized to 0-1 per gene across groups.
	
	Parameters
	------------
	adata : anndata.AnnData object
		An annotated data matrix containing counts in .X.
	gene_list : list
		A list of genes to show expression for.
	groupby : str
		A column in .obs for grouping cells into groups on the x-axis
	figsize : tuple, optional
		Control the size of the output figure, e.g. (6,10). Default: None (matplotlib default).
	"""
	
	#Obtain pseudobulk
	gene_table = sctoolbox.utilities.pseudobulk_table(adata, groupby)
	
	#Normalize across clusters
	gene_table = qnorm.quantile_normalize(gene_table, axis=1)
	
	#Normalize to 0-1 across groups
	scaler = MinMaxScaler()
	df = gene_table.T
	df[df.columns] = scaler.fit_transform(df[df.columns])
	gene_table = df
	
	#Melt to long format
	gene_table_melted = gene_table.reset_index().melt(id_vars="index", var_name="gene")
	gene_table_melted.rename(columns={"index": groupby}, inplace=True)
	
	#Subset to input gene list
	gene_table_melted = gene_table_melted[gene_table_melted["gene"].isin(gene_list)]
	
	#Sort by median 
	medians = gene_table_melted.groupby(groupby).median()
	medians.columns = ["medians"]
	gene_table_melted_sorted = gene_table_melted.merge(medians, left_on=groupby, right_index=True).sort_values("medians", ascending=False)

	#Joined figure with all
	fig, ax = plt.subplots(figsize=figsize)
	g = sns.boxplot(data=gene_table_melted_sorted, x=groupby, y="value", ax=ax, color="darkgrey")
	ax.set_ylabel("Normalized expression")
	
	ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha="right")
	
	return(g)
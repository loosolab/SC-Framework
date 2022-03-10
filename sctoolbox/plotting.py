"""
Modules for plotting single cell data
"""

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

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
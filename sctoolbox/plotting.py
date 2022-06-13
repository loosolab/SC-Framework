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


def qcf_ploting(DFCELLS, DFGENES, COLORS, DFCUTS, PLOT=None, SAVE=None, SAVE_PATH=None, FILENAME=None):
    '''
    Violin plot with cutoffs
    Parameters
    ------------
    DEFCELLs : Pandas dataframe
        Anndata.obs variables to be used for plot. The first colum MUST be the condition or sample description
    DFGENES : Pandas dataframe
        Anndata.var variables to be used for plot
    COLORS : List
        Name of colors to be used in the plot
    DFCUTS : Pandas dataframe
        Dataframe with conditions, parameters and cutoffs as columns for both DEFCELLs and DEFGENEs.
        The cutoffs must be a list
    PLOT : List. Default None
        List of parameters that the cutoff lines will be plotted.
    SAVE : Boolean
        True, save the figure. Default: None (figure is not saved).
    SAVE_PATH : String
        Pathway to save the figure. It will be used if SAVE==True. Default: None
    FILENAME : String
        Name of file to be saved. It will be used if SAVE==True. Default: None
    '''
    #Author : Guilherme Valente
    def defin_cut_lnes(NCUTS): #NCUTS define the number of cuts of X axis
        range_limits=np.linspace(0,1,2+NCUTS).tolist()
        list_limits=[]
        index, counter=0, 1
        while counter <= NCUTS+1:
            minim, maximim=round(range_limits[index],2), round(range_limits[index+1],2)
            if counter < NCUTS+1:
                maximim=maximim-0.01
            list_limits.append((minim, maximim))
            index, counter=index+1, counter+1
        return(list_limits)
#Definining the parameters to be ploted
    lst_dfcuts_cols2=DFCUTS.columns.tolist()
#Separating dataframes for the anndata obs and var information
    for_cells, for_genes = DFCUTS[DFCUTS[lst_dfcuts_cols2[3]] == "filter_cells"], DFCUTS[DFCUTS[lst_dfcuts_cols2[3]] == "filter_genes"]
#Defining the X axis lines limits
    lmts_X_for_cel, lmts_X_for_gen = defin_cut_lnes((len(for_cells[lst_dfcuts_cols2[0]].unique()))-1), defin_cut_lnes((len(for_genes[lst_dfcuts_cols2[0]].unique()))-1)
#Ploting variables in DEFCELLs and DFGENES separately
    ncols=3
    nrows=(len(DFCELLS.columns) + len(DFCELLS.columns) - 2)/ncols
    if (nrows % 2) != 0:
        nrows=int(nrows)+1
    fig, a = plt.subplots(int(nrows), ncols, figsize = (ncols*5, int(nrows)*5))
    labelsize, fontsize, a = 14, 20, a.ravel()
    def plot_cut_lines(a, limits):
        ax.axhline(y=max(a), xmin=limits[0], xmax=limits[1], c="orange", ls="dashed", lw=3, label=round(max(a), 3))
        ax.axhline(y=min(a), xmin=limits[0], xmax=limits[1], c="orange", ls="dashed", lw=3, label=round(min(a), 3))
    for idx, ax in enumerate(a):
        if idx <= len(DFCELLS.columns)-2:
            lines=for_cells[for_cells[lst_dfcuts_cols2[1]].str.contains(DFCELLS.iloc[:, idx + 1].name)]
            condi_cut=lines[[lst_dfcuts_cols2[0], lst_dfcuts_cols2[2]]]
            parameter=''.join(lines[lst_dfcuts_cols2[1]].unique().tolist())
            sns.violinplot(x=DFCELLS.iloc[:, 0], y=DFCELLS.iloc[:, idx + 1], ax=ax, palette=COLORS)
            counter=0
            for a in condi_cut[lst_dfcuts_cols2[2]].to_list():
                if PLOT != None and parameter in PLOT:
                    plot_cut_lines(a, lmts_X_for_cel[counter])
                else:
                    pass
                counter=counter+1
            ax.set_title("Cells: " + DFCELLS.columns[idx +1 ], fontsize=fontsize)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=labelsize)
        else:
            lines=for_genes[for_genes[lst_dfcuts_cols2[1]].str.contains(DFGENES.iloc[:, idx - 3].name)]
            param_cut=lines[[lst_dfcuts_cols2[1], lst_dfcuts_cols2[2]]]
            parameter=''.join(lines[lst_dfcuts_cols2[1]].unique().tolist())
            sns.violinplot(data=DFGENES.iloc[:, idx - 3], ax=ax, color="grey")
            for a in param_cut[lst_dfcuts_cols2[2]].to_list():
                if PLOT != None and parameter in PLOT:
                    plot_cut_lines(a, lmts_X_for_gen[0])
                else:
                    pass
            ax.set_title("Genes: " + DFGENES.columns[idx - 3 ], fontsize=fontsize)
            ax.tick_params(labelsize=labelsize)
    fig.tight_layout()
#Save plot
    if SAVE == True:
        path_filename="note2_" + SAVE_PATH + "/" + FILENAME + ".tiff"
        fig.savefig(path_filename, dpi=300, bbox_inches="tight")

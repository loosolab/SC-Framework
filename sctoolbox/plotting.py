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
import pandas as pd

#############################################################################
###################### PCA/tSNE/UMAP plotting functions #####################
#############################################################################

def search_umap_parameters(adata, dist_min=0.1, dist_max=0.4, dist_step=0.1,
                                  spread_min=2.0, spread_max=3.0, spread_step=0.5,
                                  metacol="Sample", n_components=2, verbose=True):
    """ 
    Plot a grid of different combinations of min_dist and spread variables for UMAP plots. 
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    dist_min : float
        Min value for the UMAP parameter 'min_dist'. Default: 0.1.
    dist_max : float
        Max value for the UMAP parameter 'min_dist'. Default: 0.4.
    dist_step : float
        Step size for the UMAP parameter 'min_dist'. Default: 0.1.
    spread_min : float
        Min value for the UMAP parameter 'spread'. Default: 2.0.
    spread_max : float
        Max value for the UMAP parameter 'spread'. Default: 3.0.
    spread_step : float
        Step size for the UMAP parameter 'spread'. Default: 0.5.
    metacol : str
        Name of the column in adata.obs to color by. Default: "Sample".
    n_components : int
        Number of components in UMAP calculation. Default: 2.
    verbose : bool
        Print progress to console. Default: True.
    """
    
    adata = adata.copy()

    #Setup parameters to loop over
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
        
            #Set legend loc for last column
            if i == 0 and j == (len(dists)-1):
                legend_loc = "left"
            else:
                legend_loc = "none"
                
            sc.tl.umap(adata, min_dist=dist, spread=spread, n_components=n_components)
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

    
def plot_group_embeddings(adata, groupby, embedding="umap", ncols=4):
    """ Plot a grid of embeddings (UMAP/tSNE/PCA) per group of cells within 'groupby'.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    groupby : str
        Name of the column in adata.obs to group by.
    embedding : str
        Embedding to plot. Must be one of "umap", "tsne", "pca". Default: "umap".
    ncols : int
        Number of columns in the figure. Default: 4.
    """
    
    #Get categories
    groups = adata.obs[groupby].cat.categories
    n_groups = len(groups)
    
    #Find out how many rows are needed
    ncols = min(ncols, n_groups) #Make sure ncols is not larger than n_groups
    nrows = int(np.ceil(len(groups) / ncols))

    #Setup subplots
    fig, axarr = plt.subplots(nrows, ncols, figsize = (ncols*5, nrows*5))
    axarr = np.array(axarr).reshape((-1, 1)) if ncols == 1 else axarr
    axarr = np.array(axarr).reshape((1, -1)) if nrows == 1 else axarr
    axes_list = [b for a in axarr for b in a]
    n_plots = len(axes_list)
    
    #Plot UMAP/tSNE/pca per group
    for i, group in enumerate(groups):
        
        ax = axes_list[i]
        
        #Plot individual embedding
        if embedding == "umap":
            sc.pl.umap(adata, color=groupby, groups=group, ax=ax, show=False, legend_loc=None)
        elif embedding == "tsne":
            sc.pl.tsne(adata, color=groupby, groups=group, ax=ax, show=False, legend_loc=None)
        elif embedding == "pca":
            sc.pl.pca(adata, color=groupby, groups=group, ax=ax, show=False, legend_loc=None)

        ax.set_title(group)
    
    #Hide last empty plots
    n_empty = n_plots - n_groups
    if n_empty > 0:
        for ax in axes_list[-n_empty:]:
            ax.set_visible(False)
    
    plt.show()


def compare_embeddings(adata_list, var_list, embedding="umap", adata_names=None):
    """ Compare embeddings across different adata objects. Plots a grid of embeddings with the different adatas on the 
    x-axis, and colored variables on the y-axis.
    
    Parameters
    ----------
    adata_list : list of anndata.AnnData
        List of AnnData objects to compare.
    var_list : list of str
        List of variables to color in plot.
    embedding : str
        Embedding to plot. Must be one of "umap", "tsne" or "pca". Default: "umap".
    adata_names : list of str
        List of names for the adata objects. Default: None (adatas will be named adata_1, adata_2, etc.).
    """
    
    embedding = embedding.lower()

    available = adata_list[0].var.index.tolist() + adata_list[0].obs.columns.tolist()
    var_list = [var for var in var_list if var in available]
    
    n_adata = len(adata_list)
    n_var = len(var_list)
    fig, axes = plt.subplots(n_var, n_adata, figsize=(4*n_adata, 4*n_var))
    
    #Fix indexing
    n_cols = n_adata
    n_rows = n_var
    axes = np.array(axes).reshape((-1, 1)) if n_cols == 1 else axes		#Fix indexing for one column figures
    axes = np.array(axes).reshape((1, -1)) if n_rows == 1 else axes		#Fix indexing for one row figures
    
    if adata_names == None:
        adata_names = [f"adata_{n+1}" for n in range(len(adata_list))]
    
    import matplotlib.colors as clr
    cmap = clr.LinearSegmentedColormap.from_list('custom umap', ['#f2f2f2','#ff4500'], N=256)
    
    for i, adata in enumerate(adata_list):
        for j, var in enumerate(var_list):
            
            if embedding == "umap":
                sc.pl.umap(adata, color=var, show=False, ax=axes[j,i])
            elif embedding == "tsne":
                sc.pl.tsne(adata, color=var, show=False, ax=axes[j,i])
            elif embedding == "pca":
                sc.pl.pca(adata, color=var, show=False, ax=axes[j,i], annotate_var_explained=True)
            
            #Set y-axis label
            if i == 0:
                axes[j,i].set_ylabel(var)
            else:
                axes[j,i].set_ylabel("")
            
            #Set title
            if j == 0:
                axes[j,i].set_title(adata_names[i])
            else:
                axes[j,i].set_title("")
            
            axes[j,i].set_xlabel("")
    
    #fig.tight_layout()


#############################################################################
#################### Other overview plots for expression  ###################
#############################################################################

def n_cells_barplot(adata, x, groupby=None):
    """
    Plot number and percentage of cells per group in a barplot.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    x : str
        Name of the column in adata.obs to group by on the x axis.
    groupby : str
        Name of the column in adata.obs to created stacked bars on the y axis. Default: None (the bars are not split).
    """
    
    #Get cell counts for groups or all
    tables = []
    if groupby is not None:
        for i, frame in adata.obs.groupby(groupby):
            count = frame.value_counts(x).to_frame(name="count").reset_index()
            count["groupby"] = i
            tables.append(count)
        counts = pd.concat(tables)
        
    else:
        counts = adata.obs[x].value_counts().to_frame(name="count").reset_index()
        counts.rename(columns={"index": x}, inplace=True)
        counts["groupby"] = "all"
    
    #Format counts
    counts_wide = counts.pivot(index=x, columns="groupby", values="count")
    counts_wide_percent = counts_wide.div(counts_wide.sum(axis=1), axis=0) * 100
    
    #Plot barplots
    fig, axarr = plt.subplots(1,2, figsize=(10,3))

    counts_wide.plot.bar(stacked=True, ax=axarr[0], legend=False)
    axarr[0].set_ylabel("Number of cells")
    axarr[0].set_xticklabels(axarr[0].get_xticklabels(), rotation=45, ha="right")

    counts_wide_percent.plot.bar(stacked=True, ax=axarr[1])
    plt.ylabel("Percent of cells")
    axarr[1].set_xticklabels(axarr[1].get_xticklabels(), rotation=45, ha="right")
    
    #Set location of legend
    if groupby is None:
        axarr[1].get_legend().remove()
    else:
        axarr[1].legend(title=groupby, bbox_to_anchor=(1,1))

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


#############################################################################
########################## Quality control plotting #########################
#############################################################################

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

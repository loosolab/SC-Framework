import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def cell_mitochondrial():
    '''
    Filter cells based on the mitochondrial content.
    Exclude >= cutoff
            Default cutoff: automatic cutoff from from distribution analysis
            Custom cutoff: user define the cutoff
    Return anndata
    '''

def cell_numgenes():
    '''
    Filter cells based on the number of genes.
    Exclude <= cutoff
            Default cutoff: automatic cutoff from from distribution analysis
            Custom cutoff: user define the cutoff
    Return anndata
    '''

def filter_genes(adata, genes):
    """ Remove genes from adata object. 
    
    Parameters
    ------------
    adata : AnnData
        Anndata object to filter
    genes : list of str
        A list of genes to remove from object.
    """

    #Check if all genes are found in adata
    not_found = list(set(genes) - set(adata.var_names))
    if len(not_found) > 0:
        print("{0} genes were not found in adata and could therefore not be removed. These genes are: {1}".format(len(not_found), not_found))
    
    #Remove genes from adata
    n_before = adata.shape[1]
    adata = adata[:, ~adata.var_names.isin(genes)]
    n_after = adata.shape[1]
    print("Filtered out {0} genes from adata. New number of genes is: {1}.".format(n_before-n_after, n_after))

    return(adata)


def quality_violin(adata, groupby, columns, header, color_list, title=None, save=None):
    """
    A function to plot quality measurements for cells in an anndata object.

    Parameters
    -------------
    adata : AnnData
        Anndata object containing quality measures in .obs.
    groupby : str
        A column in .obs to group cells on the x-axis, e.g. 'condition'. 
    columns : list
        A list of columns in .obs to show measures for.
    header : list
        A list of headers for each measure given in columns.
    color_list : list
        A list of colors to use for violins.
    title : str, optional
        The title of the full plot. Default: None (no title).
    save : str, optional
        Save the figure to the path given in 'save'. Default: None (figure is not saved).
    """

    #Violing plot to base the next filtering processess
    num_colors = len(adata.obs[groupby].cat.categories)
    n_cols = len(columns) #TODO: option for number of columns 
    n_rows = 1 
    if int(num_colors) <= int(len(color_list)):

        color_list = color_list[:num_colors]
        
        #Setting figures
        fig, axarr = plt.subplots(n_rows, n_cols, figsize = (n_cols*5, n_rows*5))
        axarr = np.array(axarr).reshape((-1, 1)) if n_cols == 1 else axarr
        axarr = np.array(axarr).reshape((1, -1)) if n_rows == 1 else axarr
        axes_list = [b for a in axarr for b in a]
        
        #Plotting
        for b in range(n_cols):
            ax = axes_list.pop(0)
            sns.violinplot(data=adata.obs, x=groupby, y=columns[b], ax=ax, palette=color_list)
            ax.set_title(header[b])
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    else:
        print("Increase the color_list variable")

    #Add title
    if title != None:
        plt.suptitle(title, y=1.05, fontsize=20)

    #Save figure
    if save != None:
        plt.savefig(save, dpi=400, bbox_inches="tight")


def genes_mincells():
    '''
    Filter genes expressed in few cells.
    Exclude <= cutoff
            Default cutoff: automatic cutoff from from distribution analysis
            Custom cutoff: user define the cutoff
    Return anndata
    '''

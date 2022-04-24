
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


def genes_mitochondrial():
    '''
    Filter mitochondrial genes based on a custom list of genes defined by the co-worker.
    Exclude based on the custom list.
    Return anndata
    '''

def genes_genderspecific():
    '''
    Filter gender-specific genes based on a custom list of genes defined by the co-worker.
    Exclude based on the custom list.
    Return anndata
    '''

def genes_others():
    '''
    Filter other genes based on a custom list of genes defined by the co-worker.
    Exclude based on the custom list.
    Return anndata
    '''

def genes_mincells():
    '''
    Filter genes expressed in few cells.
    Exclude <= cutoff
            Default cutoff: automatic cutoff from from distribution analysis
            Custom cutoff: user define the cutoff
    Return anndata
    '''

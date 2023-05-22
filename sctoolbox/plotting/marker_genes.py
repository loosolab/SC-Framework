import scanpy as sc
import numpy as np
import qnorm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

# for plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# sctoolbox functions
import sctoolbox.utils as utils
from sctoolbox.plotting.general import bidirectional_barplot, _save_figure


def rank_genes_plot(adata,
                    genes=None,
                    key=None,
                    n_genes=15,
                    dendrogram=False,
                    groupby=None,
                    title=None,
                    style="dots",
                    measure="expression",
                    save=None,
                    **kwargs):
    """
    Plot expression of genes from rank_genes_groups or from a gene list/dict.

    Parameters
    ----------
    adata : `anndata.AnnData`
        Annotated data matrix.
    genes : `list` or `dict`, optional (default: `None`)
        List of genes to plot. If a dict is passed, the keys are the group names and the values are lists of genes.
    key : `str`, optional (default: `None`)
        Key from `adata.uns` to plot. If specified, `genes` must be `None`.
    n_genes : `int`, optional (default: `15`)
        Number of genes to plot if `key` is specified.
    dendrogram : `bool`, optional (default: `False`)
        Whether to show the dendrogram for groups.
    groupby : `str`, optional (default: `None`)
        Key from `adata.obs` to group cells by.
    title : `str`, optional (default: `None`)
        Title for the plot.
    style : `str`, optional (default: `dots`)
        Style of the plot. Either `dots` or `heatmap`.
    measure : `str`, optional (default: `expression`)
        Measure to write in colorbar label. For example, `expression` or `accessibility`.
    """

    available_styles = ["dots", "heatmap"]
    if style not in available_styles:
        raise ValueError(f"style must be one of {available_styles}.")

    if genes is not None and key is not None:
        raise ValueError("Only one of genes or key can be specified.")

    # Plot genes from rank_genes_groups or from gene list
    parameters = {"swap_axes": False}  # default parameters
    parameters.update(kwargs)
    if key is not None:  # from rank_genes_groups output

        if style == "dots":
            g = sc.pl.rank_genes_groups_dotplot(adata,
                                                key=key,
                                                n_genes=n_genes,
                                                dendrogram=dendrogram,
                                                groupby=groupby,
                                                show=False,
                                                **parameters)
        elif style == "heatmap":
            g = sc.pl.rank_genes_groups_matrixplot(adata,
                                                   key=key,
                                                   n_genes=n_genes,
                                                   dendrogram=dendrogram,
                                                   groupby=groupby,
                                                   show=False,
                                                   **parameters)

    else:  # from a gene list

        if groupby is None:
            raise ValueError("The parameter 'groupby' is needed if 'genes' is given.")

        if style == "dots":
            g = sc.pl.dotplot(adata, genes,
                              dendrogram=False,
                              groupby=groupby,
                              show=False,
                              **parameters)
        elif style == "heatmap":
            g = sc.pl.matrixplot(adata, genes,
                                 dendrogram=False,
                                 groupby=groupby,
                                 show=False,
                                 **parameters)

    g["mainplot_ax"].set_xticklabels(g["mainplot_ax"].get_xticklabels(), ha="right", rotation=45)

    # Rotate gene group names
    if "gene_group_ax" in g and parameters["swap_axes"] is False:  # only rotate if axes is not swapped
        for text in g["gene_group_ax"]._children:
            text._rotation = 45
            text._horizontalalignment = "left"

    # Change title of colorbar (for example expression -> accessibility)
    default_title = g["color_legend_ax"].get_title()
    updated_title = default_title.replace("expression", measure)
    g["color_legend_ax"].set_title(updated_title)

    # Add title to plot above groups
    if title is not None:

        if "gene_group_ax" in g:

            if parameters["swap_axes"]:
                g["mainplot_ax"].set_title(title + "\n" * 2)  # \n to ensure a little space between title and plot

            else:
                fig = plt.gcf()
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()

                highest_y = 0
                for text in g["gene_group_ax"]._children:

                    # Find highest y value for all labels
                    ax = g["gene_group_ax"]
                    transf = ax.transData.inverted()
                    bb = text.get_window_extent(renderer=renderer)
                    bb_datacoords = bb.transformed(transf)

                    highest_y = bb_datacoords.y1 if bb_datacoords.y1 > highest_y else highest_y

                x_mid = np.mean(g["gene_group_ax"].get_xlim())
                g["gene_group_ax"].text(x_mid, highest_y + 0.2, title, fontsize=14, va="bottom", ha="center")

        else:
            g["mainplot_ax"].set_title(title)

    # Save figure
    _save_figure(save)

    return g


#####################################################################
#          Violin / boxplot / bar for genes between groups          #
#####################################################################

def grouped_violin(adata, x, y=None, groupby=None, figsize=None, title=None, style="violin",
                   normalize=False,
                   ax=None,
                   save=None,
                   **kwargs):
    """
    Create violinplot of values across cells in an adata object grouped by x and 'groupby'.
    Can for example show the expression of one gene across groups (x = obs_group, y = gene),
    expression of multiple genes grouped by cell type (x = gene_list, groupby = obs_cell_type),
    or values from adata.obs across cells (x = obs_group, y = obs_column).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    x : str or list
        Column name in adata.obs or gene name(s) in adata.var.index to group by on the x-axis. Multiple gene names can be given in a list.
    y : str, default None
        A column name in adata.obs or a gene in adata.var.index to plot values for. Only needed if x is a column in adata.obs.
    groupby : str, default None
        Column name in adata.obs to create grouped violins. If None, a single violin is plotted per group in 'x'.
    figsize : tuple, default None
        Figure size.
    title : str, default None
        Title of the plot. If None, no title is shown.
    style : str, default "violin"
        Plot style. Either "violin" or "boxplot" or "bar".
    normalize : bool, default False
        If True, normalize the values in 'y' to the range [0, 1] per group in 'x'.
    ax : matplotlib.axes.Axes, default None
        A matplotlib axes object to plot violinplots in. If None, a new figure and axes is created.
    save : str, default None
        Path to save the figure to. If None, the figure is not saved.
    kwargs : arguments, optional
        Additional arguments passed to seaborn.violinplot or seaborn.boxplot.

    Returns
    -------
    matplotlib.axes.Axes

    Example
    --------
    .. plot::
        :context: close-figs


    """

    if isinstance(x, str):
        x = [x]
    x = list(x)  # convert to list incase x was a numpy array or other iterable

    # Establish if x is a column in adata.obs or a gene in adata.var.index
    x_assignment = []
    for element in x:
        if element not in adata.obs.columns and element not in adata.var.index:
            raise ValueError(f"{element} is not a column in adata.obs or a gene in adata.var.index")
        else:
            if element in adata.obs.columns:
                x_assignment.append("obs")
            else:
                x_assignment.append("var")

    if len(set(x_assignment)) > 1:
        raise ValueError("x must be either a column in adata.obs or all genes in adata.var.index")
    else:
        x_assignment = x_assignment[0]

    # Establish if y is a column in adata.obs or a gene in adata.var.index
    if x_assignment == "obs" and y is None:
        raise ValueError("Because 'x' is a column in obs, 'y' must be given as parameter")

    if y is not None:
        if y in adata.obs.columns:
            y_assignment = "obs"
        elif y in adata.var.index:
            y_assignment = "var"
        else:
            raise ValueError(f"y' ({y}) was not found in either adata.obs or adata.var.index")

    # Create obs table with column
    obs_cols = [col for col in [x[0], y, groupby] if col is not None and col in adata.obs.columns]
    obs_table = adata.obs.copy()[obs_cols]  # creates a copy

    for element in x + [y]:
        if element in adata.var.index:
            gene_idx = np.argwhere(adata.var.index == element)[0][0]
            vals = adata.X[:, gene_idx].todense().A1
            obs_table[element] = vals

    # Convert table to long format if the x-axis contains gene expressions
    if x_assignment == "var":
        index_name = obs_table.index.name
        id_vars = [index_name, groupby]
        id_vars = id_vars + x if x_assignment == "obs" else id_vars
        id_vars = [v for v in id_vars if v is not None]

        value_name = "Expression" if not normalize else "Normalized expression"
        obs_table.reset_index(inplace=True)
        obs_table = obs_table.melt(id_vars=id_vars, value_vars=x,
                                   var_name="Gene", value_name=value_name)
        x_var = "Gene"
        y_var = value_name

    else:
        x_var = x[0]
        y_var = y

    # Normalize values to 0-1 per group in x_var
    if normalize:
        obs_table[y_var] = obs_table.groupby(x_var, group_keys=False)[y_var].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Plot expression from obs table
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if style == "violin":
        kwargs["scale"] = "width" if "scale" not in kwargs else kwargs["scale"]  # set defaults
        kwargs["cut"] = 0 if "cut" not in kwargs else kwargs["cut"]
        sns.violinplot(data=obs_table, x=x_var, y=y_var, hue=groupby, ax=ax, **kwargs)
    elif style == "boxplot":
        sns.boxplot(data=obs_table, x=x_var, y=y_var, hue=groupby, ax=ax, **kwargs)
    elif style == "bar":

        pass

    else:
        raise ValueError(f"Style '{style}' is not valid for this function. Style must be one of 'violin' or 'boxplot'")

    if groupby is not None:
        ax.legend(title=groupby, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)  # Set location of legend

    # Final adjustments of labels
    if x_assignment == "obs" and y_assignment == "var":
        ax.set_ylabel(ax.get_ylabel() + " expression")

    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(title)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # save figure if output is given
    _save_figure(save)

    return ax


def group_expression_boxplot(adata, gene_list, groupby, figsize=None):
    """
    Plot a boxplot showing gene expression of genes in `gene_list` across the groups in `groupby`. The total gene expression is quantile normalized
    per group, and are subsequently normalized to 0-1 per gene across groups.

    Parameters
    ------------
    adata : anndata.AnnData object
        An annotated data matrix object containing counts in .X.
    gene_list : list
        A list of genes to show expression for.
    groupby : str
        A column in .obs for grouping cells into groups on the x-axis
    figsize : tuple, optional
        Control the size of the output figure, e.g. (6,10). Default: None (matplotlib default).
    """

    # Obtain pseudobulk
    gene_table = utils.pseudobulk_table(adata, groupby)

    # Normalize across clusters
    gene_table = qnorm.quantile_normalize(gene_table, axis=1)

    # Normalize to 0-1 across groups
    scaler = MinMaxScaler()
    df = gene_table.T
    df[df.columns] = scaler.fit_transform(df[df.columns])
    gene_table = df

    # Melt to long format
    gene_table_melted = gene_table.reset_index().melt(id_vars="index", var_name="gene")
    gene_table_melted.rename(columns={"index": groupby}, inplace=True)

    # Subset to input gene list
    gene_table_melted = gene_table_melted[gene_table_melted["gene"].isin(gene_list)]

    # Sort by median value
    medians = gene_table_melted.groupby(groupby)["value"].median().to_frame()
    medians.columns = ["medians"]
    gene_table_melted_sorted = gene_table_melted.merge(medians, left_on=groupby, right_index=True).sort_values("medians", ascending=False)

    # Joined figure with all
    fig, ax = plt.subplots(figsize=figsize)
    g = sns.boxplot(data=gene_table_melted_sorted, x=groupby, y="value", ax=ax, color="darkgrey")
    ax.set_ylabel("Normalized expression")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return g


def gene_expression_heatmap(adata, genes, cluster_column,
                            title=None,
                            groupby=None,
                            row_cluster=True,
                            col_cluster=False,
                            show_row_dendrogram=False,
                            show_col_dendrogram=False,
                            figsize=None,
                            save=None,
                            **kwargs):
    """ Plot a heatmap of gene expression.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    genes : `list`
        List of genes to plot.
    cluster_column : `str`
        Key in `adata.obs` for which to cluster the x-axis.
    title : `str`, optional (default: `None`)
        Title of the plot.
    groupby : `str`, optional (default: `None`)
        Key in `adata.obs` for which to plot a colorbar per cluster.
    row_cluster : `bool`, optional (default: `True`)
        Whether to cluster the rows.
    col_cluster : `bool`, optional (default: `False`)
        Whether to cluster the columns.
    show_row_dendrogram : `bool`, optional (default: `False`)
        Whether to show the dendrogram for the rows.
    show_col_dendrogram : `bool`, optional (default: `False`)
        Whether to show the dendrogram for the columns.
    figsize : `tuple`, optional (default: `None`)
        Size of the figure. If `None`, use default size.
    save : `str`, optional (default: `None`)
        If given, save the figure to this path.

    Example
    --------
    .. plot::
        :context: close-figs

        genes = adata.var.index[:10]
        pl.gene_expression_heatmap(adata, genes, cluster_column="bulk_labels")
    """

    adata = adata[:, genes]  # Subset to genes

    # Collect counts for each gene per sample
    counts = utils.pseudobulk_table(adata, groupby=cluster_column)
    counts_z = counts.T.apply(zscore).T

    # color dict for groupby
    if groupby is not None:
        groups = adata.obs[groupby].cat.categories
        colors = sns.color_palette()[:len(groups)]
        color_dict = dict(zip(groups, colors))

        # samples = counts_z.columns.tolist()
        sample2group = adata.obs[[cluster_column, groupby]].drop_duplicates()
        samples = sample2group[cluster_column].tolist()
        groups = sample2group[groupby].tolist()
        colors = [color_dict[group] for group in groups]

        sample_info = pd.DataFrame([samples, groups, colors], index=["sample", "group", "color"]).T.set_index("sample")
        col_colors = sample_info["color"]

    else:
        col_colors = None

    nrows, ncols = counts_z.shape
    figsize = (ncols / 2, nrows / 3) if figsize is None else figsize  # (width, height)

    # Plot heatmap
    parameters = {"cmap": "bwr", "center": 0}
    parameters.update(**kwargs)
    g = sns.clustermap(counts_z,
                       xticklabels=True,
                       yticklabels=True,  # show all genes
                       row_cluster=row_cluster,
                       col_cluster=col_cluster,
                       figsize=figsize,
                       col_colors=col_colors,
                       **parameters)

    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    g.ax_heatmap.tick_params(left=True, labelleft=True, right=False, labelright=False)
    g.ax_heatmap.set_ylabel("")
    heatmap_bbox = g.ax_heatmap.get_position()
    heatmap_width = heatmap_bbox.x1 - heatmap_bbox.x0
    # heat_height = heatmap_bbox.y1 - heatmap_bbox.y0

    if show_row_dendrogram is False:
        g.ax_row_dendrogram.set_visible(False)
    if show_col_dendrogram is False:
        g.ax_col_dendrogram.set_visible(False)

    # Invert order of x-axis
    # g.ax_col_colors.invert_xaxis()
    # g.ax_col_dendrogram.invert_xaxis()
    # g.ax_heatmap.invert_xaxis()

    # Add color legend for groupby
    if groupby is not None:

        g.ax_col_colors.tick_params(right=False, labelright=False)

        handles = [Patch(facecolor=color_dict[name]) for name in color_dict]
        legend = plt.legend(handles, color_dict,
                            title=groupby,
                            bbox_to_anchor=(1, 1),
                            bbox_transform=g.ax_heatmap.transAxes,
                            loc='upper left',
                            handlelength=1, handleheight=1,
                            frameon=False,
                            )
        legend._legend_box.align = "left"

    # Move colorbar
    cbar_ax = g.ax_cbar
    bbox = cbar_ax.get_position()
    left, bottom, width, height = bbox._points.flatten()
    cbar_ax.set_position([heatmap_bbox.x1 + heatmap_width / ncols, heatmap_bbox.y0, width, height / nrows * 10])
    cbar_ax.set_ylabel("Mean expr.\nz-score")

    # Set title on top of heatmap
    if title is not None:
        if g.ax_col_dendrogram.get_visible():
            g.ax_col_dendrogram.set_title(title)
        else:
            g.ax_heatmap.text(counts_z.shape[1] / 2, -2, title,  # ensures space beetween title and heatmap
                              transform=g.ax_heatmap.transData, ha="center", va="bottom",
                              fontsize=16)
            # g.ax_heatmap.set_title(title, y=1.2)

    _save_figure(save)

    return g


def group_heatmap(adata, groupby, gene_list=None, save=None, figsize=None):
    """ Plot a heatmap of gene expression across groups in `groupby`. The rows are z-scored per gene.

    NOTE: Likely to be covered in funtionality by gene_expression_heatmap.

    Parameters
    ----------
    adata : anndata.AnnData object
        An annotated data matrix object containing counts in .X.
    groupby : str
        A column in .obs for grouping cells into groups on the x-axis
    gene_list : list, optional
        A list of genes to show expression for. Default: None (all genes)
    save : str, optional
        Save the figure to a file. Default: None (do not save)
    figsize : tuple, optional
        Control the size of the output figure, e.g. (6,10). Default: None (matplotlib default).

    Returns
    -------
    g : seaborn.clustermap
        The seaborn clustermap object
    """

    # Obtain pseudobulk
    gene_table = utils.pseudobulk_table(adata, groupby)

    # Subset to input gene list
    if gene_list is not None:
        gene_table = gene_table.loc[gene_list, :]

    # Z-score
    gene_table = utils.table_zscore(gene_table)

    # Plot heatmap
    g = sns.heatmap(gene_table, figsize=figsize, xticklabels=True, yticklabels=True, cmap="RdBu_r", center=0)  # center=0, vmin=-2, vmax=2)

    _save_figure(save)

    return g


def plot_differential_genes(rank_table, title="Differentially expressed genes",
                            save=None,
                            **kwargs):
    """ Plot number of differentially expressed genes per contrast in a barplot.
    Takes the output of mg.pairwise_rank_genes as input.

    Parameters
    ----------
    rank_table : `pandas.DataFrame`
        Output of mg.pairwise_rank_genes.
    title : `str`, optional (default: `"Differentially expressed genes"`)
        Title of the plot.
    **kwargs : keyword arguments
        Keyword arguments passed to pl.bidirectional_barplot.

    Returns
    -------
    `matplotlib.axes.Axes`
        Axes object.
    """
    group_columns = [col for col in rank_table.columns if "_group" in col]

    info = {}
    for col in group_columns:
        contrast = tuple(col.split("_")[0].split("/"))
        counts = rank_table[col].value_counts()
        info[contrast] = {"left_value": counts["C1"], "right_value": counts["C2"]}

    df = pd.DataFrame().from_dict(info, orient="index")
    df = df.reset_index(names=["left_label", "right_label"])

    ax = bidirectional_barplot(df, title=title, save=save, **kwargs)
    ax.set_xlabel("Number of genes")

    return ax

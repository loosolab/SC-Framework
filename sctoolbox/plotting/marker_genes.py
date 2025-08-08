"""Plots for marker genes e.g. as results of sc.tl.rank_genes_groups."""

import scanpy as sc
import numpy as np
import qnorm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import re

# for plotting
import seaborn as sns
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from beartype.typing import Optional, Tuple, Literal, Any
from beartype import beartype
from numpy.typing import NDArray

# sctoolbox functions
import sctoolbox.utils as utils
from sctoolbox._settings import settings
from sctoolbox.plotting.general import bidirectional_barplot, _save_figure, _make_square
import sctoolbox.utils.decorator as deco


@deco.log_anndata
@beartype
def rank_genes_plot(adata: sc.AnnData,
                    key: Optional[str] = "rank_genes_groups",
                    genes: Optional[list[str] | dict[str, list[str]]] = None,
                    n_genes: int = 15,
                    dendrogram: bool = False,
                    groupby: Optional[str] = None,
                    title: Optional[str] = None,
                    style: Literal["dots", "heatmap"] = "dots",
                    measure: str = "expression",
                    save: Optional[str] = None,
                    report: Optional[str] = None,
                    **kwargs: Any) -> dict:
    """
    Plot expression of genes from rank_genes_groups or from a gene list/dict.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    key : Optional[str], default "rank_genes_groups"
        Key from `adata.uns` to plot. For example, `rank_genes_groups` or `rank_genes_groups_filtered`.
    genes : Optional[list[str] | dict[str, list[str]]], default None
        List of genes to plot across groups in 'groupby'. If a dict is passed, the keys are the group names and the values are lists of genes. Setting 'genes' overrides the 'key' parameter.
    n_genes : int, default 15
        Number of genes to plot if `key` is specified.
    dendrogram : bool, default False
        Whether to show the dendrogram for groups.
    groupby : Optional[str], default None
        Key from `adata.obs` to group cells by.
    title : Optional[str], default None
        Title for the plot.
    style : Literal["dots", "heatmap"], default "dots"
        Style of the plot. Either `dots` or `heatmap`.
    measure : str, default "expression"
        Measure to write in colorbar label. For example, `expression` or `accessibility`.
    save : Optional[str], default None
        If given, save the figure to this path.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    **kwargs : Any
        Additional arguments passed to `sc.pl.rank_genes_groups_dotplot` or `sc.pl.rank_genes_groups_matrixplot`.

    Raises
    ------
    ValueError
        1. If `style` is not one of `dots` or `heatmap`
        2. If `groupby` is not specified when `genes` is specified.

    Returns
    -------
    g : dict
        Dictionary containing the matplotlib axes objects for the plot.

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.marker_genes.rank_genes_plot(adata, n_genes=5)

    .. plot::
        :context: close-figs

        pl.marker_genes.rank_genes_plot(adata, genes={"group1": list(adata.var.index[:10]), "group2": list(adata.var.index[10:20])}, groupby="bulk_labels")
    """

    # Key is not needed if genes are specified
    if genes is not None:
        key = None

    # Plot genes from rank_genes_groups or from gene list
    parameters = {"swap_axes": False}  # default parameters
    parameters.update(kwargs)
    if key is not None:  # from rank_genes_groups output

        # change var.index if neccessary
        if "sctoolbox_params" in adata.uns[key] and "index" in adata.uns[key]["sctoolbox_params"]:
            adata = adata[:, ~adata.var[adata.uns[key]["sctoolbox_params"]["index"]].isna()].copy()  # remove na
            # make the column unique same as .make_var_names_unique
            adata.var[adata.uns[key]["sctoolbox_params"]["index"]] = sc.anndata.utils.make_index_unique(adata.var[adata.uns[key]["sctoolbox_params"]["index"]].astype(str), join="_")
            adata.var.set_index(adata.uns[key]["sctoolbox_params"]["index"], inplace=True)

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

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True)

    return g


#####################################################################
#          Violin / boxplot / bar for genes between groups          #
#####################################################################

@deco.log_anndata
@beartype
def grouped_violin(adata: sc.AnnData,
                   x: str | list[str],
                   y: Optional[str] = None,
                   groupby: Optional[str] = None,
                   figsize: Optional[Tuple[int | float, int | float]] = None,
                   title: Optional[str] = None,
                   style: Literal["violin", "boxplot", "bar"] = "violin",
                   normalize: bool = False,
                   ax: Optional[Axes] = None,
                   save: Optional[str] = None,
                   **kwargs: Any) -> Axes:
    """
    Create violinplot of values across cells in an adata object grouped by x and 'groupby'.

    Can for example show the expression of one gene across groups (x = obs_group, y = gene),
    expression of multiple genes grouped by cell type (x = gene_list, groupby = obs_cell_type),
    or values from adata.obs across cells (x = obs_group, y = obs_column).

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    x : str | list[str]
        Column name in adata.obs or gene name(s) in adata.var.index to group by on the x-axis. Multiple gene names can be given in a list.
    y : Optional[str], default None
        A column name in adata.obs or a gene in adata.var.index to plot values for. Only needed if x is a column in adata.obs.
    groupby : Optional[str], default None
        Column name in adata.obs to create grouped violins. If None, a single violin is plotted per group in 'x'.
    figsize : Optional[Tuple[int | float, int | float]], default None
        Figure size.
    title : Optional[str], default None
        Title of the plot. If None, no title is shown.
    style : Literal["violin", "boxplot", "bar"], default "violin"
        Plot style. Either "violin" or "boxplot" or "bar".
    normalize : bool, default False
        If True, normalize the values in 'y' to the range [0, 1] per group in 'x'.
    ax : Optional[Axes], default None
        A matplotlib axes object to plot violinplots in. If None, a new figure and axes is created.
    save : Optional[str], default None
        Path to save the figure to. If None, the figure is not saved.
    **kwargs : Any
        Additional arguments passed to seaborn.violinplot or seaborn.boxplot.

    Returns
    -------
    Axes

    Raises
    ------
    ValueError
        If x or y are not columns in adata.obs or a genes in adata.var.index.

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.marker_genes.grouped_violin(adata, 'phase', y='G2M_score')
    """

    if isinstance(x, str):
        x = [x]
    x = list(x)  # convert to list incase x was a numpy array or other iterable

    # Establish if x is a column in adata.obs or a gene in adata.var.index
    x_assignment = []
    for element in x:
        if element not in adata.obs.columns and element not in adata.var.index:
            raise ValueError(f"{element} is not a column in adata.obs or a gene in adata.var.index")
        x_assignment.append("obs" if element in adata.obs.columns else "var")

    if len(set(x_assignment)) > 1:
        raise ValueError("x must be either a column in adata.obs or all genes in adata.var.index")

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

            try:
                vals = adata.X[:, gene_idx].todense().A1  # try sparse-matrix
            except AttributeError:
                vals = adata.X[:, gene_idx].ravel()  # try dense-matrix/ numpy array

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


@deco.log_anndata
@beartype
def group_expression_boxplot(adata: sc.AnnData,
                             gene_list: list[str],
                             groupby: str,
                             figsize: Optional[Tuple[int | float, int | float]] = None,
                             **kwargs: Any) -> Axes:
    """
    Plot a boxplot showing summarized gene expression of genes in `gene_list` across the groups in `groupby`.

    The total gene expression is quantile normalized per group, and are subsequently normalized to 0-1 per gene across groups.

    Parameters
    ----------
    adata : sc.AnnData
        An annotated data matrix object containing counts in .X.
    gene_list : list[str]
        A list of genes to show expression for.
    groupby : str
        A column in .obs for grouping cells into groups on the x-axis
    figsize : Optional[Tuple[int | float, int | float]], default None (matplotlib default)
        Control the size of the output figure, e.g. (6,10).
    **kwargs : Any
        Additional arguments passed to seaborn.boxplot.

    Returns
    -------
    Axes

    Examples
    --------
    .. plot::
        :context: close-figs

        gene_list=["HES4", "PRMT2", "ITGB2"]
        pl.marker_genes.group_expression_boxplot(adata, gene_list, groupby="bulk_labels")
    """

    # Obtain pseudobulk
    gene_table = utils.bioutils.pseudobulk_table(adata, groupby)

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
    g = sns.boxplot(data=gene_table_melted_sorted, x=groupby, y="value", ax=ax, color="darkgrey", **kwargs)
    ax.set_ylabel("Normalized expression")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return g


@deco.log_anndata
@beartype
def gene_expression_heatmap(adata: sc.AnnData,
                            genes: list[str],
                            cluster_column: str,
                            gene_name_column: Optional[str] = None,
                            title: Optional[str] = None,
                            groupby: Optional[str] = None,
                            row_cluster: bool = True,
                            col_cluster: bool = False,
                            show_row_dendrogram: bool = False,
                            show_col_dendrogram: bool = False,
                            figsize: Optional[Tuple[int | float, int | float]] = None,
                            save: Optional[str] = None,
                            **kwargs: Any) -> Any:  # Any since beartype cannot handle sns datatypes
    """
    Plot a heatmap of z-score normalized gene expression across clusters/groups.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    genes : list[str]
        List of genes to plot. Must match names in `adata.var.index`.
    cluster_column : str
        Key in `adata.obs` for which to cluster the x-axis.
    gene_name_column : Optional[str], default None
        Column in `adata.var` for which to use for gene row names. Default is to use the .var index.
    title : Optional[str], default None
        Title of the plot.
    groupby : Optional[str], default None
        Key in `adata.obs` for which to plot a colorbar per cluster.
    row_cluster : bool, default True
        Whether to cluster the rows.
    col_cluster : bool, default False
        Whether to cluster the columns.
    show_row_dendrogram : bool, default False
        Whether to show the dendrogram for the rows.
    show_col_dendrogram : bool, default False
        Whether to show the dendrogram for the columns.
    figsize : Optional[Tuple[int | float, int | float]], default None
        Size of the figure. If `None`, use default size.
    save : Optional[str], default None
        If given, save the figure to this path.
    **kwargs : Any
        Additional arguments passed to `seaborn.clustermap`.

    Returns
    -------
    g : Any
        sns.matrix.ClusterGrid: The seaborn ClusterGrid object containing the heatmap.
        Note: Any since sns.matrix.ClusterGrid cannot be checked by beartype.

    Raises
    ------
    KeyError
        If `gene_name_column` is not a column in `adata.var`.

    Examples
    --------
    .. plot::
        :context: close-figs

        adata.obs["samples"] = np.random.choice(["CTRL1", "CTRL2", "CTRL3", "CTRL4", "TREAT1", "TREAT2", "TREAT3", "TREAT4"], size=adata.shape[0])
        adata.obs["condition"] = adata.obs["samples"].str.extract("([A-Z]+)")

        genes = list(adata.var.index[:15])
        pl.marker_genes.gene_expression_heatmap(adata,
                                                genes,
                                                cluster_column="samples",
                                                groupby="condition",
                                                title="Gene expression",
                                                col_cluster=True,
                                                show_col_dendrogram=True,
                                                colors_ratio=0.03)
    """

    adata = adata[:, genes]  # Subset to genes

    # Decide which combination to cluster by
    groupby_col = "_cluster_by"
    if groupby is not None:
        adata.obs[groupby_col] = list(zip(adata.obs[cluster_column], adata.obs[groupby]))
    else:
        adata.obs[groupby_col] = [(s, ) for s in adata.obs[cluster_column]]

    # Collect counts for each gene per sample
    counts = utils.bioutils.pseudobulk_table(adata, groupby=groupby_col)
    counts_z = counts.T.apply(zscore).T

    # Color dict for groupby
    if groupby is not None:
        groups = adata.obs[groupby].unique().tolist()
        color_list = sns.color_palette()[:len(groups)]
        color_dict = dict(zip(groups, color_list))

        # Get color per column
        colors = [color_dict[col[-1]] for col in counts_z.columns]
        col_colors = pd.Series(index=counts_z.columns, data=colors)

    else:
        col_colors = None

    # Translation dict for row names
    if gene_name_column is not None:
        try:
            row_name_dict = dict(zip(adata.var.index, adata.var[gene_name_column]))
        except KeyError:
            raise KeyError(f"Column '{gene_name_column}' not found in adata.var")
    else:
        row_name_dict = {}

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

    yticklabes = [row_name_dict.get(s, s) for s in g.data2d.index]
    g.ax_heatmap.set_yticklabels(yticklabes)

    xticklabels = [c[0] for c in g.data2d.columns]  # first index contains the cluster_column
    g.ax_heatmap.set_xticklabels(xticklabels, rotation=45, ha="right")
    g.ax_heatmap.tick_params(left=True, labelleft=True, right=False, labelright=False)
    g.ax_heatmap.set_ylabel("")
    heatmap_bbox = g.ax_heatmap.get_position()

    if show_row_dendrogram is False:
        g.ax_row_dendrogram.set_visible(False)
    if show_col_dendrogram is False:
        g.ax_col_dendrogram.set_visible(False)

    # Invert order of x-axis
    # g.ax_col_colors.invert_xaxis()
    # g.ax_col_dendrogram.invert_xaxis()
    # g.ax_heatmap.invert_xaxis()

    # Move colorbar
    cbar_ax = g.ax_cbar
    cbar_width = heatmap_bbox.width / ncols  # width of 1 column
    cbar_height = min(heatmap_bbox.height, heatmap_bbox.height / nrows * 5)  # 5 rows high, but ensure colorbar is not taller than heatmap
    cbar_ax.set_position([heatmap_bbox.x1 + cbar_width, heatmap_bbox.y0, cbar_width, cbar_height])
    cbar_ax.set_ylabel("Mean expr.\nz-score")

    # Add color legend for groupby above cbar
    if groupby is not None:

        g.ax_col_colors.tick_params(right=False, labelright=False)

        handles = [Patch(facecolor=color_dict[name]) for name in color_dict]
        legend = plt.legend(handles, color_dict,
                            title=groupby,
                            bbox_to_anchor=(0, 1.25),  # 1.25 to make sure there is space for cbar label
                            bbox_transform=cbar_ax.transAxes,
                            loc='lower left',
                            handlelength=1, handleheight=1,
                            frameon=False,
                            borderpad=0
                            )
        legend._legend_box.align = "left"

    # Set title on top of heatmap
    if title is not None:
        if g.ax_col_dendrogram.get_visible():
            g.ax_col_dendrogram.set_title(title, fontsize=13)
        else:
            g.ax_heatmap.text(counts_z.shape[1] / 2, -2, title,  # ensures space beetween title and heatmap
                              transform=g.ax_heatmap.transData, ha="center", va="bottom",
                              fontsize=13)

    _save_figure(save)

    return g


@beartype
def plot_differential_genes(rank_table: pd.DataFrame,
                            title: str = "Differentially expressed genes",
                            save: Optional[str] = None,
                            **kwargs: Any) -> Axes:
    """
    Plot number of differentially expressed genes per contrast in a barplot.

    Parameters
    ----------
    rank_table : pd.DataFrame
        Output of sctoolbox.tools.marker_genes.pairwise_rank_genes.
    title : str, default "Differentially expressed genes"
        Title of the plot.
    save : Optional[str], default None
        If given, save the figure to this path.
    **kwargs : Any
        Keyword arguments passed to pl.general.bidirectional_barplot.

    Raises
    ------
    ValueError
        If no significant differentially expressed genes are found in the data.

    Returns
    -------
    Axes
        Axes object.

    Examples
    --------
    .. plot::
        :context: close-figs

        import sctoolbox.tools as tl
        adata.obs["groups"] = np.random.choice(["G1", "G2", "G3"], size=adata.shape[0])
        pairwise_table = tl.marker_genes.pairwise_rank_genes(adata, foldchange_threshold=0.2, groupby="groups")

        pl.marker_genes.plot_differential_genes(pairwise_table)
    """
    group_columns = [col for col in rank_table.columns if "_group" in col]

    info = {}
    for col in group_columns:
        m = re.match("(.+)/(.+)_group", col)  # tuple(col.split("_")[0].split("/"))
        contrast = tuple([m.group(1), m.group(2)])

        counts = rank_table[col].value_counts()
        if all(x in list(counts.index) for x in ['C1', 'C2']):
            info[contrast] = {"left_value": counts["C1"], "right_value": counts["C2"]}

    if not info:
        raise ValueError("No significant differentially expressed genes in the data. Abort.")

    df = pd.DataFrame().from_dict(info, orient="index")
    df = df.reset_index(names=["left_label", "right_label"])

    ax = bidirectional_barplot(df, title=title, save=save, **kwargs)
    ax.set_xlabel("Number of genes")

    return ax


@beartype
def plot_gene_correlation(adata: sc.AnnData,
                          ref_gene: str,
                          gene_list: list[str] | str,
                          ncols: int = 3,
                          figsize: Optional[Tuple[int | float, int | float]] = None,
                          save: Optional[str] = None,
                          **kwargs: Any) -> NDArray[Axes]:
    """
    Plot the gene expression of one reference gene against the expression of a set of genes.

    Parameters
    ----------
    adata : sc.AnnData
        An annotated data matrix object containing counts in .X.
    ref_gene : str
        Reference gene to which other genes are comapred to.
    gene_list : list[str] | str
        A list of genes to show expression for.
    ncols : int, default 3
        Number of columns in plot grid.
    figsize : Optional[Tuple[int | float, int | float]], default None
        Control the size of the output figure, e.g. (6,10).
    save : Optional[str], default None
        Save the figure to a file.
    **kwargs : Any
        Additional arguments passed to seaborn.regplot.

    Returns
    -------
    NDArray[Axes]
        List containing all axis objects.

    Examples
    --------
    .. plot::
        :context: close-figs

        gene_list=["HES4", "PRMT2", "ITGB2"]
        pl.marker_genes.plot_gene_correlation(adata, "SUMO3", gene_list)
    """

    if isinstance(gene_list, str):
        gene_list = [gene_list]

    # Find out how many rows we need
    nrows = int(np.ceil(len(gene_list) / ncols))

    if figsize is None:
        figsize = (ncols * 3, nrows * 3)

    fig, axarr = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    axes_list = axarr.flatten()

    # Get expression values of reference gene
    ref = adata[:, ref_gene].to_df()[ref_gene]

    for i, gene in enumerate(gene_list):
        ax = axes_list[i]
        gene_expr = adata[:, gene].to_df()[gene]
        sns.regplot(x=ref, y=gene_expr, ax=ax, **kwargs)

    # Hide axes not used
    for ax in axes_list[len(gene_list):]:
        ax.set_visible(False)
    axes_list = axes_list[:len(gene_list)]

    # Make plots square
    for ax in axes_list:
        _make_square(ax)

    fig.tight_layout()

    # Save figure if chosen
    if save:
        _save_figure(save)

    return axes_list

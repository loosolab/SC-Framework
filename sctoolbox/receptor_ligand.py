import pandas as pd
from collections import Counter
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import igraph as ig
from itertools import combinations_with_replacement
from matplotlib import cm
from tqdm import tqdm
from matplotlib.artist import Artist
from igraph import BoundingBox, Graph, palettes
import matplotlib
from matplotlib.patches import ConnectionPatch
import matplotlib.lines as lines
from sklearn.preprocessing import minmax_scale
import warnings

def download_db(adata, db_path, ligand_column, receptor_column, sep="\t", inplace=False, overwrite=False):
    """
    Download table of receptor-ligand interactions and store in adata.

    Note: This will remove all information stored in adata.uns['receptor-ligand']

    Parameters:
    ----------
        adata : AnnData
            Analysis object the database will be added to.
        dp_path : str
            Path to database table. A valid database needs a column with receptor gene ids/ symbols and ligand gene ids/ symbols.
            Human: http://tcm.zju.edu.cn/celltalkdb/download/processed_data/human_lr_pair.txt
            Mouse: http://tcm.zju.edu.cn/celltalkdb/download/processed_data/mouse_lr_pair.txt
        ligand_column : str
            Name of the column with ligand gene names.
            Use 'ligand_gene_symbol' for the urls provided above.
        receptor_column : str
            Name of column with receptor gene names.
            Use 'receptor_gene_symbol' for the urls provided above.
        sep : str, default '\t'
            Separator of database table.
        inplace : boolean, default False
            Whether to copy `adata` or modify it inplace.
        overwrite : boolean, default False
            If True will overwrite existing database.

    Returns:
    ----------
        AnnData : optional
            Copy of adata with added database path and database table to adata.uns['receptor-ligand']
    """
    # datbase already existing?
    if not overwrite and "receptor-ligand" in adata.uns and "database" in adata.uns["receptor-ligand"]:
        warnings.warn("Database already exists! Skipping. Set `overwrite=True` to replace.")

        if inplace:
            return
        else:
            return adata

    database = pd.read_csv(db_path, sep=sep)

    # check column names in table
    if ligand_column not in database.columns:
        raise ValueError(f"Ligand column '{ligand_column}' not found in database! Available columns: {database.columns}")
    if receptor_column not in database.columns:
        raise ValueError(f"Receptor column '{receptor_column}' not found in database! Available columns: {database.columns}")

    modified_adata = adata if inplace else adata.copy()

    # setup dict to store information old data will be overwriten!
    modified_adata.uns['receptor-ligand'] = dict()

    modified_adata.uns['receptor-ligand']['database_path'] = db_path
    modified_adata.uns['receptor-ligand']['database'] = database
    modified_adata.uns['receptor-ligand']['ligand_column'] = ligand_column
    modified_adata.uns['receptor-ligand']['receptor_column'] = receptor_column

    if not inplace:
        return modified_adata

def calculate_interaction_table(adata, cluster_column, gene_index=None, normalize=1000, inplace=False, overwrite=False):
    """
    Calculate an interaction table of the clusters defined in adata.
    
    Parameters:
    ----------
        adata : AnnData
            AnnData object that holds the expression values and clustering
        cluster_column : str
            Name of the cluster column in adata.obs.
        gene_index : str, default None
            Column in adata.var that holds gene symbols/ ids. Corresponds to `download_db(ligand_column, receptor_column)`. Uses index when None.
        normalize : int, default 1000
            Correct clusters to given size.
        inplace : boolean, default False
            Whether to copy `adata` or modify it inplace.
        overwrite : boolean, default False
            If True will overwrite existing interaction table.

    Returns:
    ----------
        AnnData : optional
            Copy of adata with added interactions table to adata.uns['receptor-ligand']['interactions']
    """
    if "receptor-ligand" not in adata.uns.keys():
        raise ValueError("Could not find receptor-ligand database. Please setup database with `download_db(...)` before running this function.")

    # interaction table already exists?
    if not overwrite and "receptor-ligand" in adata.uns and "interactions" in adata.uns["receptor-ligand"]:
        warnings.warn("Interaction table already exists! Skipping. Set `overwrite=True` to replace.")

        if inplace:
            return
        else:
            return adata

    r_col, l_col = adata.uns["receptor-ligand"]["receptor_column"], adata.uns["receptor-ligand"]["ligand_column"]
    index = adata.var[gene_index] if gene_index else adata.var.index

    # test if database gene columns overlap with adata.var genes
    if (not set(adata.uns["receptor-ligand"]["database"][r_col]) & set(index) or
        not set(adata.uns["receptor-ligand"]["database"][l_col]) & set(index)
        ):
        raise ValueError(f"Database columns '{r_col}', '{l_col}' don't match adata.uns['{gene_index}']. Please make sure to select gene ids or symbols in all columns.")

    ##### compute cluster means and expression percentage for each gene #####
    # gene mean expression per cluster
    cl_mean_expression = pd.DataFrame(index=index)
    # percent cells in cluster expressing gene
    cl_percent_expression = pd.DataFrame(index=index)
    # number of cells for each cluster
    clust_sizes = {}

    # fill above tables
    for cluster in tqdm(set(adata.obs[cluster_column]), desc="computing cluster gene scores"):
        # filter adata to a specific cluster
        cluster_adata = adata[adata.obs[cluster_column] == cluster]
        clust_sizes[cluster] = len(cluster_adata)
        
        ## compute cluster means
        if gene_index is None:
            cl_mean_expression.loc[cl_mean_expression.index.isin(cluster_adata.var.index), cluster] = cluster_adata.X.mean(axis=0).reshape(-1,1)
        else:
            cl_mean_expression.loc[cl_mean_expression.index.isin(cluster_adata.var[gene_index]), cluster] = cluster_adata.X.mean(axis=0).reshape(-1,1)
    
        ## compute expression percentage
        # get nonzero expression count for all genes
        _, cols = cluster_adata.X.nonzero()
        gene_occurence = Counter(cols)

        cl_percent_expression[cluster] = 0
        cl_percent_expression.iloc[list(gene_occurence.keys()), cl_percent_expression.columns.get_loc(cluster)] = list(gene_occurence.values())
        cl_percent_expression[cluster] = cl_percent_expression[cluster] / len(cluster_adata.obs) * 100
    
    # combine duplicated genes through mean (can happen due to mapping between organisms)
    if len(set(cl_mean_expression.index)) != len(cl_mean_expression):
        cl_mean_expression = cl_mean_expression.groupby(cl_mean_expression.index).mean()
        cl_percent_expression = cl_percent_expression.groupby(cl_percent_expression.index).mean()

    # cluster scaling factor for cluster size correction
    scaling_factor = {k: v / normalize for k, v in clust_sizes.items()}
    
    ########## compute zscore of cluster means for each gene ##########
    # create pandas functions that show progress bar
    tqdm.pandas(desc="computing Z-scores")

    zscores = cl_mean_expression.progress_apply(lambda x: pd.Series(scipy.stats.zscore(x, nan_policy='omit'), index=cl_mean_expression.columns), axis=1)
    
    interactions = {"receptor_cluster": [],
                    "ligand_cluster": [],
                    "receptor_gene": [],
                    "ligand_gene": [],
                    "receptor_score": [],
                    "ligand_score": [],
                    "receptor_percent": [],
                    "ligand_percent": [],
                    "receptor_scale_factor": [],
                    "ligand_scale_factor": [],
                    "receptor_cluster_size": [],
                    "ligand_cluster_size": []}
    
    ########## create interaction table ##########
    for _, (receptor, ligand) in tqdm(adata.uns["receptor-ligand"]["database"][[r_col, l_col]].iterrows(),
                                      total=len(adata.uns["receptor-ligand"]["database"]),
                                      desc="finding receptor-ligand interactions"):
        # skip interaction if not in data
        if receptor is np.nan or ligand is np.nan:
            continue
    
        if not receptor in zscores.index or not ligand in zscores.index:
            continue

        # add interactions to dict
        for receptor_cluster in zscores.columns:
            for ligand_cluster in zscores.columns:
                interactions["receptor_gene"].append(receptor)
                interactions["ligand_gene"].append(ligand)
                interactions["receptor_cluster"].append(receptor_cluster)
                interactions["ligand_cluster"].append(ligand_cluster)
                interactions["receptor_score"].append(zscores.loc[receptor, receptor_cluster])
                interactions["ligand_score"].append(zscores.loc[ligand, ligand_cluster])
                interactions["receptor_percent"].append(cl_percent_expression.loc[receptor, receptor_cluster])
                interactions["ligand_percent"].append(cl_percent_expression.loc[ligand, ligand_cluster])
                interactions["receptor_scale_factor"].append(scaling_factor[receptor_cluster])
                interactions["ligand_scale_factor"].append(scaling_factor[ligand_cluster])
                interactions["receptor_cluster_size"].append(clust_sizes[receptor_cluster])
                interactions["ligand_cluster_size"].append(clust_sizes[ligand_cluster])
    
    interactions = pd.DataFrame(interactions)

    # compute interaction score
    interactions["receptor_score"] = interactions["receptor_score"] * interactions["receptor_scale_factor"]
    interactions["ligand_score"] = interactions["ligand_score"] * interactions["ligand_scale_factor"]
    interactions["interaction_score"] = interactions["receptor_score"] + interactions["ligand_score"]

    # clean up columns
    interactions.drop(columns=["receptor_scale_factor", "ligand_scale_factor"], inplace=True)
    
    # add to adata
    modified_adata = adata if inplace else adata.copy()

    modified_adata.uns['receptor-ligand']['interactions'] = interactions

    if not inplace:
        return modified_adata

def interaction_violin_plot(adata, min_perc, output=None, figsize=(5,20), dpi=100):
    '''
    Generate violin plot of pairwise cluster interactions.

    Parameters:
    ----------
        adata : AnnData
            AnnData object
        min_perc : float
            Minimum percentage of cells in a cluster that express the respective gene. A value from 0-100.
        output : str, default None
            Path to output file.
        figsize : int tuple, default (5, 20)
            Figure size
        dpi : float, default 100
            The resolution of the figure in dots-per-inch.

    Returns:
    ----------
        matplotlib.axes.Axes : 
            Object containing all plots. As returned by matplotlib.pyplot.subplots
    '''
    # is interaction table available?
    if "receptor-ligand" not in adata.uns.keys() or "interactions" not in adata.uns["receptor-ligand"].keys():
        raise ValueError("Could not find interaction data! Please setup with `calculate_interaction_table(...)` before running this function.")
    
    interactions = adata.uns["receptor-ligand"]["interactions"]
    
    rows = len(set(interactions["receptor_cluster"]))

    fig, axs = plt.subplots(ncols=1, nrows=rows, figsize=figsize, dpi=dpi, tight_layout={'rect': (0, 0, 1, 0.95)}) # prevent label clipping; leave space for title
    #fig.suptitle('Cluster interactions', fontsize=16)
    flat_axs = axs.flatten()

    # generate violins of one cluster vs rest in each iteration
    for i, cluster in enumerate(sorted(set(interactions["receptor_cluster"].tolist() + interactions["ligand_cluster"].tolist()))):
        cluster_interactions = interactions[((interactions["receptor_cluster"] == cluster) | 
                                            (interactions["ligand_cluster"] == cluster)) &
                                            (interactions["receptor_percent"] >= min_perc) &
                                            (interactions["ligand_percent"] >= min_perc)].copy()
        
        # get column of not main clusters
        cluster_interactions["Cluster"] = cluster_interactions.apply(lambda x: x[1] if x[0] == cluster else x[0], axis=1).tolist()

        plot = sns.violinplot(x=cluster_interactions["Cluster"],
                    y=cluster_interactions["interaction_score"], 
                    ax=flat_axs[i])
        
        plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
        
        flat_axs[i].set_title(f"Cluster {cluster}")

    # save plot
    if output:
        # create path if necessary
        Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
    
    return axs

def hairball(adata, min_perc, interaction_score=0, output=None, title="Network", color_min=0, color_max=None, cbar_label="Interaction count", show_count=False):
    '''
    Generate network graph of interactions between clusters.
    
    Note: The dimensions of the jupyter view often differ from what is saved to the file.

    KNOWN ISSUE: The network graph will not show up in jupyter unless first running `matplotlib.use("cairo")`. 
    Afterwards run `matplotlib.use("module://matplotlib_inline.backend_inline")` in a new cell or the other plots won't work.
    TODO: this may be fixable when igraph>=0.10 is released. https://github.com/igraph/python-igraph/issues/426

    Parameters:
    ----------
        adata : AnnData
            AnnData object
        min_perc : float
            Minimum percentage of cells in a cluster that express the respective gene. A value from 0-100.
        interaction_score : float, default 0
            Interaction score must be above this threshold for the interaction to be counted in the graph.
        output : str, default None
            Path to output file.
        title : str, default 'Network'
            The plots title.
        color_min : float, default 0
            Min value for color range.
        color_max : float, default max
            Max value for color range.
        cbar_label : str, default 'Interaction count'
            Label above the colorbar.
        show_count : bool, default False
            Show the interaction count in the hairball.

    Returns:
    ----------
        matplotlib.axes.Axes : 
            Object containing all plots. As returned by matplotlib.pyplot.subplots
    '''
    # is interaction table available?
    if "receptor-ligand" not in adata.uns.keys() or "interactions" not in adata.uns["receptor-ligand"].keys():
        raise ValueError("Could not find interaction data! Please setup with `calculate_interaction_table(...)` before running this function.")

    interactions = adata.uns["receptor-ligand"]["interactions"]

    igraph_scale=3
    matplotlib_scale=4
    
    ########## setup class that combines igraph with matplotlib ##########
    # from https://stackoverflow.com/a/36154077
    # makes igraph compatible with matplotlib
    # so a colorbar can be added
    class GraphArtist(Artist):
        """Matplotlib artist class that draws igraph graphs.

        Only Cairo-based backends are supported.
        """

        def __init__(self, graph, bbox, palette=None, *args, **kwds):
            """Constructs a graph artist that draws the given graph within
            the given bounding box.

            `graph` must be an instance of `igraph.Graph`.
            `bbox` must either be an instance of `igraph.drawing.BoundingBox`
            or a 4-tuple (`left`, `top`, `width`, `height`). The tuple
            will be passed on to the constructor of `BoundingBox`.
            `palette` is an igraph palette that is used to transform
            numeric color IDs to RGB values. If `None`, a default grayscale
            palette is used from igraph.

            All the remaining positional and keyword arguments are passed
            on intact to `igraph.Graph.__plot__`.
            """
            Artist.__init__(self)

            if not isinstance(graph, Graph):
                raise TypeError("expected igraph.Graph, got %r" % type(graph))

            self.graph = graph
            self.palette = palette or palettes["gray"]
            self.bbox = BoundingBox(bbox)
            self.args = args
            self.kwds = kwds

        def draw(self, renderer):
            from matplotlib.backends.backend_cairo import RendererCairo
            if not isinstance(renderer, RendererCairo):
                raise TypeError("graph plotting is supported only on Cairo backends")
            self.graph.__plot__(renderer.gc.ctx, self.bbox, self.palette, *self.args, **self.kwds)
    
    ########## create igraph ##########
    graph = ig.Graph()

    # set nodes
    clusters = list(set(list(interactions["receptor_cluster"]) + list(interactions["ligand_cluster"])))
    
    graph.add_vertices(clusters)
    graph.vs['label'] = clusters
    graph.vs['size'] = 45
    graph.vs['label_size'] = 30

    # set edges
    for (a, b) in combinations_with_replacement(clusters, 2):
        subset = interactions[(((interactions["receptor_cluster"] == a) & (interactions["ligand_cluster"] == b)) |
                            ((interactions["receptor_cluster"] == b) & (interactions["ligand_cluster"] == a))) &
                            (interactions["receptor_percent"] >= min_perc) &
                            (interactions["ligand_percent"] >= min_perc) &
                            (interactions["interaction_score"] > interaction_score)]

        graph.add_edge(a, b, weight=len(subset))

    # set edge colors/ width based on weight
    colormap = cm.get_cmap('viridis', len(graph.es))
    print(f"Max weight {np.max(np.array(graph.es['weight']))}")
    max_weight = np.max(np.array(graph.es['weight'])) if color_max is None else color_max
    for e in graph.es:
        e["color"] = colormap(e["weight"] / max_weight, e["weight"] / max_weight)
        e["width"] = (e["weight"] / max_weight) * 10
        # show weights in plot
        if show_count:
            e["label"] = e["weight"]
            e["label_size"] = 25
        
    ########## setup matplotlib plot and combine with igraph ##########
    # Make Matplotlib use a Cairo backend
    matplotlib.use("cairo")

    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(8*matplotlib_scale, 6*matplotlib_scale), gridspec_kw={'width_ratios': [20, 1]})
    fig.suptitle(title, fontsize=12*matplotlib_scale)

    # hide axis under network graph
    axes[0].axis('off')

    # Draw the graph over the plot
    # Two points to note here:
    # 1) we add the graph to the axes, not to the figure. This is because
    #    the axes are always drawn on top of everything in a matplotlib
    #    figure, and we want the graph to be on top of the axes.
    # 2) we set the z-order of the graph to infinity to ensure that it is
    #    drawn above all the curves drawn by the axes object itself.
    left = 250
    top = 150
    width = 500 * igraph_scale + left
    height = 500 * igraph_scale + top
    graph_artist = GraphArtist(graph, (left, top, width, height), layout=graph.layout_circle(order=sorted(clusters)))
    graph_artist.set_zorder(float('inf'))
    axes[0].artists.append(graph_artist)

    # add colorbar
    cb = matplotlib.colorbar.ColorbarBase(axes[1], 
                                        orientation='vertical', 
                                        cmap=colormap,
                                        norm=matplotlib.colors.Normalize(0 if color_min is None else color_min, 
                                                                        max_weight))
    
    cb.ax.tick_params(labelsize=10*matplotlib_scale) 
    cb.ax.set_title(cbar_label, fontsize=10*matplotlib_scale)

    # prevent label clipping out of picture
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)

    if output:
        # create path if necessary
        Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True)
        
        # Save the figure
        fig.savefig(output)

    return axes

def progress_violins(datalist, datalabel, cluster_a, cluster_b, min_perc, output, figsize=(12, 6)):
    '''
    CURRENTLY NOT FUNCTIONAL!

    Show cluster interactions over timepoints.
    
    Parameters:
        datalist (list): List of interaction DataFrames. Each DataFrame represents a timepoint.
        datalabel (list): List of strings. Used to label the violins.
        cluster_a (str): Name of the first interacting cluster.
        cluster_b (str): Name of the second interacting cluster.
        min_perc (float): Minimum percentage of cells in a cluster each gene must be expressed in.
        output (str): Path to output file.
        figsize (int tuple): Tuple of plot (width, height).
    '''
    return "Function to be implemented"
    
    fig, axs = plt.subplots(1, len(datalist), figsize=figsize)
    fig.suptitle(f"{cluster_a} - {cluster_b}")
    
    flat_axs = axs.flatten()
    for i, (table, label) in enumerate(zip(datalist, datalabel)):
        # filter data
        subset = table[((table["cluster_a"] == cluster_a) & (table["cluster_b"] == cluster_b) |
                        (table["cluster_a"] == cluster_b) & (table["cluster_b"] == cluster_a)) &
                    (table["percentage_a"] >= min_perc) &
                    (table["percentage_b"] >= min_perc)]
        
        v = sns.violinplot(data=subset, y="interaction_score", ax=flat_axs[i])
        v.set_xticklabels([label])
        
    plt.tight_layout()
    
    if not output is None:
        fig.savefig(output)

def connectionPlot(adata, 
                   restrict_to=None,
                   figsize=(10, 15),
                   dpi=100,
                   connection_alpha="interaction_score",
                   output=None,
                   title=None,
                   # receptor params
                   receptor_cluster_col="receptor_cluster",
                   receptor_col="receptor_gene",
                   receptor_hue="receptor_score",
                   receptor_size="receptor_percent",
                   # ligand params
                   ligand_cluster_col="ligand_cluster",
                   ligand_col="ligand_gene",
                   ligand_hue="ligand_score",
                   ligand_size="ligand_percent",
                   filter=None,
                   lw_multiplier=2,
                   wspace=0.4,
                   line_colors="rainbow"
                  ):
    '''
    Show specific receptor-ligand connections between clusters.

    Parameters:
    ----------
        adata : AnnData
            AnnData object
        restrict_to : str list, default None
            Restrict plot to given cluster names.
        figsize : int tuple, default (10, 15)
            Figure size
        dpi : float, default 100
            The resolution of the figure in dots-per-inch.
        connection_alpha : str, default 'interaction_score'
            Name of column that sets alpha value of lines between plots. None to disable.
        output : str, default None
            Path to output file.
        title : str, default None
            Title of the plot
        receptor_cluster_col : str, default 'receptor_cluster'
            Name of column containing cluster names of receptors. Shown on x-axis.
        receptor_col : str, default 'receptor_gene'
            Name of column containing gene names of receptors. Shown on y-axis.
        receptor_hue : str, default 'receptor_score'
            Name of column containing receptor scores. Shown as point color.
        receptor_size : str, default 'receptor_percent'
            Name of column containing receptor expression percentage. Shown as point size.
        ligand_cluster_col : str, default 'ligand_cluster'
            Name of column containing cluster names of ligands. Shown on x-axis.
        ligand_col : str, default 'ligand_gene'
            Name of column containing gene names of ligands. Shown on y-axis.
        ligand_hue : str, default 'ligand_score'
            Name of column containing ligand scores. Shown as point color.
        ligand_size : str, default 'ligand_percent'
            Name of column containing ligand expression percentage. Shown as point size.
        filter : str, default None
            Conditions to filter the interaction table on. E.g. 'column_name > 5 & other_column < 2'. Forwarded to pandas.DataFrame.query.
        lw_multiplier : int, default 2
            Linewidth multiplier.
        wspace : float, default 0.4
            Width between plots. Fraction of total width.
        line_colors : str, default 'rainbow'
            Name of colormap used to color lines. All lines are black if None.

    Returns:
    ----------
        matplotlib.axes.Axes : 
            Object containing all plots. As returned by matplotlib.pyplot.subplots
    '''
    # is interaction table available?
    if "receptor-ligand" not in adata.uns.keys() or "interactions" not in adata.uns["receptor-ligand"].keys():
        raise ValueError("Could not find interaction data! Please setup with `calculate_interaction_table(...)` before running this function.")

    data = adata.uns["receptor-ligand"]["interactions"].copy()

    # filter interactions
    if filter:
        data.query(filter, inplace=True)

    # restrict interactions to certain clusters
    if restrict_to:
        data = data[data[receptor_cluster_col].isin(restrict_to) & data[ligand_cluster_col].isin(restrict_to)]
    if len(data) < 1:
        raise Exception(f"No interactions between clusters {restrict_to}")

    # setup subplot
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi, gridspec_kw={'wspace': wspace})
    fig.suptitle(title)

    # receptor plot
    r_plot = sns.scatterplot(data=data, 
                             y=receptor_col,
                             x=receptor_cluster_col,
                             hue=receptor_hue,
                             size=receptor_size,
                             ax=axs[0])

    r_plot.set(xlabel="Cluster", ylabel=None, title="Receptor", axisbelow=True)
    axs[0].tick_params(axis='x', rotation=90)
    axs[0].grid(alpha=0.8)

    # ligand plot
    l_plot = sns.scatterplot(data=data,
                             y=ligand_col,
                             x=ligand_cluster_col,
                             hue=ligand_hue,
                             size=ligand_size,
                             ax=axs[1])

    axs[1].yaxis.tick_right()
    l_plot.set(xlabel="Cluster", ylabel=None, title="Ligand", axisbelow=True)
    axs[1].tick_params(axis='x', rotation=90)
    axs[1].grid(alpha=0.8)

    # force tick labels to be populated
    # https://stackoverflow.com/questions/41122923/getting-empty-tick-labels-before-showing-a-plot-in-matplotlib
    fig.canvas.draw()

    # add receptor-ligand lines
    receptors = list(set(data[receptor_col]))

    # create colorramp
    if line_colors:
        cmap = cm.get_cmap(line_colors, len(receptors))
        colors = cmap(range(len(receptors)))
    else:
        colors = ["black"] * len(receptors)

    # scale connection score column between 0-1 to be used as alpha values
    if connection_alpha:
        # note: minmax_scale sometimes produces values >1. Looks like a rounding error (1.000000000002).
        data["alpha"] = minmax_scale(data[connection_alpha], feature_range=(0, 1))
        # fix values >1
        data.loc[data["alpha"] > 1, "alpha"] = 1

    for rec, color in zip(receptors, colors):
        # find receptor label location
        rec_index = None
        for i, label in enumerate(axs[0].get_yticklabels()):
            if label.get_text() == rec:
                rec_index = i
                break

        for lig in data.loc[data[receptor_col] == rec, ligand_col]:
            # find ligand label location
            lig_index = None
            for i, label in enumerate(axs[1].get_yticklabels()):
                if label.get_text() == lig:
                    lig_index = i
                    break

            # TODO
            # a r-l pair can have multiple clusters, which results in overlapping connection lines
            # add the moment these lines are plotted on top of each other
            # compute line alpha
            if connection_alpha:
                alphas = data.loc[(data[receptor_col] == rec) & (data[ligand_col] == lig), "alpha"]
            else:
                alphas = [1]
            
            for alpha in alphas:
                # stolen from https://matplotlib.org/stable/gallery/userdemo/connect_simple01.html
                # Draw a line between the different points, defined in different coordinate
                # systems.
                con = ConnectionPatch(
                    # x in axes coordinates, y in data coordinates
                    xyA=(1, rec_index), coordsA=axs[0].get_yaxis_transform(),
                    # x in axes coordinates, y in data coordinates
                    xyB=(0, lig_index), coordsB=axs[1].get_yaxis_transform(),
                    arrowstyle="-",
                    color=color,
                    zorder=-1000,
                    alpha=alpha,
                    linewidth=alpha * lw_multiplier
                )

                axs[1].add_artist(con)

    ##### legends #####
    # set receptor plot legend position
    sns.move_legend(r_plot, loc='upper right', bbox_to_anchor=(-1, 1, 0, 0))

    # create legend for connection lines
    if connection_alpha:
        step_num = 5
        s_steps, a_steps = np.linspace(min(data[connection_alpha]), max(data[connection_alpha]), step_num), np.linspace(0, 1, step_num)

        # create proxy actors https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#proxy-legend-handles
        line_list = [lines.Line2D([], [], color="black", alpha=a, linewidth=a * lw_multiplier, label=f"{np.round(s, 2)}") for a, s in zip(a_steps, s_steps)]
        line_list.insert(0, lines.Line2D([], [], alpha=0, label=connection_alpha))

        # add to current legend
        handles, _ = axs[1].get_legend_handles_labels()
        axs[1].legend(handles=handles + line_list, bbox_to_anchor=(2, 1, 0, 0), loc='upper left')
    else:
        # set ligand plot legend position
        axs[1].legend(bbox_to_anchor=(2, 1, 0, 0), loc='upper left')

    if output:
        plt.savefig(output, bbox_inches='tight')

    return axs

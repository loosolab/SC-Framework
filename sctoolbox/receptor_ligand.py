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

def download_db(adata, db_path, ligand_column, receptor_column, sep="\t", inplace=False):
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

    Returns:
    ----------
        AnnData : optional
            Copy of adata with added database path and database table to adata.uns['receptor-ligand']
    """
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

def calculate_interaction_table(adata, cluster_column, gene_index=None, normalize=1000, inplace=False):
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

    Returns:
    ----------
        AnnData : optional
            Copy of adata with added interactions table to adata.uns['receptor-ligand']['interactions']
    """
    if "receptor-ligand" not in adata.uns.keys():
        raise ValueError("Could not find receptor-ligand database. Please setup database with `download_db()` before running this function.")

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
                    "ligand_scale_factor": []}
    
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
    
    interactions = pd.DataFrame(interactions)

    # compute interaction score
    interactions["receptor_score_corrected"] = interactions["receptor_score"] * interactions["receptor_scale_factor"]
    interactions["ligand_score_corrected"] = interactions["ligand_score"] * interactions["ligand_scale_factor"]
    interactions["interaction_score"] = interactions["receptor_score_corrected"] + interactions["ligand_score_corrected"]
    
    # add to adata
    modified_adata = adata if inplace else adata.copy()

    modified_adata.uns['receptor-ligand']['interactions'] = interactions
    modified_adata.uns['receptor-ligand']['percentage'] = cl_percent_expression

    if not inplace:
        return modified_adata

def interaction_violin_plot(interactions, min_perc, output, figsize=(14,20)):
    '''
    Generate violin plot of pairwise cluster interactions.
    
    Parameters:
        interactions (DataFrame): Interactions table as given by calculate_interaction_table().
        min_perc (float): Minimum percentage of cells in a cluster each gene must be expressed in.
        output (str): Path to output file.
        figsize (int tuple): Tuple of plot (width, height).
    '''
    
    rows = len(set(interactions["cluster_a"]))

    fig, axs = plt.subplots(ncols=1, nrows=rows, figsize=figsize, tight_layout={'rect': (0, 0, 1, 0.95)}) # prevent label clipping; leave space for title
    #fig.suptitle('Cluster interactions', fontsize=16)
    flat_axs = axs.flatten()

    # generate violins of one cluster vs rest in each iteration
    for i, cluster in enumerate(sorted(set(interactions["cluster_a"].tolist() + interactions["cluster_b"].tolist()))):
        cluster_interactions = interactions[((interactions["cluster_a"] == cluster) | 
                                            (interactions["cluster_b"] == cluster)) &
                                            (interactions["percentage_a"] >= min_perc) &
                                            (interactions["percentage_b"] >= min_perc)].copy()    
        cluster_interactions["Cluster"] = cluster_interactions.apply(lambda x: x[1] if x[0] == cluster else x[0], axis=1).tolist()

        plot = sns.violinplot(x=cluster_interactions["Cluster"],
                    y=cluster_interactions["interaction_score"], 
                    ax=flat_axs[i])
        
        plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
        
        flat_axs[i].set_title(f"Cluster {cluster}")

    # create path if necessary
    Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True)
    fig.savefig(output)

def hairball(interactions, min_perc, interaction_score, output, title, color_min=None, color_max=None):
    '''
    Generate network graph of interactions between clusters.
    
    Parameters:
        interactions (DataFrame): Interactions table as given by calculate_interaction_table().
        min_perc (float): Minimum percentage of cells in a cluster each gene must be expressed in.
        interaction_score (float): Interaction score must be above this threshold for the interaction to be counted in the graph.
        output (str): Path to output file.
        title (str): The plots title.
        color_min (float): Min value for color range. If None = 0
        color_max (float): Max value for color range. If None = np.max
    '''
    igraph_scale=3
    matplotlib_scale=4
    
    ########## setup class that combines igraph with matplotlib ##########
    # from https://stackoverflow.com/a/36154077
    # makes igraph compatible with matplotlib
    # so a colorbar can be added

    from matplotlib.artist import Artist
    from igraph import BoundingBox, Graph, palettes

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
    clusters = list(set(list(interactions["cluster_a"]) + list(interactions["cluster_b"])))
    
    graph.add_vertices(clusters)
    graph.vs['label'] = clusters
    graph.vs['size'] = 45
    graph.vs['label_size'] = 30

    # set edges
    for (a, b) in combinations_with_replacement(clusters, 2):
        subset = interactions[(((interactions["cluster_a"] == a) & (interactions["cluster_b"] == b)) |
                            ((interactions["cluster_a"] == b) & (interactions["cluster_b"] == a))) &
                            (interactions["percentage_a"] >= min_perc) &
                            (interactions["percentage_b"] >= min_perc) &
                            (interactions["interaction_score"] > interaction_score)]

        graph.add_edge(a, b, weight=len(subset))#, label=len(subset)) # add label to show edge labels

    # set edge colors/ width based on weight
    colormap = cm.get_cmap('viridis', len(graph.es))
    print(f"Max weight {np.max(np.array(graph.es['weight']))}")
    max_weight = np.max(np.array(graph.es['weight'])) if color_max is None else color_max
    for e in graph.es:
        e["color"] = colormap(e["weight"] / max_weight, e["weight"] / max_weight)
        e["width"] = (e["weight"] / max_weight) * 10
        
    ########## setup matplotlib plot and combine with igraph ##########
    # Make Matplotlib use a Cairo backend
    import matplotlib
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
                                        #label='Receptor-Ligand Interactions',
                                        norm=matplotlib.colors.Normalize(0 if color_min is None else color_min, 
                                                                        max_weight))
    
    cb.ax.tick_params(labelsize=10*matplotlib_scale) 

    # prevent label clipping out of picture
    plt.tight_layout()

    # create path if necessary
    Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    fig.savefig(output)

def progress_violins(datalist, datalabel, cluster_a, cluster_b, min_perc, output, figsize=(12, 6)):
    '''
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

def connectionPlot(data, 
                   restrict_to=None,
                   figsize=(25, 50),
                   connection_alpha=None,
                   output=None,
                   title=None,
                   # receptor params
                   receptor_cluster_col="cluster_a",
                   receptor_col="partner_a",
                   receptor_hue="Receptor score",
                   receptor_size="Genotype difference",
                   # ligand params
                   ligand_cluster_col="cluster_b",
                   ligand_col="partner_b",
                   ligand_hue="Ligand score",
                   ligand_size="Genotype difference"
                  ):
    # restrict interactions to certain clusters
    if restrict_to:
        data = data[data[receptor_cluster_col].isin(restrict_to) & data[ligand_cluster_col].isin(restrict_to)]
    if len(data) < 1:
        raise Exception(f"No interactions between clusters {restrict_to}")

    # setup subplot
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)

    # receptor plot
    r_plot = sns.scatterplot(data=data, 
                           y=receptor_col,
                           x=receptor_cluster_col,
                           hue=receptor_hue,
                           size=receptor_size,
                           ax=axs[0])

    axs[0].yaxis.set_label_position("right")
    sns.move_legend(r_plot, loc='best', bbox_to_anchor=(-1, 1, 0, 0))
    r_plot.set(xlabel="Cluster", ylabel=None, title="Receptor")
    #r_plot.set_xticklabels(r_plot.get_xticklabels(), rotation = 90)
    axs[0].tick_params(axis='x', rotation=90)

    # ligand plot
    l_plot = sns.scatterplot(data=data,
                           y=ligand_col,
                           x=ligand_cluster_col,
                           hue=ligand_hue,
                           size=ligand_size,
                          ax=axs[1])

    axs[1].yaxis.tick_right()
    sns.move_legend(l_plot, bbox_to_anchor=(2, 1, 0, 0), loc='best')
    l_plot.set(xlabel="Cluster", ylabel=None, title="Ligand")
    #l_plot.set_xticklabels(l_plot.get_xticklabels(), rotation = 90)
    axs[1].tick_params(axis='x', rotation=90)
    fig.tight_layout(pad=5)

    # draw receptor-ligand lines
    receptors = list(set(data[receptor_col]))
    diff_max = max(data[connection_alpha]) if connection_alpha else None
    # create colorramp
    cmap = cm.get_cmap('rainbow', len(receptors))
    colors = cmap(range(len(receptors)))
    for rec, color in zip(receptors, colors):
        rec_index = [i for i, label in enumerate(axs[0].get_yticklabels()) if label.get_text() == rec][0]

        ligands = list(data.loc[data[receptor_col] == rec, ligand_col])
        for lig_index, lig in [(i, label.get_text()) for i, label in enumerate(axs[1].get_yticklabels()) if label.get_text() in ligands]:
            # compute line alpha
            if connection_alpha:
                alpha = data.loc[(data[receptor_col] == rec) & (data[ligand_col] == lig), connection_alpha].values[0] / diff_max
                alpha **= 2
            else:
                alpha=1
            
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
                linewidth=alpha**2
            )

            axs[1].add_artist(con)
            
    if output:
        plt.savefig(output)

    return axs

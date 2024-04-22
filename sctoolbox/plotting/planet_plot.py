#load libraries
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import rgb2hex
import sctoolbox.utilities as utils
import sctoolbox.tools as tools
import sctoolbox.plotting as pl
from beartype import beartype
from beartype.typing import Iterable, Optional, Literal, Tuple, Union, Any




#count for values > threshold
def count_greater_than_threshold(group, threshold):
    return (group > threshold).sum()


#calculate dot sizes function
def calculate_dot_sizes(values, max_value, min_dot_size, max_dot_size, use_log_scale=False):
    if use_log_scale:
        values = np.log1p(values)
        max_value = np.log1p(max_value)
    #size relative to the largest dot on the plot
    fraction_count = values/max_value
    # Scale to [min_size, max_size] range
    sizes = fraction_count * (max_dot_size - min_dot_size) + min_dot_size
    return sizes


#calculate plot size
def calculate_figure_size_planet_mode(x_range, y_range, max_size, dot_scale):
    width = x_range*(3*max_size)*dot_scale
    height = y_range*(3*max_size)*dot_scale
    return width, height


def calculate_figure_size_mean_mode(x_range, y_range, max_size, dot_scale):
    width = x_range*(1.5*max_size)*dot_scale
    height = y_range*(1.5*max_size)*dot_scale
    return width, height


def planet_plot_anndata_preprocess(adata: sc.AnnData,
                                  x_col: str,
                                  y_col: str,
                                  genes: list | str,
                                  gene_symbols: str,
                                  x_col_subset: list = None,
                                  y_col_subset: list = None,
                                  input_layer: str = None,
                                  fillna: int | float = 0.0,
                                  expression_threshold: int | float = 0.0):
    """
    Preprocesses the annData object for the planet plot.
    
    Parameters
    ----------
    expression_threshold : int, default 0.
        Value of expression greater than which the gene is regarded as expressed.
    adata : sc.AnnData
        Annotated data matrix object.
    x_col : str
        Name of adata column to be used for x-axis.
    y_col : str
        Name of adata column to be used for y-axis.
    x_col_subset : str
        Name of adata column to be used for x-axis.
    y_col_subset : str
        Name of adata column to be used for y-axis.
    genes : np.ndarray
        Names of the relavant genes.
    gene_symbols : str
        Column of adata.var to search in the genes for.
    input_layer : str, default 'raw'
        Name of the layer used for the data.
        
    Returns
    -------
    axes : npt.ArrayLike (?)
        Array of axis objects

    Raises
    ------
    KeyError (?)
        If the given method is not found in adata.obsm.
    ValueError (?)
        If the 'components' given is larger than the number of components in the embedding.

    Examples
    --------
    TODO
    
    """
    #initialize use_raw
    use_raw = False
    
    if input_layer == "raw":
        use_raw = True
        input_layer = None
    
    if isinstance(genes, str):
        genes = [genes]
    
    #get the genex values from the adata
    df_all = sc.get.obs_df(adata, [*genes, x_col, y_col], gene_symbols = gene_symbols, layer = input_layer, use_raw = use_raw)
        
    #get the count from the adata
    df_counts = sc.get.obs_df(adata, [x_col, y_col])
    df_counts = df_counts.groupby([x_col, y_col]).size().reset_index(name='total_count')
    
    #get the count of the expressed
    df_expressed_counts = df_all.groupby([x_col, y_col]).apply(lambda x: count_greater_than_threshold(x[genes], expression_threshold)).reset_index()
    
    #get mean expressions  (mode: mean)
    df_mean_expressions = df_all.groupby([x_col, y_col]).mean().reset_index()
    
    #merge the dataframes. Note that after merging, the 'genename_x' is the gene count , whereas 'genename_y' is the gene expression.
    plot_vars = pd.merge(df_counts, df_expressed_counts, on=[x_col, y_col])
    plot_vars = pd.merge(plot_vars, df_mean_expressions, on=[x_col, y_col])
    
    #perform subsetting
    if x_col_subset is not None:
        plot_vars = plot_vars[plot_vars[x_col].isin(x_col_subset)].reset_index(drop=True)
        plot_vars[x_col] = plot_vars[x_col].cat.remove_unused_categories()
    if y_col_subset is not None:
        plot_vars = plot_vars[plot_vars[y_col].isin(y_col_subset)].reset_index(drop=True)
        plot_vars[y_col] = plot_vars[y_col].cat.remove_unused_categories()
    
    #calculate expression weighted counts mean, mean expression, percentage expressed (mode: mean, count, percentage)
    cols_set1 = [col + '_x' for col in genes]
    cols_set2 = [col + '_y' for col in genes]

    plot_vars['expression_weighted_count'] = sum(plot_vars[col1] * plot_vars[col2] for col1, col2 in zip(cols_set1, cols_set2))/sum(plot_vars[col2] for col2 in cols_set2)
    plot_vars['mean_expression'] = sum(plot_vars[col2] for col2 in cols_set2)/len(cols_set2)
    #REVIEW the requirement of percentage_expressed
    plot_vars['percentage_expressed'] = plot_vars['expression_weighted_count']*100/plot_vars['total_count']
    plot_vars['percentage_max_expression'] = plot_vars['mean_expression']*100/np.max(plot_vars['mean_expression'])
    
    #calculate percentage expressed and percentage max expression observed for each gene (mode: planet, percentage)
    for gene in genes:
        plot_vars[gene+'_percentage_expressed'] = plot_vars[gene+'_x']*100/plot_vars['total_count']
        plot_vars[gene+'_percentage_max_expression'] = plot_vars[gene+'_y']*100/np.max(plot_vars[gene+'_y'])
    
    # Convert the numeric columns to float and fill NaN with 0
    # List of columns to exclude from conversion
    exclude_columns = [x_col, y_col]
    # Convert all other columns to float and fillna
    for col in plot_vars.columns:
        if col not in exclude_columns:
            plot_vars[col] = pd.to_numeric(plot_vars[col], errors='coerce').astype(float)
            plot_vars[col] = plot_vars[col].fillna(fillna)

    return plot_vars
    
    
def planet_plot_anndata_preprocess_advanced(adata: sc.AnnData,
                                          x_col: str,
                                          y_col: str,
                                          genes: list | str,
                                          gene_symbols: str,
                                          x_col_subset: list = None,
                                          y_col_subset: list = None,
                                          input_layer: str = None,
                                          fillna: int | float = 0.0,
                                          expression_threshold: int | float = 0.0,
                                          planet_columns: list = None,
                                          planet_thresholds: list = None,
                                          planet_color_schemas: list = None):
    """
    This function has additional functionality to use columns other than genes for the dots.
    
    Parameters
    ----------
    expression_threshold : int, default 0.
        Value of expression greater than which the gene is regarded as expressed.
    adata : sc.AnnData
        Annotated data matrix object.
    x_col : str
        Name of adata column to be used for x-axis.
    y_col : str
        Name of adata column to be used for y-axis.
    x_col_subset : str
        Name of adata column to be used for x-axis.
    y_col_subset : str
        Name of adata column to be used for y-axis.
    genes : np.ndarray
        Names of the relavant genes.
    gene_symbols : str
        Column of adata.var to search in the genes for.
    input_layer : str, default 'raw'
        Name of the layer used for the data.
        
    Returns
    -------
    axes : npt.ArrayLike (?)
        Array of axis objects

    Raises
    ------
    KeyError (?)
        If the given method is not found in adata.obsm.
    ValueError (?)
        If the 'components' given is larger than the number of components in the embedding.

    Examples
    --------
    TODO
    
    """
    #initialize use_raw
    use_raw = False
    
    if input_layer == "raw":
        use_raw = True
        input_layer = None
    
    if isinstance(genes, str):
        genes = [genes]
        
    total_columns = [*genes, *planet_columns]
    total_thresholds = [*([expression_threshold]*len(genes)), *planet_thresholds]
    
    #get the genex values from the adata
    df_all = sc.get.obs_df(adata, [*genes, *planet_columns, x_col, y_col], gene_symbols = gene_symbols, layer = input_layer, use_raw = use_raw)
    
    #get the count from the adata
    df_counts = sc.get.obs_df(adata, [x_col, y_col])
    df_counts = df_counts.groupby([x_col, y_col]).size().reset_index(name='total_count')
    
    #get the count of the expressed
    df_expressed_counts = df_all.groupby([x_col, y_col]).apply(lambda x: pd.Series({total_columns[i]: count_greater_than_threshold(x[total_columns[i]], total_thresholds[i]) for i in range(len(total_columns))})).reset_index()
    
    
    #get mean expressions  (mode: mean)
    df_mean_expressions = df_all.groupby([x_col, y_col]).mean().reset_index()
    
    #merge the dataframes. Note that after merging, the 'genename_x' is the gene count , whereas 'genename_y' is the gene expression.
    plot_vars = pd.merge(df_counts, df_expressed_counts, on=[x_col, y_col])
    plot_vars = pd.merge(plot_vars, df_mean_expressions, on=[x_col, y_col])
    
    #perform subsetting
    if x_col_subset is not None:
        plot_vars = plot_vars[plot_vars[x_col].isin(x_col_subset)].reset_index(drop=True)
        plot_vars[x_col] = plot_vars[x_col].cat.remove_unused_categories()
    if y_col_subset is not None:
        plot_vars = plot_vars[plot_vars[y_col].isin(y_col_subset)].reset_index(drop=True)
        plot_vars[y_col] = plot_vars[y_col].cat.remove_unused_categories()
    
    #calculate expression weighted counts mean, mean expression, percentage expressed (mode: mean, count, percentage)
    cols_set1 = [col + '_x' for col in genes]
    cols_set2 = [col + '_y' for col in genes]

    plot_vars['expression_weighted_count'] = sum(plot_vars[col1] * plot_vars[col2] for col1, col2 in zip(cols_set1, cols_set2))/sum(plot_vars[col2] for col2 in cols_set2)
    plot_vars['mean_expression'] = sum(plot_vars[col2] for col2 in cols_set2)/len(cols_set2)
    #REVIEW the requirement of percentage_expressed
    plot_vars['percentage_expressed'] = plot_vars['expression_weighted_count']*100/plot_vars['total_count']
    plot_vars['percentage_max_expression'] = plot_vars['mean_expression']*100/np.max(plot_vars['mean_expression'])
    
    #calculate percentage expressed and percentage max expression observed for each gene (mode: planet, percentage)
    for column in total_columns:
        plot_vars[column+'_percentage_expressed'] = plot_vars[column+'_x']*100/plot_vars['total_count']
        plot_vars[column+'_percentage_max_expression'] = plot_vars[column+'_y']*100/np.max(plot_vars[column+'_y'])
    
    # Convert the numeric columns to float and fill NaN with 0
    # List of columns to exclude from conversion
    exclude_columns = [x_col, y_col]
    # Convert all other columns to float and fillna
    for col in plot_vars.columns:
        if col not in exclude_columns:
            plot_vars[col] = pd.to_numeric(plot_vars[col], errors='coerce').astype(float)
            plot_vars[col] = plot_vars[col].fillna(fillna)
            
    return plot_vars
            
    
def planet_plot_render(plot_vars: pd.DataFrame,
                        x_col: str,
                        y_col: str,
                        x_label: str = None,
                        y_label: str = None,
                        mode : Literal["mean", "planet"] = "mean",
                        size_value : Literal["count", "percentage"] = "count",
                        color_value : Literal["expression" , "percentage_max"] = "expression",
                        use_log_scale : bool = False,
                        planet_columns: list = None,
                        color_schema: str = "Blues",
                        planet_color_schemas: list = None,
                        OUTER_SIZE_COUNT_COLUMN: str = 'total_count',
                        INNER_SIZE_COUNT_COLUMN: str = 'expression_weighted_count',
                        INNER_SIZE_PERCENTAGE_COLUMN: str = 'percentage_expressed',
                        DOT_COLOR_VALUE_COLUMN: str = 'mean_expression',
                        DOT_COLOR_PERCENTAGE_MAX_VALUE_COLUMN: str = 'percentage_max_expression',
                        PLANET_SIZE_COUNT_SUFFIX: str = '_x',
                        PLANET_SIZE_PERCENTAGE_SUFFIX: str = '_percentage_expressed',
                        PLANET_COLOR_VALUE_SUFFIX: str = '_y',
                        PLANET_COLOR_PERCENTAGE_MAX_VALUE_SUFFIX: str = '_percentage_max_expression',
                        COLOR_BAR_TITLE: str = 'Gene Expression',
                        COLOR_BAR_TITLE_PMV: str = 'Percentage of max expression\nobserved for the gene',
                        SIZE_LEGEND_TITLE: str = 'Cell Count',
                        SIZE_LEGEND_TITLE_PERCENTAGE: str = 'Percentage cells expressed',
                        OUTER_CIRCLE_LABEL: str = 'Total count',
                        INNER_DOT_LABEL: str = 'Expressed count',
                        ORIENTATION_LEGEND_TITLE: str = 'Genes',
                        ORIENTATION_LEGEND_CENTER_LABEL: str = 'Mean'):
    """
    Renders the planet plot on the basis of the preprocessed data.
    
    Parameters
    ----------
    mode : Literal["mean", "planet"], default "mean"
        Type of plot. "mean" mode displays count weighted mean of the gene expression as the middle dot. "planet" mode additionaly displays gene expression for up to 6 genes as small dots surrounding the middle dot. Note: that if there are more than 6 genes in the list, the first 6 mentioned genes are displayed as planets.
    size_value : Literal["count", "percentage"], default "count".
        The size of the dots can either correspond to counts of the cells or the percentage of the cells with > threshold expression. When the size_value is set to "count", then the plots are shown in 'Orbit' mode instead of 'Dot' mode.
    expression_threshold : int, default 0.
        Value of expression greater than which the gene is regarded as expressed.
    color_value : Literal["expression" , "percentage_max"], default "expression".
        When set to "percentage_max" the color of the dots corresponds to the percentage gene expression relative to the max gene expression observed for that particular gene. In case of "mean" mode it is relative to the max mean observed.
    log_scale : bool, default false.
        Use log scale for dot sizes. Log scale can be used when the disparity between the vales is very high.
    adata : sc.AnnData
        Annotated data matrix object.
    x_col : str
        Name of adata column to be used for x-axis.
    y_col : str
        Name of adata column to be used for y-axis.
    x_label: str
        Label for the x-axis.
    y_label: str
        Label for the y-axis.
    genes : np.ndarray
        Names of the relavant genes.
    gene_symbols : str
        Column of adata.var to search in the genes for.
    color : str
        Color schema for the gene expression.
    legend : Optional[Literal["reduced", "full"]], default "reduced"
        Toggle between reduced or full legend.
    cellnumbers : bool, default false
        Give bars on the right with total numbers.
    save : Optional[str], default None
        Filename to save the figure.
        
    Returns
    -------
    axes : npt.ArrayLike (?)
        Array of axis objects

    Raises
    ------
    KeyError (?)
        If the given method is not found in adata.obsm.
    ValueError (?)
        If the 'components' given is larger than the number of components in the embedding.

    Examples
    --------
    TODO
    
    """
    #plotting parameters:
    #df column mapping defaults
#     OUTER_SIZE_COUNT_COLUMN = 'total_count'
#     INNER_SIZE_COUNT_COLUMN = 'expression_weighted_count'
#     INNER_SIZE_PERCENTAGE_COLUMN = 'percentage_expressed'
#     DOT_COLOR_VALUE_COLUMN = 'mean_expression'
#     DOT_COLOR_PERCENTAGE_MAX_VALUE_COLUMN = 'percentage_max_expression'
#     PLANET_SIZE_COUNT_SUFFIX = '_x'
#     PLANET_SIZE_PERCENTAGE_SUFFIX = '_percentage_expressed'
#     PLANET_COLOR_VALUE_SUFFIX = '_y'
#     PLANET_COLOR_PERCENTAGE_MAX_VALUE_SUFFIX = '_percentage_max_expression'
        
    #Init Params:
    DOT_SCALE = 1/72
    MAX_DOT_SIZE = 30
    MIN_DOT_SIZE = 3
    LINE_WIDTH = 1

    #legend dimensions in inches:
    LEGEND_COLOR_HEIGHT = 0.2
    LEGEND_COLOR_WIDTH = 2
    LEGEND_DOT_HEIGHT = 2.5
    LEGEND_DOT_WIDTH = 2
    LEGEND_PLANET_HEIGHT = 2
    LEGEND_PLANET_WIDTH = 2
    
    #default legend labels    
#     COLOR_BAR_TITLE = 'Gene Expression'
#     COLOR_BAR_TITLE_PMV = 'Percentage of max expression\nobserved for the gene'
#     SIZE_LEGEND_TITLE = 'Cell Count'
#     SIZE_LEGEND_TITLE_PERCENTAGE = 'Percentage cells expressed'
#     OUTER_CIRCLE_LABEL = 'Total count'
#     INNER_DOT_LABEL = 'Expressed count'
#     ORIENTATION_LEGEND_TITLE = 'Genes'
#     ORIENTATION_LEGEND_CENTER_LABEL = 'Mean'
        
    #planet extras
    if isinstance(planet_columns, str):
        planet_columns = [planet_columns]
    
    #planet details
    planet_x = [0,-0.866,-0.866,0,0.866,0.866]
    planet_y = [1,0.5,-0.5,-1,-0.5,0.5]
    planet_text = [{'va': 'bottom'},{'ha': 'right', 'va': 'bottom'},{'ha': 'right', 'va': 'top'},{'va': 'top'},{'ha': 'left', 'va': 'top'},{'ha': 'left', 'va': 'bottom'}]
    
    
    #max cell count
    max_count = np.max(plot_vars[OUTER_SIZE_COUNT_COLUMN])
    min_count = 0
    
    #calculating vmin and vmax
    vmin = min(plot_vars[DOT_COLOR_VALUE_COLUMN])
    vmax = max(plot_vars[DOT_COLOR_VALUE_COLUMN])
    if mode == "planet" and planet_color_schemas is None:
        vmin = np.min(plot_vars[[planet_column + PLANET_COLOR_VALUE_SUFFIX for planet_column in planet_columns]])
        vmax = np.max(plot_vars[[planet_column + PLANET_COLOR_VALUE_SUFFIX for planet_column in planet_columns]])
    if planet_color_schemas is not None:
        vmin_array = np.min(plot_vars[[planet_column + PLANET_COLOR_VALUE_SUFFIX for planet_column in planet_columns]], axis=0)
        vmax_array = np.max(plot_vars[[planet_column + PLANET_COLOR_VALUE_SUFFIX for planet_column in planet_columns]], axis=0)
        
    # Convert categories to numerical values for plotting
    plot_vars['x_steps'] = plot_vars[x_col].astype('category').cat.codes
    plot_vars['y_steps'] = plot_vars[y_col].astype('category').cat.codes

    # Get unique categories for each annotation
    x_categories = plot_vars[x_col].astype('category').cat.categories
    y_categories = plot_vars[y_col].astype('category').cat.categories

    # Create mappings from codes to original labels
    # Note: The codes are integers from 0 to n_categories-1, directly usable as indices for the categories
    x_labels = {code: label for code, label in enumerate(x_categories)}
    y_labels = {code: label for code, label in enumerate(y_categories)}
    
    #dot size calculation
    if size_value == 'count':
        plot_vars['outer_size'] = calculate_dot_sizes(plot_vars[OUTER_SIZE_COUNT_COLUMN], max_count, MIN_DOT_SIZE, MAX_DOT_SIZE, use_log_scale)
        plot_vars['inner_size'] = calculate_dot_sizes(plot_vars[INNER_SIZE_COUNT_COLUMN], max_count, MIN_DOT_SIZE, MAX_DOT_SIZE, use_log_scale)
        if mode == 'planet':
            for planet_column in planet_columns:
                plot_vars[planet_column+'_dot_size'] = calculate_dot_sizes(plot_vars[planet_column+PLANET_SIZE_COUNT_SUFFIX], max_count, MIN_DOT_SIZE, MAX_DOT_SIZE, use_log_scale)
            
    if size_value == 'percentage':
        plot_vars['inner_size'] = calculate_dot_sizes(plot_vars[INNER_SIZE_PERCENTAGE_COLUMN], 100, MIN_DOT_SIZE, MAX_DOT_SIZE, use_log_scale)
        if mode == 'planet':
            for planet_column in planet_columns:
                plot_vars[planet_column+'_dot_size'] = calculate_dot_sizes(plot_vars[planet_column+PLANET_SIZE_PERCENTAGE_SUFFIX], 100, MIN_DOT_SIZE, MAX_DOT_SIZE, use_log_scale)
        
    #figure size calculation
    if mode == 'planet':
        w, h = calculate_figure_size_planet_mode(len(plot_vars['x_steps'].unique()), len(plot_vars['y_steps'].unique()), MAX_DOT_SIZE, DOT_SCALE)
    
    if mode == 'mean':
        w, h = calculate_figure_size_mean_mode(len(plot_vars['x_steps'].unique()), len(plot_vars['y_steps'].unique()), MAX_DOT_SIZE, DOT_SCALE)
                
    height_per_unit = 1/h
    width_per_unit = 1/w

    # Plot main dots
    fig, ax = plt.subplots(figsize=(w, h))

    plt.xticks(ticks=range(len(x_categories)), labels=x_categories, rotation='vertical')
    plt.yticks(ticks=range(len(y_categories)), labels=y_categories)

    #size value condition
    if size_value == 'count':
        ax.scatter(plot_vars['x_steps'], plot_vars['y_steps'], s=plot_vars['outer_size']**2, facecolors='none', edgecolors='0',linewidths=LINE_WIDTH)
    if color_value == 'expression':
        sc = ax.scatter(plot_vars['x_steps'], plot_vars['y_steps'], s=plot_vars['inner_size']**2, c=plot_vars[DOT_COLOR_VALUE_COLUMN], cmap=color_schema, vmin=vmin, vmax=vmax)
    if color_value == 'percentage_max':
        sc = ax.scatter(plot_vars['x_steps'], plot_vars['y_steps'], s=plot_vars['inner_size']**2, c=plot_vars[DOT_COLOR_PERCENTAGE_MAX_VALUE_COLUMN], cmap=color_schema, vmin=0, vmax=100)

    if mode == 'planet':
        sc_array = {}
        for i, row in plot_vars.iterrows():
            p=0
            outer_size = row['outer_size']
            for planet_column in planet_columns:
                #test if planet capacity is reached
                if p>5: continue
                size = row[planet_column+'_dot_size']
                size_factor = (3*LINE_WIDTH+ (outer_size+size)/2)*DOT_SCALE
                offset_x = size_factor*planet_x[p]
                offset_y = size_factor*planet_y[p]
                secondary_x = row['x_steps'] + offset_x
                secondary_y = row['y_steps'] + offset_y
                if planet_color_schemas is not None:
                    if color_value == 'expression':
                        sc_array[p] = ax.scatter(secondary_x, secondary_y, s=row[planet_column+'_dot_size']**2, c=row[planet_column+PLANET_COLOR_VALUE_SUFFIX],cmap=planet_color_schemas[p], vmin=vmin_array[p], vmax=vmin_array[p])
                    if color_value == 'percentage_max':
                        sc_array[p] = ax.scatter(secondary_x, secondary_y, s=row[planet_column+'_dot_size']**2, c=row[planet_column+PLANET_COLOR_PERCENTAGE_MAX_VALUE_SUFFIX],cmap=planet_color_schemas[p], vmin=0, vmax=100)
                else:
                    if color_value == 'expression':
                        ax.scatter(secondary_x, secondary_y, s=row[planet_column+'_dot_size']**2, c=row[planet_column+PLANET_COLOR_VALUE_SUFFIX],cmap=color_schema, vmin=vmin, vmax=vmax)
                    if color_value == 'percentage_max':
                        ax.scatter(secondary_x, secondary_y, s=row[planet_column+'_dot_size']**2, c=row[planet_column+PLANET_COLOR_PERCENTAGE_MAX_VALUE_SUFFIX],cmap=color_schema, vmin=0, vmax=100)
                p=p+1

    plt.xlim(-0.5,len(x_labels)-0.5)
    plt.ylim(-0.5,len(y_labels)-0.5)

    if x_label is None:
        x_label = x_col
    ax.set_xlabel(x_label)
    if y_label is None:
        y_label = y_col
    ax.set_ylabel(y_label)

    
    #color_legend
    cbar_ax = fig.add_axes([1 + 0.25*LEGEND_COLOR_WIDTH*width_per_unit, 0.1, LEGEND_COLOR_WIDTH*width_per_unit, LEGEND_COLOR_HEIGHT*height_per_unit])

    cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal', fraction=0.046, pad=0.04)
    if color_value == 'expression':
        cbar.set_label(COLOR_BAR_TITLE, labelpad=-50)
    if color_value == 'percentage_max':
        cbar.set_label(COLOR_BAR_TITLE_PMV, labelpad=-60)

    #get a color value from the given schema for the other legends
    cmap = getattr(plt.cm, color_schema)
    #Specify the value you want to query in the colormap, between 0 and 1
    legend_color_value = 0.7
    # Get the color corresponding to the value
    legend_color = cmap(legend_color_value)
    legend_hex_color = rgb2hex(legend_color)
    
    cbar_count = 1
    planet_cbar_ax = {}
    planet_legend_hex_color = {}
    if planet_color_schemas is not None:
        for i in range(len(sc_array)):
            planet_cbar_ax[i] = fig.add_axes([1 + 0.25*LEGEND_COLOR_WIDTH*width_per_unit, 0.1 + cbar_count*5*LEGEND_COLOR_HEIGHT*height_per_unit, LEGEND_COLOR_WIDTH*width_per_unit, LEGEND_COLOR_HEIGHT*height_per_unit])

            cbar = plt.colorbar(sc_array[i], cax=planet_cbar_ax[i], orientation='horizontal', fraction=0.046, pad=0.04)
            if color_value == 'expression':
                cbar.set_label(planet_columns[i], labelpad=-50)
            if color_value == 'percentage_max':
                cbar.set_label(planet_columns[i], labelpad=-60)

            #get a color value from the given schema for the other legends
            cmap = getattr(plt.cm, planet_color_schemas[i])
            #Specify the value you want to query in the colormap, between 0 and 1
            legend_color_value = 0.7
            # Get the color corresponding to the value
            legend_color = cmap(legend_color_value)
            planet_legend_hex_color[i] = rgb2hex(legend_color)
            cbar_count += 1

    
    #dots_legend
    if mode == 'planet':
        dot_size_ax = fig.add_axes([1 + 0.25*LEGEND_DOT_WIDTH*width_per_unit, 0.1+ cbar_count*5*LEGEND_COLOR_HEIGHT*height_per_unit+ 1.1*LEGEND_PLANET_HEIGHT*height_per_unit, LEGEND_DOT_WIDTH*width_per_unit, LEGEND_DOT_HEIGHT*height_per_unit])
    if mode == 'mean':
        dot_size_ax = fig.add_axes([1 + 0.25*LEGEND_DOT_WIDTH*width_per_unit, 0.1+ cbar_count*5*LEGEND_COLOR_HEIGHT*height_per_unit, LEGEND_DOT_WIDTH*width_per_unit, LEGEND_DOT_HEIGHT*height_per_unit])

    values = np.linspace(min_count, max_count, num=5)
    #scale sizes
    sizes = calculate_dot_sizes(values, max_count, MIN_DOT_SIZE, MAX_DOT_SIZE, use_log_scale)

    #size value condition
    if size_value == 'percentage':
        labels = np.linspace(0, 100, num=5).astype(int)
    if size_value == 'count':
        labels = np.linspace(min_count, max_count, num =5).astype(int)

    for i in range(len(sizes)):
        #size value condition
        if size_value == 'percentage':
            dot_size_ax.scatter(0, i, s= sizes[i]**2, c=legend_hex_color, alpha=1)
        if size_value == 'count':
            dot_size_ax.scatter(-0.25, i, s= sizes[i]**2, c=legend_hex_color, alpha=1)
            dot_size_ax.scatter(-0.25, i, s=sizes[i]**2, facecolors='none', edgecolors='0',linewidths=LINE_WIDTH)
            if i == len(sizes)-1:
                dot_size_ax.annotate(OUTER_CIRCLE_LABEL, 
                            xy=(-0.25 + DOT_SCALE*sizes[i]/(1.5*np.pi), i), xycoords='data',
                            xytext=(-0.25 + 0.25, i), textcoords='data',
                            arrowprops=dict(arrowstyle="-", color="black", lw=LINE_WIDTH),
                            fontsize=8, ha='left', va='center')
            if i == len(sizes)-2:
                dot_size_ax.annotate(INNER_DOT_LABEL, 
                            xy=(-0.25, i), xycoords='data',
                            xytext=(-0.25 + 0.25, i), textcoords='data',
                            arrowprops=dict(arrowstyle="-", color="black", lw=LINE_WIDTH),
                            fontsize=8, va = 'center')

    for spine in dot_size_ax.spines.values():
        spine.set_visible(False)
    dot_size_ax.set_xticks([])
    dot_size_ax.set_xticklabels([])
    dot_size_ax.set_yticks(range(len(sizes)))
    dot_size_ax.set_yticklabels(labels)
    dot_size_ax.margins(x=0, y=0.2)
    dot_size_ax.set_xlim(-0.5, 0.5)
    #size value condition
    if size_value == 'count':
        dot_size_ax.tick_params(axis='y', which='major', pad=-15, length=0)
        dot_size_ax.set_title(SIZE_LEGEND_TITLE, fontsize=10)
    if size_value == 'percentage':
        dot_size_ax.tick_params(axis='y', which='major', pad=-55, length=0)
        dot_size_ax.set_title(SIZE_LEGEND_TITLE_PERCENTAGE, fontsize=10)


    #planet legend
    if mode == 'planet':
        dot_orientation_ax = fig.add_axes([1 + 0.25*LEGEND_PLANET_WIDTH*width_per_unit,0.1+ cbar_count*5*LEGEND_COLOR_HEIGHT*height_per_unit, LEGEND_PLANET_WIDTH*width_per_unit, LEGEND_PLANET_HEIGHT*height_per_unit])
        size = (MIN_DOT_SIZE+MAX_DOT_SIZE)/2

        #size value condition
        if size_value == 'count':
            dot_orientation_ax.scatter(0, 0, s=(size*2)**2, facecolors='none', edgecolors='0',linewidths=LINE_WIDTH, alpha=1)

        dot_orientation_ax.scatter(0, 0, s=(size*1.5)**2, c=legend_hex_color, alpha=1)
        dot_orientation_ax.text(0,0, ORIENTATION_LEGEND_CENTER_LABEL, fontsize=8, ha='center', va='center')

        p=0
        for planet_column in planet_columns:
            #test if planet capacity is reached
            if p>5: continue
            size_factor = (LINE_WIDTH+size)*DOT_SCALE
            offset_x = size_factor*planet_x[p]
            offset_y = size_factor*planet_y[p]
            secondary_x = offset_x
            secondary_y = offset_y
            if planet_color_schemas is not None:
                dot_orientation_ax.scatter(secondary_x, secondary_y, s=size**2, c=planet_legend_hex_color[p], alpha=1)
            else:
                dot_orientation_ax.scatter(secondary_x, secondary_y, s=size**2, c=legend_hex_color, alpha=1)
            dot_orientation_ax.text(1.5*secondary_x, 1.5*secondary_y, planet_column, fontsize=8, **planet_text[p])
            p=p+1

        for spine in dot_orientation_ax.spines.values():
            spine.set_visible(False)


        # Remove tick marks and labels
        dot_orientation_ax.set_xticks([])
        dot_orientation_ax.set_yticks([])

        dot_orientation_ax.set_xlabel('')
        dot_orientation_ax.set_ylabel('')

        # Turn off tick labels
        dot_orientation_ax.set_xticklabels([])
        dot_orientation_ax.set_yticklabels([])

        dot_orientation_ax.margins(x=0.9, y=0.9)
        dot_orientation_ax.set_aspect('equal')
        dot_orientation_ax.set_title(ORIENTATION_LEGEND_TITLE, fontsize=10, y=0.9)


    
    
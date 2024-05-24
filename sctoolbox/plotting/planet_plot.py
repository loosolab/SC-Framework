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
def calculate_figure_size_planet_mode(x_range, y_range, max_size, ipd, scaler):
    width = x_range*(6*max_size)*ipd*scaler
    height = y_range*(6*max_size)*ipd*scaler
    return width, height


def calculate_figure_size_mean_mode(x_range, y_range, max_size, ipd, scaler):
    width = x_range*(2*max_size)*ipd*scaler
    height = y_range*(2*max_size)*ipd*scaler
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
                                  expression_threshold: int | float = 0.0,
                                  layer_value_aggregator : str = "mean",
                                  gene_count_aggregator : str = "median",
                                  gene_expression_aggregator : str = "median",
                                  **kwargs):
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
    #initialize use_raw=False
    defaultargs = {
        "use_raw":False
    }
    defaultargs.update(kwargs)
    
    if isinstance(genes, str):
        genes = [genes]

    if gene_count_aggregator == "expression_weighted_count" and gene_expression_aggregator == "count_weighted_expression":
        raise ValueError("'gene_count_aggregator' cannot be set to 'expression_weighted_count' when 'gene_expression_aggregator' is set to 'count_weighted_expression'")
    
    #get the genex values from the adata
    df_all = sc.get.obs_df(adata, [*genes, x_col, y_col], gene_symbols = gene_symbols, layer = input_layer, use_raw = defaultargs['use_raw'])
        
    #get the count from the adata
    df_counts = sc.get.obs_df(adata, [x_col, y_col])
    df_counts = df_counts.groupby([x_col, y_col]).size().reset_index(name='total_count')
    
    #get the count of observations exceeding threshold per gene per given clcuster
    df_exceedance_counts = df_all.groupby([x_col, y_col]).apply(lambda x: count_greater_than_threshold(x[genes], expression_threshold)).reset_index()
    
    #get aggregate expression values per gene per given cluster
    df_aggregate_values = df_all.groupby([x_col, y_col]).agg(layer_value_aggregator).reset_index()
    
    #merge the dataframes. Note that after merging, the 'genename_x' is the gene count , whereas 'genename_y' is the gene expression.
    plot_vars = pd.merge(df_counts, df_exceedance_counts, on=[x_col, y_col])
    plot_vars = pd.merge(plot_vars, df_aggregate_values, on=[x_col, y_col])
    
    #perform subsetting
    if x_col_subset is not None:
        plot_vars = plot_vars[plot_vars[x_col].isin(x_col_subset)].reset_index(drop=True)
        plot_vars[x_col] = plot_vars[x_col].cat.remove_unused_categories()
    if y_col_subset is not None:
        plot_vars = plot_vars[plot_vars[y_col].isin(y_col_subset)].reset_index(drop=True)
        plot_vars[y_col] = plot_vars[y_col].cat.remove_unused_categories()
    
    
    #calculate aggregates of all the given genes for the center dot
    cols_count = [col + '_x' for col in genes]
    cols_expression = [col + '_y' for col in genes]

    if gene_count_aggregator == "expression_weighted_count":
        plot_vars['agg_count'] = sum(plot_vars[col1] * plot_vars[col2] for col1, col2 in zip(cols_count, cols_expression))/sum(plot_vars[col2] for col2 in cols_expression)
    else:
        plot_vars['agg_count'] = plot_vars[cols_count].agg(gene_count_aggregator, axis=1)
    
    if gene_count_aggregator == "count_weighted_expression":
        plot_vars['agg_expression'] = sum(plot_vars[col1] * plot_vars[col2] for col1, col2 in zip(cols_count, cols_expression))/sum(plot_vars[col1] for col1 in cols_count)
    else:
        plot_vars['agg_expression'] = plot_vars[cols_expression].agg(gene_expression_aggregator, axis=1)

    #percentage exceedance for the dot size, when size_value = 'percentage'
    plot_vars['percentage_exceedance'] = plot_vars['agg_count']*100/plot_vars['total_count']

    #percentage max expression for the dot color, when color_value = 'percentage_max_value'
    plot_vars['percentage_max_value'] = plot_vars['agg_expression']*100/np.max(plot_vars['agg_expression'])
    
    #calculate percentage expressed and percentage max expression observed for each gene (mode: planet, percentage)
    for gene in genes:
        plot_vars[gene+'_percentage_exceedance'] = plot_vars[gene+'_x']*100/plot_vars['total_count']
        plot_vars[gene+'_percentage_max_value'] = plot_vars[gene+'_y']*100/np.max(plot_vars[gene+'_y'])
    
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
                                          obs_columns: list = None,
                                          obs_thresholds: list = None,
                                          obs_aggregator_array: list = None,
                                          layer_value_aggregator : str = "mean",
                                          gene_count_aggregator : str = "median",
                                          gene_expression_aggregator : str = "median",
                                          **kwargs):
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
    #initialize use_raw=False
    defaultargs = {
        "use_raw":False
    }
    defaultargs.update(kwargs)
    
    if isinstance(genes, str):
        genes = [genes]

    if gene_count_aggregator == "expression_weighted_count" and gene_expression_aggregator == "count_weighted_expression":
        raise ValueError("'gene_count_aggregator' cannot be set to 'expression_weighted_count' when 'gene_expression_aggregator' is set to 'count_weighted_expression'")
        
    if len(obs_columns) != len(obs_thresholds):
        raise ValueError("obs_columns and obs_thresholds should have the same lengths")
    
    total_columns = [*genes, *obs_columns]
    total_thresholds = [*([expression_threshold]*len(genes)), *obs_thresholds]
    
    #get the genex values from the adata
    df_all = sc.get.obs_df(adata, [*genes, *obs_columns, x_col, y_col], gene_symbols = gene_symbols, layer = input_layer, use_raw = defaultargs['use_raw'])
    
    #get the count from the adata
    df_counts = sc.get.obs_df(adata, [x_col, y_col])
    df_counts = df_counts.groupby([x_col, y_col]).size().reset_index(name='total_count')
    
    #get the count of the obs exceeding the threshold per cluster for genes as well as other obs columns
    df_exceedance_counts = df_all.groupby([x_col, y_col]).apply(lambda x: pd.Series({total_columns[i]: count_greater_than_threshold(x[total_columns[i]], total_thresholds[i]) for i in range(len(total_columns))})).reset_index()
    
    
    #get aggregate values per cluster for genes as well as other obs columns
    if obs_aggregator_array is not None and len(obs_aggregator_array) != len(obs_columns):
        raise ValueError("obs_columns and obs_aggregator_array should have the same lengths or obs_aggregator_array should not be passed")
    elif obs_aggregator_array is None:
        df_aggregate_values = df_all.groupby([x_col, y_col]).agg(layer_value_aggregator).reset_index()
    else:
        full_aggregator_array = [*([layer_value_aggregator]*len(genes)), *obs_aggregator_array]
        df_aggregate_values = df_all.groupby([x_col, y_col]).agg({total_columns[i]: full_aggregator_array[i] for i in range(len(total_columns))}).reset_index()
    
    #merge the dataframes. Note that after merging, the 'genename_x' is the gene count , whereas 'genename_y' is the gene expression.
    plot_vars = pd.merge(df_counts, df_exceedance_counts, on=[x_col, y_col])
    plot_vars = pd.merge(plot_vars, df_aggregate_values, on=[x_col, y_col])
    
    #perform subsetting
    if x_col_subset is not None:
        plot_vars = plot_vars[plot_vars[x_col].isin(x_col_subset)].reset_index(drop=True)
        plot_vars[x_col] = plot_vars[x_col].cat.remove_unused_categories()
    if y_col_subset is not None:
        plot_vars = plot_vars[plot_vars[y_col].isin(y_col_subset)].reset_index(drop=True)
        plot_vars[y_col] = plot_vars[y_col].cat.remove_unused_categories()
    
    #calculate aggregates of all the given genes for the center dot
    cols_count = [col + '_x' for col in genes]
    cols_expression = [col + '_y' for col in genes]

    if gene_count_aggregator == "expression_weighted_count":
        plot_vars['agg_count'] = sum(plot_vars[col1] * plot_vars[col2] for col1, col2 in zip(cols_count, cols_expression))/sum(plot_vars[col2] for col2 in cols_expression)
    else:
        plot_vars['agg_count'] = plot_vars[cols_count].agg(gene_count_aggregator, axis=1)
    
    if gene_expression_aggregator == "count_weighted_expression":
        plot_vars['agg_expression'] = sum(plot_vars[col1] * plot_vars[col2] for col1, col2 in zip(cols_count, cols_expression))/sum(plot_vars[col1] for col1 in cols_count)
    else:
        plot_vars['agg_expression'] = plot_vars[cols_expression].agg(gene_expression_aggregator, axis=1)

    #percentage exceedance for the dot size, when size_value = 'percentage'
    plot_vars['percentage_exceedance'] = plot_vars['agg_count']*100/plot_vars['total_count']

    #percentage max expression for the dot color, when color_value = 'percentage_max_value'
    plot_vars['percentage_max_value'] = plot_vars['agg_expression']*100/np.max(plot_vars['agg_expression'])
    
    #calculate percentage expressed and percentage max expression observed for each gene (mode: planet, percentage)
    for column in total_columns:
        plot_vars[column+'_percentage_exceedance'] = plot_vars[column+'_x']*100/plot_vars['total_count']
        plot_vars[column+'_percentage_max_value'] = plot_vars[column+'_y']*100/np.max(plot_vars[column+'_y'])
    
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
                        mode : Literal["aggregate", "planet"] = "aggregate",
                        size_value : Literal["count", "percentage"] = "count",
                        color_value : Literal["value" , "percentage_max"] = "value",
                        use_log_scale : bool = False,
                        planet_columns: list = None,
                        color_schema: str = "Blues",
                        planet_color_schemas: list = None,
                        **kwargs):
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
    OUTER_SIZE_COUNT_COLUMN = kwargs.get('OUTER_SIZE_COUNT_COLUMN','total_count')
    INNER_SIZE_COUNT_COLUMN = kwargs.get('INNER_SIZE_COUNT_COLUMN','agg_count')
    INNER_SIZE_PERCENTAGE_COLUMN = kwargs.get('INNER_SIZE_PERCENTAGE_COLUMN','percentage_exceedance')
    DOT_COLOR_VALUE_COLUMN = kwargs.get('DOT_COLOR_VALUE_COLUMN','agg_expression')
    DOT_COLOR_PERCENTAGE_MAX_VALUE_COLUMN = kwargs.get('DOT_COLOR_PERCENTAGE_MAX_VALUE_COLUMN','percentage_max_value')
    PLANET_SIZE_COUNT_SUFFIX = kwargs.get('PLANET_SIZE_COUNT_SUFFIX','_x')
    PLANET_SIZE_PERCENTAGE_SUFFIX = kwargs.get('PLANET_SIZE_PERCENTAGE_SUFFIX','_percentage_exceedance')
    PLANET_COLOR_VALUE_SUFFIX = kwargs.get('PLANET_COLOR_VALUE_SUFFIX','_y')
    PLANET_COLOR_PERCENTAGE_MAX_VALUE_SUFFIX = kwargs.get('PLANET_COLOR_PERCENTAGE_MAX_VALUE_SUFFIX','_percentage_max_value')
        
    #Init Params:
    dpi = kwargs.get('DPI',100)
    ipd = 1/dpi
    MAX_DOT_SIZE = kwargs.get('MAX_DOT_SIZE',300)
    MAX_DOT_RADIUS = np.sqrt(MAX_DOT_SIZE/np.pi)
    MIN_DOT_SIZE = kwargs.get('MIN_DOT_SIZE',10)
    MIN_DOT_RADIUS = np.sqrt(MIN_DOT_SIZE/np.pi)
    LINE_WIDTH = kwargs.get('LINE_WIDTH',1)
    FIG_SIZE_SCALER = kwargs.get('FIG_SIZE_SCALER',2)
    PLANET_DIST_SCALER = kwargs.get('PLANET_DIST_SCALER',1.25)

    #legend dimensions in inches:
    LEGEND_COLOR_HEIGHT = kwargs.get('LEGEND_COLOR_HEIGHT',0.2)
    LEGEND_COLOR_WIDTH = kwargs.get('LEGEND_COLOR_WIDTH',2)
    LEGEND_DOT_HEIGHT = kwargs.get('LEGEND_DOT_HEIGHT',2.5)
    LEGEND_DOT_WIDTH = kwargs.get('LEGEND_DOT_WIDTH',2)
    LEGEND_PLANET_HEIGHT = kwargs.get('LEGEND_PLANET_HEIGHT',2)
    LEGEND_PLANET_WIDTH = kwargs.get('LEGEND_PLANET_WIDTH',2)
    
    #default legend labels    
    COLOR_BAR_TITLE = kwargs.get('COLOR_BAR_TITLE','Gene Expression')
    COLOR_BAR_TITLE_PMV = kwargs.get('COLOR_BAR_TITLE_PMV','Percentage of max expression\nobserved for the gene')
    SIZE_LEGEND_TITLE = kwargs.get('SIZE_LEGEND_TITLE','Cell Count')
    SIZE_LEGEND_TITLE_PERCENTAGE = kwargs.get('SIZE_LEGEND_TITLE_PERCENTAGE','Percentage cells expressed')
    OUTER_CIRCLE_LABEL = kwargs.get('OUTER_CIRCLE_LABEL','Total count')
    INNER_DOT_LABEL = kwargs.get('INNER_DOT_LABEL','Expressed count')
    ORIENTATION_LEGEND_TITLE = kwargs.get('ORIENTATION_LEGEND_TITLE','Genes')
    ORIENTATION_LEGEND_CENTER_LABEL = kwargs.get('ORIENTATION_LEGEND_CENTER_LABEL','Aggregate value')
        
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
        plot_vars['outer_area'] = calculate_dot_sizes(plot_vars[OUTER_SIZE_COUNT_COLUMN], max_count, MIN_DOT_SIZE, MAX_DOT_SIZE, use_log_scale)
        plot_vars['outer_radius'] = np.sqrt(plot_vars['outer_area']/np.pi)
        plot_vars['inner_area'] = calculate_dot_sizes(plot_vars[INNER_SIZE_COUNT_COLUMN], max_count, MIN_DOT_SIZE, MAX_DOT_SIZE, use_log_scale)
        plot_vars['inner_radius'] = np.sqrt(plot_vars['inner_area']/np.pi)
        if mode == 'planet':
            for planet_column in planet_columns:
                plot_vars[planet_column+'_dot_area'] = calculate_dot_sizes(plot_vars[planet_column+PLANET_SIZE_COUNT_SUFFIX], max_count, MIN_DOT_SIZE, MAX_DOT_SIZE, use_log_scale)
                plot_vars[planet_column+'_dot_radius'] = np.sqrt(plot_vars[planet_column+'_dot_area']/np.pi)
            
    if size_value == 'percentage':
        plot_vars['inner_area'] = calculate_dot_sizes(plot_vars[INNER_SIZE_PERCENTAGE_COLUMN], 100, MIN_DOT_SIZE, MAX_DOT_SIZE, use_log_scale)
        plot_vars['inner_radius'] = np.sqrt(plot_vars['inner_area']/np.pi)
        if mode == 'planet':
            for planet_column in planet_columns:
                plot_vars[planet_column+'_dot_area'] = calculate_dot_sizes(plot_vars[planet_column+PLANET_SIZE_PERCENTAGE_SUFFIX], 100, MIN_DOT_SIZE, MAX_DOT_SIZE, use_log_scale)
                plot_vars[planet_column+'_dot_radius'] = np.sqrt(plot_vars[planet_column+'_dot_area']/np.pi)
        
    #figure size calculation
    if mode == 'planet':
        w, h = calculate_figure_size_planet_mode(len(plot_vars['x_steps'].unique()), len(plot_vars['y_steps'].unique()), MAX_DOT_RADIUS, ipd, FIG_SIZE_SCALER)
    
    if mode == 'aggregate':
        w, h = calculate_figure_size_mean_mode(len(plot_vars['x_steps'].unique()), len(plot_vars['y_steps'].unique()), MAX_DOT_RADIUS, ipd, FIG_SIZE_SCALER)
                
    height_per_unit = 1/h
    width_per_unit = 1/w

    # print(f"Plot width: {w} inches")
    # print(f"Plot height: {h} inches")

    # # Plot main dots
    # fig, ax = plt.subplots(figsize=(w, h))

    # plt.xticks(ticks=range(len(x_categories)), labels=x_categories, rotation='vertical')
    # plt.yticks(ticks=range(len(y_categories)), labels=y_categories)

    # # Step 2: Render the plot
    # # plt.tight_layout()
    # fig.canvas.draw()

    # # Access the bounding box of the label, axes, etc.
    # x_label_bbox = ax.xaxis.label.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # y_label_bbox = ax.yaxis.label.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # # Get the bounding box of the axes plotting area in display coordinates
    # plotting_area_bbox = ax.bbox

    # # Display the bounding box coordinates in display units (pixels)
    # print("Plotting Area Bounding Box (display units):", plotting_area_bbox)

    # print("DPI:", dpi)

    # # Convert the display units to inches
    # plotting_area_bbox_inches = plotting_area_bbox.transformed(fig.dpi_scale_trans.inverted())

    # # Display the bounding box coordinates in inches
    # print("Plotting Area Bounding Box (inches):", plotting_area_bbox_inches.bounds) 

    # diff_w = w-plotting_area_bbox_inches.bounds[2]
    # diff_h = h-plotting_area_bbox_inches.bounds[3]

    # # Print the bounding box coordinates
    # print("X Label Bounding Box:", x_label_bbox.bounds)
    # print("Y Label Bounding Box:", y_label_bbox.bounds)
    # # print("Legend Bounding Box:", legend_bbox)

    # # Step 3: Retrieve the bounding box dimensions of the plot area
    # renderer = fig.canvas.get_renderer()
    # bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())

    # # Extract the dimensions
    # plot_width = bbox.width
    # plot_height = bbox.height

    # print(f"Plot area width: {plot_width} inches")
    # print(f"Plot area height: {plot_height} inches")

    # w = w+diff_w
    # h = h+diff_h

    # height_per_unit = 1/h
    # width_per_unit = 1/w

    # print(f"Plot width: {w} inches")
    # print(f"Plot height: {h} inches")

    x_inches_per_step = len(plot_vars['x_steps'].unique())/w
    y_inches_per_step = len(plot_vars['y_steps'].unique())/h

    # Plot main dots
    fig, ax = plt.subplots(figsize=(w, h))

    plt.xticks(ticks=range(len(x_categories)), labels=x_categories, rotation='vertical')
    plt.yticks(ticks=range(len(y_categories)), labels=y_categories)



    #size value condition
    if size_value == 'count':
        ax.scatter(plot_vars['x_steps'], plot_vars['y_steps'], s=plot_vars['outer_area'], facecolors='none', edgecolors='0',linewidths=LINE_WIDTH)
    if color_value == 'value':
        sc = ax.scatter(plot_vars['x_steps'], plot_vars['y_steps'], s=plot_vars['inner_area'], c=plot_vars[DOT_COLOR_VALUE_COLUMN], cmap=color_schema, vmin=vmin, vmax=vmax)
    if color_value == 'percentage_max':
        sc = ax.scatter(plot_vars['x_steps'], plot_vars['y_steps'], s=plot_vars['inner_area'], c=plot_vars[DOT_COLOR_PERCENTAGE_MAX_VALUE_COLUMN], cmap=color_schema, vmin=0, vmax=100)

    if mode == 'planet':
        sc_array = {}
        for i, row in plot_vars.iterrows():
            p=0
            outer_radius = row['outer_radius']
            for planet_column in planet_columns:
                #test if planet capacity is reached
                if p>5: continue
                radius = row[planet_column+'_dot_radius']
                distance_from_center = PLANET_DIST_SCALER*(LINE_WIDTH+outer_radius+radius)*ipd
                offset_x = distance_from_center*planet_x[p]/x_inches_per_step
                offset_y = distance_from_center*planet_y[p]/y_inches_per_step
                secondary_x = row['x_steps'] + offset_x
                secondary_y = row['y_steps'] + offset_y
                if planet_color_schemas is not None:
                    if color_value == 'value':
                        sc_array[p] = ax.scatter(secondary_x, secondary_y, s=row[planet_column+'_dot_area'], c=row[planet_column+PLANET_COLOR_VALUE_SUFFIX],cmap=planet_color_schemas[p], vmin=vmin_array[p], vmax=vmin_array[p])
                    if color_value == 'percentage_max':
                        sc_array[p] = ax.scatter(secondary_x, secondary_y, s=row[planet_column+'_dot_area'], c=row[planet_column+PLANET_COLOR_PERCENTAGE_MAX_VALUE_SUFFIX],cmap=planet_color_schemas[p], vmin=0, vmax=100)
                else:
                    if color_value == 'value':
                        ax.scatter(secondary_x, secondary_y, s=row[planet_column+'_dot_area'], c=row[planet_column+PLANET_COLOR_VALUE_SUFFIX],cmap=color_schema, vmin=vmin, vmax=vmax)
                    if color_value == 'percentage_max':
                        ax.scatter(secondary_x, secondary_y, s=row[planet_column+'_dot_area'], c=row[planet_column+PLANET_COLOR_PERCENTAGE_MAX_VALUE_SUFFIX],cmap=color_schema, vmin=0, vmax=100)
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
    if color_value == 'value':
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
            if color_value == 'value':
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
    if mode == 'aggregate':
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
            dot_size_ax.scatter(0, i, s= sizes[i], c=legend_hex_color, alpha=1)
        if size_value == 'count':
            dot_size_ax.scatter(-0.25, i, s= sizes[i], c=legend_hex_color, alpha=1)
            dot_size_ax.scatter(-0.25, i, s=sizes[i], facecolors='none', edgecolors='0',linewidths=LINE_WIDTH)
            if i == len(sizes)-1:
                dot_size_ax.annotate(OUTER_CIRCLE_LABEL, 
                            xy=(-0.25 + ipd*np.sqrt(sizes[i]/np.pi)/LEGEND_DOT_WIDTH, i), xycoords='data',
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
        legend_circle_size = (MIN_DOT_SIZE+MAX_DOT_SIZE)/2
        legend_circle_radius = np.sqrt(legend_circle_size/np.pi)
        legend_dot_size = 0.75*legend_circle_size
        legend_dot_radius = np.sqrt(legend_dot_size/np.pi)

        #size value condition
        if size_value == 'count':
            dot_orientation_ax.scatter(0, 0, s=legend_circle_size, facecolors='none', edgecolors='0',linewidths=LINE_WIDTH, alpha=1)

        dot_orientation_ax.scatter(0, 0, s=legend_dot_size, c=legend_hex_color, alpha=1)
        dot_orientation_ax.annotate(ORIENTATION_LEGEND_CENTER_LABEL, 
                            xy=(0, 0), xycoords='data',
                            xytext=(-0.15*PLANET_DIST_SCALER, -0.15*PLANET_DIST_SCALER), textcoords='data',
                            arrowprops=dict(arrowstyle="-", color="black", lw=LINE_WIDTH),
                            fontsize=8, va = 'center')
        # dot_orientation_ax.text(0,0, ORIENTATION_LEGEND_CENTER_LABEL, fontsize=8, ha='center', va='center')

        p=0
        for planet_column in planet_columns:
            #test if planet capacity is reached
            if p>5: continue
            distance_from_center = PLANET_DIST_SCALER*(LINE_WIDTH+legend_circle_radius+legend_dot_radius)*ipd/LEGEND_PLANET_WIDTH
            offset_x = distance_from_center*planet_x[p]
            offset_y = distance_from_center*planet_y[p]
            secondary_x = offset_x
            secondary_y = offset_y
            if planet_color_schemas is not None:
                dot_orientation_ax.scatter(secondary_x, secondary_y, s=legend_dot_size, c=planet_legend_hex_color[p], alpha=1)
            else:
                dot_orientation_ax.scatter(secondary_x, secondary_y, s=legend_dot_size, c=legend_hex_color, alpha=1)
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


    
    
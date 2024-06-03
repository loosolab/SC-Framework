"""Funtions for high dimensional bubble plot a.k.a planet plot."""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from sctoolbox.plotting.general import _save_figure
from beartype import beartype
from beartype.typing import Literal


#############################################################################
#                                  Utilities                                #
#############################################################################

@beartype
def count_greater_than_threshold(group: pd.DataFrame | pd.Series,
                                 threshold: int | float) -> pd.Series | np.int64:
    """
    Calculates count of values exceeding a given threshold value.
    
    Parameters
    ----------
    group : pd.DataFrame | pd.Series
        A dataframe or a series to apply the threshold condition over.
    threshold : int | float
        The threshold value to be used.
        
    Returns
    -------
    threshold_exceedence_count : pd.Series | np.int64
        Returns pd.Series type output when input for group is a pd.DataFrame type and a np.int64 when input for group is a pd.Series type.
    
    """
    threshold_exceedence_count = (group > threshold).sum()
    return threshold_exceedence_count

@beartype
def calculate_dot_sizes(values: pd.Series | np.ndarray,
                        max_value: int | float,
                        min_dot_size: int,
                        max_dot_size: int,
                        use_log_scale: bool = False) -> pd.Series | np.ndarray:
    """
    Calculates the sizes of dots for plotting.
    
    Parameters
    ----------
    values : pd.Series | np.ndarray
        A series or an array containing the values to be plotted.
    max_value : int | float
        The biggest value observed to correspond to the max_dot_size.
    min_dot_size : int
        A value for the minimum size of the dot/circle.
    max_dot_size : int
        A value for the maximum size of the dot/circle.
    use_log_scale : bool, default False
        Set True to use a logarithmic instead of linear scale to calculate the dot sizes.
        
    Returns
    -------
    sizes : pd.Series | np.ndarray
        Returns a series or an array containing the sizes for the dots depending upon the input type for values.
    
    """
    if use_log_scale:
        values = np.log1p(values)
        max_value = np.log1p(max_value)
    fraction_count = values/max_value
    sizes = fraction_count * (max_dot_size - min_dot_size) + min_dot_size
    return sizes


#############################################################################
#                             Pre-processing                                #
#############################################################################

@beartype
def planet_plot_anndata_preprocess(adata: sc.AnnData,
                                    x_col: str,
                                    y_col: str,
                                    genes: list | str,
                                    gene_symbols: str | None = None,
                                    x_col_subset: list | None = None,
                                    y_col_subset: list | None = None,
                                    input_layer: str | None = None,
                                    fillna: int | float = 0.0,
                                    expression_threshold: int | float = 0.0,
                                    layer_value_aggregator : str = "mean",
                                    gene_count_aggregator : str = "median",
                                    gene_expression_aggregator : str = "median",
                                    **kwargs) -> pd.DataFrame:
    """
    Preprocesses the annData object for the planet plot.
    
    Parameters
    ----------
    adata : sc.AnnData
        The input AnnData object.
    x_col : str
        Name of the obs column of the AnnData to be shown on the x-axis of the plot.
    y_col : str
        Name of the obs column of the AnnData to be shown on the y-axis of the plot.
    genes : list | str
        List of genes to be to consider for plotting and aggregation.
    gene_symbols : str | None, default None
        Name of the var column of the AnnData object used for the gene names. If set None, the index of AnnData.var is used.
    x_col_subset: list | None, default None
        To plot a specific subset of the entries in x_col instead of all the entries.
    y_col_subset: list | None, default None
        To plot a specific subset of the entries in y_col instead of all the entries.
    input_layer: str | None, default None
        Layer of the AnnData object to be considered for the gene expression values.
    fillna: int | float, default 0.0
        Value to fill up the NaN values that are created during aggregation.
    expression_threshold: int | float, default 0.0
        The threshold value to calculate the threshold exceedence count. This count is then used to calculate the size of the dots.
    layer_value_aggregator : str, default  "mean"
        A standard numpy aggregator (eg. 'mean', 'median', etc.) to aggregate the values in the input_layer for the corresponding gene.
    gene_count_aggregator : str, default "median"
        A standard numpy aggregator (eg. 'mean', 'median', etc.) to aggregate the values for multiple genes into a single value which is used to calculate the size of the center dot.
        Apart from the standard numpy aggregators there is 'expression_weighted_count' option to calculate a expression weighted mean of the counts.
        eg. id c1, c2, e1 and e2 are the counts and expressions of 2 genes, then expression_weighted_count = (c1*e1+c2*e2)/(e1+e2).
        This is used to reduce the disparity between the size and the color of the center dots.
    gene_expression_aggregator : str, default "median"
        A standard numpy aggregator (eg. 'mean', 'median', etc.) to aggregate the values for multiple genes into a single value which is used to calculate the color of the center dot.
        Apart from the standard numpy aggregators there is 'count_weighted_expression' option to calculate a count weighted mean of the expressions.
        eg. id c1, c2, e1 and e2 are the counts and expressions of 2 genes, then count_weighted_expression = (e1*c1+e2*c2)/(c1+c2).
        This is used to reduce the disparity between the size and the color of the center dots. 
        Note that 'count_weighted_expression' cannot be used as gene_expression_aggregator when 'expression_weighted_count' is used as gene_count_aggregator!
    **kwargs : Any
        Additional keyword arguments are passed to :func:`scanpy.get.obs_df`.
        
    Returns
    -------
    plot_vars : pd.DataFrame
        A dataframe containing all the parameters to be plotted.

    Raises
    ------
    KeyError (?)
        If the given method is not found in adata.obsm.
    ValueError (?)
        If the 'components' given is larger than the number of components in the embedding.
    
    """
    #initialize use_raw=False
    defaultargs = {
        "use_raw":False
    }
    defaultargs.update(kwargs)
    
    if isinstance(genes, str):  # convert to array if string
        genes = [genes]
    # check conflicting condition
    if gene_count_aggregator == "expression_weighted_count" and gene_expression_aggregator == "count_weighted_expression":
        raise ValueError("'gene_count_aggregator' cannot be set to 'expression_weighted_count' when 'gene_expression_aggregator' is set to 'count_weighted_expression'")
    
    #get the genex values and obs values from the adata
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

    #percentage exceedance for the center dot size, when size_value = 'percentage'
    plot_vars['percentage_exceedance'] = plot_vars['agg_count']*100/plot_vars['total_count']
    #percentage max value for the center dot color, when color_value = 'percentage_max'
    plot_vars['percentage_max_value'] = plot_vars['agg_expression']*100/np.max(plot_vars['agg_expression'])
    #calculate percentage exceedance and percentage max value observed for each gene
    for gene in genes:
        plot_vars[gene+'_percentage_exceedance'] = plot_vars[gene+'_x']*100/plot_vars['total_count']
        plot_vars[gene+'_percentage_max_value'] = plot_vars[gene+'_y']*100/np.max(plot_vars[gene+'_y'])
    
    # List of columns to exclude from conversion
    exclude_columns = [x_col, y_col]
    # Convert all other columns to float and fillna
    for col in plot_vars.columns:
        if col not in exclude_columns:
            plot_vars[col] = pd.to_numeric(plot_vars[col], errors='coerce').astype(float)
            plot_vars[col] = plot_vars[col].fillna(fillna)

    return plot_vars
    

@beartype    
def planet_plot_anndata_preprocess_advanced(adata: sc.AnnData,
                                            x_col: str,
                                            y_col: str,
                                            genes: list | str,
                                            gene_symbols: str | None = None,
                                            x_col_subset: list | None = None,
                                            y_col_subset: list | None = None,
                                            input_layer: str | None = None,
                                            fillna: int | float = 0.0,
                                            expression_threshold: int | float = 0.0,
                                            obs_columns: list | None = None,
                                            obs_thresholds: list | None = None,
                                            obs_aggregator_array: list | None = None,
                                            layer_value_aggregator : str = "mean",
                                            gene_count_aggregator : str = "median",
                                            gene_expression_aggregator : str = "median",
                                            **kwargs) -> pd.DataFrame:
    """
    This function has additional functionality to use columns other than genes for the dots.
    
    Parameters
    ----------
    adata : sc.AnnData
        The input AnnData object.
    x_col : str
        Name of the obs column of the AnnData to be shown on the x-axis of the plot.
    y_col : str
        Name of the obs column of the AnnData to be shown on the y-axis of the plot.
    genes : list | str
        List of genes to be to consider for plotting and aggregation.
    gene_symbols : str | None, default None
        Name of the var column of the AnnData object used for the gene names. If set None, the index of AnnData.var is used.
    x_col_subset: list | None, default None
        To plot a specific subset of the entries in x_col instead of all the entries.
    y_col_subset: list | None, default None
        To plot a specific subset of the entries in y_col instead of all the entries.
    input_layer: str | None, default None
        Layer of the AnnData object to be considered for the gene expression values.
    fillna: int | float, default 0.0
        Value to fill up the NaN values that are created during aggregation.
    expression_threshold: int | float, default 0.0
        The threshold value to calculate the threshold exceedence count. This count is then used to calculate the size of the dots.
    obs_columns: list | None, default None
        The obs columns of the AnnData object to be additionaly considered for aggregation.
        Note that these values must have numeric values!
    obs_thresholds: list | None, default None
        A corresponding list of threshold values for the obs_columns.
        Make sure that len(obs_thresholds) == len(obs_columns).
        If this argument is not passed then expression_threshold is used instead.
    obs_aggregator_array: list | None, default None
        A list of standard numpy aggregators (eg. 'mean', 'median', etc.) to aggregate the values mentioned in the obs_columns.
        Make sure that len(obs_aggregator_array) == len(obs_columns).
        If this argument is not passed then layer_value_aggregator is used instead.
    layer_value_aggregator : str, default  "mean"
        A standard numpy aggregator (eg. 'mean', 'median', etc.) to aggregate the values in the input_layer for the corresponding gene.
    gene_count_aggregator : str, default "median"
        A standard numpy aggregator (eg. 'mean', 'median', etc.) to aggregate the values for multiple genes into a single value which is used to calculate the size of the center dot.
        Apart from the standard numpy aggregators there is 'expression_weighted_count' option to calculate a expression weighted mean of the counts.
        eg. id c1, c2, e1 and e2 are the counts and expressions of 2 genes, then expression_weighted_count = (c1*e1+c2*e2)/(e1+e2).
        This is used to reduce the disparity between the size and the color of the center dots.
    gene_expression_aggregator : str, default "median"
        A standard numpy aggregator (eg. 'mean', 'median', etc.) to aggregate the values for multiple genes into a single value which is used to calculate the color of the center dot.
        Apart from the standard numpy aggregators there is 'count_weighted_expression' option to calculate a count weighted mean of the expressions.
        eg. id c1, c2, e1 and e2 are the counts and expressions of 2 genes, then count_weighted_expression = (e1*c1+e2*c2)/(c1+c2).
        This is used to reduce the disparity between the size and the color of the center dots. 
        Note that 'count_weighted_expression' cannot be used as gene_expression_aggregator when 'expression_weighted_count' is used as gene_count_aggregator!
    **kwargs : Any
        Additional keyword arguments are passed to :func:`scanpy.get.obs_df`.

    Returns
    -------
    plot_vars : pd.DataFrame
        A dataframe containing all the parameters to be plotted.

    Raises
    ------
    KeyError (?)
        If the given method is not found in adata.obsm.
    ValueError (?)
        If the 'components' given is larger than the number of components in the embedding.
    
    """
    #initialize use_raw=False
    defaultargs = {
        "use_raw":False
    }
    defaultargs.update(kwargs)
    
    if isinstance(genes, str):  # convert to array if string
        genes = [genes]
    #check conflicting conditions
    if gene_count_aggregator == "expression_weighted_count" and gene_expression_aggregator == "count_weighted_expression":
        raise ValueError("'gene_count_aggregator' cannot be set to 'expression_weighted_count' when 'gene_expression_aggregator' is set to 'count_weighted_expression'")    
    if obs_thresholds is not None and len(obs_columns) != len(obs_thresholds):
        raise ValueError("obs_columns and obs_thresholds should have the same lengths")
    
    total_columns = [*genes, *obs_columns]
    if obs_thresholds is not None:
        total_thresholds = [*([expression_threshold]*len(genes)), *obs_thresholds]
    else:
        total_thresholds = [*([expression_threshold]*(len(genes)+len(obs_columns)))]
    #get the genex values and obs values from the adata
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

    if gene_count_aggregator == "count_weighted_expression":
        plot_vars['agg_expression'] = sum(plot_vars[col1] * plot_vars[col2] for col1, col2 in zip(cols_count, cols_expression))/sum(plot_vars[col1] for col1 in cols_count)
    else:
        plot_vars['agg_expression'] = plot_vars[cols_expression].agg(gene_expression_aggregator, axis=1)

    #percentage exceedance for the center dot size, when size_value = 'percentage'
    plot_vars['percentage_exceedance'] = plot_vars['agg_count']*100/plot_vars['total_count']
    #percentage max value for the center dot color, when color_value = 'percentage_max'
    plot_vars['percentage_max_value'] = plot_vars['agg_expression']*100/np.max(plot_vars['agg_expression'])
    #calculate percentage exceedance and percentage max value observed for each obs column and gene
    for column in total_columns:
        plot_vars[column+'_percentage_exceedance'] = plot_vars[column+'_x']*100/plot_vars['total_count']
        plot_vars[column+'_percentage_max_value'] = plot_vars[column+'_y']*100/np.max(plot_vars[column+'_y'])
    # List of columns to exclude from conversion
    exclude_columns = [x_col, y_col]
    # Convert all other columns to float and fillna
    for col in plot_vars.columns:
        if col not in exclude_columns:
            plot_vars[col] = pd.to_numeric(plot_vars[col], errors='coerce').astype(float)
            plot_vars[col] = plot_vars[col].fillna(fillna)

    return plot_vars


#############################################################################
#                                Plotting                                   #
#############################################################################

@beartype    
def planet_plot_render(plot_vars: pd.DataFrame,
                        x_col: str,
                        y_col: str,
                        x_label: str | None = None,
                        y_label: str | None = None,
                        mode : Literal["aggregate", "planet"] = "aggregate",
                        size_value : Literal["count", "percentage"] = "count",
                        color_value : Literal["value" , "percentage_max"] = "value",
                        use_log_scale : bool = False,
                        planet_columns: list | None = None,
                        color_schema: str = "Blues",
                        planet_color_schemas: list | None = None,
                        FIG_SIZE_SCALER: int | float = 2,
                        PLANET_DIST_SCALER: int | float = 1.25,
                        MAX_DOT_SIZE: int = 300,
                        MIN_DOT_SIZE: int = 10,
                        LINE_WIDTH: int = 1,
                        OUTER_SIZE_COUNT_COLUMN: str = 'total_count',
                        INNER_SIZE_COUNT_COLUMN: str = 'agg_count',
                        INNER_SIZE_PERCENTAGE_COLUMN: str = 'percentage_exceedance',
                        DOT_COLOR_VALUE_COLUMN: str = 'agg_expression',
                        DOT_COLOR_PERCENTAGE_MAX_VALUE_COLUMN: str = 'percentage_max_value',
                        PLANET_SIZE_COUNT_SUFFIX: str = '_x',
                        PLANET_SIZE_PERCENTAGE_SUFFIX: str = '_percentage_exceedance',
                        PLANET_COLOR_VALUE_SUFFIX: str = '_y',
                        PLANET_COLOR_PERCENTAGE_MAX_VALUE_SUFFIX: str = '_percentage_max_value',
                        LEGEND_COLOR_HEIGHT: int | float = 0.15,
                        LEGEND_COLOR_WIDTH: int | float = 2,
                        LEGEND_DOT_HEIGHT: int | float = 2,
                        LEGEND_DOT_WIDTH: int | float = 2,
                        LEGEND_PLANET_HEIGHT: int | float = 2,
                        LEGEND_PLANET_WIDTH: int | float = 2,
                        LEGEND_COLOR_X_ALIGNMENT: int | float = 0,
                        LEGEND__DOT_X_ALIGNMENT: int | float = 0,
                        LEGEND_PLANET_X_ALIGNMENT: int | float = 0,
                        LEGEND_COLOR_Y_ALIGNMENT: int | float = 0,
                        LEGEND_DOT_Y_ALIGNMENT: int | float = 0,
                        LEGEND_PLANET_Y_ALIGNMENT: int | float = 0,
                        LEGEND_FONT_SIZE: int = 8,
                        COLOR_BAR_TITLE: str = 'Gene Expression',
                        COLOR_BAR_TITLE_PMV: str = 'Percentage of max expression\nobserved for the gene',
                        colorbar_label_array: list | None = None,
                        SIZE_LEGEND_TITLE: str = 'Cell Count',
                        SIZE_LEGEND_TITLE_PERCENTAGE: str = 'Percentage cells expressed',
                        OUTER_CIRCLE_LABEL: str = 'Total count',
                        INNER_DOT_LABEL: str = 'Expressed count',
                        ORIENTATION_LEGEND_TITLE: str = 'Genes',
                        ORIENTATION_LEGEND_CENTER_LABEL: str = 'Aggregate value',
                        orientation_labels_array: list | None = None,
                        save: str | None = None,
                        **kwargs):
    """
    Renders the planet plot on the basis of the preprocessed data.
    
    Parameters
    ----------
    plot_vars : pd.DataFrame
        A pandas.DataFrame object containing all the required columns for plotting.
    x_col : str
        Name of the column representing the x-axis of the plot.
    y_col : str
        Name of the column representing the y-axis of the plot.
    x_label : str | None, default None
        Label for the x-axis of the plot.
    y_label : str | None, default None
        Label fot the y-axis of the plot.
    mode : Literal["aggregate", "planet"], default "aggregate"
        There are two modes:
        1)"aggregate" mode to show the aggregate of values per cluster as the center dot. 
        2)"planet" mode to show gene specific or other additional values as planets surrounding the aggregate dot. Upto 6 planets are supported. 
    size_value : Literal["count", "percentage"], default "count"
        There are two settings:
        1)"count": here the exact count of the cells exceeding the given threshold is represented as the size of the dots (applicable for both aggregate as well as planets). Additionally circle is around the center dot represents the total number of cells present in the cluster.
        2)"percentage": here the percentage of cells per cluster exceeding the given threshold is represented as the size of the dots (applicable for both aggregate as well as planets).
    color_value : Literal["value" , "percentage_max"], default "value"
        There are two settings:
        1)"value": here the exact value of the gene expression or other values is represented as the color of the dots.
        2)"percentage_max": here the percentage value of the maximum value observed per gene (when planets correspond to genes) or max value observed for that column in case of other aggregate values.
    use_log_scale : bool, default False
        Set the value to true in order to use the log scale to represent sizes.
    planet_columns: list | None, default None
        Give a list of up to 6 genes(or columns) to be represented as planets.
    color_schema: str, default "Blues"
        The color schema for the color of the dots.
    planet_color_schemas: list | None, default None
        Color schemas for individual planets, use this parameter when entities other than genes are represented as planets. Make sure that len(planet_color_schemas) == len(planet_columns).
        Multiple color bars are displayed for the legend in this case.
    FIG_SIZE_SCALER: int | float, default 2
        A factor used to scale the size of the figure. Use this to fine tune and adjust the size of the figure.
    PLANET_DIST_SCALER: int | float, default 1.25
        A factor used to scale the distance between the center and planets. Use this to fine tune and adjust the figure.
    MAX_DOT_SIZE: int, default 300
        Maximum size of the dot or circle (when size_value == "count").
    MIN_DOT_SIZE: int, default 10
        Minimum size of the dot or circle (when size_value == "count").
    LINE_WIDTH: int, default 1
        Width of the line drawing the circle (when size_value == "count").
    OUTER_SIZE_COUNT_COLUMN: str, default 'total_count'
        Name of the df column to be used for the size of the outer circle (when size_value == "count").
    INNER_SIZE_COUNT_COLUMN: str, default 'agg_count'
        Name of the df column to be used for the size of the center dot (when size_value == "count").
    INNER_SIZE_PERCENTAGE_COLUMN: str, default 'percentage_exceedance'
        Name of the df column to be used for the size of the center dot (when size_value == "percentage").
    DOT_COLOR_VALUE_COLUMN: str, default 'agg_expression'
        Name of the df column to be used for the color of the center dot (when color_value == "value").
    DOT_COLOR_PERCENTAGE_MAX_VALUE_COLUMN: str, default 'percentage_max_value'
        Name of the df column to be used for the color of the center dot (when color_value == "percentage_max").
    PLANET_SIZE_COUNT_SUFFIX: str, default '_x'
        Name of the suffix to be to be added to the column names for the planets mentioned in the planet_columns that are used for the size of the planets (when size_value == "count").
        e.g. in the default case "FLT1_x" would be used as the size column for the gene "FLT1".
    PLANET_SIZE_PERCENTAGE_SUFFIX: str, default '_percentage_exceedance'
        Name of the suffix to be to be added to the column names for the planets mentioned in the planet_columns that are used for the size of the planets (when size_value == "percentage").
        e.g. in the default case "FLT1_percentage_exceedance" would be used as the size column for the gene "FLT1".
    PLANET_COLOR_VALUE_SUFFIX: str, default '_y'
        Name of the suffix to be to be added to the column names for the planets mentioned in the planet_columns that are used for the color of the planets (when color_value == "value").
        e.g. in the default case "FLT1_y" would be used as the color column for the gene "FLT1".
    PLANET_COLOR_PERCENTAGE_MAX_VALUE_SUFFIX: str, default '_percentage_max_value'
        Name of the suffix to be to be added to the column names for the planets mentioned in the planet_columns that are used for the color of the planets (when color_value == "percentage_max").
        e.g. in the default case "FLT1_percentage_max_value" would be used as the color column for the gene "FLT1".
    LEGEND_COLOR_HEIGHT: int | float, default 0.15
        Height in inches for the color bar(s) legend.
    LEGEND_COLOR_WIDTH: int | float, default 2
        Width in inches for the color bar(s) legend.
    LEGEND_DOT_HEIGHT: int | float, default 2
        Height in inches for the dot size legend.
    LEGEND_DOT_WIDTH: int | float, default 2
        Width in inches for the dot size legend.
    LEGEND_PLANET_HEIGHT: int | float, default 2
        Height in inches for the planet legend.
    LEGEND_PLANET_WIDTH: int | float, default 2
        Width in inches for the planet legend.
    LEGEND_COLOR_X_ALIGNMENT: int | float, default 0
        A scalar to adjust the x alignment of the color bar(s) legend.
    LEGEND__DOT_X_ALIGNMENT: int | float, default 0
        A scalar to adjust the x alignment of the dot size legend.
    LEGEND_PLANET_X_ALIGNMENT: int | float, default 0
        A scalar to adjust the x alignment of the planet legend.
    LEGEND_COLOR_Y_ALIGNMENT: int | float, default 0
        A scalar to adjust the y alignment of the color bar(s) legend.
    LEGEND_DOT_Y_ALIGNMENT: int | float, default 0
        A scalar to adjust the x alignment of the dot size legend.
    LEGEND_PLANET_Y_ALIGNMENT: int | float, default 0
        A scalar to adjust the x alignment of the planet legend.
    LEGEND_FONT_SIZE: int, default 8
        Font size for text inside the legends.
    COLOR_BAR_TITLE: str, default 'Gene Expression'
        Title of the color bar legend (when color_value == "value"). Note that the planet names in planet_columns are used as the titles for the extra legends by default (when planet_color_schemas != None).
    COLOR_BAR_TITLE_PMV: str, default 'Percentage of max expression\nobserved for the gene'
        Title of the color bar legend (when color_value == "percentage_max").
    colorbar_label_array: list | None, default = None
        Give a list of the cbar titles for the planets (when planet_color_schemas != None).
    SIZE_LEGEND_TITLE: str, default 'Cell Count'
        Title for the size legend (when size_value == "count").
    SIZE_LEGEND_TITLE_PERCENTAGE: str, default 'Percentage cells expressed'
        Title for the size legend (when size_value == "percentage").
    OUTER_CIRCLE_LABEL: str, default 'Total count'
        Annotation for the circle in the size legend (when size_value == "count").
    INNER_DOT_LABEL: str, default 'Expressed count'
        Annotation for the dot in the size legend (when size_value == "count").
    ORIENTATION_LEGEND_TITLE: str, default 'Genes'
        Title for the planet legend.
    ORIENTATION_LEGEND_CENTER_LABEL: str, default 'Aggregate value'
        Annotation for the centre dot in the planet legend. Note that the planets are by default annotated as per the names present in planet_columns.
    orientation_labels_array: list | None, default None
        Give a list of annotations for individual planets.
    save : Optional[str], default None
        Filename to save the figure.
        
    Returns
    -------
    fig : matplotlib.figure (?)
        Array of axis objects

    Raises
    ------
    KeyError (?)
        If the given method is not found in adata.obsm.
    ValueError (?)
        If the 'components' given is larger than the number of components in the embedding.

    Examples
    --------
    .. plot::
        :context: close-figs

        plot_vars = pl.planet_plot_anndata_preprocess(adata, x_col="bulk_labels", y_col="phase", genes=["SSU72","S100B","ITGB2"])
        pl.planet_plot_render(plot_vars, x_col="bulk_labels", y_col="phase", mode="planet")
    
    """
     
    # ---- Initialization parameters ---- #
    dpi = kwargs.get('dpi',100)     # Dots per inches
    ipd = 1/dpi
    MAX_DOT_RADIUS = np.sqrt(MAX_DOT_SIZE/np.pi)
    #planet config
    planet_x = [0,-0.866,-0.866,0,0.866,0.866]      # For x orientation
    planet_y = [1,0.5,-0.5,-1,-0.5,0.5]     # For y orientation
    planet_text = [{'va': 'bottom'},{'ha': 'right', 'va': 'bottom'},{'ha': 'right', 'va': 'top'},{'va': 'top'},{'ha': 'left', 'va': 'top'},{'ha': 'left', 'va': 'bottom'}]      # For annotation orientation 

    if isinstance(planet_columns, str):     # convert to array if string
        planet_columns = [planet_columns]
    if len(planet_columns)>6:
          raise ValueError("planet_columns should contain 6 elements at max")
    if planet_color_schemas is not None:
        if len(planet_color_schemas) != len(planet_columns):
            raise ValueError("planet_columns and planet_color_schemas should have the same lengths")
    if colorbar_label_array is not None:
        if len(colorbar_label_array) != len(planet_columns):
            raise ValueError("planet_columns and colorbar_label_array should have the same lengths")
    if orientation_labels_array is not None:
        if len(orientation_labels_array) != len(planet_columns):
            raise ValueError("planet_columns and orientation_labels_array should have the same lengths")
    
    #max cell count
    max_count = np.max(plot_vars[OUTER_SIZE_COUNT_COLUMN])
    min_count = 0
    
    #calculating vmin and vmax
    vmin = np.min(plot_vars[DOT_COLOR_VALUE_COLUMN])
    vmax = np.max(plot_vars[DOT_COLOR_VALUE_COLUMN])
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
    
    # ---- Dot size calculation ---- #
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
    if mode == 'planet':    # 3 times the size is used for planet mode.
        w = len(plot_vars['x_steps'].unique())*(6*MAX_DOT_RADIUS)*ipd*FIG_SIZE_SCALER
        h = len(plot_vars['y_steps'].unique())*(6*MAX_DOT_RADIUS)*ipd*FIG_SIZE_SCALER
    if mode == 'aggregate':
        w = len(plot_vars['x_steps'].unique())*(2*MAX_DOT_RADIUS)*ipd*FIG_SIZE_SCALER
        h = len(plot_vars['y_steps'].unique())*(2*MAX_DOT_RADIUS)*ipd*FIG_SIZE_SCALER
    height_per_unit = 1/h
    width_per_unit = 1/w
    x_inches_per_step = len(plot_vars['x_steps'].unique())/w
    y_inches_per_step = len(plot_vars['y_steps'].unique())/h

    # Plot
    fig, ax = plt.subplots(figsize=(w, h))
    plt.xticks(ticks=range(len(x_categories)), labels=x_categories, rotation='vertical')
    plt.yticks(ticks=range(len(y_categories)), labels=y_categories)

    # ---- Center dot and circle scatter ---- #
    if size_value == 'count':   # draw circle
        ax.scatter(plot_vars['x_steps'], plot_vars['y_steps'], s=plot_vars['outer_area'], facecolors='none', edgecolors='0',linewidths=LINE_WIDTH)
    if color_value == 'value':     # draw center dot
        sc = ax.scatter(plot_vars['x_steps'], plot_vars['y_steps'], s=plot_vars['inner_area'], c=plot_vars[DOT_COLOR_VALUE_COLUMN], cmap=color_schema, vmin=vmin, vmax=vmax)
    if color_value == 'percentage_max':     # draw center dot
        sc = ax.scatter(plot_vars['x_steps'], plot_vars['y_steps'], s=plot_vars['inner_area'], c=plot_vars[DOT_COLOR_PERCENTAGE_MAX_VALUE_COLUMN], cmap=color_schema, vmin=0, vmax=100)

    # ---- Planet scatter ---- #
    if mode == 'planet':
        sc_array = {}
        for i, row in plot_vars.iterrows():
            p=0     #initialize planet count
            if size_value == 'count':
                outer_radius = row['outer_radius']
            else:
                outer_radius = MAX_DOT_RADIUS
            for planet_column in planet_columns:
                radius = row[planet_column+'_dot_radius']
                distance_from_center = PLANET_DIST_SCALER*(LINE_WIDTH+outer_radius+radius)*ipd      # Formula for planet distance from center
                offset_x = distance_from_center*planet_x[p]/x_inches_per_step       # divide to compensate for irregular figure sizes
                offset_y = distance_from_center*planet_y[p]/y_inches_per_step
                secondary_x = row['x_steps'] + offset_x
                secondary_y = row['y_steps'] + offset_y
                if planet_color_schemas is not None:
                    if color_value == 'value':
                        sc_array[p] = ax.scatter(secondary_x, secondary_y, s=row[planet_column+'_dot_area'], c=row[planet_column+PLANET_COLOR_VALUE_SUFFIX],cmap=planet_color_schemas[p], vmin=vmin_array[p], vmax=vmax_array[p])
                    if color_value == 'percentage_max':
                        sc_array[p] = ax.scatter(secondary_x, secondary_y, s=row[planet_column+'_dot_area'], c=row[planet_column+PLANET_COLOR_PERCENTAGE_MAX_VALUE_SUFFIX],cmap=planet_color_schemas[p], vmin=0, vmax=100)
                else:
                    if color_value == 'value':
                        ax.scatter(secondary_x, secondary_y, s=row[planet_column+'_dot_area'], c=row[planet_column+PLANET_COLOR_VALUE_SUFFIX],cmap=color_schema, vmin=vmin, vmax=vmax)
                    if color_value == 'percentage_max':
                        ax.scatter(secondary_x, secondary_y, s=row[planet_column+'_dot_area'], c=row[planet_column+PLANET_COLOR_PERCENTAGE_MAX_VALUE_SUFFIX],cmap=color_schema, vmin=0, vmax=100)
                p=p+1
    # adjust labels
    plt.xlim(-0.5,len(x_labels)-0.5)
    plt.ylim(-0.5,len(y_labels)-0.5)

    if x_label is None:     # set labels
        x_label = x_col
    ax.set_xlabel(x_label)
    if y_label is None:
        y_label = y_col
    ax.set_ylabel(y_label)

    # ---- Dot colour legend (primary) ---- #
    # create new axis for color bar
    cbar_ax = fig.add_axes([1+LEGEND_COLOR_X_ALIGNMENT*width_per_unit + 0.25*LEGEND_COLOR_WIDTH*width_per_unit, 0.1+LEGEND_COLOR_Y_ALIGNMENT*height_per_unit, LEGEND_COLOR_WIDTH*width_per_unit, LEGEND_COLOR_HEIGHT*height_per_unit])
    cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=LEGEND_FONT_SIZE)
    if color_value == 'value':
        cbar.set_label(COLOR_BAR_TITLE)
    if color_value == 'percentage_max':
        cbar.set_label(COLOR_BAR_TITLE_PMV)
    cmap = getattr(plt.cm, color_schema)    # get the colormap
    legend_color_value = 0.7    # use 70% value for the dot size and planet legend
    legend_color = cmap(legend_color_value)
    legend_hex_color = rgb2hex(legend_color)    # get the hex value
    
    # ---- Planet colour legends for multiple color schemas ---- #
    cbar_count = 1    # initialize color bar count
    planet_cbar_ax = {}     # array for the axises of the planet specific colorbars
    planet_legend_hex_color = {}    # array to store the 70% hex value for the planet legend.
    if planet_color_schemas is not None:
        for i in range(len(sc_array)):
            planet_cbar_ax[i] = fig.add_axes([1+LEGEND_COLOR_X_ALIGNMENT*width_per_unit + 0.25*LEGEND_COLOR_WIDTH*width_per_unit, 0.1+LEGEND_COLOR_Y_ALIGNMENT*height_per_unit + cbar_count*5*LEGEND_COLOR_HEIGHT*height_per_unit, LEGEND_COLOR_WIDTH*width_per_unit, LEGEND_COLOR_HEIGHT*height_per_unit])
            cbar = plt.colorbar(sc_array[i], cax=planet_cbar_ax[i], orientation='horizontal', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=LEGEND_FONT_SIZE)
            if colorbar_label_array is not None:
                cbar.set_label(colorbar_label_array[i])
            else:
                cbar.set_label(planet_columns[i])
            cmap = getattr(plt.cm, planet_color_schemas[i])
            legend_color_value = 0.7
            legend_color = cmap(legend_color_value)
            planet_legend_hex_color[i] = rgb2hex(legend_color)
            cbar_count += 1

    # ---- Dot size legend ---- #
    # create new axis for dot size legend.
    # Note that we create a separate plot on this axis to show our legend.
    if mode == 'planet':
        dot_size_ax = fig.add_axes([1+LEGEND__DOT_X_ALIGNMENT*width_per_unit + 0.25*LEGEND_DOT_WIDTH*width_per_unit, 0.1+LEGEND_DOT_Y_ALIGNMENT*height_per_unit+ cbar_count*5*LEGEND_COLOR_HEIGHT*height_per_unit+ 1.1*LEGEND_PLANET_HEIGHT*height_per_unit, LEGEND_DOT_WIDTH*width_per_unit, LEGEND_DOT_HEIGHT*height_per_unit])
    if mode == 'aggregate':
        dot_size_ax = fig.add_axes([1+LEGEND__DOT_X_ALIGNMENT*width_per_unit + 0.25*LEGEND_DOT_WIDTH*width_per_unit, 0.1+LEGEND_DOT_Y_ALIGNMENT*height_per_unit+ cbar_count*5*LEGEND_COLOR_HEIGHT*height_per_unit, LEGEND_DOT_WIDTH*width_per_unit, LEGEND_DOT_HEIGHT*height_per_unit])
    if use_log_scale:
        values = np.logspace(np.log1p(min_count)/np.log(10), np.log1p(max_count)/np.log(10), num=5) - 1   # create 5 values to be displayed on the legend
        if size_value == 'percentage':
            labels = np.logspace(np.log1p(0)/np.log(10), np.log1p(100)/np.log(10), num=5).astype(int) - 1   # create the 5 corresponding labels
        if size_value == 'count':
            labels = np.logspace(np.log1p(min_count)/np.log(10), np.log1p(max_count)/np.log(10), num =5).astype(int) - 1
    else:
        values = np.linspace(min_count, max_count, num=5)
        if size_value == 'percentage':
            labels = np.linspace(0, 100, num=5).astype(int)
        if size_value == 'count':
            labels = np.linspace(min_count, max_count, num =5).astype(int)
    
    sizes = calculate_dot_sizes(values, max_count, MIN_DOT_SIZE, MAX_DOT_SIZE, use_log_scale)   # dot sizes for the legend
    for i in range(len(sizes)):
        if size_value == 'percentage':
            dot_size_ax.scatter(0, i, s= sizes[i], c=legend_hex_color, alpha=1)
        if size_value == 'count':
            dot_size_ax.scatter(-0.25, i, s= sizes[i], c=legend_hex_color, alpha=1)
            dot_size_ax.scatter(-0.25, i, s=sizes[i], facecolors='none', edgecolors='0',linewidths=LINE_WIDTH)
            if i == len(sizes)-1:   # annotation for the circle
                dot_size_ax.annotate(OUTER_CIRCLE_LABEL, 
                            xy=(-0.25 + ipd*np.sqrt(sizes[i]/np.pi)/LEGEND_DOT_WIDTH, i), xycoords='data',
                            xytext=(-0.25 + 0.25, i), textcoords='data',
                            arrowprops=dict(arrowstyle="-", color="black", lw=LINE_WIDTH),
                            fontsize=LEGEND_FONT_SIZE, ha='left', va='center')
            if i == len(sizes)-2:   # annotation for the dot
                dot_size_ax.annotate(INNER_DOT_LABEL, 
                            xy=(-0.25, i), xycoords='data',
                            xytext=(-0.25 + 0.25, i), textcoords='data',
                            arrowprops=dict(arrowstyle="-", color="black", lw=LINE_WIDTH),
                            fontsize=LEGEND_FONT_SIZE, va = 'center')
    
    #remove exerything else from the dot legend axis
    for spine in dot_size_ax.spines.values():
        spine.set_visible(False)
    dot_size_ax.set_xticks([])
    dot_size_ax.set_xticklabels([])

    dot_size_ax.set_yticks(range(len(sizes)))
    dot_size_ax.set_yticklabels(labels)    
    dot_size_ax.margins(x=0, y=0.2)
    dot_size_ax.set_xlim(-0.5, 0.5)
    if size_value == 'count':
        dot_size_ax.tick_params(axis='y', which='major', pad=-15, length=0)
        dot_size_ax.set_title(SIZE_LEGEND_TITLE, fontsize=10)
    if size_value == 'percentage':
        dot_size_ax.tick_params(axis='y', which='major', pad=-55, length=0)
        dot_size_ax.set_title(SIZE_LEGEND_TITLE_PERCENTAGE, fontsize=10)

    # ---- Planet orientation legend ---- #
    # create new axis for planet orientation legend.
    # Note that we create a separate plot on this axis to show our legend.
    if mode == 'planet':
        dot_orientation_ax = fig.add_axes([1+LEGEND_PLANET_X_ALIGNMENT*width_per_unit + 0.25*LEGEND_PLANET_WIDTH*width_per_unit,0.1+LEGEND_PLANET_Y_ALIGNMENT*height_per_unit+ cbar_count*5*LEGEND_COLOR_HEIGHT*height_per_unit, LEGEND_PLANET_WIDTH*width_per_unit, LEGEND_PLANET_HEIGHT*height_per_unit])
        legend_circle_size = (MIN_DOT_SIZE+MAX_DOT_SIZE)/2      # Define size of the circle
        legend_circle_radius = np.sqrt(legend_circle_size/np.pi)
        legend_dot_size = 0.75*legend_circle_size       # Define relative size of the dots
        legend_dot_radius = np.sqrt(legend_dot_size/np.pi)
        if size_value == 'count':
            dot_orientation_ax.scatter(0, 0, s=legend_circle_size, facecolors='none', edgecolors='0',linewidths=LINE_WIDTH, alpha=1)    # draw circle
        dot_orientation_ax.scatter(0, 0, s=legend_dot_size, c=legend_hex_color, alpha=1)    # draw center dot
        p=0
        for planet_column in planet_columns:
            distance_from_center = PLANET_DIST_SCALER*(LINE_WIDTH+legend_circle_radius+legend_dot_radius)*ipd   # planet distance formula
            offset_x = distance_from_center*planet_x[p]
            offset_y = distance_from_center*planet_y[p]
            secondary_x = offset_x
            secondary_y = offset_y
            if planet_color_schemas is not None:    # plot planet
                dot_orientation_ax.scatter(secondary_x, secondary_y, s=legend_dot_size, c=planet_legend_hex_color[p], alpha=1)
            else:
                dot_orientation_ax.scatter(secondary_x, secondary_y, s=legend_dot_size, c=legend_hex_color, alpha=1)
            if orientation_labels_array is not None:    # plot planet label
                dot_orientation_ax.text(1.5*secondary_x, 1.5*secondary_y, orientation_labels_array[p], fontsize=LEGEND_FONT_SIZE, **planet_text[p])    
            else:
                dot_orientation_ax.text(1.5*secondary_x, 1.5*secondary_y, planet_column, fontsize=LEGEND_FONT_SIZE, **planet_text[p])
            p=p+1
        if p < 6:   # Dynamic annotation for center dot
            distance_from_center = PLANET_DIST_SCALER*(LINE_WIDTH+legend_circle_radius+legend_dot_radius)*ipd
            offset_x = distance_from_center*planet_x[p]
            offset_y = distance_from_center*planet_y[p]
            secondary_x = offset_x
            secondary_y = offset_y
            dot_orientation_ax.annotate(ORIENTATION_LEGEND_CENTER_LABEL, 
                            xy=(0, 0), xycoords='data',
                            xytext=(secondary_x, secondary_y), textcoords='data',
                            arrowprops=dict(arrowstyle="-", color="black", lw=LINE_WIDTH),
                            fontsize=LEGEND_FONT_SIZE, **planet_text[p])
        else:
            center_annotation_position = PLANET_DIST_SCALER*(LINE_WIDTH+5*legend_circle_radius)*ipd
            dot_orientation_ax.annotate(ORIENTATION_LEGEND_CENTER_LABEL, 
                                xy=(0, 0), xycoords='data',
                                xytext=(-center_annotation_position, -center_annotation_position), textcoords='data',
                                arrowprops=dict(arrowstyle="-", color="black", lw=LINE_WIDTH),
                                fontsize=LEGEND_FONT_SIZE, va = 'center')
        
        # remove everything else from the planet legend axis
        for spine in dot_orientation_ax.spines.values():
            spine.set_visible(False)
        dot_orientation_ax.set_xticks([])
        dot_orientation_ax.set_yticks([])
        dot_orientation_ax.set_xlabel('')
        dot_orientation_ax.set_ylabel('')
        dot_orientation_ax.set_xticklabels([])
        dot_orientation_ax.set_yticklabels([])

        dot_orientation_ax.margins(x=0.9, y=0.9)
        dot_orientation_ax.set_aspect('equal')
        dot_orientation_ax.set_title(ORIENTATION_LEGEND_TITLE, fontsize=10, y=0.9)

    # Save figure
    _save_figure(save)

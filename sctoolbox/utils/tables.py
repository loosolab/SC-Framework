import warnings
import pandas as pd
from scipy.stats import zscore
import sctoolbox.utils as utils


def rename_categories(series):
    """
    Rename categories in a pandas series to numbers between 1-(number of categories).

    Parameters
    ----------
    series : pandas.Series
        Series to rename categories in.

    Returns
    -------
    pandas.Series
        Series with renamed categories.
    """

    n_categories = series.cat.categories
    new_names = [str(i) for i in range(1, len(n_categories) + 1)]
    translate_dict = dict(zip(series.cat.categories.tolist(), new_names))
    series = series.cat.rename_categories(translate_dict)

    return series


def fill_na(df, inplace=True, replace={"bool": False, "str": "-", "float": 0, "int": 0, "category": ""}):
    """
    Fill all NA values in pandas depending on the column data type

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame object with NA values over multiple columns
    inplace : boolean, default True
        Whether the DataFrame object is modified inplace.
    replace :  dict, default {"bool": False, "str": "-", "float": 0, "int": 0, "category": ""}
        dict that contains default values to replace nas depedning on data type

    Returns
    -------
    pd.DataFrame or None
        DataFrame with replaced NA values.
    """

    if not inplace:
        df = df.copy()

    # Set default of missing replace value
    replace_def = {"bool": False, "str": "-", "float": 0, "int": 0, "category": ""}
    for t in ["bool", "str", "float", "int", "category"]:
        if t not in replace:
            warnings.warn(f"Value for replace key '{t}' not given. Set to default value: '{replace_def[t]}'")
            replace[t] = replace_def[t]

    for nan_col in df.columns[df.isna().any()]:
        col_type = df[nan_col].dtype.name
        if col_type == "category":
            df[nan_col] = df[nan_col].cat.add_categories(replace[col_type])
            df[nan_col].fillna(replace[col_type], inplace=True)
        elif col_type.startswith("float"):
            df[nan_col].fillna(replace["float"], inplace=True)
        elif col_type.startswith("int"):
            df[nan_col].fillna(replace["int"], inplace=True)
        elif col_type == "object":
            value_set = list({x for x in set(df[nan_col]) if x == x})
            o_type = type(value_set[0]).__name__ if value_set else "str"
            df[nan_col].fillna(replace[o_type], inplace=True)
    if not inplace:
        return df


def _sanitize_sheetname(s, replace="_"):
    """
    Alters given string to produce a valid excel sheetname.
    https://www.excelcodex.com/2012/06/worksheets-naming-conventions/

    Parameters
    ----------
    s : str
        String to sanitize
    replace : str, default "_"
        Replacement of substrings.

    Returns
    -------
    str :
        Valid excel sheetname
    """

    return utils.sanitize_string(s, char_list=["\\", "/", "*", "?", ":", "[", "]"], replace=replace)[0:31]


def write_excel(table_dict, filename, index=False):
    """
    Write a dictionary of tables to a single excel file with one table per sheet.

    Parameters
    ----------
    table_dict : dict
        Dictionary of tables in the format {<sheet_name1>: table, <sheet_name2>: table, (...)}.
    filename : str
        Path to output file.
    index : bool, default False
        Whether to include the index of the tables in file.

    Raises
    ------
    Exception
        If `table_dict` contains items not of type DataFrame.
    """

    # Check if tables are pandas dataframes
    for name, table in table_dict.items():
        if not isinstance(table, pd.DataFrame):
            raise Exception(f"Table {name} is not a pandas DataFrame!")

    # Write to excel
    with pd.ExcelWriter(filename) as writer:
        for name, table in table_dict.items():
            table.to_excel(writer, sheet_name=_sanitize_sheetname(f'{name}'), index=index, engine='xlsxwriter')  # faster than openpyxl


def table_zscore(table, how="row"):
    """
    Z-score a table.

    Parameters
    ----------
    table : pandas.DataFrame
        Table to z-score.
    how : {'row', 'col'}
        Whether to z-score rows or columns.

    Returns
    -------
    pandas.DataFrame :
        Z-scored table.

    Raises
    ------
    Exception
        If `how` has invalid selection.

    """

    if how == "row":
        counts_z = table.T.apply(zscore).T
    elif how == "col":
        counts_z = table.apply(zscore)
    else:
        raise Exception(f"'{how}' is invalid for 'how' - it must be 'row' or 'col'.")

    return counts_z

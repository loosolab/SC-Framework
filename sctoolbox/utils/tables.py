"""Table related functions."""

import warnings
import pandas as pd
from scipy.stats import zscore
import sctoolbox.utils as utils

# type hint imports
from typing import Optional, Any, Literal
from beartype import beartype


@beartype
def rename_categories(series: pd.Series) -> pd.Series:
    """
    Rename categories in a pandas series to numbers between 1-(number of categories).

    Parameters
    ----------
    series : pd.Series
        Pandas Series to rename categories in.

    Returns
    -------
    pd.Series
        Series with renamed categories.
    """

    series_cat = series.astype("category")
    n_categories = series_cat.cat.categories
    new_names = [str(i) for i in range(1, len(n_categories) + 1)]
    translate_dict = dict(zip(series_cat.cat.categories.tolist(), new_names))
    series_cat = series_cat.cat.rename_categories(translate_dict)

    return series_cat


@beartype
def fill_na(df: pd.DataFrame,
            inplace: bool = True,
            replace: dict[str, Any] = {"bool": False, "str": "-", "float": 0, "int": 0, "category": ""}) -> Optional[pd.DataFrame]:
    """
    Fill all NA values in a pandas DataFrame depending on the column data type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame object with NA values over multiple columns
    inplace : boolean, default True
        Whether the DataFrame object is modified inplace.
    replace :  dict[str, Any], default {"bool": False, "str": "-", "float": 0, "int": 0, "category": ""}
        dict that contains default values to replace nas depedning on data type

    Returns
    -------
    Optional[pd.DataFrame]
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


@beartype
def _sanitize_sheetname(s: str,
                        replace: str = "_") -> str:
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
    str
        Valid excel sheetname
    """

    return utils.sanitize_string(s, char_list=["\\", "/", "*", "?", ":", "[", "]"], replace=replace)[0:31]


@beartype
def write_excel(table_dict: dict[str, Any],
                filename: str, index: bool = False) -> None:
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


@beartype
def table_zscore(table: pd.DataFrame,
                 how: Literal["row", "col"] = "row") -> pd.DataFrame:
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
    pd.DataFrame
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
        # Will not be called due to beartype checking for input
        raise Exception(f"'{how}' is invalid for 'how' - it must be 'row' or 'col'.")

    return counts_z

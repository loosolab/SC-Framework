""" Test tables functions"""

import pytest
import numpy as np
import pandas as pd
import sctoolbox.utilities as utils


@pytest.fixture
def na_dataframe():
    """Return DataFrame with columns of multiple types containing NA."""
    data = {'int': [3, 2, 1, np.nan],
            'float': [1.2, 3.4, 5.6, np.nan],
            'string': ['a', 'b', 'c', np.nan],
            'boolean': [True, False, True, np.nan],
            'category_str': ['cat1', 'cat2', 'cat3', np.nan],
            'category_num': [10, 20, 30, np.nan]}
    df = pd.DataFrame.from_dict(data)

    df['category_str'] = df['category_str'].astype('category')
    df['category_num'] = df['category_num'].astype('category')
    return df


def test_rename_categories():
    """Assert if categories were renamed."""

    data = np.random.choice(["C1", "C2", "C3"], size=100)
    series = pd.Series(data).astype("category")
    renamed_series = utils.rename_categories(series)

    assert renamed_series.cat.categories.tolist() == ["1", "2", "3"]


def test_fill_na(na_dataframe):
    """Test if na values in dataframe are filled correctly."""
    utils.fill_na(na_dataframe)
    assert not na_dataframe.isna().any().any()
    assert list(na_dataframe.iloc[3, :]) == [0.0, 0.0, '-', False, '', '']

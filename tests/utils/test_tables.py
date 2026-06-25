"""Test tables functions."""

import numpy as np
import pandas as pd
import sctoolbox.utils.tables as tables


# --------------------------- TESTS --------------------------------- #


def test_rename_categories():
    """Assert if categories were renamed."""

    data = np.random.choice(["C1", "C2", "C3"], size=100)
    series = pd.Series(data).astype("category")
    renamed_series = tables.rename_categories(series)

    assert renamed_series.cat.categories.tolist() == ["1", "2", "3"]


def test_fill_na(na_dataframe):
    """Test if na values in dataframe are filled correctly."""
    tables.fill_na(na_dataframe)
    assert not na_dataframe.isna().any().any()
    assert list(na_dataframe.iloc[3, :]) == [0.0, 0.0, '-', False, '', '']

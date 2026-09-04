"""Shared fixtures for tests under ``tests/utils/``.

Consolidates fixtures that were duplicated byte-for-byte across multiple
``tests/utils/test_*.py`` files (and the ``adata2`` alias of the shared
``pbmc3k_processed`` fixture). Names and (function) scopes match the original
local definitions so consumers need no change.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def named_var_adata(adata_atac):
    """Return a adata object with a prefix attached to the .var index.

    Returns
    -------
    anndata.AnnData
        AnnData object with a prefixed coordinate column and a string
        RangeIndex as .var index.
    """
    adata = adata_atac.copy()
    adata.var['coordinate_col'] = ('prefix-' + adata.var['chr'].astype(str)
                                   + ':' + adata.var['start'].astype(str)
                                   + '-' + adata.var['end'].astype(str))
    adata.var.index = [str(i) for i in range(adata.n_vars)]
    return adata


@pytest.fixture
def na_dataframe():
    """Return DataFrame with columns of multiple types containing NA.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns of multiple types containing NA.
    """
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


@pytest.fixture
def adata2(pbmc3k_processed):
    """Load scanpy processed pbmc3k adata.

    Returns
    -------
    sc.AnnData
        Processed PBMC3k dataset from scanpy.
    """
    return pbmc3k_processed

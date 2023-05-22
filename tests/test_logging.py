import os
import scanpy as sc
import pytest
from sctoolbox.utils import logging


@pytest.fixture
def adata():
    """ Load and returns an anndata object. """
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")

    return sc.read_h5ad(f)


def test_add_uns_info(adata):
    """ Test if add_uns_info works on both string and list keys. """

    logging.add_uns_info(adata, "akey", "info")
    print(adata.uns)
    assert "akey" in adata.uns["sctoolbox"]
    assert adata.uns["sctoolbox"]["akey"] == "info"

    logging.add_uns_info(adata, ["upper", "lower"], "info")
    assert "upper" in adata.uns["sctoolbox"]
    assert adata.uns["sctoolbox"]["upper"]["lower"] == "info"

    logging.add_uns_info(adata, ["upper", "lower"], "info2", how="append")
    assert adata.uns["sctoolbox"]["upper"]["lower"] == ["info", "info2"]

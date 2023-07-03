import os
import scanpy as sc
import pytest
from sctoolbox.utils import decorator as deco


@pytest.fixture
def adata():
    """ Load and returns an anndata object. """
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")

    return sc.read_h5ad(f)


def test_log_anndata(adata):
    """ Test if log_anndata  decorator works. """

    # define a function with the decorator
    @deco.log_anndata
    def test_func(adata, param1, param2, param3, param4, param5):
        param_array = [param1, param2, param3, param4, param5]
        adata.uns["test_func"] = param_array
        return adata

    # run the function
    adata = test_func(adata, param1=1, param2=None, param3="test", param4=1.0, param5=True)
    # run the function again
    adata = test_func(adata, param1=1, param2=None, param3="test", param4=1.0, param5=True)

    # check if log is in adata.uns
    assert "sctoolbox" in adata.uns
    assert "test_func" in adata.uns["sctoolbox"]
    # check if run_1 and run_2 are in adata.uns
    assert adata.uns["sctoolbox"]["test_func"]["run_1"]['kwargs']["param1"] == 1
    assert adata.uns["sctoolbox"]["test_func"]["run_1"]['kwargs']["param2"] is None
    assert adata.uns["sctoolbox"]["test_func"]["run_1"]['kwargs']["param3"] == "test"
    assert adata.uns["sctoolbox"]["test_func"]["run_1"]['kwargs']["param4"] == 1.0
    assert adata.uns["sctoolbox"]["test_func"]["run_1"]['kwargs']["param5"] is True

    assert adata.uns["sctoolbox"]["test_func"]["run_2"]['kwargs']["param1"] == 1

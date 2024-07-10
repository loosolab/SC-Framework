"""Test decorator functions."""

import os
import scanpy as sc
import pytest
from sctoolbox.utils import decorator as deco

import sctoolbox.tools.qc_filter as qc
import sctoolbox.utils as utils


@pytest.fixture
def adata():
    """Load and returns an anndata object."""
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")

    return sc.read_h5ad(f)


def test_log_anndata(adata):
    """Test if log_anndata  decorator works."""

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
    assert "log" in adata.uns["sctoolbox"]
    assert "test_func" in adata.uns["sctoolbox"]["log"]
    # check if run_1 and run_2 are in adata.uns
    assert adata.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param1"] == 1
    assert adata.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param2"] is None
    assert adata.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param3"] == "test"
    assert adata.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param4"] == 1.0
    assert adata.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param5"] is True
    assert adata.uns["sctoolbox"]["log"]["test_func"]["run_2"]['kwargs']["param1"] == 1


def test_get_parameter_table(adata):
    """Test if get_parameter_table works."""

    # Run a few functions on the adata
    qc.calculate_qc_metrics(adata)
    qc.predict_sex(adata, "sample", threshold=0.1)  # threshold is kwargs

    table = utils.decorator.get_parameter_table(adata)

    assert table.shape[0] == 2  # two functions were run
    assert table.loc[1, "kwargs"] == {"threshold": 0.1}  # check if kwargs are correctly stored for predict_sex
    assert set(["func", "args", "kwargs", "user", "timestamp"]).issubset(table.columns)

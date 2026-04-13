"""Test decorator functions."""

import pytest
from sctoolbox.utils import decorator as deco

import sctoolbox.tools.qc_filter as qc
import sctoolbox.utils as utils


# --------------------------- TESTS --------------------------------- #


def test_log_anndata(adata_fun_scope):
    """Test if log_anndata  decorator works."""

    # define a function with the decorator
    @deco.log_anndata
    def test_func(adata, param1, param2, param3, param4, param5, param6):
        param_array = [param1, param2, param3, param4, param5, param6]
        adata.uns["test_func"] = param_array
        return adata

    # run the function
    adata_fun_scope = test_func(adata_fun_scope, param1=1, param2=None, param3="test", param4=1.0, param5=True, param6={"nested": {"inner": True}})
    # run the function again
    adata_fun_scope = test_func(adata_fun_scope, param1=1, param2=None, param3="test", param4=1.0, param5=True, param6={"nested": {"inner": True}})

    # check if log is in adata.uns
    assert "sctoolbox" in adata_fun_scope.uns
    assert "log" in adata_fun_scope.uns["sctoolbox"]
    assert "test_func" in adata_fun_scope.uns["sctoolbox"]["log"]
    # check if run_1 and run_2 are in adata.uns
    assert adata_fun_scope.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param1"] == 1
    assert adata_fun_scope.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param2"] is None
    assert adata_fun_scope.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param3"] == "test"
    assert adata_fun_scope.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param4"] == 1.0
    assert adata_fun_scope.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param5"] is True
    assert "param6" in adata_fun_scope.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']
    assert adata_fun_scope.uns["sctoolbox"]["log"]["test_func"]["run_2"]['kwargs']["param1"] == 1


def test_get_parameter_table(adata_fun_scope):
    """Test if get_parameter_table works."""

    # Run a few functions on the adata
    qc.calculate_qc_metrics(adata_fun_scope)
    qc.predict_sex(adata_fun_scope, "condition", threshold=0.1)  # threshold is kwargs

    table = utils.decorator.get_parameter_table(adata_fun_scope)

    assert table.shape[0] == 2  # two functions were run
    assert table.loc[1, "kwargs"] == {"threshold": 0.1}  # check if kwargs are correctly stored for predict_sex
    assert set(["func", "args", "kwargs", "user", "timestamp"]).issubset(table.columns)

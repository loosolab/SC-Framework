import os
import scanpy as sc
import pytest
from sctoolbox.utils import decorator as deco
import sctoolbox.utils as utils

from sctoolbox._settings import settings
logger = settings.logger


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
    assert "log" in adata.uns["sctoolbox"]
    assert "test_func" in adata.uns["sctoolbox"]["log"]
    # check if run_1 and run_2 are in adata.uns
    assert adata.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param1"] == 1
    assert adata.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param2"] is None
    assert adata.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param3"] == "test"
    assert adata.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param4"] == 1.0
    assert adata.uns["sctoolbox"]["log"]["test_func"]["run_1"]['kwargs']["param5"] is True

    assert adata.uns["sctoolbox"]["log"]["test_func"]["run_2"]['kwargs']["param1"] == 1


def test_add_uns_info(adata):
    """ Test if add_uns_info works on both string and list keys. """

    utils.add_uns_info(adata, "akey", "info")
    print(adata.uns)
    assert "akey" in adata.uns["sctoolbox"]
    assert adata.uns["sctoolbox"]["akey"] == "info"

    utils.add_uns_info(adata, ["upper", "lower"], "info")
    assert "upper" in adata.uns["sctoolbox"]
    assert adata.uns["sctoolbox"]["upper"]["lower"] == "info"

    utils.add_uns_info(adata, ["upper", "lower"], "info2", how="append")
    assert adata.uns["sctoolbox"]["upper"]["lower"] == ["info", "info2"]


def test_user_logging():
    """ Test is logfile is correctly overwritten """

    settings.log_file = "test.log"
    logger.info("test_info")
    assert "test_info" in open(settings.log_file).read()

    # Set again to the same file
    settings.log_file = "test.log"
    logger.info("test_info2")
    content = open(settings.log_file).read()
    assert "test_info" in content  # check that the first log is still there
    assert "test_info2" in content

    # Set to overwrite the file
    settings.overwrite_log = True
    logger.info("test_info3")
    content = open(settings.log_file).read()
    assert "test_info2" not in content  # previous log was overwritten
    assert "test_info3" in content      # new log is there

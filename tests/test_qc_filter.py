
import pytest
import sctoolbox.qc_filter as qc
import scanpy as sc
import numpy as np
import os


# --------------------------- Fixtures ------------------------------ #

@pytest.fixture
def adata():
    """ Load and returns an anndata object. """
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")
    return sc.read_h5ad(f)


@pytest.fixture
def adata_qc(adata):
    """ Add qc variables to an adata object """

    n1 = int(adata.shape[0] * 0.8)
    n2 = adata.shape[0] - n1

    adata.obs["qc_variable1"] = np.append(np.random.normal(size=n1), np.random.normal(size=n2, loc=1, scale=2))
    adata.obs["qc_variable2"] = np.append(np.random.normal(size=n1), np.random.normal(size=n2, loc=1, scale=3))

    return adata


@pytest.fixture
def threshold_dict():
    d = {"qc_variable1": {"min": 0.5, "max": 1.5},
         "qc_variable2": {"min": 0.5, "max": 1}}
    return d


@pytest.fixture
def invalid_threshold_dict():
    d = {"not_present": {"notmin": 0.5, "max": 1.5},
         "qc_variable2": {"min": 0.5, "max": 1}}
    return d


# --------------------------- Tests --------------------------------- #

def test_estimate_doublets(adata):
    """ Test whether 'doublet_score' was added to adata.obs """
    qc.estimate_doublets(adata, plot=False)  # turn plot off to avoid block during testing

    assert "doublet_score" in adata.obs.columns


def test_get_thresholds():
    """ Test whether min/max threshold can be found for a normal distribution """
    data_arr = np.random.normal(size=1000)

    threshold = qc.get_thresholds(data_arr)

    assert "min" in threshold and "max" in threshold


def test_automatic_thresholds(adata_qc):
    """ Test whether automatic thresholds are successfully calculated and added to the threshold dict """
    thresholds = qc.automatic_thresholds(adata_qc, columns=["qc_variable1", "qc_variable2"])
    threshold_keys = sorted(thresholds.keys())

    assert threshold_keys == ["qc_variable1", "qc_variable2"]


def test_thresholds_as_table(threshold_dict):
    """ Test whether treshold dict is successfully converted to pandas table """

    table = qc.thresholds_as_table(threshold_dict)

    assert type(table).__name__ == "DataFrame"


def test_apply_qc_thresholds(adata_qc):
    """ Check whether adata cells were filtered by thresholds """

    thresholds = qc.automatic_thresholds(adata_qc, columns=["qc_variable1", "qc_variable2"])
    adata_filter = qc.apply_qc_thresholds(adata_qc, thresholds, inplace=False)

    assert adata_filter.shape[0] < adata_qc.shape[0]


def test_validate_threshold_dict(adata_qc, threshold_dict):
    """ Test whether threshold dict is successfully validated """
    ret = qc.validate_threshold_dict(adata_qc.obs, threshold_dict)
    assert ret is None


def test_validate_threshold_dict_invalid(adata_qc, invalid_threshold_dict):
    """ Test if an invalid threshold dict raises an error """
    with pytest.raises(ValueError):
        qc.validate_threshold_dict(adata_qc.obs, invalid_threshold_dict)


def test_filter_genes(adata):
    """ Test whether genes were filtered out based on a boolean column"""

    adata.var["gene_bool"] = np.random.choice(a=[False, True], size=adata.shape[1])
    n_false = sum(adata.var["gene_bool"] is False)
    qc.filter_genes(adata, "gene_bool", inplace=True)  # removes all genes with boolean True

    assert adata.shape[1] == n_false

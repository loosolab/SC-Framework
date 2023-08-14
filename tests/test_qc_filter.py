import logging

import pytest
import sctoolbox.qc_filter as qc
import scanpy as sc
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt

# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")


# --------------------------- Fixtures ------------------------------ #

@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """ Load and returns an anndata object. """
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")
    adata = sc.read_h5ad(f)
    adata.obs['sample'] = np.random.choice(["sample1", "sample2"], size=len(adata))

    # Add fake qc variables to anndata
    n1 = int(adata.shape[0] * 0.8)
    n2 = adata.shape[0] - n1

    adata.obs["qc_variable1"] = np.append(np.random.normal(size=n1), np.random.normal(size=n2, loc=1, scale=2))
    adata.obs["qc_variable2"] = np.append(np.random.normal(size=n1), np.random.normal(size=n2, loc=1, scale=3))

    # set gene names as index instead of ensemble ids
    adata.var.reset_index(inplace=True)
    adata.var['gene'] = adata.var['gene'].astype('str')
    adata.var.set_index('gene', inplace=True, drop=False)  # keep gene column
    adata.var_names_make_unique()

    # Make sure adata contains at least some cell cycle genes (mouse)
    genes = ["Gmnn", "Rad51", "Tmpo", "Cdk1"]
    adata.var.index = adata.var.index[:-len(genes)].tolist() + genes  # replace last genes with cell cycle genes

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


@pytest.fixture
def s_list(adata):
    return adata.var.index[:int(len(adata.var) / 2)].tolist()


@pytest.fixture
def g2m_list(adata):
    return adata.var.index[int(len(adata.var) / 2):].tolist()


@pytest.fixture
def g2m_file(g2m_list):
    """ Write a tmp file, which is deleted after usage. """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = os.path.join(tmpdir, "g2m_genes.txt")

        with open(tmp, "w") as f:
            f.writelines([g + "\n" for g in g2m_list])

        yield tmp


@pytest.fixture
def s_file(s_list):
    """ Write a tmp file, which is deleted after usage. """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = os.path.join(tmpdir, "s_genes.txt")

        with open(tmp, "w") as f:
            f.writelines([g + "\n" for g in s_list])

        yield tmp


# --------------------------- Tests --------------------------------- #

@pytest.mark.parametrize("groupby,threads", [(None, 1), ("sample", 1), ("sample", 4)])
def test_estimate_doublets(adata, groupby, threads):
    """ Test whether 'doublet_score' was added to adata.obs """

    adata = adata.copy()  # copy adata to avoid inplace changes
    qc.estimate_doublets(adata, groupby=groupby, plot=False, threads=threads, n_prin_comps=10)  # turn plot off to avoid block during testing

    assert "doublet_score" in adata.obs.columns


def test_get_thresholds():
    """ Test whether min/max threshold can be found for a normal distribution """
    data_arr = np.random.normal(size=1000)

    threshold = qc._get_thresholds(data_arr)

    assert "min" in threshold and "max" in threshold


def test_automatic_thresholds(adata):
    """ Test whether automatic thresholds are successfully calculated and added to the threshold dict """
    thresholds = qc.automatic_thresholds(adata, columns=["qc_variable1", "qc_variable2"])
    threshold_keys = sorted(thresholds.keys())

    assert threshold_keys == ["qc_variable1", "qc_variable2"]


def test_thresholds_as_table(threshold_dict):
    """ Test whether treshold dict is successfully converted to pandas table """

    table = qc.thresholds_as_table(threshold_dict)

    assert type(table).__name__ == "DataFrame"


def test_apply_qc_thresholds(adata):
    """ Check whether adata cells were filtered by thresholds """

    thresholds = qc.automatic_thresholds(adata, columns=["qc_variable1", "qc_variable2"])
    adata_filter = qc.apply_qc_thresholds(adata, thresholds, inplace=False)

    assert adata_filter.shape[0] < adata.shape[0]


def test_validate_threshold_dict(adata, threshold_dict):
    """ Test whether threshold dict is successfully validated """
    ret = qc.validate_threshold_dict(adata.obs, threshold_dict)
    assert ret is None


def test_validate_threshold_dict_invalid(adata, invalid_threshold_dict):
    """ Test if an invalid threshold dict raises an error """
    with pytest.raises(ValueError):
        qc.validate_threshold_dict(adata.obs, invalid_threshold_dict)


def test_filter_genes(adata):
    """ Test whether genes were filtered out based on a boolean column"""
    adata_c = adata.copy()
    adata_c.var["gene_bool"] = np.random.choice(a=[False, True], size=adata_c.shape[1])
    n_false = sum(~adata_c.var["gene_bool"])
    qc.filter_genes(adata_c, "gene_bool", inplace=True)  # removes all genes with boolean True

    assert adata_c.shape[1] == n_false


def test_filter_cells(adata):
    """ Test whether cells were filtered out based on a boolean column"""

    adata.obs["cell_bool"] = np.random.choice(a=[False, True], size=adata.shape[0])
    n_false = sum(~adata.obs["cell_bool"])
    qc.filter_cells(adata, "cell_bool", inplace=True)  # removes all genes with boolean True

    assert adata.shape[0] == n_false


@pytest.mark.parametrize("threshold", [0.3, 0.0])
def test_predict_sex(caplog, adata, threshold):
    """Test if predict_sex warns on invalid gene and succeeds."""
    adata = adata.copy()  # copy adata to avoid inplace changes

    # gene not in data
    with caplog.at_level(logging.INFO):
        qc.predict_sex(adata, groupby='sample')
        assert "Selected gene is not present in the data. Prediction is skipped." in caplog.records[1].message

    # gene in data
    qc.predict_sex(adata, gene='Xkr4', gene_column='gene', groupby='sample', threshold=threshold)
    assert 'predicted_sex' in adata.obs.columns


def test_predict_sex_diff_types(caplog, adata):
    """Test predict_sex for different adata.X types."""

    adata_ndarray = adata.copy()
    adata_ndarray.X = adata_ndarray.X.toarray()
    adata_matrix = adata_ndarray.copy()
    adata_matrix.X = np.asmatrix(adata_matrix.X)

    # ndarray
    qc.predict_sex(adata_ndarray, gene='Xkr4', gene_column='gene', groupby='sample')
    assert 'predicted_sex' in adata_ndarray.obs.columns

    # matrix
    qc.predict_sex(adata_matrix, gene='Xkr4', gene_column='gene', groupby='sample')
    assert 'predicted_sex' in adata_matrix.obs.columns


@pytest.mark.parametrize(
    "species, s_genes, g2m_genes, inplace",
    [
        ("mouse", None, None, False),
        (None, "s_file", "g2m_file", True),
        (None, "s_list", "g2m_list", True),
        ("unicorn", None, None, False)
    ],
)
def test_predict_cell_cycle(adata, species, s_genes, g2m_genes, inplace, request):
    """ Test if cell cycle is predicted and added to adata.obs """

    # Get value of s_genes and g2m_genes fixtures
    s_genes = request.getfixturevalue(s_genes) if s_genes is not None else None
    g2m_genes = request.getfixturevalue(g2m_genes) if g2m_genes is not None else None

    # Remove columns if already present
    expected_columns = ["S_score", "G2M_score", "phase"]
    adata.obs = adata.obs.drop(columns=[c for c in expected_columns if c in adata.obs.columns])

    assert not any(c in adata.obs.columns for c in expected_columns)

    if species == "unicorn":
        with pytest.raises(ValueError):
            qc.predict_cell_cycle(adata, species=species)
    else:
        # For other organisms or if s_genes / g2m_genes are given
        out = qc.predict_cell_cycle(adata, species=species, s_genes=s_genes, g2m_genes=g2m_genes, inplace=inplace)

        if inplace:
            assert out is None
            assert all(c in adata.obs.columns for c in expected_columns)
        else:
            assert not any(c in adata.obs.columns for c in expected_columns)
            assert all(c in out.obs.columns for c in expected_columns)

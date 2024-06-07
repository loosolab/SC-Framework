"""Test quality control functions."""
import logging

import pytest
import sctoolbox.qc_filter as qc
import numpy as np

# --------------------------- Tests --------------------------------- #


@pytest.mark.parametrize("groupby,threads", [(None, 1), ("sample", 1), ("sample", 4)])
def test_estimate_doublets(adata, groupby, threads):
    """Test whether 'doublet_score' was added to adata.obs."""

    adata = adata.copy()  # copy adata to avoid inplace changes
    qc.estimate_doublets(adata, groupby=groupby, plot=False, threads=threads, n_prin_comps=10)  # turn plot off to avoid block during testing

    assert "doublet_score" in adata.obs.columns


def test_get_thresholds():
    """Test whether min/max threshold can be found for a normal distribution."""
    data_arr = np.random.normal(size=1000)

    threshold = qc._get_thresholds(data_arr)

    assert "min" in threshold and "max" in threshold


@pytest.mark.parametrize("groupby", [None, "group"])
@pytest.mark.parametrize("columns", [None, ["qc_variable1", "qc_variable2"]])
@pytest.mark.parametrize("which", ["obs", "var"])
def test_automatic_thresholds(adata, which, columns, groupby):
    """Test whether automatic thresholds are successfully calculated and added to the threshold dict."""
    thresholds = qc.automatic_thresholds(adata, which=which, columns=columns, groupby=groupby)
    threshold_keys = sorted(thresholds.keys())

    if columns:
        assert threshold_keys == columns


def test_automatic_thresholds_failure(adata):
    """Test automatic_thresholds failure."""

    with pytest.raises(ValueError):
        qc.automatic_thresholds(adata, groupby="INVALID")


def test_thresholds_as_table(threshold_dict):
    """Test whether treshold dict is successfully converted to pandas table."""

    table = qc.thresholds_as_table(threshold_dict)

    assert type(table).__name__ == "DataFrame"


def test_apply_qc_thresholds(adata):
    """Check whether adata cells were filtered by thresholds."""

    thresholds = qc.automatic_thresholds(adata, columns=["qc_variable1", "qc_variable2"])
    adata_filter = qc.apply_qc_thresholds(adata, thresholds, inplace=False)

    assert adata_filter.shape[0] < adata.shape[0]


def test_validate_threshold_dict(adata, threshold_dict):
    """Test whether threshold dict is successfully validated."""
    ret = qc.validate_threshold_dict(adata.obs, threshold_dict)
    assert ret is None


def test_validate_threshold_dict_invalid(adata, invalid_threshold_dict):
    """Test if an invalid threshold dict raises an error."""
    with pytest.raises(ValueError):
        qc.validate_threshold_dict(adata.obs, invalid_threshold_dict)


def test_filter_genes(adata):
    """Test whether genes were filtered out based on a boolean column."""
    adata_c = adata.copy()
    adata_c.var["gene_bool"] = np.random.choice(a=[False, True], size=adata_c.shape[1])
    n_false = sum(~adata_c.var["gene_bool"])
    qc.filter_genes(adata_c, "gene_bool", inplace=True)  # removes all genes with boolean True

    assert adata_c.shape[1] == n_false


def test_filter_cells(adata):
    """Test whether cells were filtered out based on a boolean column."""
    adata = adata.copy()  # copy adata to avoid inplace changes
    adata.obs["cell_bool"] = np.random.choice(a=[False, True], size=adata.shape[0])
    n_false = sum(~adata.obs["cell_bool"])
    qc.filter_cells(adata, "cell_bool", inplace=True)  # removes all genes with boolean True

    assert adata.shape[0] == n_false


@pytest.mark.parametrize("which, to_filter", [("obs", ["AAACCCACAGCCTATA", "AAACCCACAGGGCTTC"]),
                                              ("var", ["ENSMUSG00000051951", "ENSMUSG00000102851"])])
def test_filter_object(adata, which, to_filter):
    """Test whether cells/genes are filtered based on a list of cells/genes."""
    adata = adata.copy()  # copy adata to avoid inplace changes
    qc._filter_object(adata, to_filter, which=which)
    table = adata.obs if which == "obs" else adata.var
    assert all([i not in table.index for i in to_filter])


def test_filter_object_fail(adata):
    """Test whether invalid input raises the correct errors."""
    adata = adata.copy()
    adata.obs["notbool"] = np.random.choice(a=[False, True, np.nan], size=adata.shape[0])
    with pytest.raises(ValueError, match="Column notbool contains values that are not of type boolean"):
        qc._filter_object(adata, "notbool", which="obs")

    with pytest.raises(ValueError, match="Column invalid not found"):
        qc._filter_object(adata, "invalid", which="obs")


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
    """Test if cell cycle is predicted and added to adata.obs."""

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

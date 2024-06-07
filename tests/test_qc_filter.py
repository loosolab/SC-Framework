"""Test quality control functions."""

import pytest
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
    """Load and returns an anndata object."""
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")
    adata = sc.read_h5ad(f)
    adata.obs['sample'] = np.random.choice(["sample1", "sample2"], size=len(adata))

    # add random groups
    adata.obs["group"] = np.random.choice(["grp1", "grp2", "grp3"], size=len(adata))
    adata.var["group"] = np.random.choice(["grp1", "grp2", "grp3"], size=len(adata.var))

    # Add fake qc variables to anndata
    n1 = int(adata.shape[0] * 0.8)
    n2 = adata.shape[0] - n1

    adata.obs["qc_variable1"] = np.append(np.random.normal(size=n1), np.random.normal(size=n2, loc=1, scale=2))
    adata.obs["qc_variable2"] = np.append(np.random.normal(size=n1), np.random.normal(size=n2, loc=1, scale=3))

    n1 = int(adata.shape[1] * 0.8)
    n2 = adata.shape[1] - n1

    adata.var["qc_variable1"] = np.append(np.random.normal(size=n1), np.random.normal(size=n2, loc=1, scale=2))
    adata.var["qc_variable2"] = np.append(np.random.normal(size=n1), np.random.normal(size=n2, loc=1, scale=3))

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
    """Create dict with qc thresholds."""
    d = {"qc_variable1": {"min": 0.5, "max": 1.5},
         "qc_variable2": {"min": 0.5, "max": 1}}
    return d


@pytest.fixture
def invalid_threshold_dict():
    """Create invalid qc threshold dict."""
    d = {"not_present": {"notmin": 0.5, "max": 1.5},
         "qc_variable2": {"min": 0.5, "max": 1}}
    return d


@pytest.fixture
def s_list(adata):
    """Return a list of first half of adata genes."""
    return adata.var.index[:int(len(adata.var) / 2)].tolist()


@pytest.fixture
def g2m_list(adata):
    """Return a list of second half of adata genes."""
    return adata.var.index[int(len(adata.var) / 2):].tolist()


@pytest.fixture
def g2m_file(g2m_list):
    """Write a tmp file, which is deleted after usage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = os.path.join(tmpdir, "g2m_genes.txt")

        with open(tmp, "w") as f:
            f.writelines([g + "\n" for g in g2m_list])

        yield tmp


@pytest.fixture
def s_file(s_list):
    """Write a tmp file, which is deleted after usage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = os.path.join(tmpdir, "s_genes.txt")

        with open(tmp, "w") as f:
            f.writelines([g + "\n" for g in s_list])

        yield tmp

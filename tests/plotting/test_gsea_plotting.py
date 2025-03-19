"""Test gsea plotting functions."""

import pytest
import scanpy as sc
import pandas as pd
import numpy as np
from sctoolbox.plotting import gsea


# ------------------------------ FIXTURES --------------------------------- #


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """Minimal adata file for testing."""
    return sc.datasets.pbmc68k_reduced()


@pytest.fixture()
def term_table():
    """Minimal term table for testing."""
    term_table = pd.DataFrame({
        "Term": "Actin Filament Organization (GO:0007015)",
        "Lead_genes": ["COBL", "WIPF1;SH3KBP1"]})
    return term_table


@pytest.fixture()
def term_table_clustered():
    """Clustered term table."""
    term_table = pd.DataFrame({
        "Term": ["Myofibril Assembly (GO:0030239)", "Actomyosin Structure Organization (GO:0031032)", "Sarcomere Organization (GO:0045214)",
                 "Regulation Of GTPase Activity (GO:0043087)", "Positive Regulation Of Gene Expression", "Regulation Of Angiogenesis (GO:0045765)"],
        "NES": [2.51025, 2.51025, 2.404361,
                2.51025, 2.51025, 2.404361],
        "FDR q-val": [0.001371, 0.001371, 0.001371,
                      0.001045, 0.003421, 0.008325],
        "Lead_genes": ["ANKRD1;MYH6;ACTN2;MYOZ2;MYOM2;LMOD2;MYPN;TNNT2;MYLK3;TTN;FHOD3",
                       "ANKRD1;MYH6;ACTN2;MYOZ2;MYOM2;LMOD2;MYPN;TNNT2;MYLK3;TTN;FHOD3",
                       "ANKRD1;MYH6;MYOZ2;MYOM2;LMOD2;MYPN;TNNT2;MYLK3;TTN;FHOD3",
                       "DOCK10;DOCK8;ARHGAP15;RAP1GDS1;SEMA4D;PICALM",
                       "DOCK10;DOCK8;ARHGAP15;RAP1GDS1;SEMA4D;PICALM",
                       "DOCK10;DOCK8;"],
        "Cluster": ["1", "1", "1",
                    "2", "2", "2"]})
    return term_table


# ------------------------------ TESTS --------------------------------- #


def test_term_dotplot(adata, term_table):
    """Test term_dotplot success."""
    axes = gsea.term_dotplot(term="Actin Filament Organization (GO:0007015)",
                             term_table=term_table,
                             adata=adata,
                             groupby="louvain")

    assert isinstance(axes, np.ndarray)
    ax_type = type(axes[0]).__name__
    assert ax_type.startswith("Axes")


def test_gsea_network(term_table_clustered):
    """Test tsea_network success."""
    gsea.gsea_network(term_table_clustered)


def test_gsea_network_fail(term_table_clustered):
    """Test tsea_network success."""
    with pytest.raises(ValueError):
        gsea.gsea_network(term_table_clustered, cutoff=0.0000005)

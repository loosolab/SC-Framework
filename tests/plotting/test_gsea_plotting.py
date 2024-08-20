"""Test gsea plotting functions."""

import pytest
import scanpy as sc
import pandas as pd
from sctoolbox.plotting import gsea


# ------------------------------ FIXTURES --------------------------------- #


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """Minimal adata file for testing."""
    return sc.datasets.pbmc68k_reduced()


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def term_table():
    """Minimal term table for testing."""
    term_table = pd.DataFrame({
        "Term": "Actin Filament Organization (GO:0007015)",
        "Genes": ["COBL", "WIPF1;SH3KBP1"]})
    return term_table


# ------------------------------ TESTS --------------------------------- #


def test_term_dotplot(adata, term_table):
    """Test term_dotplot success."""
    axes = gsea.term_dotplot(term="Actin Filament Organization (GO:0007015)",
                             term_table=term_table,
                             adata=adata,
                             groupby="louvain")

    assert isinstance(axes, list)
    ax_type = type(axes[0]).__name__
    assert ax_type.startswith("Axes")

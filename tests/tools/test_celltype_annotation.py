"""Test functions related to cell type annotation."""

import os
import pytest
import anndata as ad
from sctoolbox.tools import celltype_annotation
from tests.conftest import DATA_DIR


# --------------------------- FIXTURES ------------------------------ #


@pytest.fixture
def test_adata():
    """Load adata.

    Returns
    -------
    anndata.AnnData
        AnnData object for SCSA testing.
    """
    adata_dir = os.path.join(DATA_DIR, 'scsa')
    adata = ad.read_h5ad(adata_dir + '/adata_scsa.h5ad')
    return adata


# --------------------------- TESTS --------------------------------- #


def fetch_adata_uns(test_adata):
    """Return precalculated gene ranking dict.

    Returns
    -------
    dict
        Dictionary containing rank_genes_groups results from adata.uns.
    """
    # fetches adata.uns['rank_genes_groups] as dict from a test adata to use in
    # test_get_rank_genes
    d = test_adata.uns['rank_genes_groups']
    return d


def test_get_rank_genes(test_adata):
    """Test _get_rank_genes success."""
    d = fetch_adata_uns(test_adata)
    genes = celltype_annotation._get_rank_genes(d)
    assert len(genes) == len(set(genes))


@pytest.mark.parametrize("column", ["SCSA_pred_celltype", "test_1", "test_2"])
def test_run_scsa(test_adata, column):
    """Test run_scsa success."""
    adata = celltype_annotation.run_scsa(test_adata, species='Mouse', inplace=False, column_added=column)
    assert column in adata.obs.columns


def test_add_cellxgene_annotation(adata_fun_scope, tmp_path):
    """Test if 'cellxgene' column is added to adata.obs."""

    # Create a CSV with matching barcodes
    csv_f = tmp_path / "cellxgene_anno.csv"
    csv_f.write_text(
        "index,cellxgene_clusters\n"
        + "\n".join(f"{bc},cluster{i % 3}" for i, bc in enumerate(adata_fun_scope.obs.index))
    )

    celltype_annotation.add_cellxgene_annotation(adata_fun_scope, str(csv_f))

    assert "cellxgene_clusters" in adata_fun_scope.obs.columns

import os
import pytest
import anndata as ad
from sctoolbox import celltype_annotation


@pytest.fixture
def test_adata():
    adata_dir = os.path.join(os.path.dirname(__file__), 'data', 'scsa')
    adata = ad.read_h5ad(adata_dir + '/adata_scsa.h5ad')
    return adata


def fetch_adata_uns(test_adata):
    # fetches adata.uns['rank_genes_groups] as dict from a test adata to use in
    # test_get_rank_genes
    d = test_adata.uns['rank_genes_groups']
    return d


@pytest.mark.parametrize("test_species", ["Mouse", "Human"])
def test_read_scsa_database(test_species):
    gene_names, gene_ids = celltype_annotation.read_scsa_database(test_species)

    assert isinstance(gene_names, list) and isinstance(gene_ids, list)


def test_get_rank_genes(test_adata):
    d = fetch_adata_uns(test_adata)
    genes = celltype_annotation.get_rank_genes(d)
    assert len(genes) == len(set(genes))


@pytest.mark.parametrize("column", ["SCSA_pred_celltype", "test_1", "test_2"])
def test_run_scsa(test_adata, column):
    adata = celltype_annotation.run_scsa(test_adata, species='Mouse', inplace=False, column_added=column)
    assert column in adata.obs.columns

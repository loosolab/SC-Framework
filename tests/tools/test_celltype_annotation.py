"""Test functions related to cell type annotation."""

import pytest
from sctoolbox.tools import celltype_annotation
from tests.conftest import _load_adata_scsa_h5ad


# --------------------------- FIXTURES ------------------------------ #


@pytest.fixture
def test_adata():
    """Load adata.

    Returns
    -------
    anndata.AnnData
        AnnData object for SCSA testing.
    """
    return _load_adata_scsa_h5ad()


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

    # derive the cluster -> celltype mapping from the returned adata; with inplace=False
    # the input object carries neither column_added nor uns['SCSA']['results']
    groupby = adata.uns['rank_genes_groups']['params']['groupby']
    annotated = adata.obs[[groupby, column]].dropna().drop_duplicates()  # clusters 5 and 6 get no SCSA row
    mapping = dict(zip(annotated[groupby].astype(str), annotated[column]))

    # Golden-output pin, captured from a run on the adata_scsa fixture with the bundled
    # cellmarker_mouse.tsv: a mismatch means the annotations changed, not that these are stale
    assert mapping == {'1': 'Fibroblast',
                       '2': 'Stage I neutrophil',
                       '3': 'Hepatocellular cell',
                       '4': 'Endothelial cell',
                       '7': 'Epithelial cell',
                       '8': 'Podocyte'}

    results = adata.uns['SCSA']['results']
    assert list(results.columns) == ['Cell Type', 'Z-score', 'Cluster']
    assert len(results) == 8604


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

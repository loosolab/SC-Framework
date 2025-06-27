"""Test tools/gsea.py functions."""

import pytest
import scanpy as sc
import sctoolbox.tools as tools
import pandas as pd
from sctoolbox.utils.adata import in_uns


# ---------------------------- FIXTURES -------------------------------- #


@pytest.fixture
def adata():
    """Return clustered dataset with ranked genes."""
    obj = sc.datasets.pbmc3k_processed()
    tools.marker_genes.run_rank_genes(obj, "louvain")

    return obj


# ------------------------------ TESTS --------------------------------- #

@pytest.mark.parametrize("method, res_col", [("prerank",
                                              ['Name', 'Term', 'ES', 'NES',
                                               'NOM p-val', 'FDR q-val',
                                               'FWER p-val', 'Tag %', 'Gene %',
                                               'Lead_genes', 'Cluster']),
                                             ("enrichr",
                                              ['Gene_set', 'Term', 'Overlap',
                                               'P-value', 'Adjusted P-value',
                                               'Odds Ratio', 'Combined Score',
                                               'Genes'])])
def test_gene_set_enrichment(adata, method, res_col):
    """Test enrichr_marker_genes."""
    adata_mod = tools.gsea.gene_set_enrichment(adata,
                                               marker_key="rank_genes_louvain_filtered",
                                               organism="human",
                                               method=method,
                                               inplace=False)
    assert in_uns(adata_mod, ['sctoolbox', 'gsea'])
    assert not in_uns(adata, ['sctoolbox', 'gsea'])
    assert 'enrichment_table' in adata_mod.uns['sctoolbox']['gsea']
    assert isinstance(adata_mod.uns['sctoolbox']['gsea']['enrichment_table'], pd.DataFrame)
    assert len(adata_mod.uns['sctoolbox']['gsea']['enrichment_table'].columns) > 0
    assert set(res_col).issubset(set(adata_mod.uns['sctoolbox']['gsea']['enrichment_table'].columns))


def test_fail_gene_set_enrichment(adata):
    """Test if invalid marker key is caught by enrichr_marker_genes."""

    with pytest.raises(KeyError):
        tools.gsea.gene_set_enrichment(adata,
                                       marker_key="invalid",
                                       organism="human")

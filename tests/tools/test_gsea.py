"""Test tools/gsea.py functions."""

import pytest
import os
import scanpy as sc
import sctoolbox.tools as tools
import gseapy as gp
import pandas as pd


@pytest.fixture
def adata():
    """Load and returns an anndata object."""
    f = os.path.join(os.path.dirname(__file__), '..', 'data', "adata.h5ad")

    obj = sc.read_h5ad(f)

    # add cluster column
    def repeat_items(list, count):
        """
        Repeat list until size reached.

        https://stackoverflow.com/a/54864336/19870975
        """
        return list * (count // len(list)) + list[:(count % len(list))]

    obj.obs["cluster"] = repeat_items([f"cluster {i}" for i in range(10)], len(obj))
    
    tools.marker_genes.run_rank_genes(obj, "cluster")

    return obj


def test_enrichr_marker_genes(adata):
    """Test enrichr_marker_genes."""

    organism = "human"

    # download a library or read a .gmt file
    go_mf = gp.get_library(name="GO_Biological_Process_2023", organism=organism)
    # list of all genes as background
    flat_list = set([item for sublist in go_mf.values() for item in sublist])

    result = tools.gsea.enrichr_marker_genes(adata,
                                             marker_key="rank_genes_cluster_filtered",
                                             gene_sets=go_mf,
                                             organism=organism,
                                             background=flat_list)

    assert isinstance(result, pd.DataFrame)


def test_fail_enrichr_marker_genes(adata):
    """Test if invalid marker key is caught by enrichr_marker_genes"""

    organism = "human"

    ## download a library or read a .gmt file
    go_mf = gp.get_library(name="GO_Biological_Process_2023", organism=organism)
    # list of all genes as background
    flat_list = set([item for sublist in go_mf.values() for item in sublist])
    
    with pytest.raises(KeyError):
        tools.gsea.enrichr_marker_genes(adata,
                                        marker_key="invalid",
                                        gene_sets=go_mf,
                                        organism=organism,
                                        background=flat_list)

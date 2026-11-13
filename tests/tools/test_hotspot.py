"""Test hotspot functions."""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import sctoolbox.tools.hotspot as hs_tools


# ----------------------------- FIXTURES ------------------------------- #


@pytest.fixture
def adata_hotspot(adata_raw_small):
    """Provide raw PBMC3k counts prepared for a Hotspot run.

    Parameters
    ----------
    adata_raw_small : anndata.AnnData
        Shared ``300 x 3000`` raw PBMC3k fixture from ``tests/conftest.py``.

    Returns
    -------
    anndata.AnnData
        AnnData object with the counts in ``layers["raw"]`` and ``obsm["X_pca"]``.
    """
    adata = adata_raw_small
    adata.layers["raw"] = adata.X.copy()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10)

    return adata


# ------------------------------ TESTS --------------------------------- #


def test_run_hotspot_modules(adata_hotspot):
    """Test that run_hotspot creates modules and module scores."""
    results = hs_tools.run_hotspot(adata_hotspot,
                                   layer_key="raw",
                                   latent_obsm_key="X_pca",
                                   gene_selection_method="top_n_z",
                                   top_n=300,
                                   min_gene_threshold=5,
                                   n_neighbors=15,
                                   jobs=1,
                                   compute_modules=True,
                                   compute_module_scores=True)

    hs = results["hotspot"]
    selected_genes = results["selected_genes"]

    assert isinstance(hs.modules, pd.Series)
    assert hs.modules.index.equals(selected_genes)
    assert (hs.modules != -1).any()

    assert isinstance(hs.linkage, np.ndarray)
    assert hs.linkage.shape == (len(selected_genes) - 1, 4)

    assert len(results["modules"]) > 0
    assert results["modules"].equals(hs.modules)

    module_scores = results["module_scores"]
    assert isinstance(module_scores, pd.DataFrame)
    assert not module_scores.empty
    assert module_scores.shape[0] == adata_hotspot.n_obs

    assert "hotspot_module" in adata_hotspot.var.columns
    assert any(c.startswith("Hotspot module ") for c in adata_hotspot.obs.columns)

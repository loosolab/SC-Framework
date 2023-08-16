import pytest
import scanpy as sc
import scipy

import sctoolbox.utils.adata as ad


@pytest.fixture(scope="session")
def adata_icxg():
    """Load and returns an cellxgene incompatible anndata object."""

    # has .X of type numpy.array
    obj = sc.datasets.pbmc3k_processed()

    # make 8-digit hex colors
    obj.uns["louvain_colors"] = [e + "00" for e in obj.uns["louvain_colors"]]

    # add Int32 column to .obs and .var
    obj.obs["Int32"] = 1
    obj.obs["Int32"] = obj.obs["Int32"].astype("Int32")
    obj.var["Int32"] = 1
    obj.var["Int32"] = obj.var["Int32"].astype("Int32")

    return obj


def test_prepare_for_cellxgene(adata_icxg):
    """Test the prepare_for_cellxgene function."""

    obs_names = list(adata_icxg.obs.columns)
    var_names = list(adata_icxg.var.columns)

    cxg_adata = ad.prepare_for_cellxgene(
        adata_icxg,
        keep_obs=obs_names[1:],
        keep_var=var_names[1:],
        rename_obs={e: e.upper() for e in obs_names[1:]},
        rename_var={e: e.upper() for e in var_names[1:]},
        inplace=False
    )

    # obs are removed
    assert len(obs_names) > len(cxg_adata.obs.columns)

    # obs are renamed
    assert all(e.upper() in cxg_adata.obs.columns for e in obs_names[1:])

    # no Int32 in obs
    assert all(cxg_adata.obs.dtypes != "Int32")

    # var are removed
    assert len(var_names) > len(cxg_adata.var.columns)

    # var are renamed
    assert all(e.upper() in cxg_adata.var.columns for e in var_names[1:])

    # no Int32 in var
    assert all(cxg_adata.var.dtypes != "Int32")

    # .X is sparse float32
    assert scipy.sparse.isspmatrix(cxg_adata.X)
    assert cxg_adata.X.dtype == "float32"

    # colors are stored as 6-digit hex code
    for key in cxg_adata.uns.keys():
        if key.endswith('colors'):
            assert all(len(c) <= 7 for c in cxg_adata.uns[key])

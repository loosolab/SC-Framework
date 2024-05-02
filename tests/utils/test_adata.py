"""Test adata.py functions."""

import pytest
import scanpy as sc
import scipy
import os

import sctoolbox.utilities as utils


@pytest.fixture
def adata():
    """Load and returns an anndata object."""
    # remove "utils"-folder (last folder) from path 
    f = os.path.join("/".join(os.path.dirname(__file__).split("/")[:-1]), 'data', "adata.h5ad") 

    return sc.read_h5ad(f)


def test_get_adata_subsets(adata):
    """Test if adata subsets are returned correctly."""

    subsets = utils.get_adata_subsets(adata, "group")

    for group, sub_adata in subsets.items():
        assert sub_adata.obs["group"][0] == group
        assert sub_adata.obs["group"].nunique() == 1


def test_save_h5ad(adata):
    """Test if h5ad file is saved correctly."""

    path = "test.h5ad"
    utils.save_h5ad(adata, path)

    assert os.path.isfile(path)
    os.remove(path)  # clean up after tests


@pytest.fixture(scope="session")
def adata_icxg():
    """Load and returns an cellxgene incompatible anndata object."""

    # has .X of type numpy.array
    obj = sc.datasets.pbmc3k_processed()

    # make broken colormap
    obj.obs["broken_louvain"] = obj.obs["louvain"]
    obj.uns["broken_louvain_colors"] = obj.uns["louvain_colors"][1:]

    # make 8-digit hex colors
    obj.uns["louvain_colors"] = [e + "00" for e in obj.uns["louvain_colors"]]

    # add colors that don't match a .obs column
    obj.uns["no_match_colors"] = obj.uns["louvain_colors"]

    # add Int32 column to .obs and .var
    obj.obs["Int32"] = 1
    obj.obs["Int32"] = obj.obs["Int32"].astype("Int32")
    obj.var["Int32"] = 1
    obj.var["Int32"] = obj.var["Int32"].astype("Int32")

    return obj


def test_add_uns_info(adata):
    """Test if add_uns_info works on both string and list keys."""

    utils.add_uns_info(adata, "akey", "info")

    assert "akey" in adata.uns["sctoolbox"]
    assert adata.uns["sctoolbox"]["akey"] == "info"

    utils.add_uns_info(adata, ["upper", "lower"], "info")
    assert "upper" in adata.uns["sctoolbox"]
    assert adata.uns["sctoolbox"]["upper"]["lower"] == "info"

    utils.add_uns_info(adata, ["upper", "lower"], "info2", how="append")
    assert adata.uns["sctoolbox"]["upper"]["lower"] == ["info", "info2"]


def test_prepare_for_cellxgene(adata_icxg):
    """Test the prepare_for_cellxgene function."""

    obs_names = list(adata_icxg.obs.columns)
    var_names = list(adata_icxg.var.columns)

    cxg_adata = utils.prepare_for_cellxgene(
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

    # broken color mapping is fixed also checks if renaming worked
    assert len(cxg_adata.uns["BROKEN_LOUVAIN_colors"]) == len(set(cxg_adata.obs["BROKEN_LOUVAIN"]))

    # colors are stored as 6-digit hex code
    for key in cxg_adata.uns.keys():
        if key.endswith('colors'):
            assert all(len(c) <= 7 for c in cxg_adata.uns[key])

    # check if redundant colors are deleted
    assert "no_match_colors" not in cxg_adata.uns.keys()


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("embedding_names", [["umap", "pca", "tsne"], ["invalid"]])
def test_prepare_cellxgene_emb(adata_icxg, inplace, embedding_names):
    """Test inplace and embedding check."""
    adata = adata_icxg.copy() if inplace else adata_icxg

    if "invalid" in embedding_names:
        with pytest.raises(ValueError):
            utils.prepare_for_cellxgene(adata, embedding_names=embedding_names)
    else:
        out = utils.prepare_for_cellxgene(
            adata,
            embedding_names=embedding_names,
            keep_obs=[adata.obs.columns[0]],
            keep_var=[adata.var.columns[0]],
            inplace=inplace
        )

        if inplace:
            assert len(adata.obs.columns) == 1
            assert len(adata.var.columns) == 1
            assert len(adata_icxg.obs.columns) > 1
            assert len(adata_icxg.var.columns) > 1
        else:
            assert len(out.obs.columns) == 1
            assert len(out.var.columns) == 1
            assert len(adata.obs.columns) > 1
            assert len(adata.var.columns) > 1

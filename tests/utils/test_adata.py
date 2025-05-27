"""Test adata.py functions."""

import pytest
import scanpy as sc
import scipy
import os
import numpy as np

import sctoolbox.utils.adata as utils


# --------------------------- FIXTURES ------------------------------ #


@pytest.fixture
def adata1():
    """Load scanpy moignard15 adata."""
    return sc.datasets.moignard15()


@pytest.fixture
def adata2():
    """Load scanpy processed pbmc3k adata."""
    return sc.datasets.pbmc3k_processed()


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """Return adata object with 3 groups."""

    adata = sc.AnnData(np.random.randint(0, 100, (100, 100)))
    adata.obs["group"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])

    return adata


# --------------------------- TESTS ------------------------------ #


def test_get_adata_subsets(adata):
    """Test if adata subsets are returned correctly."""

    subsets = utils.get_adata_subsets(adata, "group")

    for group, sub_adata in subsets.items():
        assert sub_adata.obs["group"][0] == group
        assert sub_adata.obs["group"].nunique() == 1


@pytest.mark.parametrize("raw", [True, False])
def test_save_and_load_h5ad(adata, raw, caplog):
    """Test if h5ad file is saved correctly. Then test loading."""
    path = "test.h5ad"

    # add raw layer
    if raw:
        adata = adata.copy()  # copy to avoid overwriting the adata (side-effects)
        adata.raw = adata

    try:
        utils.save_h5ad(adata, path)

        assert os.path.isfile(path)

        loaded = utils.load_h5ad(path)

        assert isinstance(loaded, sc.AnnData)

        # assume the last record is the warning
        log_rec = caplog.records[-1]
        assert raw == (log_rec.levelname == "WARNING" and log_rec.message.startswith("Found AnnData.raw!"))

    finally:
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

    # add layer
    obj.layers["layer"] = obj.X.copy()

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


@pytest.mark.parametrize("key", ["a", ["a", "b", "c"]])
def test_in_uns(adata, key):
    """Test in_uns success."""
    if isinstance(key, str):
        assert not utils.in_uns(adata, key)

        adata.uns[key] = "placeholder"

        assert utils.in_uns(adata, key)
    else:
        assert not utils.in_uns(adata, ["sctoolbox"] + key)

        utils.add_uns_info(adata=adata, key=key, value="placeholder")

        assert utils.in_uns(adata, ["sctoolbox"] + key)


@pytest.mark.parametrize("key", ["a", ["a", "b", "c"]])
def test_get_uns(adata, key):
    """Test get_uns success."""
    if isinstance(key, str):
        adata.uns[key] = "placeholder"
        res = utils.get_uns(adata, key)
    else:
        utils.add_uns_info(adata=adata, key=key, value="placeholder")
        res = utils.get_uns(adata, ['sctoolbox'] + key)
    assert res == "placeholder"


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


def test_prepare_cellxgene_delete(adata2):
    """Test delete_obs and delete_var parameters."""
    with pytest.raises(ValueError):
        utils.prepare_for_cellxgene(adata2, delete_obs=[], keep_obs=[])
    with pytest.raises(ValueError):
        utils.prepare_for_cellxgene(adata2, delete_var=[], keep_var=[])

    obs_cols = list(adata2.obs.columns)[:-1]
    var_cols = list(adata2.var.columns)[:-1]
    prepare_out = utils.prepare_for_cellxgene(adata2,
                                              delete_obs=[adata2.obs.columns[-1]],
                                              delete_var=[adata2.var.columns[-1]],
                                              inplace=False)
    assert obs_cols == list(prepare_out.obs.columns)
    assert var_cols == list(prepare_out.var.columns)


@pytest.mark.parametrize("layer", ["layer", None, "invalid"])
def test_prepare_cellxgene_layer(adata_icxg, layer):
    """Test layer parameter."""
    if layer == "invalid":
        with pytest.raises(ValueError):
            utils.prepare_for_cellxgene(adata_icxg, layer=layer)
    else:
        utils.prepare_for_cellxgene(adata_icxg, layer=layer)


@pytest.mark.parametrize("adatas,label", [(["adata1", "adata2"], "list"), ({"a": "adata1", "b": "adata2"}, "dict")])
def test_concadata(adatas, label, request):
    """Test the concadata function."""
    # enable fixture in parametrize https://engineeringfordatascience.com/posts/pytest_fixtures_with_parameterize/
    if isinstance(adatas, list):
        adatas = [request.getfixturevalue(a) for a in adatas]
    elif isinstance(adatas, dict):
        adatas = {k: request.getfixturevalue(v) for k, v in adatas.items()}

    total_obs = sum(a.shape[0] for a in (adatas if label == "list" else adatas.values()))
    total_var = len({v for a in (adatas if label == "list" else adatas.values()) for v in a.var.index})

    result = utils.concadata(adatas, label=label)

    # assert the correct amount of obs
    assert total_obs == result.shape[0]
    # assert the correct amounf of var
    assert total_var == result.shape[1]

    # batch column exists and is correct
    assert label in result.obs.columns
    assert len(set(result.obs[label])) == len(adatas)
    if label == "dict":
        assert all(result.obs[label].isin([k]).any() for k in adatas.keys())

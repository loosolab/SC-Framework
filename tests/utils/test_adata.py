"""Test adata.py functions."""

import pytest
import scanpy as sc

import sctoolbox.utils.adata as ad

# --------------------------- Fixtures ------------------------------ #


@pytest.fixture
def adata1():
    """Load scanpy moignard15 adata."""
    return sc.datasets.moignard15()


@pytest.fixture
def adata2():
    """Load scanpy processed pbmc3k adata."""
    return sc.datasets.pbmc3k_processed()

# --------------------------- Tests --------------------------------- #


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

    result = ad.concadata(adatas, label=label)

    # assert the correct amount of obs
    assert total_obs == result.shape[0]
    # assert the correct amounf of var
    assert total_var == result.shape[1]

    # batch column exists and is correct
    assert label in result.obs.columns
    assert len(set(result.obs[label])) == len(adatas)
    if label == "dict":
        assert all(result.obs[label].isin([k]).any() for k in adatas.keys())

"""Test functions to assemble adata objects."""

import os
import pytest
import sctoolbox.utils.assemblers as assemblers
import scanpy as sc

# --------------------------- Fixtures ------------------------------ #


@pytest.fixture()
def h5ad_file1():
    """Return path to h5ad file."""
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'adata.h5ad')


@pytest.fixture()
def h5ad_file2():
    """Return path to h5ad file."""
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'scsa', 'adata_scsa.h5ad')

# --------------------------- Tests --------------------------------- #


@pytest.mark.parametrize("files", [
    "h5ad_file1",
    ["h5ad_file1", "h5ad_file2"],
    {"a": "h5ad_file1", "b": "h5ad_file2"}
])
def test_from_h5ad(files, request):
    """Test the from_h5ad function."""
    # enable fixture in parametrize https://engineeringfordatascience.com/posts/pytest_fixtures_with_parameterize/
    if isinstance(files, list):
        files = [request.getfixturevalue(f) for f in files]
    elif isinstance(files, dict):
        files = {k: request.getfixturevalue(v) for k, v in files.items()}
    else:
        files = request.getfixturevalue(files)

    assert isinstance(assemblers.from_h5ad(files), sc.AnnData)

"""Test utility functions."""
import sctoolbox.utils as utils


def test_is_notebook():
    """Test if the function is run in a notebook."""

    boolean = utils.jupyter._is_notebook()
    assert boolean is False  # testing environment is not notebook

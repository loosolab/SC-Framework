"""Test checker functions."""

import sctoolbox.checker as ch


def test_in_range():
    """Test if int is in given range."""
    assert ch.in_range(value=100, limits=(1, 1000))

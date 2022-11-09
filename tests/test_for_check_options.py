import pytest
import sctoolbox.checker as ch


@pytest.mark.parametrize("answer,opts,result", [("y", ["y", "yes", "n", "no"], True), ("mouse", ["y", "yes"], False), ({}, ["y", "n"], False)])
def test_check_options(answer, opts, result):
    """ Check if answer is correctly replied to. """
    assert ch.check_options(answer, opts) == result
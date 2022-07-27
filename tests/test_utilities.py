import pytest
import sctoolbox.utilities as utils

@pytest.fixture
def berries():
    return ["blueberry", "strawberry", "blackberry"]


def test_longest_common_suffix(berries):

    suffix = utils.longest_common_suffix(berries)
    assert suffix == "berry"


def test_remove_prefix():

    strings = ["abcd", "abce", "abcf"]
    noprefix = [utils.remove_prefix(s, "abc") for s in strings]
    assert noprefix == ["d", "e", "f"]


def test_remove_suffix(berries):

    nosuffix = [utils.remove_suffix(s, "berry") for s in berries]
    assert nosuffix == ["blue", "straw", "black"]


def test_split_list(berries):

    split = utils.split_list(berries, 2)
    assert split == [["blueberry", "blackberry"], ["strawberry"]]

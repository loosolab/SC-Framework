import pytest
import numpy as np
import sctoolbox.utilities as utils

@pytest.fixture
def berries():
    return ["blueberry", "strawberry", "blackberry"]


arr_ints = np.random.randint(10, size=(10,10))
arr_ints2 = arr_ints.astype(float)
arr_floats = np.random.rand(10, 10)

@pytest.mark.parametrize("arr, boolean", [(arr_ints, True), (arr_ints2, True), (arr_floats, False)])
def test_is_integer_array(arr, boolean):
    assert utils.is_integer_array(arr) is boolean


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

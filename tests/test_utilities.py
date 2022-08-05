import pytest
import os
import numpy as np
import sctoolbox.utilities as utils

@pytest.fixture
def berries():
    return ["blueberry", "strawberry", "blackberry"]


arr_ints = np.random.randint(10, size=(10,10))
arr_ints2 = arr_ints.astype(float)
arr_floats = np.random.rand(10, 10)

@pytest.mark.parametrize("arr,boolean", [(arr_ints, True), (arr_ints2, True), (arr_floats, False)])
def test_is_integer_array(arr, boolean):

    result = utils.is_integer_array(arr)
    assert result == boolean


def test_clean_flanking_strings():

    paths = ["path_a.txt", "path_b.txt", "path_c.txt"]
    cleaned = utils.clean_flanking_strings(paths)

    assert cleaned == ["a", "b", "c"]


def test_longest_common_suffix(berries):
    suffix = utils.longest_common_suffix(berries)
    assert suffix == "berry"


def test_create_dir():
    utils.create_dir("testdir")
    assert os.path.isdir("testdir")


def test_is_notebook():
    boolean = utils._is_notebook()
    assert boolean is False  # testing environment is not notebook


def test_is_str_numeric():

    expect_true = utils.is_str_numeric("1.3")
    expect_false = utils.is_str_numeric("astring")

    assert expect_true
    assert expect_false is False


def test_check_module():
    with pytest.raises(Exception):
        utils.check_module("nonexisting_module")


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

def test_read_list_file(berries):

    path = "berries.txt"
    utils.write_list_file(berries, path)
    berries_read = utils.read_list_file(path)

    assert berries == berries_read

def test_write_list_file(berries):

    path = "berries.txt"
    utils.write_list_file(berries, path)

    assert os.path.isfile(path)

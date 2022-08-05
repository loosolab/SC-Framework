import pytest
import os
import numpy as np
import shutil
import sctoolbox.utilities as utils

@pytest.fixture
def berries():
    return ["blueberry", "strawberry", "blackberry"]


arr_ints = np.random.randint(10, size=(10, 10))
arr_ints2 = arr_ints.astype(float)
arr_floats = np.random.rand(10, 10)

@pytest.mark.parametrize("arr,boolean", [(arr_ints, True), (arr_ints2, True), (arr_floats, False)])
def test_is_integer_array(arr, boolean):
    """ Get boolean of whether an array is an integer array """

    result = utils.is_integer_array(arr)
    assert result == boolean


def test_clean_flanking_strings():
    """ Test if longest common pre- and suffix is removed. """

    paths = ["path_a.txt", "path_b.txt", "path_c.txt"]
    cleaned = utils.clean_flanking_strings(paths)

    assert cleaned == ["a", "b", "c"]


def test_longest_common_suffix(berries):
    """ Test if longest common suffix is found correctly """

    suffix = utils.longest_common_suffix(berries)
    assert suffix == "berry"


def test_create_dir():
    """ Test if the directory is created. """

    utils.create_dir("testdir")
    assert os.path.isdir("testdir")

    shutil.rmtree("testdir")  # clean up after tests


def test_is_notebook():
    """ Test if the function is run in a notebook """

    boolean = utils._is_notebook()
    assert boolean is False  # testing environment is not notebook


@pytest.mark.parametrize("str,boolean", [("1.3", True), ("astring", False)])
def test_is_str_numeric(string, boolean):
    """ Test if a string can be converted to numeric """

    result = utils.is_str_numeric(string)

    assert result == boolean


def test_check_module():
    """ Test if check_moduel raises an error for a non-existing module """

    with pytest.raises(Exception):
        utils.check_module("nonexisting_module")


def test_remove_prefix():
    """ Test if prefix is removed from a string """

    strings = ["abcd", "abce", "abcf"]
    noprefix = [utils.remove_prefix(s, "abc") for s in strings]
    assert noprefix == ["d", "e", "f"]


def test_remove_suffix(berries):
    """ Test if suffix is removed from a string """

    nosuffix = [utils.remove_suffix(s, "berry") for s in berries]
    assert nosuffix == ["blue", "straw", "black"]


def test_split_list(berries):
    """ Test if list is split correctly """

    split = utils.split_list(berries, 2)
    assert split == [["blueberry", "blackberry"], ["strawberry"]]


def test_read_list_file(berries):
    """ Test if read_list_file returns the correct list from a file """

    path = "berries.txt"
    utils.write_list_file(berries, path)
    berries_read = utils.read_list_file(path)
    os.remove(path)  # file no longer needed

    assert berries == berries_read


def test_write_list_file(berries):
    """ Test if write_list_file writes a file """

    path = "berries.txt"
    utils.write_list_file(berries, path)

    assert os.path.isfile(path)
    os.remove(path)  # clean up after tests

import pytest
import os
import numpy as np
import shutil
import pandas as pd
import sctoolbox.utilities as utils
import scanpy as sc


@pytest.fixture
def adata():
    """ Returns adata object with 3 groups """

    adata = sc.AnnData(np.random.randint(0, 100, (100, 100)))
    adata.obs["group"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])

    return adata


@pytest.fixture
def berries():
    return ["blueberry", "strawberry", "blackberry"]


@pytest.fixture
def na_dataframe():
    data = {'int': [3, 2, 1, np.nan],
            'float': [1.2, 3.4, 5.6, np.nan],
            'string': ['a', 'b', 'c', np.nan],
            'boolean': [True, False, True, np.nan],
            'category_str': ['cat1', 'cat2', 'cat3', np.nan],
            'category_num': [10, 20, 30, np.nan]}
    df = pd.DataFrame.from_dict(data)

    df['category_str'] = df['category_str'].astype('category')
    df['category_num'] = df['category_num'].astype('category')
    return df


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

    # Ensure that testdir is not already existing
    if os.path.isdir("testdir"):
        shutil.rmtree("testdir")

    # create the dir with the utils function
    utils.create_dir("testdir")
    assert os.path.isdir("testdir")

    shutil.rmtree("testdir")  # clean up after tests


def test_is_notebook():
    """ Test if the function is run in a notebook """

    boolean = utils._is_notebook()
    assert boolean is False  # testing environment is not notebook


@pytest.mark.parametrize("string,expected", [("1.3", True), ("astring", False)])
def test_is_str_numeric(string, expected):
    """ Test if a string can be converted to numeric """

    result = utils.is_str_numeric(string)

    assert result == expected


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


def test_fill_na(na_dataframe):
    """ Test if na values in dataframe are filled correctly """
    utils.fill_na(na_dataframe)
    assert not na_dataframe.isna().any().any()
    assert list(na_dataframe.iloc[3, :]) == [0.0, 0.0, '-', False, '', '']


def test_get_adata_subsets(adata):
    """ Test if adata subsets are returned correctly """

    subsets = utils.get_adata_subsets(adata, "group")

    for group, sub_adata in subsets.items():
        assert sub_adata.obs["group"][0] == group
        assert sub_adata.obs["group"].nunique() == 1


def test_remove_files():
    """ Remove files from list """

    if not os.path.isfile("afile.txt"):
        os.mknod("afile.txt")

    files = ["afile.txt", "notfound.txt"]
    utils.remove_files(files)

    assert os.path.isfile("afile.txt") is False


def test_pseudubulk_table(adata):
    """ Test if pseudobulk table is returned correctly """

    pseudobulk = utils.pseudobulk_table(adata, "group")

    assert pseudobulk.shape[0] == adata.shape[0]
    assert pseudobulk.shape[1] == 3  # number of groups

"""Test general utility functions."""

import pytest
import os
import subprocess
import numpy as np
import pandas as pd
import sctoolbox.utilities as utils


@pytest.fixture
def berries():
    """Return list of berries."""
    return ["blueberry", "strawberry", "blackberry"]


@pytest.fixture
def na_dataframe():
    """Return DataFrame with columns of multiple types containing NA."""
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


def test_run_cmd_valid():
    """Test if the command is run."""
    utils.run_cmd("echo hello world")


def test_run_cmd_invalid():
    """Check that invalid commands raise an error."""
    with pytest.raises(subprocess.CalledProcessError):
        utils.run_cmd("ecccho hello world")


def test_split_list(berries):
    """Test if list is split correctly."""

    split = utils.split_list(berries, 2)
    assert split == [["blueberry", "blackberry"], ["strawberry"]]


def test_read_list_file(berries):
    """Test if read_list_file returns the correct list from a file."""

    path = "berries.txt"
    utils.write_list_file(berries, path)
    berries_read = utils.read_list_file(path)
    os.remove(path)  # file no longer needed

    assert berries == berries_read


def test_write_list_file(berries):
    """Test if write_list_file writes a file."""

    path = "berries.txt"
    utils.write_list_file(berries, path)

    assert os.path.isfile(path)
    os.remove(path)  # clean up after tests


def test_clean_flanking_strings():
    """Test if longest common pre- and suffix is removed."""

    paths = ["path_a.txt", "path_b.txt", "path_c.txt"]
    cleaned = utils.clean_flanking_strings(paths)

    assert cleaned == ["a", "b", "c"]


def test_longest_common_suffix(berries):
    """Test if longest common suffix is found correctly."""

    suffix = utils.longest_common_suffix(berries)
    assert suffix == "berry"


def test_remove_prefix():
    """Test if prefix is removed from a string."""

    strings = ["abcd", "abce", "abcf"]
    noprefix = [utils.remove_prefix(s, "abc") for s in strings]
    assert noprefix == ["d", "e", "f"]


def test_remove_suffix(berries):
    """Test if suffix is removed from a string."""

    nosuffix = [utils.remove_suffix(s, "berry") for s in berries]
    assert nosuffix == ["blue", "straw", "black"]


@pytest.mark.parametrize("regex, result", [(".*_str",
                                            ["category_str"]),
                                           ([".*_str", ".*_num"],
                                            ["category_str", "category_num"]),
                                           (["INVALID"], [])])
def test_identify_columns(na_dataframe, regex, result):
    """Test if identify returns matching columns."""

    assert utils.identify_columns(na_dataframe, regex) == result


@pytest.mark.parametrize("array,mini,maxi", [(np.array([1, 2, 3]), 0, 1),
                                             (np.array([[1, 2, 3], [1, 2, 3]]), 1, 100),
                                             (np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]]), 1, 5)])
def test_scale_values(array, mini, maxi):
    """Test that scaled values are in given range."""
    result = utils.scale_values(array, mini, maxi)

    assert len(result) == len(array)
    if len(result.shape) == 1:
        assert all((mini <= result) & (result <= maxi))
    else:
        for i in range(len(result)):
            assert all((mini <= result[i]) & (result[i] <= maxi))

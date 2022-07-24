import sctoolbox.utilities as utils


def test_longest_common_suffix():

    strings = ["blueberry", "strawberry", "blackberry"]
    suffix = utils.longest_common_suffix(strings)
    assert suffix == "berry"


def test_remove_prefix():

    strings = ["abcd", "abce", "abcf"]
    noprefix = utils.remove_prefix(strings, "abc")
    assert noprefix == ["d", "e", "f"]


def test_remove_suffix():

    strings = ["blueberry", "strawberry", "blackberry"]
    nosuffix = utils.remove_suffix(strings, "berry")
    assert nosuffix == ["blue", "straw", "black"]

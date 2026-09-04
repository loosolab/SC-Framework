"""Test checker functions."""

import pytest
import sctoolbox.utils.checker as ch
import numpy as np
import pandas as pd
import logging
import os
import re
import sys
from contextlib import contextmanager


# --------------------------- HELPER -------------------------------- #


@contextmanager
def add_logger_handler(logger, handler):
    """Temporarily add a handler to the given logger."""
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


# --------------------------- FIXTURES ------------------------------ #


@pytest.fixture
def marker_dict(adata):
    """Return a dict of cell type markers using genes from the shared adata.

    Returns
    -------
    dict
        Dictionary of cell type markers.
    """
    genes = adata.var_names.tolist()
    return {"Celltype A": [genes[0], genes[1]],
            "Celltype B": [genes[2], 'invalid_gene'],
            "Celltype C": ['invalid_gene_1', 'invalid_gene_2']}


# --------------------------- TESTS --------------------------------- #


def test_check_module():
    """Test if check_moduel raises an error for a non-existing module."""

    with pytest.raises(Exception):
        ch.check_module("nonexisting_module")


def test_add_path():
    """Test if _add_path adds the path correctly."""
    python_exec_dir = os.path.dirname(sys.executable)  # get path to python executable

    assert python_exec_dir == ch._add_path()
    assert python_exec_dir in os.environ['PATH']

    ori_PATH = os.environ['PATH']  # save the original PATH
    os.environ['PATH'] = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'  # mock notebook like path
    assert python_exec_dir == ch._add_path()
    os.environ['PATH'] = ori_PATH  # restore the original PATH
    assert python_exec_dir in os.environ['PATH']


@pytest.mark.parametrize("string,expected", [("1.3", True), ("astring", False)])
def test_is_str_numeric(string, expected):
    """Test if a string can be converted to numeric."""

    result = ch.is_str_numeric(string)

    assert result == expected


arr_ints = np.random.randint(10, size=(10, 10))
arr_ints2 = arr_ints.astype(float)
arr_floats = np.random.rand(10, 10)


@pytest.mark.parametrize("arr,boolean", [(arr_ints, True), (arr_ints2, True), (arr_floats, False)])
def test_is_integer_array(arr, boolean):
    """Get boolean of whether an array is an integer array."""

    result = ch.is_integer_array(arr)
    assert result == boolean


def test_check_marker_lists(adata, marker_dict):
    """Test that check_marker_lists intersects lists correctly."""
    genes = adata.var_names.tolist()

    filtered_marker = ch.check_marker_lists(adata, marker_dict)

    assert filtered_marker == {"Celltype A": [genes[0], genes[1]],
                               "Celltype B": [genes[2]]}


def test_in_range() -> object:
    """Test if int is in given range."""
    assert ch.in_range(value=100, limits=(1, 1000))


def test_var_index_from_single_col(named_var_adata):
    """Test if var_index_from_single_col works correctly."""
    ch._var_index_from_single_col(named_var_adata,
                                  index_type='prefix',
                                  from_column='coordinate_col')

    # regex pattern to match the var coordinate
    coordinate_pattern = re.compile(r"(chr[0-9XYM]+)+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+")

    # match the first var index
    match = coordinate_pattern.match(named_var_adata.var.index[0])

    # check if the match is not None
    assert match


def test_var_column_to_index_coordinate_cols(adata_atac):
    """Test if var_column_to_index works correctly with coordinate columns."""
    # regex pattern to match the var coordinate
    coordinate_pattern = r"^(chr[0-9XYM]+)[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+$"

    adata = adata_atac.copy()
    adata.var = adata.var.reset_index(drop=True)

    # test if the function formats the var index correctly from the coordinate columns
    ch.var_column_to_index(adata, coordinate_cols=['chr', 'start', 'end'])

    # check if the first var index is in the correct format
    assert bool(re.fullmatch(coordinate_pattern, adata.var.index[0]))


def test_var_column_to_index(adata_atac):
    """Test if var_column_to_index works correctly."""
    adata = adata_atac.copy()
    # add string to var index
    adata.var.index = 'name_' + adata.var.index
    # test if the function formats the var index correctly
    ch.var_column_to_index(adata)

    # regex pattern to match the var coordinate
    coordinate_pattern = r"^(chr[0-9XYM]+)[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+$"

    # check if the first var index is in the correct format
    assert bool(re.fullmatch(coordinate_pattern, adata.var.index[0]))

    # prepare adata for the next test, add a column to var
    adata.var['index_copy'] = 'name_' + adata.var.index
    adata.var = adata.var.reset_index(drop=True)

    # test if the function formats the var index correctly
    ch.var_column_to_index(adata, coordinate_cols='index_copy')

    # check if the first var index is in the correct format
    assert bool(re.fullmatch(coordinate_pattern, adata.var.index[0]))


@pytest.mark.parametrize("coordinate_columns, expected", [(["chr", "start", "end"], ['chr', 'start', 'end']),  # expects any container to become a list
                                                          (("chr", "start", "end"), ['chr', 'start', 'end']),
                                                          (np.array(["chr", "start", "end"]), ['chr', 'start', 'end']),
                                                          (pd.Index(["chr", "start", "end"]), ['chr', 'start', 'end']),
                                                          (None, ['chr', 'start', 'end']),  # expects a fallback to the default names
                                                          ("coordinate_col", ['chr', 'start', 'end']),  # a single string is not a three column spec
                                                          (("chr", "start"), ValueError),  # expects a valueerror due to length
                                                          (("chr", "start", "end", "name"), ValueError)])
def test_normalize_coordinate_columns(coordinate_columns, expected):
    """Test that _normalize_coordinate_columns converts any accepted input into three column names."""

    if isinstance(expected, type):
        with pytest.raises(expected, match="length 3"):
            ch._normalize_coordinate_columns(coordinate_columns)

    else:
        assert ch._normalize_coordinate_columns(coordinate_columns) == expected


@pytest.mark.parametrize("coordinate_columns, expected", [(['chr', 'start', 'end'], True),  # expects var tables to be unchanged
                                                          (['chr', 'end', 'start'], False)])  # expects a valueerror due to format of columns
def test_validate_regions(adata_atac, coordinate_columns, expected):
    """Test if validate_regions works correctly."""

    assert ch.validate_regions(adata_atac, coordinate_columns=coordinate_columns) == expected


@pytest.mark.parametrize("n_names, expected", [(3, True),  # expects a non-list container to be honoured
                                               (2, ValueError)])  # expects the length check to be delegated
def test_validate_regions_normalization(adata_atac, n_names, expected):
    """Test that validate_regions delegates coordinate_columns to _normalize_coordinate_columns."""
    coordinate_columns = tuple(adata_atac.var.columns[:3])[:n_names]

    if isinstance(expected, type):
        with pytest.raises(expected, match="length 3"):
            ch.validate_regions(adata_atac, coordinate_columns=coordinate_columns)

    else:
        assert ch.validate_regions(adata_atac, coordinate_columns=coordinate_columns) is expected


def test_validate_regions_default(adata_atac):
    """Test that validate_regions defaults to the coordinate columns ['chr', 'start', 'end']."""

    assert ch.validate_regions(adata_atac) is True


def test_validate_regions_error(adata_atac, adata_atac_invalid, adata_atac_emptyvar):
    """Test that validate_regions raises the matching error type per invalidity."""
    coordinate_columns = list(adata_atac_invalid.var.columns[:3])

    # the default is unchanged; malformed regions are reported as False
    assert ch.validate_regions(adata_atac_invalid, coordinate_columns) is False

    # malformed regions raise a ValueError
    with pytest.raises(ValueError, match="do not contain valid genome regions"):
        ch.validate_regions(adata_atac_invalid, coordinate_columns, error=True)

    # columns absent from adata.var raise a KeyError
    with pytest.raises(KeyError, match="are not found in adata.var"):
        ch.validate_regions(adata_atac_emptyvar, error=True)

    # a valid object must not raise
    assert ch.validate_regions(adata_atac, tuple(adata_atac.var.columns[:3]), error=True) is True


def test_validate_regions_verbose(adata_atac_invalid, caplog, add_logger_handler):
    """Test that verbose=False silences the invalid region message."""
    coordinate_columns = list(adata_atac_invalid.var.columns[:3])

    with caplog.at_level(logging.INFO), add_logger_handler(ch.logger, caplog.handler):
        assert ch.validate_regions(adata_atac_invalid, coordinate_columns, verbose=False) is False
        assert "is not a valid genome region" not in caplog.text

        assert ch.validate_regions(adata_atac_invalid, coordinate_columns) is False
        assert "is not a valid genome region" in caplog.text


# TODO(0.18.0): remove together with the 'stop' acceptance in sctoolbox.utils.checker.validate_regions
def test_validate_regions_verbose_deprecated(adata_atac_stop, caplog, add_logger_handler):
    """Test that verbose=False silences the 'stop' deprecation warning."""

    with caplog.at_level(logging.INFO), add_logger_handler(ch.logger, caplog.handler):
        assert ch.validate_regions(adata_atac_stop, ["chr", "start", "stop"], verbose=False) is True
        assert "deprecated" not in caplog.text

        assert ch.validate_regions(adata_atac_stop, ["chr", "start", "stop"]) is True
        assert len([msg for _, level, msg in caplog.record_tuples if level == logging.WARNING]) == 1


def test_get_index_type():
    """Test if get_index_type works correctly."""
    start_with_name_index = "some_name-chr1:12343-76899 "
    regex = r"(chr[0-9XYM])+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+"

    assert ch._get_index_type(start_with_name_index, regex) == 'prefix'


def test_check_columns(adata_atac, adata_atac_invalid):
    """Test if check_columns works correctly."""
    assert ch.check_columns(adata_atac.var, ['chr', 'start', 'end'], error=False)
    assert ch.check_columns(adata_atac.var, 'chr', error=False)

    assert ch.validate_regions(adata_atac_invalid, ['chr', 'start', 'end']) is False

    with pytest.raises(KeyError):
        ch.check_columns(adata_atac.var, ['chr', 'start', 'end', 'name'], error=True)

    with pytest.raises(KeyError):
        ch.check_columns(adata_atac.var, 'name', error=True)


@pytest.mark.parametrize("fixture, expected", [("adata_atac", True),  # expects var tables to be unchanged
                                               ("adata_atac_emptyvar", False),
                                               # expects var tables to be changed
                                               ("adata", ValueError),
                                               # expects a valueerror due to missing columns
                                               ("adata_atac_invalid",
                                                ValueError)])  # expects a valueerror due to format of columns
def test_var_index_to_column(fixture, expected, request):
    """Test whether adata regions can be formatted (or raise an error if not)."""

    adata_orig = request.getfixturevalue(fixture)  # fix for using fixtures in parametrize
    adata_cp = adata_orig.copy()  # make a copy to avoid changing the fixture

    if isinstance(expected, type):
        with pytest.raises(expected):
            ch.var_index_to_column(adata_cp, coordinate_columns=["chr", "start", "end"])

    else:
        ch.var_index_to_column(adata_cp, coordinate_columns=["chr", "start", "end"])

        assert np.array_equal(adata_orig.var.values,
                              adata_cp.var.values) == expected  # check if the original adata was changed or not


@pytest.mark.parametrize("coordinate_columns, expected", [(("seqname", "begin", "finish"), ['seqname', 'begin', 'finish']),  # expects a non-list container to be honoured
                                                          (None, ['chr', 'start', 'end']),  # expects a fallback to the default names
                                                          (("chr", "start"), ValueError)])  # expects the length check to be delegated
def test_var_index_to_column_normalization(adata_atac_emptyvar, coordinate_columns, expected):
    """Test that var_index_to_column delegates coordinate_columns to _normalize_coordinate_columns."""

    if isinstance(expected, type):
        with pytest.raises(expected, match="length 3"):
            ch.var_index_to_column(adata_atac_emptyvar, coordinate_columns=coordinate_columns)

    else:
        ch.var_index_to_column(adata_atac_emptyvar, coordinate_columns=coordinate_columns)

        assert list(adata_atac_emptyvar.var.columns) == expected


def test_var_index_to_column_unchanged(adata_atac):
    """Test that var_index_to_column leaves valid coordinate columns untouched."""
    coordinate_columns = tuple(adata_atac.var.columns[:3])
    var_before = adata_atac.var.copy()

    ch.var_index_to_column(adata_atac, coordinate_columns=coordinate_columns)

    assert list(adata_atac.var.columns) == list(var_before.columns)
    assert np.array_equal(var_before.values, adata_atac.var.values)


def test_var_index_to_column_no_deprecation(adata_atac, caplog, add_logger_handler):
    """Test that the 'end' coordinate column does not emit a deprecation warning."""

    with caplog.at_level(logging.INFO), add_logger_handler(ch.logger, caplog.handler):
        ch.var_index_to_column(adata_atac)

    assert list(adata_atac.var.columns) == ['chr', 'start', 'end']
    assert not [msg for _, level, msg in caplog.record_tuples if level == logging.WARNING]


# TODO(0.18.0): remove together with the 'stop' acceptance in sctoolbox.utils.checker.validate_regions
@pytest.mark.parametrize("coordinate_columns, expected", [(["chr", "start", "stop"], ['chr', 'start', 'stop']),  # expects the deprecated columns to be honoured
                                                          (["chr", "start", "end"], ['chr', 'start', 'end', 'stop'])])  # expects the deprecated column to be kept alongside
def test_var_index_to_column_deprecated(adata_atac_stop, coordinate_columns, expected, caplog, add_logger_handler):
    """Test that the deprecated 'stop' coordinate column still works and warns exactly once."""

    with caplog.at_level(logging.INFO), add_logger_handler(ch.logger, caplog.handler):
        ch.var_index_to_column(adata_atac_stop, coordinate_columns=coordinate_columns)

    assert list(adata_atac_stop.var.columns) == expected

    warnings = [msg for _, level, msg in caplog.record_tuples if level == logging.WARNING]
    assert len(warnings) == 1
    assert "stop" in warnings[0] and "end" in warnings[0] and "0.18.0" in warnings[0]

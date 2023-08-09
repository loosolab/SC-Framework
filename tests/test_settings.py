"""Test settings class."""

import pytest
import os
import numpy as np
from io import StringIO
import sys

from sctoolbox.utils.adata import load_h5ad, get_adata_subsets
from sctoolbox._settings import settings, settings_from_config

adata_path = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")
config_path = os.path.join(os.path.dirname(__file__), 'data', "test_config.yaml")
config_path_nokey = os.path.join(os.path.dirname(__file__), 'data', "test_config_nokey.yaml")


@pytest.mark.parametrize("key, value",
                         [("figure_dir", "figures"),
                          ("figure_prefix", "test_"),
                          ("adata_input_dir", "data"),
                          ("threads", 4)])
def test_valid_settings(key, value):
    """Test that valid settings can be set."""
    setattr(settings, key, value)
    assert hasattr(settings, key)


def test_invalid_keys():
    """Assert if invalid settings raise ValueError or KeyError."""

    with pytest.raises(ValueError):
        settings.invalid_key = "value"

    for key in ["full_adata_input_prefix", "full_adata_output_prefix", "full_figure_prefix"]:
        with pytest.raises(KeyError):
            setattr(settings, key, "value")


@pytest.mark.parametrize("key, value",
                         [("figure_dir", 1),   # should be string
                          ("create_dirs", 1),  # should be bool
                          ("threads", "4")])
def test_invalid_values(key, value):
    """Assert that invalid values raise TypeError."""

    with pytest.raises(TypeError):
        setattr(settings, key, value)


def test_logfile_verbosity():
    """Check that info messages to stdout respect verbosity, and that all messages (including debug) are written to log file."""

    sys.stdout = mystdout = StringIO()  # for capturing stdout

    settings.reset()
    settings.verbosity = 1  # info
    settings.log_file = "test.log"

    # Read adata and run get_adata_subsets
    adata = load_h5ad(adata_path)
    adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
    _ = get_adata_subsets(adata, "condition")

    # Check stdout
    sys.stdout = sys.__stdout__
    captured = mystdout.getvalue()
    assert "[INFO]" in captured       # check that info message from load_h5ad is in written log
    assert "[DEBUG]" not in captured  # check that debug message from get_adata_subsets is NOT in written log, since verbosity=1

    settings.close_logfile()

    # Read log file
    with open(settings.log_file, "r") as f:
        log = f.read()

        assert "[INFO]" in log   # check that info message from load_h5ad is in log file
        assert "[DEBUG]" in log  # check that debug message from get_adata_subsets is in log file


@pytest.mark.parametrize("key, path", [(None, config_path_nokey), ("01", config_path)])
def test_settings_from_config(key, path):
    """Tests that the function is able to read the config yaml and and applies the settings."""
    settings_from_config(path, key=key)
    assert getattr(settings, "overwrite_log")
    assert getattr(settings, "log_file") == "pipeline_output/logs/01_log.txt"
    settings.reset()


def test_invalid_key_settings_from_config():
    """Test that apropriate Error is returned if the given key is not found in the yaml."""
    with pytest.raises(KeyError, match="Key 01 not found in config file"):
        settings_from_config(config_path_nokey, key="01")
    settings.reset()

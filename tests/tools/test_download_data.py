"""Test tools/download_data.py functions."""

import pytest
import os
import shutil
import botocore

from sctoolbox.tools import download_data


def test_run_downloaddata():
    """Test data download."""

    download_data.run_downloaddata(pattern="danioheart_atlas.h5ad")

    is_downloaded = os.path.isfile("data-sc-framework-2025/danioheart_atlas.h5ad")
    if is_downloaded:
        shutil.rmtree("data-sc-framework-2025")
    assert is_downloaded


def test_run_downloaddata_fail():
    """Test data download."""

    with pytest.raises(FileNotFoundError):
        download_data.run_downloaddata(pattern="invalid_file")

    with pytest.raises(botocore.exceptions.ClientError):
        download_data.run_downloaddata(bucket="invalid_bucket")

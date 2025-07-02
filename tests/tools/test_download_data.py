"""Test tools/download_data.py functions."""

import pytest
import os
import shutil
import botocore

from sctoolbox.tools import download_data


def test_download_dataset():
    """Test data download."""

    download_data.download_dataset("danioheart_atlas.h5ad")

    is_downloaded = os.path.isfile("data-sc-framework-2025/danioheart_atlas.h5ad")
    if is_downloaded:
        shutil.rmtree("data-sc-framework-2025")
    assert is_downloaded


def test_download_dataset_fail():
    """Test data download."""

    with pytest.raises(FileNotFoundError):
        download_data.download_dataset("invalid_file")

    with pytest.raises(botocore.exceptions.ClientError):
        download_data.download_dataset("danioheart_atlas.h5ad", bucket="invalid_bucket")

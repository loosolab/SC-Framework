"""Test tools/download_data.py functions."""

import pytest
import botocore

from sctoolbox.tools import download_data


def test_download_dataset(tmp_path):
    """Test data download."""

    download_data.download_dataset("danioheart_atlas.h5ad", download_path=str(tmp_path))
    assert (tmp_path / "danioheart_atlas.h5ad").is_file()


def test_download_dataset_fail():
    """Test data download."""

    with pytest.raises(FileNotFoundError):
        download_data.download_dataset("invalid_file")

    with pytest.raises(botocore.exceptions.ClientError):
        download_data.download_dataset("danioheart_atlas.h5ad", bucket="invalid_bucket")

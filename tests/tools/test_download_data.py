"""Test tools/download_data.py functions."""

from pathlib import Path
import pytest
import botocore
from unittest.mock import patch, Mock

from sctoolbox.tools import download_data


def _make_client_mock(keys, raise_on_bucket=None):
    """Build a stubbed boto3 S3 client mocking the calls download_dataset makes.

    Parameters
    ----------
    keys : list[str]
        Keys returned by ``list_objects_v2`` under ``Contents``.
    raise_on_bucket : Optional[str]
        If set, ``list_objects_v2`` raises ``botocore.exceptions.ClientError``
        when called with this bucket name (emulates an invalid bucket).

    Returns
    -------
    S3
        A stub S3 client exposing ``list_objects_v2``, ``download_file`` and a
        no-op ``meta.events.register``. Its class is named ``S3`` to satisfy the
        ``s3_downloader`` beartype validator.
    """
    # The class name must be "S3" to pass the s3_downloader Client validator.
    class S3:
        def __init__(self):
            self.meta = Mock()  # client.meta.events.register(...) is a no-op

        def list_objects_v2(self, Bucket):
            if raise_on_bucket is not None and Bucket == raise_on_bucket:
                raise botocore.exceptions.ClientError(
                    {'Error': {'Code': 'NoSuchBucket', 'Message': 'invalid bucket'}},
                    'ListObjectsV2')
            return {'Contents': [{'Key': key} for key in keys]}

        def download_file(self, bucket, key, target, **kwargs):
            # Mirror a real download by creating the requested target file.
            Path(target).touch()

    return S3()


def test_download_dataset(tmp_path):
    """Test data download (S3 client mocked, no network)."""

    client = _make_client_mock(keys=["danioheart_atlas.h5ad"])
    session = Mock()
    session.client.return_value = client

    with patch("sctoolbox.tools.download_data.boto3.Session", return_value=session):
        download_data.download_dataset("danioheart_atlas.h5ad", download_path=str(tmp_path))

    assert (tmp_path / "danioheart_atlas.h5ad").is_file()


def test_download_dataset_fail():
    """Test data download failure paths (S3 client mocked, no network)."""

    # Pattern matches no key in the bucket -> download_dataset raises FileNotFoundError.
    client = _make_client_mock(keys=["danioheart_atlas.h5ad"])
    session = Mock()
    session.client.return_value = client

    with patch("sctoolbox.tools.download_data.boto3.Session", return_value=session):
        with pytest.raises(FileNotFoundError):
            download_data.download_dataset("invalid_file")

    # Invalid bucket -> list_objects_v2 raises botocore ClientError.
    client = _make_client_mock(keys=["danioheart_atlas.h5ad"],
                               raise_on_bucket="invalid_bucket")
    session = Mock()
    session.client.return_value = client

    with patch("sctoolbox.tools.download_data.boto3.Session", return_value=session):
        with pytest.raises(botocore.exceptions.ClientError):
            download_data.download_dataset("danioheart_atlas.h5ad", bucket="invalid_bucket")

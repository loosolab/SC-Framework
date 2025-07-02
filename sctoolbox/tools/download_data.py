"""Tools to download data from S3 storage."""
import boto3
import botocore
import fnmatch
import warnings
from pathlib import Path

from beartype import beartype
from typing import Annotated
from beartype.vale import Is
from beartype.typing import Optional

Client = Annotated[object, Is[
    lambda obj: obj.__class__.__name__ == "S3"]]


@beartype
def s3_downloader(client: Client,
                  bucket: str,
                  download_list: list[str],
                  force: bool,
                  download_path: Optional[str] = None):
    """
    Target is the name of the folder to save files to.

    Parameters
    ----------
    client : botocore.client.S3
        boto3 Client
    bucket : str
        bucket name
    download_list : list[str]
        List of filenames in bucket
    force : bool, default False
        Force download of already exisiting files
    download_path : Optional[str]
        Download path. Creates directory with same name as bucket in current directory if None.
    """

    for s3_file in download_list:

        # Create folder Path if doesn't exists
        target_path = (Path(download_path) if download_path else Path(bucket)) / s3_file

        # Create directory if it is not already there
        if not target_path.parent.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists
        if target_path.is_file():
            # If file exists and force is off
            if force is False:
                warnings.warn("File already exists: " + str(target_path)
                              + ". Skipping download. Use force=True to overwrite.")
                continue

        # Download file
        client.download_file(bucket, s3_file, str(target_path))


@beartype
def download_dataset(pattern: str,
                     endpoint: str = "https://s3.mpi-bn.mpg.de",
                     bucket: str = "data-sc-framework-2025",
                     download_path: Optional[str] = None,
                     force: bool = False):
    """
    Download data from an S3 storage.

    Parameters
    ----------
    pattern : str
        Pattern for files to download e.g. '*.txt'
    endpoint : str, default "https://s3.mpi-bn.mpg.de"
        Link to the s3 server (default: The loosolab s3 server)
    bucket : str, default "data-sc-framework-2025"
        Name of bucket to download from
    download_path : Optional[str], default None
        Download path. Creates directory with same name as bucket in current directory if None
    force : bool, default False
        Force download of already exisiting files

    Raises
    ------
    FileNotFoundError
        If pattern does not match any file in S3 bucket
    """

    bsession = boto3.Session()

    # Create client
    client = bsession.client('s3',
                             endpoint_url=endpoint,
                             aws_access_key_id=None,
                             aws_secret_access_key=None)

    # Disable sigin with credentials for public buckets
    client.meta.events.register('choose-signer.s3.*', botocore.handlers.disable_signing)
    bucket_objects = [obj['Key'] for obj in client.list_objects_v2(Bucket=bucket)['Contents']]

    # Check if bucket has files which match the pattern
    pattern_files = fnmatch.filter(bucket_objects, pattern)
    if len(pattern_files) == 0:
        raise FileNotFoundError('Could not find file for pattern: ' + pattern)

    # Download files which match the pattern
    s3_downloader(client, bucket, pattern_files, force, download_path)

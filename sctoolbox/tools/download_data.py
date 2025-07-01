import boto3
import botocore
import fnmatch
from pathlib import Path

from beartype import beartype
from typing import Annotated
from beartype.vale import Is
from beartype.typing import Optional, Any

Client = Annotated[object, Is[
    lambda obj: obj.__class__.__name__ == "S3"]]


@beartype
def s3_downloader(
        client: Client,
        bucket: str,
        download_list: list[str],
        force:  bool,
        path: Optional[str] = None):
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
    path : Optional[str]
        Output path
    """

    for s3_file in download_list:

        #Create folder Path if doesn't exists
        target_path = (Path(path) if path else Path(bucket)) / s3_file

        #Create directory if it is not already there
        if not target_path.parent.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)

        #Check if file already exists
        if target_path.is_file():
            if force == False:    #if file exists and force is off
                continue

        #Download file
        client.download_file(bucket, s3_file, str(target_path))


@beartype
def s3_client(
    config_dict : dict,
    force :  bool = False):
    """
    Create S3 client

    Parameters
    ----------
    config_dict : dict
        Dictionary containing parameter for client:
         - endpoint
         - bucket
         - pattern
         - path
    force : bool, default False
        Force download of already exisiting files
    """

    bsession = boto3.Session()

    #Create client 
    client = bsession.client('s3',
                             endpoint_url = config_dict['endpoint'],
                             aws_access_key_id = None,
                             aws_secret_access_key = None)

    client.meta.events.register('choose-signer.s3.*', botocore.handlers.disable_signing)#disable sigin with credentials for public buckets
    bucket_objects = [obj['Key'] for obj in client.list_objects_v2(Bucket = config_dict['bucket'])['Contents']]

    #Download files
    #Check if bucket has files which match the pattern
    pattern_files = fnmatch.filter(bucket_objects, config_dict['pattern'])
    if len(pattern_files) == 0:
        raise FileNotFoundError('Could not find file for pattern: ' + config_dict['pattern'])

    #Download files which match the pattern
    s3_downloader(client, config_dict['bucket'], pattern_files, force, config_dict['path'])


@beartype
def run_downloaddata(
    endpoint: str = "https://s3.mpi-bn.mpg.de",
    bucket: str = "bcu-sc-framework-2025",
    pattern: str = "*",
    path: Optional[str] = None,
    force: bool = False):
    """
    Download data from an S3 storage.

    Parameters
    ----------
    endpoint : str, default "https://s3.mpi-bn.mpg.de"
        Link to the s3 server (default: The loosolab s3 server)
    bucket : str, default "bcu-sc-framework-2025"
        Name of bucket to download from
    pattern : str, default "*"
        Pattern for files to download e.g. '*.txt' (default: *)
    force : bool, default False
        Force download of already exisiting files
    """

    #Create config dict
    config = {"endpoint": endpoint, "bucket": bucket, "pattern": pattern, "path": path}

    #Download data using s3 client
    s3_client(config, force)

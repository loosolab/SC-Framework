"""Tools to download data from S3 storage."""
import boto3
import botocore
import fnmatch
import warnings
from pathlib import Path
import tarfile
import tqdm

from beartype import beartype
from typing import Annotated
from beartype.vale import Is
from beartype.typing import Optional

from sctoolbox._settings import settings
logger = settings.logger

Client = Annotated[object, Is[
    lambda obj: obj.__class__.__name__ == "S3"]]


@beartype
def s3_downloader(client: Client,
                  bucket: str,
                  download_list: list[str],
                  force: bool = False,
                  download_path: Optional[str] = None,
                  progress: bool = False):
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
    progress : bool, default False
        Display a progress bar.
    """

    for i, s3_file in enumerate(download_list, start=1):

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

        if progress:
            # https://stackoverflow.com/a/70263266
            # get file size
            meta_data = client.head_object(Bucket=bucket, Key=s3_file)
            size = int(meta_data.get('ContentLength', 0))  # in bytes

            # format progressbar
            with tqdm.tqdm(
                total=size,
                desc='' if len(download_list) < 2 else f'| File {i}/{len(download_list)}',
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            ) as pbar:
                client.download_file(bucket, s3_file, str(target_path), Callback=pbar.update)
        else:
            # Download file
            client.download_file(bucket, s3_file, str(target_path))


@beartype
def download_dataset(pattern: str,
                     endpoint: str = "https://s3.mpi-bn.mpg.de",
                     bucket: str = "data-sc-framework-2025",
                     download_path: Optional[str] = None,
                     force: bool = False,
                     progress: bool = False):
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
    progress : bool, default False
        Display a progress bar.

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

    # Disable sign-in with credentials for public buckets
    client.meta.events.register('choose-signer.s3.*', botocore.handlers.disable_signing)
    bucket_objects = [obj['Key'] for obj in client.list_objects_v2(Bucket=bucket)['Contents']]

    # Check if bucket has files which match the pattern
    pattern_files = fnmatch.filter(bucket_objects, pattern)
    if len(pattern_files) == 0:
        raise FileNotFoundError('Could not find file for pattern: ' + pattern)

    # Download files which match the pattern
    s3_downloader(client, bucket, pattern_files, force, download_path, progress=progress)


@beartype
def _download_and_unpack(archive_name: str, path: str | Path = ".", keep_archive: bool = False):
    """
    Download and unpack a single archive.

    Parameters
    ----------
    archive_name : str
        Name of the archive to download.
    path : str, default "."
        Download the file to this directory.
    keep_archive : bool, default False
        Whether to keep the archive file after unpacking.
    """
    bucket_path = Path("paper")

    # convert string to Path object
    if not isinstance(path, Path):
        path = Path(path)

    # download archive
    # does this have a progress bar?
    logger.info("Downloading...")
    download_dataset(pattern=str(bucket_path / archive_name),
                     bucket="data-sc-framework-2025",
                     download_path=str(path),
                     progress=True)

    # unpack archive
    # the archive is a folder named RNA that contains everything
    logger.info("Extracting archive...")
    with tarfile.open(path / bucket_path / archive_name, mode="r:gz") as tar:
        # get a list of all files in the archive
        members = tar.getmembers()

        # extract one after the other to allow a progressbar
        for member in tqdm.tqdm(members):
            tar.extract(member, path=path)

    # remove the .tar.gz file
    if not keep_archive:
        (path / bucket_path / archive_name).unlink()
        try:
            (path / bucket_path).rmdir()  # delete folder if empty
        except OSError:
            pass
        logger.info(f"Deleted {archive_name}")

# ---------------------------------------- Datasets ----------------------------------------
# Functions to download specific datasets from S3.


@beartype
def rna_fabian22(path: str | Path = ".", keep_archive: bool = False) -> None:
    """
    Download the scRNA zebrafish cranial neural crest data of `Fabian et al.`_ , which was used in the SC-Framework paper.

    .. _Fabian et al.: https://doi.org/10.1038/s41467-021-27594-w
    .. _GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178969

    .. note::
        The total size is ~260 MB.

    Downloads a folder containing one .h5ad per sample/replicate. The .h5ad files are assembled from the supplementary files of each sample provided in `GEO`_.
    The files were assembled using :func:`~sctoolbox.utils.assemblers.from_single_mtx`.

    Parameters
    ----------
    path : str, default "."
        Download the files to this directory.
    keep_archive : bool, default False
        Whether to keep the archive file after unpacking.
    """
    _download_and_unpack("RNA.tar.gz", path=path, keep_archive=keep_archive)


@beartype
def atac_pbmc10k(path: str = ".", keep_archive: bool = False) -> None:
    """
    Download the snATAC `human peripheral mononuclear cells of 10X Genomics`_ and additional data, which was used in the SC-Framework paper.

    .. _human peripheral mononuclear cells of 10X Genomics: https://www.10xgenomics.com/datasets/10k-human-pbmcs-atac-v2-chromium-controller-2-standard
    .. _source: https://doi.org/10.5281/zenodo.1491733
    .. _JASPAR: https://doi.org/10.1093/nar/gkad1059
    .. _ENSEMBL: https://doi.org/10.1093/nar/gkac958

    .. note::
        The total size is ~52 GB.

    Files/ folders originating from 10X Genomics are:
        - filtered_peak_bc_matrix
        - 10k_pbmc_ATACv2_nextgem_Chromium_Controller_fragments.tsv
        - 10k_pbmc_ATACv2_nextgem_Chromium_Controller_possorted_bam.bam
    Additional files:
        - hg38.blacklist.v2.bed (source_)
        - JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt (JASPAR_)
        - homo_sapiens.110.mainChr.fa (ENSEMBL_ 110)
        - homo_sapiens.110.genes.gtf (ENSEMBL_ 110)
        - homo_sapiens.110.promoters2000.gtf (corresponds to the genes of the above file. Assigns the 2000bp upstream of each gene as promoter.)

    Parameters
    ----------
    path : str, default "."
        Download the files to this directory.
    keep_archive : bool, default False
        Whether to keep the archive file after unpacking.
    """
    _download_and_unpack("ATAC.tar.gz", path=path, keep_archive=keep_archive)

"""Test io functions."""

import sctoolbox.utils as utils
import os


def touch_file(f):
    """Create file if not existing."""
    try:
        open(f, 'x')
    except FileExistsError:
        pass


def test_rm_tmp():
    """Test create_dir and rm_tmp success."""

    temp_dir = "tempdir"
    utils.create_dir(temp_dir)
    touch_file("tempdir/afile.gtf")
    touch_file("tempdir/tempfile1.txt")
    touch_file("tempdir/tempfile2.txt")

    # Remove tempfile in tempdir (but tempdir should still exist)
    tempfiles = ["tempdir/tempfile1.txt", "tempdir/tempfile2.txt"]
    utils.rm_tmp(temp_dir=temp_dir, temp_files=tempfiles, rm_dir=False)

    dir_exists = os.path.exists(temp_dir)
    files_removed = sum([os.path.exists(f) for f in tempfiles]) == 0

    assert dir_exists and files_removed

    # Check that tempdir is removed if it is empty
    utils.rm_tmp(temp_dir=temp_dir, rm_dir=True, force=True)
    dir_exists = os.path.exists(temp_dir)

    assert dir_exists is False

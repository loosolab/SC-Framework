"""Test io functions."""

import sctoolbox.utils as utils
import os
import pathlib
import shutil


# --------------------------- TESTS --------------------------------- #


def test_create_dir():
    """Test if the directory is created."""

    # Ensure that testdir is not already existing
    if os.path.isdir("testdir"):
        shutil.rmtree("testdir")

    # create the dir with the utils function
    utils.io.create_dir("testdir")
    assert os.path.isdir("testdir")

    shutil.rmtree("testdir")  # clean up after tests


def test_remove_files():
    """Remove files from list."""

    if not os.path.isfile("afile.txt"):
        os.mknod("afile.txt")

    files = ["afile.txt", "notfound.txt"]
    utils.io.remove_files(files)

    assert os.path.isfile("afile.txt") is False


def test_rm_tmp():
    """Test create_dir and rm_tmp success."""

    temp_dir = "tempdir"
    utils.io.create_dir(temp_dir)
    pathlib.Path("tempdir/afile.gtf").touch(exist_ok=True)
    pathlib.Path("tempdir/tempfile1.txt").touch(exist_ok=True)
    pathlib.Path("tempdir/tempfile2.txt").touch(exist_ok=True)

    # Remove tempfile in tempdir (but tempdir should still exist)
    tempfiles = ["tempdir/tempfile1.txt", "tempdir/tempfile2.txt"]
    utils.io.rm_tmp(temp_dir=temp_dir, temp_files=tempfiles, rm_dir=False)

    dir_exists = os.path.exists(temp_dir)
    files_removed = sum([os.path.exists(f) for f in tempfiles]) == 0

    assert dir_exists and files_removed

    # Check that tempdir is removed if it is empty
    utils.io.rm_tmp(temp_dir=temp_dir, rm_dir=True, all=True)
    dir_exists = os.path.exists(temp_dir)

    assert dir_exists is False

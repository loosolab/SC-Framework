"""Test io functions."""

import sctoolbox.utils as utils
import os
import pathlib


# --------------------------- TESTS --------------------------------- #


def test_create_dir(tmp_path):
    """Test if the directory is created."""

    testdir = str(tmp_path / "testdir")

    # create the dir with the utils function
    utils.io.create_dir(testdir)
    assert os.path.isdir(testdir)


def test_rm_tmp(tmp_path):
    """Test create_dir and rm_tmp success."""

    temp_dir = str(tmp_path / "tempdir")
    utils.io.create_dir(temp_dir)
    pathlib.Path(temp_dir, "a_file.gtf").touch(exist_ok=True)
    pathlib.Path(temp_dir, "tempfile1.txt").touch(exist_ok=True)
    pathlib.Path(temp_dir, "tempfile2.txt").touch(exist_ok=True)

    # Remove tempfile in tempdir (but tempdir should still exist)
    tempfiles = [os.path.join(temp_dir, "tempfile1.txt"), os.path.join(temp_dir, "tempfile2.txt")]
    utils.io.rm_tmp(temp_dir=temp_dir, temp_files=tempfiles, rm_dir=False)

    dir_exists = os.path.exists(temp_dir)
    files_removed = sum([os.path.exists(f) for f in tempfiles]) == 0

    assert dir_exists and files_removed

    # Check that tempdir is removed if it is empty
    utils.io.rm_tmp(temp_dir=temp_dir, rm_dir=True, all=True)
    dir_exists = os.path.exists(temp_dir)

    assert dir_exists is False

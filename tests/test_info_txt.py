from unittest import mock
import pytest
import builtins
import sctoolbox.checker as checker


@mock.patch.object(builtins, "open", new_callable=mock.mock_open, read_data="/test/path/")
def test_fetch_info_path(tmpdir):
    """ Test if path is correctly returned. """
    result_path = checker.fetch_info_txt("test")

    assert result_path == '/test/path/'


def test_write_info_txt(tmpdir):
    """ Test if path is correctly written to file. """
    file = tmpdir.join("info.txt")
    checker.write_info_txt("/test/path/", tmpdir.strpath)

    assert file.read() == '/test/path/'


def test_write_info_txt_val_err():
    """ Test error for missing output directory. """
    with pytest.raises(ValueError, match="Invalid directory given."):
        checker.write_info_txt("/test/path/", "invalid_dir")


@pytest.mark.parametrize("invalid_char", [">", ":", "|", "\\", "<", "\""])
def test_write_info_txt_invalid_char(invalid_char):
    """ Test error for invalid charachters in path. """
    with pytest.raises(ValueError, match="Invalid character in directory string."):
        checker.write_info_txt("/test/path/" + invalid_char + "/", "invalid_dir")

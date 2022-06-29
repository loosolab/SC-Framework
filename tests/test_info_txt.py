from unittest import mock
import pytest
import builtins
import sctoolbox.checker as checker


@mock.patch.object(builtins, "open", new_callable=mock.mock_open, read_data="/test/path/")
def test_fetch_info_path(tmpdir):
    result_path = checker.fetch_info_txt("test")
    assert result_path == '/test/path/'

def test_write_info_txt(tmpdir):
    file = tmpdir.join("info.txt")
    checker.write_info_txt("/test/path/", tmpdir.strpath)
    assert file.read() == '/test/path/'

def test_write_info_txt_val_err():
    with pytest.raises(ValueError, match="Invalid directory given."):
        checker.write_info_txt("/test/path/", "invalid_dir")